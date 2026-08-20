//
//  FeatureTracker.swift
//  fable — 掃描時同步抽取特徵並建立跨幀對應（給 BundleAdjuster 用）
//
//  這一步是「跳過從零重建軌跡」的關鍵：ARKit 已經給了位姿、LiDAR 已經給了深度，
//  所以我們不需要三角化、也不需要全域描述子搜尋。
//
//    · 每個特徵的 3D 位置直接由 LiDAR 讀出 → 不必三角化
//    · 已知位姿 ⇒ 前一幀的特徵投影到本幀的**預期位置**可以算出來
//      → 匹配退化成 ±R 像素的局部搜尋，而不是 N×M 的描述子比對
//
//  這兩件事合起來，把 SfM 最貴的兩步（搜尋 + 三角化）整個拿掉，
//  剩下的只有「局部微調」——也就是把對應關係丟進 BA 解一次。
//
//  ── 為什麼在掃描時做而不是停止後 ────────────────────────────────
//  停止後做要重新解碼 JPEG（120 張 × ~20ms = 2.4s，比 BA 本身還貴），
//  而掃描時影像已經在記憶體裡（captureKeyframe 已經 clone 了一份 pixel buffer）。
//  抽取＋匹配約 5ms / 關鍵幀，而關鍵幀速率只有 ~2Hz —— 成本攤掉後看不見。
//
//  ── 精度考量 ──────────────────────────────────────────────────
//  Harris 響應在 stride 2 的網格上算（1920×1440 → 960×720），特徵座標仍記全解析度，
//  故量化誤差 2px。以 ~300 個特徵做最小平方，等效位姿誤差 ≈ 2/√300 ≈ 0.12px，
//  遠低於我們要打的目標（把重投影殘差壓到 <2px），所以不值得為此付全解析度的代價。
//

import Foundation
import CoreVideo
import simd

/// 一幀裡的一個特徵。patch 用來做 ZNCC 匹配 —— 引導式局部搜尋不需要旋轉不變性，
/// 而 ZNCC 對曝光變化免疫（掃描中 AE 會變），比 SSD 可靠。
nonisolated struct TrackedFeature: Sendable {
    /// 全解析度影像座標
    var u: Float
    var v: Float
    /// 該像素的 LiDAR 深度（公尺）——位姿無關，BA 每輪重算世界座標用
    var depth: Float
    /// 以抽取當時的位姿反投影出的世界座標（匹配時的引導投影用）
    var world: SIMD3<Float>
    /// 9×9 patch（在 stride 網格上取樣）
    var patch: [UInt8]
    /// 預先算好的均值與 1/範數，讓 ZNCC 不必每次重算
    var mean: Float
    var invNorm: Float
    /// 所屬 track（同一個 3D 點）。-1 = 尚未歸屬
    var trackID: Int = -1
}

/// BA 要吃的扁平觀測：某一幀看到某個 track 的影像位置
nonisolated struct FeatureObservation: Sendable {
    var frameID: Int
    var trackID: Int
    var u: Float
    var v: Float
    /// 該像素的 LiDAR 深度（公尺）。**存深度而不是世界座標**是關鍵：
    /// 世界座標是用「拍攝當時的位姿」算出來的，BA 一改位姿它就過期了；
    /// 深度與位姿無關，所以每輪都能用最新位姿重算世界座標。
    var depth: Float
}

nonisolated enum FeatureParams {
    /// Harris/Shi-Tomasi 響應的取樣步長（全解析度像素）
    static let stride = 2
    /// patch 邊長（以 stride 網格計）。9 → 全解析度覆蓋 18px，
    /// 對室內紋理是合適的尺度：小了容易誤匹配、大了對視角變化不耐
    static let patchSide = 9
    /// 特徵數上限（依響應強度取前 N 個）。
    ///
    /// **不用「每格取最強」的網格法。** 網格取極值跨幀不穩定：同一個物理角點在下一幀
    /// 若跨到隔壁格子、而那格另有更強的角點，它就完全不會被偵測到。
    /// 實機實測其後果：匹配失敗有 72% 是「半徑內無候選」——
    /// 不是投影不準，是對應的角點在新幀根本沒被抽出來。
    /// 改用非極大值抑制：局部極大值在下一幀仍然是局部極大值，與網格對齊無關。
    ///
    /// **600 → 2400，因為 600 是先天不足的。** 實機每幀只存活 259 個特徵
    /// （600 上限 → 深度過濾存活 43%），在 1920×1440 上平均最近鄰 52px，
    /// 而引導搜尋半徑只有 28px —— 位姿完美也只有 21% 的投影找得到候選。
    /// 「半徑內無候選 37%」這個最大的失配桶，主因是我們自己抽得太稀，不是位姿不準。
    ///
    /// 2400 是反推的，不是猜的：要讓平均最近鄰（Poisson 下 = 0.5/√密度）掉到
    /// 搜尋半徑以內，1920×1440 上需要 ≥882 個**存活**特徵；深度過濾的實機存活率
    /// 是 43%（600 抽出 → 259 存活），所以 NMS 要產出 ≥2051。取 2400 留一點餘裕
    /// （→ 約 1030 存活、最近鄰 26px）。tools/test_feature_index.swift 會驗這條不變式，
    /// 所以調小它時測試會直接告訴你密度論證破了。
    ///
    /// 這個數字只有在下面兩件事同時成立才付得起：
    ///   · NMS 改成真半徑檢定（格子佔用法的容量只有 1/4，撐不到 2400）
    ///   · 匹配改成空間索引（原本每個投影點掃過全部特徵，2400 個會變 1.2G 次檢查）
    static let maxFeatures = 2400
    /// 深度過濾的存活率（實機量到 259/600 ≈ 0.43）。只用於上面那條密度論證的自我檢查；
    /// 實際存活數每次掃描都會印在 stats() 的「抽取關卡」那一行。
    static let depthSurvivalRate: Double = 0.43
    /// NMS 的最小間距（stride 網格單位）。8 → 全解析度 16px，
    /// 約等於 patch 邊長，避免同一個角落被重複收好幾次
    static let nmsRadius = 8
    /// 匹配用空間索引的格邊長（stride 網格單位）。取 searchRadius/stride 向上取整，
    /// 這樣「半徑內」的候選一定落在 3×3 個格子內。
    static let bucketSide = 15
    /// Shi-Tomasi 最小特徵值門檻（8-bit 影像的經驗值）。太低會收進平坦區的雜訊
    static let minCornerResponse: Float = 120
    /// 引導搜尋半徑（全解析度像素）。ARKit 位姿殘差約十幾 px，
    /// 但深度取樣誤差也會讓引導位置偏掉，故留到 28（實測 16 太窄，匹配產出率只有 ~15/幀）
    static let searchRadius: Float = 28
    /// ZNCC 接受門檻。0.80 對「9×9 patch ＋ 6° 視角變化」偏嚴 ——
    /// patch 沒有旋轉補償，6° 就足以讓相關性掉一截。0.70 仍遠高於雜訊水準。
    static let minZNCC: Float = 0.70
    /// 深度邊緣拒絕比例：特徵所在的 3×3 深度鄰域若彼此差異超過 depth×此值就不要。
    ///
    /// **這是匹配產出率的關鍵。** 角點天生偏好長在深度不連續處（物體輪廓），
    /// 而深度圖只有 256×192 —— 一個深度像素橫跨 7.5 個影像像素。
    /// 於是「我們最愛偵測的那些點」正好是深度最不可信的地方：
    /// 取到前景或背景差之毫釐，3D 位置就差很多，引導投影跟著錯掉幾十個像素、
    /// 落在搜尋半徑外 → 匹配失敗。實機實測：2224 個特徵只長出 68 條 track。
    /// 只留深度局部平滑的角點，數量少一些但每一個都可靠。
    static let depthEdgeReject: Float = 0.03
    /// 最佳/次佳比值檢定：次佳太接近就視為模糊匹配、丟棄
    static let maxSecondBestRatio: Float = 0.9
    /// 每幀往回匹配幾幀。3~5 幀給出足夠長的 track，又不會讓成本線性爆掉
    static let matchAgainstRecent = 4
    /// track 至少要被幾幀看到才進 BA。2 幀就能約束，但 3 幀起才穩
    static let minTrackLength = 3
}

// MARK: - 抽取

nonisolated enum FeatureExtractor {

    /// 每一關卡的存活數。**這是必要的診斷，不是可有可無的統計** ——
    /// 特徵太稀是匹配率最大的失敗來源（實機「半徑內無候選」佔 37%），
    /// 而「太稀」可能卡在三個完全不同的地方：角點響應門檻、NMS 容量、深度過濾。
    /// 沒有這三個數字就只能猜哪一關該調，而我已經因為猜錯改過一次方向。
    struct ExtractStats { var candidates = 0; var afterNMS = 0; var afterDepth = 0 }

    /// 從 ARKit capturedImage 的 luma plane 抽網格化的 Shi-Tomasi 角點，
    /// 並用深度圖給每個角點一個世界座標。
    ///
    /// - depth: 該幀的深度（float32, dw×dh），conf 為信心圖（可 nil）
    /// - c2w: 該幀的 camera-to-world（ARKit GL 慣例）
    static func extract(luma pb: CVPixelBuffer,
                        depth: Data, conf: [UInt8]?, dw: Int, dh: Int,
                        K: CameraIntrinsics, c2w: simd_float4x4,
                        minDepth: Float, maxDepth: Float)
        -> (features: [TrackedFeature], stats: ExtractStats) {
        var stats = ExtractStats()
        guard CVPixelBufferGetPlaneCount(pb) >= 1 else { return ([], stats) }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else { return ([], stats) }
        let w = CVPixelBufferGetWidthOfPlane(pb, 0)
        let h = CVPixelBufferGetHeightOfPlane(pb, 0)
        let rowBytes = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let p = base.assumingMemoryBound(to: UInt8.self)

        let s = FeatureParams.stride
        let half = FeatureParams.patchSide / 2
        // 網格座標範圍：留出 patch 與差分所需的邊界
        let margin = half + 2
        let gw = w / s, gh = h / s
        guard gw > 2 * margin, gh > 2 * margin else { return ([], stats) }

        @inline(__always) func lum(_ gx: Int, _ gy: Int) -> Int {
            Int((p + gy * s * rowBytes)[gx * s])
        }

        // 收集所有「3×3 局部極大值且超過門檻」的候選，稍後做 NMS
        var cand: [(resp: Float, gx: Int, gy: Int)] = []
        cand.reserveCapacity(4096)
        // 響應先算成一整張圖，才能做局部極大值判定（逐點算無法比較鄰居）
        var resp = [Float](repeating: 0, count: gw * gh)

        var gy = margin
        while gy < gh - margin {
            var gx = margin
            while gx < gw - margin {
                // 3×3 網格窗上的結構張量（＝全解析度 6×6）
                var a: Float = 0, b: Float = 0, c: Float = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let ix = Float(lum(gx + dx + 1, gy + dy) - lum(gx + dx - 1, gy + dy))
                        let iy = Float(lum(gx + dx, gy + dy + 1) - lum(gx + dx, gy + dy - 1))
                        a += ix * ix; b += ix * iy; c += iy * iy
                    }
                }
                // Shi-Tomasi：最小特徵值（比 Harris 的 det-k·trace² 少一個要調的 k）
                let t = (a + c) * 0.5
                let d = (((a - c) * 0.5) * ((a - c) * 0.5) + b * b).squareRoot()
                resp[gy * gw + gx] = t - d
                gx += 1
            }
            gy += 1
        }

        // 3×3 局部極大值
        gy = margin + 1
        while gy < gh - margin - 1 {
            var gx = margin + 1
            while gx < gw - margin - 1 {
                let r = resp[gy * gw + gx]
                if r >= FeatureParams.minCornerResponse {
                    var isMax = true
                    for dy in -1...1 {
                        for dx in -1...1 where !(dx == 0 && dy == 0) {
                            if resp[(gy + dy) * gw + (gx + dx)] > r { isMax = false; break }
                        }
                        if !isMax { break }
                    }
                    if isMax { cand.append((r, gx, gy)) }
                }
                gx += 1
            }
            gy += 1
        }

        // 依響應由強到弱貪婪取點，彼此至少相隔 nmsRadius（空間 hash 做 O(1) 鄰域查詢）
        stats.candidates = cand.count
        cand.sort { $0.resp > $1.resp }
        let nms = FeatureParams.nmsRadius
        let nms2 = nms * nms
        // **真正的距離檢定，不是格子佔用檢定。**
        //
        // 先前只判「自己或 8 個鄰格是否已被佔用」。格邊長 = nms = 8 網格單位，
        // 所以那實際上排除到 3 格 ＝ 24 網格 ＝ 48 全解析度像素 ——
        // 遠大於 nmsRadius 想表達的 16px，可容納密度因此只有預期的 1/4
        // （2700 vs 10800）。而 maxFeatures 從 600 提到 2000 需要那個容量。
        var buckets: [Int32: [(Int, Int)]] = [:]
        var bestPos: [(Int, Int)] = []
        bestPos.reserveCapacity(FeatureParams.maxFeatures)
        for c in cand {
            if bestPos.count >= FeatureParams.maxFeatures { break }
            let cx = c.gx / nms, cy = c.gy / nms
            var clash = false
            // 格邊長等於 nms ⇒ 距離 < nms 的既有點一定在這 3×3 格之內
            search: for dy in -1...1 {
                for dx in -1...1 {
                    guard let b = buckets[Int32((cy + dy) << 16 | (cx + dx))] else { continue }
                    for (px, py) in b {
                        let ddx = px - c.gx, ddy = py - c.gy
                        if ddx * ddx + ddy * ddy < nms2 { clash = true; break search }
                    }
                }
            }
            if clash { continue }
            buckets[Int32(cy << 16 | cx), default: []].append((c.gx, c.gy))
            bestPos.append((c.gx, c.gy))
        }

        stats.afterNMS = bestPos.count

        // 取 patch + 由深度賦予世界座標
        let side = FeatureParams.patchSide
        var out: [TrackedFeature] = []
        out.reserveCapacity(bestPos.count)
        let sx = Float(dw) / Float(w), sy = Float(dh) / Float(h)
        let fx = Float(K.fx), fy = Float(K.fy), cx = Float(K.cx), cy = Float(K.cy)

        depth.withUnsafeBytes { raw in
            let dep = raw.bindMemory(to: Float32.self)
            for (gx, gy) in bestPos {
                let uFull = Float(gx * s), vFull = Float(gy * s)
                // 深度圖座標（深度解析度遠低於影像，故最近取樣即可）
                let du = Int(uFull * sx), dv = Int(vFull * sy)
                guard du >= 0, dv >= 0, du < dw, dv < dh else { continue }
                let di = dv * dw + du
                let z0 = dep[di]
                // BA 的 3D 只用 high confidence：這一步求位姿，寧可少也要準
                guard z0.isFinite, z0 > minDepth, z0 < maxDepth,
                      (conf?[di] ?? 2) >= 2 else { continue }
                // 深度邊緣拒絕：3×3 鄰域必須一致，否則這個角點的 3D 不可信（見 depthEdgeReject）
                guard du >= 1, dv >= 1, du < dw - 1, dv < dh - 1 else { continue }
                var edgeOK = true
                let tol = z0 * FeatureParams.depthEdgeReject
                for ny in -1...1 {
                    for nx in -1...1 {
                        let nz = dep[(dv + ny) * dw + (du + nx)]
                        if !nz.isFinite || abs(nz - z0) > tol { edgeOK = false; break }
                    }
                    if !edgeOK { break }
                }
                guard edgeOK else { continue }

                // **雙線性內插深度**，不用最近鄰。
                //
                // 深度圖 256×192 對影像 1920×1440 是 7.5 倍落差 —— 一個深度像素橫跨
                // 7.5 個影像像素。用最近鄰等於「拿一個 7.5px 方塊的深度代表一個
                // 次像素定位的特徵」，在傾斜表面上誤差可達公分級。
                // 實機量到深度殘差 2.0cm，遠大於 LiDAR 本身的 σ≈1cm，差額就是這個。
                // 上面的邊緣拒絕已保證局部是平滑的（3% 內），所以線性項可靠、值得內插。
                let fu = uFull * sx - Float(du), fv = vFull * sy - Float(dv)
                let z = (1 - fv) * ((1 - fu) * dep[di] + fu * dep[di + 1])
                      + fv * ((1 - fu) * dep[di + dw] + fu * dep[di + dw + 1])
                guard z.isFinite, z > minDepth, z < maxDepth else { continue }

                var patch = [UInt8](repeating: 0, count: side * side)
                var sum: Float = 0
                for py in 0..<side {
                    for px in 0..<side {
                        let val = UInt8(lum(gx + px - half, gy + py - half))
                        patch[py * side + px] = val
                        sum += Float(val)
                    }
                }
                let mean = sum / Float(side * side)
                var sq: Float = 0
                for val in patch { let d = Float(val) - mean; sq += d * d }
                guard sq > 1 else { continue }          // 平坦 patch 無法匹配
                let invNorm = 1 / sq.squareRoot()

                let xc = (uFull - cx) / fx * z
                let yc = (vFull - cy) / fy * z
                let wp = c2w * SIMD4<Float>(xc, -yc, -z, 1)
                out.append(TrackedFeature(u: uFull, v: vFull, depth: z,
                                          world: SIMD3<Float>(wp.x, wp.y, wp.z),
                                          patch: patch, mean: mean, invNorm: invNorm))
            }
        }
        stats.afterDepth = out.count
        return (out, stats)
    }

    /// ZNCC。兩個 patch 都已預先算好 mean 與 invNorm，故只剩一次點積。
    /// 對曝光變化免疫 —— 掃描中 AE 會變，SSD 在這種情況下不可靠。
    @inline(__always)
    static func zncc(_ a: TrackedFeature, _ b: TrackedFeature) -> Float {
        var acc: Float = 0
        for i in 0..<a.patch.count {
            acc += (Float(a.patch[i]) - a.mean) * (Float(b.patch[i]) - b.mean)
        }
        return acc * a.invNorm * b.invNorm
    }
}

// MARK: - 追蹤（跨幀建立 track）

/// 掃描期間逐幀累積。actor 讓它離開主執行緒 —— 不能餓死 ARKit 的 VIO。
actor FeatureTracker {

    private struct FrameFeatures {
        let frameID: Int
        let c2w: simd_float4x4
        let K: CameraIntrinsics
        var features: [TrackedFeature]
    }

    private var frames: [FrameFeatures] = []
    private var nextTrackID = 0
    private(set) var matchCount = 0
    /// 匹配失敗的原因統計 —— 產出率不足時要能指出是哪一關卡住的，
    /// 而不是只看到「BA 沒跑」
    private var attempted = 0
    private var outOfView = 0
    private var noCandidate = 0
    private var lowScore = 0
    private var ambiguous = 0
    /// 抽取三關卡的累計存活數（角點候選 → NMS → 深度過濾），用來判斷特徵稀疏卡在哪
    private var candTotal = 0, nmsTotal = 0, depthTotal = 0

    func reset() {
        frames.removeAll()
        nextTrackID = 0
        matchCount = 0
        attempted = 0; outOfView = 0; noCandidate = 0; lowScore = 0; ambiguous = 0
        candTotal = 0; nmsTotal = 0; depthTotal = 0
    }

    /// 加入一幀：抽取 → 對最近數幀做引導式匹配 → 歸入既有 track 或開新 track
    func add(frameID: Int, luma: CVPixelBuffer, depth: Data, conf: [UInt8]?,
             dw: Int, dh: Int, K: CameraIntrinsics, c2w: simd_float4x4,
             minDepth: Float, maxDepth: Float) {
        let (extracted, es) = FeatureExtractor.extract(luma: luma, depth: depth, conf: conf,
                                                       dw: dw, dh: dh, K: K, c2w: c2w,
                                                       minDepth: minDepth, maxDepth: maxDepth)
        var feats = extracted
        candTotal += es.candidates
        nmsTotal += es.afterNMS
        depthTotal += es.afterDepth
        guard !feats.isEmpty else { return }

        let w2c = c2w.inverse
        let r2 = FeatureParams.searchRadius * FeatureParams.searchRadius

        let index = Self.buildIndex(feats)

        for prev in frames.suffix(FeatureParams.matchAgainstRecent) {
            for pf in prev.features {
                attempted += 1
                // 用**已知位姿**把前一幀的 3D 特徵投影到本幀 → 預期位置
                guard let (pu, pv) = Self.project(pf.world, w2c: w2c, K: K) else {
                    outOfView += 1; continue
                }
                let (bestIdx, best, second) = Self.bestMatch(for: pf, at: (pu, pv),
                                                            in: feats, index: index)
                guard bestIdx >= 0 else { noCandidate += 1; continue }
                guard best >= FeatureParams.minZNCC else { lowScore += 1; continue }
                // 比值檢定：次佳太接近代表這區域自相似（磁磚、格紋），寧可不要
                if second > 0, second / best > FeatureParams.maxSecondBestRatio {
                    ambiguous += 1; continue
                }

                if pf.trackID >= 0 {
                    feats[bestIdx].trackID = pf.trackID
                } else {
                    // 前一幀那個特徵還沒歸屬 → 兩者一起開一個新 track
                    let id = nextTrackID
                    nextTrackID += 1
                    feats[bestIdx].trackID = id
                    if let pi = frames.indices.last(where: { frames[$0].frameID == prev.frameID }),
                       let fi = frames[pi].features.firstIndex(where: {
                           $0.u == pf.u && $0.v == pf.v && $0.trackID < 0
                       }) {
                        frames[pi].features[fi].trackID = id
                    }
                }
                matchCount += 1
            }
        }
        frames.append(FrameFeatures(frameID: frameID, c2w: c2w, K: K, features: feats))
    }

    /// 本幀特徵的空間索引：格座標 → 特徵下標。
    ///
    /// **沒有它，提高 maxFeatures 是負收益的。** 原本每個投影點都要掃過本幀全部特徵
    /// 才知道哪些落在半徑內：600 個特徵時是 78M 次距離檢查（54 幀、回溯 4 幀），
    /// 2000 個就變成 864M —— 而抽取＋匹配的預算是每關鍵幀 ~5ms。
    ///
    /// 格邊長取 ≥ searchRadius（30px vs 28px），所以半徑內的候選一定落在 3×3 格之內，
    /// 每次查詢只碰到常數個特徵，成本與 maxFeatures 幾乎無關。
    static func buildIndex(_ feats: [TrackedFeature]) -> [Int32: [Int]] {
        var index: [Int32: [Int]] = [:]
        for (i, f) in feats.enumerated() {
            let c = cell(f.u, f.v)
            index[cellKey(c.x, c.y), default: []].append(i)
        }
        return index
    }

    /// 格座標。**加了偏置讓它恆為非負**：project 允許投影落在畫面外
    /// （到 -searchRadius），而負索引在 `y << 16 | x` 的打包裡會讓符號位吃掉另一個欄位。
    /// 那種情況能不能自圓其說要推一段理，加偏置比推理便宜也比推理可靠。
    @inline(__always)
    static func cell(_ u: Float, _ v: Float) -> (x: Int, y: Int) {
        let side = Float(FeatureParams.bucketSide * FeatureParams.stride)
        return (Int((u / side).rounded(.down)) + 2, Int((v / side).rounded(.down)) + 2)
    }

    @inline(__always)
    static func cellKey(_ x: Int, _ y: Int) -> Int32 { Int32(y << 16 | x) }

    /// 在投影位置附近找最佳與次佳 ZNCC。回傳 (下標, 最佳, 次佳)；找不到候選時下標為 -1。
    ///
    /// 抽成函式是為了**可測**：索引化的搜尋若少看了某一格，症狀是「匹配數變少」而不是
    /// 崩潰 —— 而少看的那些正是原本就稀少的匹配，很可能被當成場景難度而不是 bug。
    /// tools/test_feature_index.swift 拿它跟暴力搜尋逐點對照。
    static func bestMatch(for pf: TrackedFeature, at p: (Float, Float),
                          in feats: [TrackedFeature],
                          index: [Int32: [Int]]) -> (idx: Int, best: Float, second: Float) {
        let r2 = FeatureParams.searchRadius * FeatureParams.searchRadius
        let (pu, pv) = p
        let b = cell(pu, pv)
        var bestIdx = -1
        var best: Float = -1, second: Float = -1
        for dy in -1...1 {
            for dx in -1...1 {
                guard let bucket = index[cellKey(b.x + dx, b.y + dy)] else { continue }
                for i in bucket where feats[i].trackID < 0 {
                    let f = feats[i]
                    let du = f.u - pu, dv = f.v - pv
                    if du * du + dv * dv > r2 { continue }      // 只搜半徑內的角點
                    let s = FeatureExtractor.zncc(pf, f)
                    if s > best { second = best; best = s; bestIdx = i }
                    else if s > second { second = s }
                }
            }
        }
        return (bestIdx, best, second)
    }

    /// 世界座標 → 影像座標。ARKit GL 慣例：相機看 -Z、Y 朝上，
    /// 與 RefusionEngine 的反投影 (xc, -yc, -z) 互為逆運算。
    ///
    /// **會檢查畫面邊界。** 先前只判「在相機後方」，於是投影到畫面外的特徵
    /// 全被計入「半徑內無候選」—— 匹配率的分母因此嚴重高估，
    /// 實機診斷出現過「出畫面 0、無候選 57%」這種不可能的分佈
    /// （往回比對 4 幀＝最多相隔 0.4m/24°，大量特徵理應飛出視野）。
    static func project(_ world: SIMD3<Float>, w2c: simd_float4x4,
                       K: CameraIntrinsics) -> (Float, Float)? {
        let pc = w2c * SIMD4<Float>(world, 1)
        guard pc.z < -1e-4 else { return nil }        // 在相機後方
        let d = -pc.z
        let u = Float(K.cx) + pc.x * Float(K.fx) / d
        let v = Float(K.cy) - pc.y * Float(K.fy) / d
        // 邊界留一點餘裕（搜尋半徑）：剛好在邊上的特徵仍可能匹配到畫面內的角點
        let m = FeatureParams.searchRadius
        guard u >= -m, v >= -m,
              u < Float(K.width) + m, v < Float(K.height) + m else { return nil }
        return (u, v)
    }

    /// 匯出給 BA 的觀測。只留長度足夠的 track —— 短 track 對位姿幾乎沒有約束力，
    /// 卻會把離群匹配帶進求解。
    func observations() -> [FeatureObservation] {
        var lengths: [Int: Int] = [:]
        for f in frames { for t in f.features where t.trackID >= 0 {
            lengths[t.trackID, default: 0] += 1
        } }
        var out: [FeatureObservation] = []
        for f in frames {
            for t in f.features where t.trackID >= 0 {
                guard (lengths[t.trackID] ?? 0) >= FeatureParams.minTrackLength else { continue }
                out.append(FeatureObservation(frameID: f.frameID, trackID: t.trackID,
                                              u: t.u, v: t.v, depth: t.depth))
            }
        }
        return out
    }

    /// 診斷用摘要。含匹配失敗原因分解 —— 產出率不足時必須能指出卡在哪一關，
    /// 否則只會看到「BA 沒跑」而不知道要改什麼。
    func stats() -> String {
        let obs = observations()
        let feats = frames.reduce(0) { $0 + $1.features.count }
        let perFrame = frames.isEmpty ? 0 : obs.count / frames.count
        var s = "特徵追蹤: \(frames.count) 幀 / \(feats) 特徵 / "
        s += "\(Set(obs.map(\.trackID)).count) tracks / \(obs.count) 觀測"
        s += "（每幀 \(perFrame)，BA 需要 ≥\(BundleAdjuster.kMinObsPerFrame)）"
        if attempted > 0 {
            s += String(format: "\n  匹配 %d/%d (%.0f%%)：出畫面 %d、半徑內無候選 %d、"
                        + "ZNCC 不足 %d、模糊匹配 %d",
                        matchCount, attempted, Double(matchCount) * 100 / Double(attempted),
                        outOfView, noCandidate, lowScore, ambiguous)
        }
        // 特徵稀疏卡在哪一關。附上「平均最近鄰 vs 搜尋半徑」——
        // 前者大於後者時，即使位姿完美也有大半的投影找不到候選，
        // 那時該調的是密度而不是 BA。
        if !frames.isEmpty, depthTotal > 0 {
            let perFrame = Double(depthTotal) / Double(frames.count)
            // Poisson 下的平均最近鄰距離 = 0.5/√密度
            let nn = 0.5 / (perFrame / (1920.0 * 1440)).squareRoot()
            s += String(format: "\n  抽取關卡: 角點候選 %.0f → NMS %.0f（上限 %d）→ 深度過濾 %.0f /幀"
                        + "；平均最近鄰 %.0f px vs 搜尋半徑 %.0f px%@",
                        Double(candTotal) / Double(frames.count),
                        Double(nmsTotal) / Double(frames.count),
                        FeatureParams.maxFeatures, perFrame,
                        nn, Double(FeatureParams.searchRadius),
                        nn > Double(FeatureParams.searchRadius)
                            ? " ⚠️ 特徵太稀，匹配率的上限由密度決定" : "")
        }
        return s
    }
}
