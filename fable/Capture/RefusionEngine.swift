//
//  RefusionEngine.swift
//  fable — 掃描後點雲重融合（Scaniverse 式「Processing」階段）
//
//  停止掃描後，以「錨點修正後的姿態」把所有關鍵幀的原始深度重新反投影，
//  做逐 voxel 加權平均融合：
//    - 深度雜訊隨觀測數 ~1/√N 收斂（LiDAR 單幀 σ≈1-2cm → 融合後 <1cm）
//    - 顏色加權平均，去除曝光閃爍與斜視取樣的色偏
//    - 輸出的 points3D 與 images.bin 的姿態完全一致（同一組修正後姿態）
//  本檔不依賴 ARKit，可在 macOS 編譯，供 scratchpad/check harness 交叉驗證。
//

import Foundation
import CoreGraphics
import ImageIO
import simd

// MARK: - 共用幾何工具

nonisolated enum PointCloudMath {

    /// 21 bits/軸 精確格子索引（±2^20 格，1cm 格距下 ≈ ±10km）
    static func voxelKey(_ p: SIMD3<Float>, size: Float) -> Int64? {
        let ix = Int64((p.x / size).rounded(.down)) &+ (1 << 20)
        let iy = Int64((p.y / size).rounded(.down)) &+ (1 << 20)
        let iz = Int64((p.z / size).rounded(.down)) &+ (1 << 20)
        let limit: Int64 = 1 << 21
        guard ix >= 0, ix < limit, iy >= 0, iy < limit, iz >= 0, iz < limit else { return nil }
        return (ix << 42) | (iy << 21) | iz
    }

    /// voxelKey 反解 → 該格中心的世界座標
    static func cellCenter(_ key: Int64, size: Float) -> SIMD3<Float> {
        let mask: Int64 = (1 << 21) - 1
        let ix = ((key >> 42) & mask) - (1 << 20)
        let iy = ((key >> 21) & mask) - (1 << 20)
        let iz = (key & mask) - (1 << 20)
        return SIMD3<Float>((Float(ix) + 0.5) * size,
                            (Float(iy) + 0.5) * size,
                            (Float(iz) + 0.5) * size)
    }

    /// 分層擇優下採樣：逐級加粗格子、每格保留最高分，直到 ≤ target。
    /// 相比全域 top-K 不會把點擠在單一區域 —— 密度均勻且每處都是最佳樣本。
    static func stratifiedBest(_ input: [CloudPoint], startCell: Float, target: Int) -> [CloudPoint] {
        var points = input
        var cellSize = startCell
        while points.count > target {
            var cells: [Int64: CloudPoint] = Dictionary(minimumCapacity: points.count / 4)
            for pt in points {
                guard let key = voxelKey(SIMD3<Float>(pt.x, pt.y, pt.z), size: cellSize) else { continue }
                if let old = cells[key], old.score >= pt.score { continue }
                cells[key] = pt
            }
            points = Array(cells.values)
            cellSize *= 2
        }
        return points
    }
}

// MARK: - 加權平均 voxel 融合格

nonisolated struct FusedVoxelGrid {

    struct Cell {
        var mean: SIMD3<Float>      // 加權平均位置
        var color: SIMD3<Float>     // 加權平均顏色（0-255）
        var weight: Float           // 累積權重（封頂 → 指數移動平均，晚到的好觀測仍能修正）
        var bestScore: Float
        /// 是否曾收到「量測」點（LiDAR 直接反投影）。false ＝ 這格只有推論來源（mesh/MDE）撐著。
        /// 用來量化補充來源的實際貢獻：只有 measured==false 的格子才是真的補到新覆蓋。
        /// 預設 true —— 即時預覽的 TiledFusedGrid 共用本型別但不追蹤來源。
        var measured: Bool = true
        /// 這格被「哪些方向」觀測過：16 個 bin 的 bitmask（方位 8 × 仰角 2）。
        /// 用方向多樣性而非觀測次數，是因為次數會給假綠燈——同一角度看 20 次，
        /// 視差為零、幾何完全沒被約束，但次數計量會判定為充分。
        /// 3DGS 的高斯深度/形狀靠視差約束（同 SfM 三角化），視角相依外觀靠角度多樣性，
        /// 兩者都不是次數能取代的。popcount 也天然涵蓋次數（1 次不可能有 3 個方向）。
        var dirMask: UInt16 = 0
    }

    private(set) var cells: [Int64: Cell] = [:]
    private(set) var voxelSize: Float
    private let maxCells: Int
    private let weightCap: Float = 8

    init(voxelSize: Float, maxCells: Int) {
        self.voxelSize = voxelSize
        self.maxCells = maxCells
    }

    var count: Int { cells.count }
    /// 只有推論來源（mesh/MDE）覆蓋、LiDAR 完全沒打到的格子數 ＝ 補充來源的實際新增覆蓋
    var inferredOnlyCount: Int { cells.values.reduce(0) { $1.measured ? $0 : $0 + 1 } }

    /// - measured: true ＝ LiDAR 直接反投影（量測）；false ＝ mesh / MDE（推論）
    mutating func insert(_ candidates: [CloudPoint], measured: Bool = true) {
        for pt in candidates {
            let pos = SIMD3<Float>(pt.x, pt.y, pt.z)
            guard let key = PointCloudMath.voxelKey(pos, size: voxelSize) else { continue }
            let rgb = SIMD3<Float>(Float(pt.r), Float(pt.g), Float(pt.b))
            if var cell = cells[key] {
                let w = max(0.01, pt.score)
                let total = cell.weight + w
                cell.mean += (pos - cell.mean) * (w / total)
                cell.color += (rgb - cell.color) * (w / total)
                cell.weight = min(total, weightCap)
                cell.bestScore = max(cell.bestScore, pt.score)
                cell.measured = cell.measured || measured
                cells[key] = cell
            } else {
                cells[key] = Cell(mean: pos, color: rgb,
                                  weight: max(0.01, pt.score), bestScore: pt.score,
                                  measured: measured)
                if cells.count >= maxCells { coarsen() }
            }
        }
    }

    /// 觸頂自動粗化：voxel ×2、加權合併 —— 長掃描記憶體有界且不停止收點
    private mutating func coarsen() {
        voxelSize *= 2
        var merged: [Int64: Cell] = Dictionary(minimumCapacity: cells.count / 4)
        for cell in cells.values {
            guard let key = PointCloudMath.voxelKey(cell.mean, size: voxelSize) else { continue }
            if var m = merged[key] {
                let total = m.weight + cell.weight
                m.mean += (cell.mean - m.mean) * (cell.weight / total)
                m.color += (cell.color - m.color) * (cell.weight / total)
                m.weight = min(total, weightCap)
                m.bestScore = max(m.bestScore, cell.bestScore)
                m.measured = m.measured || cell.measured
                merged[key] = m
            } else {
                merged[key] = cell
            }
        }
        cells = merged
    }

    /// 26 鄰域方向（單位格offset）
    private static let neighborOffsets: [SIMD3<Float>] = {
        var out: [SIMD3<Float>] = []
        for dz in -1...1 { for dy in -1...1 { for dx in -1...1 where !(dx == 0 && dy == 0 && dz == 0) {
            out.append(SIMD3<Float>(Float(dx), Float(dy), Float(dz)))
        } } }
        return out
    }()

    /// 匯出：孤立點移除（飄浮雜點）+ 單次觀測降權，再分層擇優到 target。
    /// minNeighbors>0 時，26 鄰域占據數不足的 voxel 視為雜訊剔除。
    func exportPoints(target: Int, minNeighbors: Int) -> [CloudPoint] {
        func c8(_ f: Float) -> UInt8 { UInt8(min(255, max(0, f))) }
        let vs = voxelSize
        var points: [CloudPoint] = []
        points.reserveCapacity(cells.count)
        for cell in cells.values {
            if minNeighbors > 0 {
                var n = 0
                for o in Self.neighborOffsets {
                    let np = cell.mean + o * vs
                    if let k = PointCloudMath.voxelKey(np, size: vs), cells[k] != nil {
                        n += 1
                        if n >= minNeighbors { break }
                    }
                }
                if n < minNeighbors { continue }   // 孤立 → 飄浮雜點，丟棄
            }
            points.append(CloudPoint(x: cell.mean.x, y: cell.mean.y, z: cell.mean.z,
                                     r: c8(cell.color.x), g: c8(cell.color.y), b: c8(cell.color.z),
                                     score: cell.bestScore * min(1, cell.weight / 1.5)))
        }
        return PointCloudMath.stratifiedBest(points, startCell: voxelSize * 2, target: target)
    }
}

// MARK: - 重融合引擎

nonisolated enum RefusionEngine {

    /// 由磁碟上的關鍵幀（深度 .bin + JPEG + 修正後姿態）重建高品質融合點雲。
    /// 在背景執行緒同步執行；progress ∈ 0...1。
    /// - meshVertices: ARKit 場景重建網格的世界座標頂點（可空）。用來補上關鍵幀沒拍到的表面
    ///   —— ARKit 的 mesh 融合每一幀（60fps）的深度，而本函式只吃 ~120 個關鍵幀。
    static func refuse(records: [FrameRecord], sessionDir: URL, config: CaptureConfig,
                       meshVertices: [SIMD3<Float>] = [],
                       progress: @Sendable (Double) -> Void) -> [CloudPoint] {
        var grid = FusedVoxelGrid(voxelSize: config.refuseVoxelSizeM, maxCells: config.refuseMaxCells)
        let depthDir = sessionDir.appendingPathComponent("depth", isDirectory: true)
        let imagesDir = sessionDir.appendingPathComponent("images", isDirectory: true)
        let total = max(1, records.count)
        // 模型不在 bundle 裡就是 nil → 整條 MDE 路徑靜默略過，行為與改動前相同
        let mde: DepthDensifier? = config.useMDEHoleFill ? DepthDensifier() : nil
        var mdeAccepted = 0, mdePoints = 0

        for (i, r) in records.enumerated() {
            defer { progress(Double(i + 1) / Double(total)) }
            guard let depthFile = r.depthFile,
                  let dw = r.depthWidth, let dh = r.depthHeight, dw > 0, dh > 0,
                  let depth = try? Data(contentsOf: depthDir.appendingPathComponent(depthFile)),
                  depth.count == dw * dh * 4 else { continue }

            var conf: [UInt8]?
            if let confFile = r.confidenceFile,
               let confData = try? Data(contentsOf: depthDir.appendingPathComponent(confFile)),
               confData.count == dw * dh {
                conf = [UInt8](confData)
            }
            guard let rgba = decodeRGBA(url: imagesDir.appendingPathComponent(r.imageFile),
                                        width: dw, height: dh) else { continue }

            // 幾何不可信的幀直接跳過：它的深度會被反投影到錯的世界座標，疊出殘影／雙層殼。
            // 殘影比破洞更糟 —— 破洞看得出來，殘影會被當成真的幾何。
            // 注意這裡**不**跳過 .demote：那些只是顏色糊，幾何來自 LiDAR、照樣可信，
            // 丟了只會白白開洞。它們改以降權併入（見下）。
            if r.blurVerdict == .drop { continue }

            let K = r.intrinsics.scaled(toWidth: dw, height: dh)
            let c2w = float4x4(rowMajor: r.transform)
            // 權重同時吃兩個來源：估計的幾何劣化（運動/捲簾）與實測的清晰度判定。
            // 原本只看 estimatedBlurPx，於是「相機拿得很穩但失焦」的幀拿到滿分權重，
            // 它糊掉的顏色會主導那格的加權平均 —— 這是實測清晰度才看得到的破口。
            let sharpness = 1 / (1 + Float(r.estimatedBlurPx) / 4)
            grid.insert(unprojectStored(depth: depth, conf: conf, rgba: rgba,
                                        dw: dw, dh: dh, K: K, c2w: c2w,
                                        config: config, sharpness: sharpness))
            // mesh 頂點：投影進本幀取色。同一頂點會被多幀命中 → 由 voxel 加權平均做多視角混色。
            if !meshVertices.isEmpty {
                grid.insert(projectMesh(meshVertices, depth: depth, rgba: rgba, dw: dw, dh: dh,
                                        K: K, c2w: c2w, config: config, sharpness: sharpness),
                            measured: false)
            }
            // MDE 補洞：只在 LiDAR 無回波處產生點
            if let mde, config.useMDEHoleFill,
               let hiRGBA = decodeRGBA(url: imagesDir.appendingPathComponent(r.imageFile),
                                       width: mde.width, height: mde.height),
               let disp = mde.inferDisparity(rgba: hiRGBA) {
                let hiK = r.intrinsics.scaled(toWidth: mde.width, height: mde.height)
                let pts = mdeHoleFill(disp: disp, hiRGBA: hiRGBA, mw: mde.width, mh: mde.height,
                                      hiK: hiK, depth: depth, conf: conf, dw: dw, dh: dh,
                                      c2w: c2w, config: config, sharpness: sharpness)
                mdeAccepted += pts.isEmpty ? 0 : 1
                mdePoints += pts.count
                grid.insert(pts, measured: false)
            }
        }
        // 診斷：這條鏈上有三處會悄悄粗化解析度（融合格觸頂、匯出擇優下採樣、訓練高斯預算），
        // 而初始點距直接決定初始高斯大小（msplat 的初始 scale = 3-NN 距離）。
        // 過去完全沒有數字，訓練端看到 15cm 的初始高斯卻無從得知是哪一段造成的。
        let rawCells = grid.count
        let inferredOnly = grid.inferredOnlyCount
        let gridVoxel = grid.voxelSize
        let out = grid.exportPoints(target: config.exportMaxPoints,
                                    minNeighbors: config.refuseMinNeighbors)
        var msg = "Refusion: \(records.count) frames"
        if !meshVertices.isEmpty { msg += " + \(meshVertices.count) mesh verts" }
        if config.useMDEHoleFill {
            msg += mde == nil ? " (MDE: 模型未載入)"
                              : " + MDE \(mdePoints) 點/\(mdeAccepted) 幀"
        }
        msg += " -> \(rawCells) cells @ "
        msg += String(format: "%.3f", gridVoxel) + "m"
        if gridVoxel > config.refuseVoxelSizeM {
            let steps = Int((log2(Double(gridVoxel / config.refuseVoxelSizeM))).rounded())
            msg += String(format: " (觸頂粗化 %d 次，設定值 %.3fm)", steps, config.refuseVoxelSizeM)
        }
        msg += " -> 匯出 \(out.count) 點（上限 \(config.exportMaxPoints)）"
        if inferredOnly > 0 {
            let pct = Double(inferredOnly) * 100 / Double(max(1, rawCells))
            msg += String(format: "；其中 %d 格(%.1f%%) 是 LiDAR 沒覆蓋、只靠 mesh/MDE 撐著",
                          inferredOnly, pct)
        }
        if out.count < rawCells {
            msg += String(format: "，匯出端又粗化 %.1fx",
                          (Double(rawCells) / Double(max(1, out.count))).squareRoot())
        }
        FileHandle.standardError.write(Data((msg + "\n").utf8))
        return out
    }

    static func float4x4(rowMajor m: [Double]) -> simd_float4x4 {
        simd_float4x4(columns: (
            SIMD4<Float>(Float(m[0]), Float(m[4]), Float(m[8]), Float(m[12])),
            SIMD4<Float>(Float(m[1]), Float(m[5]), Float(m[9]), Float(m[13])),
            SIMD4<Float>(Float(m[2]), Float(m[6]), Float(m[10]), Float(m[14])),
            SIMD4<Float>(Float(m[3]), Float(m[7]), Float(m[11]), Float(m[15]))))
    }

    /// 與 PointExtractor 相同的過濾/評分（信心、範圍、飛點、中心/距離/清晰分數），
    /// 來源改為緊湊 Data（無 stride padding）
    /// MDE 補洞：用同幀 LiDAR 擬合仿射還原（1/z ≈ a·d + b），只在 LiDAR 無回波處產生點。
    ///
    /// ROI-aware 取樣（論文用語意分割，我們用更便宜的等價物）：補洞點的取樣密度 ∝ 局部
    /// RGB 梯度能量 —— 平坦牆面補稀、紋理複雜處補密。同一個訊號也用在 MRNF 的細節引導上。
    private static func mdeHoleFill(disp: [Float], hiRGBA: [UInt8], mw: Int, mh: Int,
                                    hiK: CameraIntrinsics,
                                    depth: Data, conf: [UInt8]?, dw: Int, dh: Int,
                                    c2w: simd_float4x4, config: CaptureConfig,
                                    sharpness: Float) -> [CloudPoint] {
        let minD = config.pointMinDepthM, maxD = config.pointMaxDepthM
        let minConf = config.minDepthConfidence

        return depth.withUnsafeBytes { raw -> [CloudPoint] in
            let d = raw.bindMemory(to: Float32.self)

            // ── 1) 收集監督配對：LiDAR 有效的像素 → (相對逆深度, 度量深度) ──
            @inline(__always) func lidarAt(_ mu: Int, _ mv: Int) -> Float {
                let u = mu * dw / mw, v = mv * dh / mh
                guard u >= 0, v >= 0, u < dw, v < dh else { return .nan }
                let i = v * dw + u
                let z = d[i]
                guard z.isFinite, z > minD, z < maxD, (conf?[i] ?? 2) >= minConf else { return .nan }
                return z
            }
            var pairs: [(d: Float, z: Float)] = []
            pairs.reserveCapacity(4096)
            var mv = 0
            while mv < mh {                     // 取樣擬合即可，不需全像素
                var mu = 0
                while mu < mw {
                    let z = lidarAt(mu, mv)
                    if z.isFinite { pairs.append((disp[mv * mw + mu], z)) }
                    mu += 4
                }
                mv += 4
            }
            guard let fit = DepthScaleFit.fit(pairs: pairs, minSamples: config.mdeMinSamples),
                  fit.rmse <= config.mdeMaxRMSE else { return [] }

            // ── 只做「內插」，不做「外插」──
            // 每幀各自擬合 (a,b)。有 LiDAR 的地方擬合被錨定 → 各幀一致；但在整片沒有回波的
            // 區域（窗戶/鏡面/>maxD）沒有任何約束讓不同幀一致 → 同一表面被各幀放到不同深度
            // → 落進不同 voxel → 疊成多層殼，也就是殘影。兩道局部守衛：
            //   (1) 深度必須落在本幀 LiDAR 實際觀測到的範圍內（外插到窗外就擋掉）
            //   (2) 該像素附近必須有 LiDAR 回波（＝這是被已知深度包圍的小洞，不是大片未知區）
            var zLo = Float.greatestFiniteMagnitude, zHi: Float = 0
            for p in pairs { zLo = min(zLo, p.z); zHi = max(zHi, p.z) }
            guard zHi > zLo else { return [] }
            zLo *= 0.8; zHi *= 1.2                      // 留一點外插餘裕，但不放行到無限遠

            // LiDAR 有效遮罩（深度解析度），供鄰域支撐檢核
            var valid = [Bool](repeating: false, count: dw * dh)
            for i in 0..<(dw * dh) {
                let z = d[i]
                valid[i] = z.isFinite && z > minD && z < maxD && (conf?[i] ?? 2) >= minConf
            }
            let r = config.mdeSupportRadiusPx
            @inline(__always) func hasNearbyLiDAR(_ mu: Int, _ mv: Int) -> Bool {
                let du = mu * dw / mw, dv = mv * dh / mh
                var y = max(0, dv - r)
                let yEnd = min(dh - 1, dv + r), xEnd = min(dw - 1, du + r)
                while y <= yEnd {
                    var x = max(0, du - r)
                    while x <= xEnd {
                        if valid[y * dw + x] { return true }
                        x += 1
                    }
                    y += 1
                }
                return false
            }

            // ── 2) 只在 LiDAR 無效處補點，密度依局部梯度能量 ──
            let fx = Float(hiK.fx), fy = Float(hiK.fy), cx = Float(hiK.cx), cy = Float(hiK.cy)
            @inline(__always) func luma(_ i: Int) -> Float {
                let p = i * 4
                return 0.299 * Float(hiRGBA[p]) + 0.587 * Float(hiRGBA[p + 1]) + 0.114 * Float(hiRGBA[p + 2])
            }
            var out: [CloudPoint] = []
            out.reserveCapacity(config.mdeMaxPointsPerFrame)
            // 基礎步長：讓「整幀都是洞」的最壞情況剛好落在每幀上限內
            let base = max(2, Int((Float(mw * mh) / Float(max(1, config.mdeMaxPointsPerFrame))).squareRoot()))
            var v = 1
            while v < mh - 1 && out.count < config.mdeMaxPointsPerFrame {
                var u = 1
                while u < mw - 1 && out.count < config.mdeMaxPointsPerFrame {
                    let i = v * mw + u
                    if lidarAt(u, v).isFinite { u += base; continue }   // 有 LiDAR → 不碰
                    guard let z = fit.depth(disp[i]), z > minD, z < maxD,
                          z >= zLo, z <= zHi,            // (1) 不外插到本幀沒量到的深度
                          hasNearbyLiDAR(u, v)           // (2) 必須是被已知深度包圍的小洞
                    else { u += base; continue }
                    // CV 反投影 → 翻 Y/Z 回 GL 相機系 → 世界（與 unprojectStored 同慣例）
                    let xc = (Float(u) - cx) / fx * z
                    let yc = (Float(v) - cy) / fy * z
                    let w4 = c2w * SIMD4<Float>(xc, -yc, -z, 1)
                    let px = i * 4
                    let ru = (Float(u) - cx) / Float(mw), rv = (Float(v) - cy) / Float(mh)
                    let central = 1 - min(1, (ru * ru + rv * rv).squareRoot() * 1.4) * 0.5
                    let near = 1 / (0.2 + z * z)
                    out.append(CloudPoint(x: w4.x, y: w4.y, z: w4.z,
                                          r: hiRGBA[px], g: hiRGBA[px + 1], b: hiRGBA[px + 2],
                                          score: central * near * sharpness * config.mdeScore))
                    // ROI-aware：梯度能量高 → 步長縮到一半（補密）；平坦區維持基礎步長
                    let e = abs(luma(i + 1) - luma(i)) + abs(luma(i + mw) - luma(i))
                    u += e > 12 ? max(1, base / 2) : base
                }
                v += base
            }
            return out
        }
    }

    /// 把 mesh 頂點投影進一個關鍵幀取色，並用該幀的深度圖做可見性檢核。
    /// 分數刻意壓低（×kMeshScore）：同格若有 LiDAR 直接觀測，加權平均由 LiDAR 主導；
    /// mesh 只在「關鍵幀沒拍到」的空格補洞，不會稀釋既有的良好觀測。
    private static let kMeshScore: Float = 0.25

    // 註：先前這裡有一個 kDemotedColorWeight = 0.2，註解寫「只降顏色權重」——
    // 但 score 是**單一權重**，同時決定位置與顏色的加權平均，所以它其實也把幾何
    // 一起壓到 1/5。在覆蓋率吃緊（實機 26.4% 的格子沒有 LiDAR）的情況下這是反效果，
    // 故移除。運動模糊本來就有物理權重 1/(1+blur/4) 壓著；
    // 失焦則完全不影響幾何，本來就不該罰。
    private static func projectMesh(_ verts: [SIMD3<Float>], depth: Data, rgba: [UInt8],
                                    dw: Int, dh: Int, K: CameraIntrinsics,
                                    c2w: simd_float4x4, config: CaptureConfig,
                                    sharpness: Float) -> [CloudPoint] {
        let fx = Float(K.fx), fy = Float(K.fy), cx = Float(K.cx), cy = Float(K.cy)
        let w2c = c2w.inverse
        let minD = config.pointMinDepthM, maxD = config.pointMaxDepthM
        let tol = config.meshColorDepthTolM

        return depth.withUnsafeBytes { raw -> [CloudPoint] in
            let d = raw.bindMemory(to: Float32.self)
            var out: [CloudPoint] = []
            out.reserveCapacity(verts.count / 8)
            for p in verts {
                let cam = w2c * SIMD4<Float>(p, 1)
                // 相機系為 GL 慣例（-Z 前方）→ 可視深度 z = -cam.z
                let z = -cam.z
                if !(z > minD && z < maxD) { continue }
                let u = fx * (cam.x / z) + cx
                let v = fy * (-cam.y / z) + cy
                let iu = Int(u), iv = Int(v)
                if iu < 0 || iv < 0 || iu >= dw || iv >= dh { continue }
                // 可見性：與該幀量到的深度一致才算「這一幀真的看到它」，否則是被遮擋的背面
                let zm = d[iv * dw + iu]
                if !zm.isFinite || abs(zm - z) > tol { continue }
                let px = (iv * dw + iu) * 4
                let ru = (u - cx) / Float(dw), rv = (v - cy) / Float(dh)
                let central = 1 - min(1, (ru * ru + rv * rv).squareRoot() * 1.4) * 0.5
                let near = 1 / (0.2 + z * z)
                out.append(CloudPoint(x: p.x, y: p.y, z: p.z,
                                      r: rgba[px], g: rgba[px + 1], b: rgba[px + 2],
                                      score: central * near * sharpness * kMeshScore))
            }
            return out
        }
    }

    private static func unprojectStored(depth: Data, conf: [UInt8]?, rgba: [UInt8],
                                        dw: Int, dh: Int, K: CameraIntrinsics,
                                        c2w: simd_float4x4, config: CaptureConfig,
                                        sharpness: Float) -> [CloudPoint] {
        let fx = Float(K.fx), fy = Float(K.fy), cx = Float(K.cx), cy = Float(K.cy)
        let minD = config.pointMinDepthM
        let maxD = config.pointMaxDepthM
        let minConf = config.minDepthConfidence
        let edgeRatio = config.depthEdgeRejectRatio
        let stride = max(1, config.refuseSampleStride)

        return depth.withUnsafeBytes { raw -> [CloudPoint] in
            let d = raw.bindMemory(to: Float32.self)
            var out: [CloudPoint] = []
            out.reserveCapacity((dw / stride) * (dh / stride))
            var v = 0
            while v < dh {
                var u = 0
                while u < dw {
                    let i = v * dw + u
                    let z = d[i]
                    let cv = conf?[i] ?? 2
                    if z.isFinite, z > minD, z < maxD, cv >= minConf {
                        var ok = true
                        if u + 1 < dw {
                            let dr = d[i + 1]
                            if !dr.isFinite || abs(dr - z) > z * edgeRatio { ok = false }
                        }
                        if ok, v + 1 < dh {
                            let db = d[i + dw]
                            if !db.isFinite || abs(db - z) > z * edgeRatio { ok = false }
                        }
                        if ok {
                            let xc = (Float(u) - cx) / fx * z
                            let yc = (Float(v) - cy) / fy * z
                            let w4 = c2w * SIMD4<Float>(xc, -yc, -z, 1)
                            let px = i * 4
                            let ru = (Float(u) - cx) / Float(dw)
                            let rv = (Float(v) - cy) / Float(dh)
                            let central = 1 - min(1, (ru * ru + rv * rv).squareRoot() * 1.4) * 0.5
                            let near = 1 / (0.2 + z * z)   // 反變異數：LiDAR 雜訊 ∝ z²，遠點大幅降權
                            let confW: Float = cv >= 2 ? 1 : config.mediumConfidenceWeight
                            out.append(CloudPoint(x: w4.x, y: w4.y, z: w4.z,
                                                  r: rgba[px], g: rgba[px + 1], b: rgba[px + 2],
                                                  score: central * near * sharpness * confW))
                        }
                    }
                    u += stride
                }
                v += stride
            }
            return out
        }
    }

    /// JPEG →（縮圖解碼）→ 深度解析度的緊湊 RGBA buffer。
    /// CGBitmapContext 第 0 列對應影像頂列，與深度圖的 v 方向一致
    /// （由 check harness 的顏色-座標相關性測試驗證）。
    private static func decodeRGBA(url: URL, width: Int, height: Int) -> [UInt8]? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height),
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return nil
        }
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let ok = buffer.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                return false
            }
            ctx.interpolationQuality = .medium
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return ok ? buffer : nil
    }
}
