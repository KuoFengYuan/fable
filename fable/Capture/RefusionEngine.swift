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
    ///
    /// 格距的成長率是**解析算出來的，不是固定 ×2**。點雲是嵌在 3D 裡的 2D 流形，
    /// 占據格數 ∝ 1/格距²，所以格距加倍等於點數砍成 1/4 —— 固定 ×2 會嚴重過衝。
    /// 實機出現過：332,251 格、上限 250,000，只需要砍 25%，結果一步跳到
    /// 4.2cm 只剩 74,836 點（砍掉 77%）。
    ///
    /// 這不只是少了點：msplat 的初始高斯尺寸就等於 3-NN 點距，
    /// 匯出端把點距放大 2 倍，初始高斯就跟著大 2 倍，密集化未必追得回來 ——
    /// 與先前修過的「重融合靜默粗化」是同一種失效，只是發生在匯出端。
    ///
    /// 改為每輪由 √(count/target) 估出需要的格距，一兩步就收斂到接近上限。
    static func stratifiedBest(_ input: [CloudPoint], startCell: Float, target: Int) -> [CloudPoint] {
        var points = input
        guard target > 0, points.count > target else { return points }
        var cellSize = startCell
        var rounds = 0
        while points.count > target, rounds < 12 {   // rounds：防呆，正常 1~3 輪就結束
            rounds += 1
            // ×1.02 留一點餘裕（估計是統計性的，剛好壓線會多跑一輪）；
            // 下限 1.03 保證一定有進展，不會卡死
            let shrink = max(1.03, (Float(points.count) / Float(target)).squareRoot() * 1.02)
            cellSize *= shrink
            var cells: [Int64: CloudPoint] = Dictionary(minimumCapacity: points.count / 2)
            for pt in points {
                guard let key = voxelKey(SIMD3<Float>(pt.x, pt.y, pt.z), size: cellSize) else { continue }
                if let old = cells[key], old.score >= pt.score { continue }
                cells[key] = pt
            }
            points = Array(cells.values)
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
        /// 是否曾收到「量測」點（LiDAR 直接反投影）。false ＝ 這格只有推論來源（ARKit mesh）撐著。
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
    /// 只有推論來源（ARKit mesh）覆蓋、LiDAR 完全沒打到的格子數 ＝ 補充來源的實際新增覆蓋
    var inferredOnlyCount: Int { cells.values.reduce(0) { $1.measured ? $0 : $0 + 1 } }

    /// - measured: true ＝ LiDAR 直接反投影（量測）；false ＝ ARKit 場景網格（推論）
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
        // 起始格距用原生 voxelSize：加粗多少交給解析步長決定。
        // 先前預設 ×2 等於還沒開始就先砍掉 4 倍的點。
        return PointCloudMath.stratifiedBest(points, startCell: voxelSize, target: target)
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
        // 位姿在進來之前就已經定案（ARKit ＋ 錨點修正，必要時再加 BA ——
        // 見 CaptureController.processScan）。這裡只負責融合。
        //
        // 先前這裡還有一條「以融合點雲為固定結構做幾何對齊」的路徑，已刪除：
        // 它用 voxel 佔用建立對應，而離線測試證明那在最該修正的方向（平面法向）
        // 完全收不到訊號 —— 沿法向偏 3cm 時 0/40000 點有對應。留著只是死碼。
        // 重投影誤差沒有這個盲點，那條路走 BundleAdjuster。

        var grid = FusedVoxelGrid(voxelSize: config.refuseVoxelSizeM, maxCells: config.refuseMaxCells)
        let depthDir = sessionDir.appendingPathComponent("depth", isDirectory: true)
        let imagesDir = sessionDir.appendingPathComponent("images", isDirectory: true)
        let total = max(1, records.count)
        // 分段計時：先前只能靠估算猜哪一段慢，實際數字才有依據。
        // tProduce 與 tInsert 一定要分開 —— 前者可以平行、後者不行（共享 grid），
        // 先前兩者混在同一個 tUnproject 裡，等於看不出並行化的上限在哪。
        var tProduce: Double = 0, tInsert: Double = 0, tMesh: Double = 0, tDecode: Double = 0

        /// 一幀的產出。純函式、不碰共享狀態 —— 所以可以平行跑。
        struct FrameYield {
            var measured: [CloudPoint] = []
            var mesh: [CloudPoint] = []
            var decodeSec: Double = 0
            var meshSec: Double = 0
        }

        func produce(_ r: FrameRecord) -> FrameYield {
            var y = FrameYield()
            // 幾何不可信的幀直接跳過：它的深度會被反投影到錯的世界座標，疊出殘影／雙層殼。
            // 殘影比破洞更糟 —— 破洞看得出來，殘影會被當成真的幾何。
            // 注意這裡**不**跳過 .demote：那些只是顏色糊，幾何來自 LiDAR、照樣可信，
            // 丟了只會白白開洞。它們改以降權併入（見下）。
            //
            // 這個檢查先前排在 JPEG 解碼**之後** —— 被排除的幀白白付了一次解碼。
            if r.blurVerdict == .drop { return y }
            guard let depthFile = r.depthFile,
                  let dw = r.depthWidth, let dh = r.depthHeight, dw > 0, dh > 0,
                  let depth = try? Data(contentsOf: depthDir.appendingPathComponent(depthFile)),
                  depth.count == dw * dh * 4 else { return y }

            var conf: [UInt8]?
            if let confFile = r.confidenceFile,
               let confData = try? Data(contentsOf: depthDir.appendingPathComponent(confFile)),
               confData.count == dw * dh {
                conf = [UInt8](confData)
            }
            let tD = Date()
            guard let rgba = decodeRGBA(url: imagesDir.appendingPathComponent(r.imageFile),
                                        width: dw, height: dh) else { return y }
            y.decodeSec = Date().timeIntervalSince(tD)

            let K = r.intrinsics.scaled(toWidth: dw, height: dh)
            let c2w = float4x4(rowMajor: r.transform)
            // 權重同時吃兩個來源：估計的幾何劣化（運動/捲簾）與實測的清晰度判定。
            // 原本只看 estimatedBlurPx，於是「相機拿得很穩但失焦」的幀拿到滿分權重，
            // 它糊掉的顏色會主導那格的加權平均 —— 這是實測清晰度才看得到的破口。
            let sharpness = 1 / (1 + Float(r.estimatedBlurPx) / 4)
            y.measured = unprojectStored(depth: depth, conf: conf, rgba: rgba,
                                         dw: dw, dh: dh, K: K, c2w: c2w,
                                         config: config, sharpness: sharpness)
            // mesh 頂點：投影進本幀取色。同一頂點會被多幀命中 → 由 voxel 加權平均做多視角混色。
            if !meshVertices.isEmpty {
                let tM = Date()
                y.mesh = projectMesh(meshVertices, depth: depth, rgba: rgba, dw: dw, dh: dh,
                                     K: K, c2w: c2w, config: config, sharpness: sharpness)
                y.meshSec = Date().timeIntervalSince(tM)
            }
            return y
        }

        // 分批：批內平行產出、批間序列插入。
        //
        // **為什麼分批而不是一次全平行。** 一次全平行要同時持有所有幀的點：
        // 54 幀 × 49152 點 × 20B ≈ 53MB，而整屋掃描的 200 幀會到 220MB ——
        // 這個 App 已經對記憶體敏感（點雲＋訓練都在同一台手機上）。
        // 分批把峰值壓在「批大小 × 每幀點數」，同時仍然吃滿核心。
        let lanes = max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 1))
        var done = 0
        var i = 0
        while i < records.count {
            let n = min(lanes, records.count - i)
            var batch = [FrameYield](repeating: FrameYield(), count: n)
            let tP = Date()
            if n == 1 {
                batch[0] = produce(records[i])
            } else {
                // 每條 lane 只寫自己那一格 → 沒有交疊，不需要鎖
                batch.withUnsafeMutableBufferPointer { buf in
                    DispatchQueue.concurrentPerform(iterations: n) { k in
                        buf[k] = produce(records[i + k])
                    }
                }
            }
            tProduce += Date().timeIntervalSince(tP)

            let tI = Date()
            for y in batch {
                grid.insert(y.measured)
                if !y.mesh.isEmpty { grid.insert(y.mesh, measured: false) }
                tDecode += y.decodeSec
                tMesh += y.meshSec
                done += 1
                progress(Double(done) / Double(total))
            }
            tInsert += Date().timeIntervalSince(tI)
            i += n
        }
        // 診斷：這條鏈上有三處會悄悄粗化解析度（融合格觸頂、匯出擇優下採樣、訓練高斯預算），
        // 而初始點距直接決定初始高斯大小（msplat 的初始 scale = 3-NN 距離）。
        // 過去完全沒有數字，訓練端看到 15cm 的初始高斯卻無從得知是哪一段造成的。
        // 產出（可平行，牆鐘時間已除以 lanes）與插入（不可平行，共享 grid）分開報。
        // 解碼/mesh 是各 lane 的 CPU 時間總和，會大於牆鐘 —— 那正是被並行吃掉的部分。
        print(String(format: "  重融合分段: 產出 %.2fs（%d 路平行；其中 CPU 時間 "
                     + "JPEG 解碼 %.2fs、mesh 投影 %.2fs）、插入 grid %.2fs（序列）、%d 幀",
                     tProduce, lanes, tDecode, tMesh, tInsert, records.count))
        let rawCells = grid.count
        let inferredOnly = grid.inferredOnlyCount
        let gridVoxel = grid.voxelSize
        let tE = Date()
        let out = grid.exportPoints(target: config.exportMaxPoints,
                                    minNeighbors: config.refuseMinNeighbors)
        print(String(format: "  匯出擇優 %.2fs（%d 格 → %d 點）",
                     Date().timeIntervalSince(tE), rawCells, out.count))
        var msg = "Refusion: \(records.count) frames"
        if !meshVertices.isEmpty { msg += " + \(meshVertices.count) mesh verts" }
        msg += " -> \(rawCells) cells @ "
        msg += String(format: "%.3f", gridVoxel) + "m"
        if gridVoxel > config.refuseVoxelSizeM {
            let steps = Int((log2(Double(gridVoxel / config.refuseVoxelSizeM))).rounded())
            msg += String(format: " (觸頂粗化 %d 次，設定值 %.3fm)", steps, config.refuseVoxelSizeM)
        }
        msg += " -> 匯出 \(out.count) 點（上限 \(config.exportMaxPoints)）"
        if inferredOnly > 0 {
            let pct = Double(inferredOnly) * 100 / Double(max(1, rawCells))
            msg += String(format: "；其中 %d 格(%.1f%%) 是 LiDAR 沒覆蓋、只靠 ARKit mesh 撐著",
                          inferredOnly, pct)
        }
        if out.count < rawCells {
            msg += String(format: "，匯出端又粗化 %.1fx",
                          (Double(rawCells) / Double(max(1, out.count))).squareRoot())
        }
        FileHandle.standardError.write(Data((msg + "\n").utf8))
        return out
    }

    /// simd_float4x4 → row-major 16（FrameRecord.transform 的格式）
    static func rowMajor(_ m: simd_float4x4) -> [Double] {
        (0..<4).flatMap { r in (0..<4).map { c in Double(m[c][r]) } }
    }

    static func float4x4(rowMajor m: [Double]) -> simd_float4x4 {
        simd_float4x4(columns: (
            SIMD4<Float>(Float(m[0]), Float(m[4]), Float(m[8]), Float(m[12])),
            SIMD4<Float>(Float(m[1]), Float(m[5]), Float(m[9]), Float(m[13])),
            SIMD4<Float>(Float(m[2]), Float(m[6]), Float(m[10]), Float(m[14])),
            SIMD4<Float>(Float(m[3]), Float(m[7]), Float(m[11]), Float(m[15]))))
    }

    // 註：先前這裡有一個 kDemotedColorWeight = 0.2，註解寫「只降顏色權重」——
    // 但 score 是**單一權重**，同時決定位置與顏色的加權平均，所以它其實也把幾何
    // 一起壓到 1/5。在覆蓋率吃緊（實機 26.4% 的格子沒有 LiDAR）的情況下這是反效果，
    // 故移除。運動模糊本來就有物理權重 1/(1+blur/4) 壓著；
    // 失焦則完全不影響幾何，本來就不該罰。
    /// 把 mesh 頂點投影進一個關鍵幀取色，並用該幀的深度圖做可見性檢核。
    /// 分數刻意壓低（×kMeshScore）：同格若有 LiDAR 直接觀測，加權平均由 LiDAR 主導；
    /// mesh 只在「關鍵幀沒拍到」的空格補洞，不會稀釋既有的良好觀測。
    private static let kMeshScore: Float = 0.25

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
