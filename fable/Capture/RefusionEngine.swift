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

    mutating func insert(_ candidates: [CloudPoint]) {
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
                cells[key] = cell
            } else {
                cells[key] = Cell(mean: pos, color: rgb,
                                  weight: max(0.01, pt.score), bestScore: pt.score)
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

            let K = r.intrinsics.scaled(toWidth: dw, height: dh)
            let c2w = float4x4(rowMajor: r.transform)
            let sharpness = 1 / (1 + Float(r.estimatedBlurPx) / 4)
            grid.insert(unprojectStored(depth: depth, conf: conf, rgba: rgba,
                                        dw: dw, dh: dh, K: K, c2w: c2w,
                                        config: config, sharpness: sharpness))
            // mesh 頂點：投影進本幀取色。同一頂點會被多幀命中 → 由 voxel 加權平均做多視角混色。
            if !meshVertices.isEmpty {
                grid.insert(projectMesh(meshVertices, depth: depth, rgba: rgba, dw: dw, dh: dh,
                                        K: K, c2w: c2w, config: config, sharpness: sharpness))
            }
        }
        // 診斷：這條鏈上有三處會悄悄粗化解析度（融合格觸頂、匯出擇優下採樣、訓練高斯預算），
        // 而初始點距直接決定初始高斯大小（msplat 的初始 scale = 3-NN 距離）。
        // 過去完全沒有數字，訓練端看到 15cm 的初始高斯卻無從得知是哪一段造成的。
        let rawCells = grid.count
        let gridVoxel = grid.voxelSize
        let out = grid.exportPoints(target: config.exportMaxPoints,
                                    minNeighbors: config.refuseMinNeighbors)
        var msg = "Refusion: \(records.count) frames"
        if !meshVertices.isEmpty { msg += " + \(meshVertices.count) mesh verts" }
        msg += " -> \(rawCells) cells @ "
        msg += String(format: "%.3f", gridVoxel) + "m"
        if gridVoxel > config.refuseVoxelSizeM {
            let steps = Int((log2(Double(gridVoxel / config.refuseVoxelSizeM))).rounded())
            msg += String(format: " (觸頂粗化 %d 次，設定值 %.3fm)", steps, config.refuseVoxelSizeM)
        }
        msg += " -> 匯出 \(out.count) 點（上限 \(config.exportMaxPoints)）"
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
                    if z.isFinite, z > minD, z < maxD,
                       (conf?[i] ?? 2) >= minConf {
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
                            out.append(CloudPoint(x: w4.x, y: w4.y, z: w4.z,
                                                  r: rgba[px], g: rgba[px + 1], b: rgba[px + 2],
                                                  score: central * near * sharpness))
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
