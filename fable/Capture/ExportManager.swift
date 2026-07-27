//
//  ExportManager.swift
//  fable — COLMAP binary sparse model 匯出（LichtFeld-Studio / Inria 3DGS 直讀）
//         ＋ PLY 點雲 ＋ 零依賴 zip 打包
//
//  匯出後的掃描資料夾本身就是標準 COLMAP 資料集：
//      scan_x/
//      ├── images/                  ← 訓練影像
//      └── sparse/0/
//          ├── cameras.bin          ← PINHOLE 內參（逐幀一組，吸收連續對焦的內參呼吸）
//          ├── images.bin           ← w2c 姿態（qw qx qy qz + t，OpenCV 相機慣例）
//          └── points3D.bin         ← LiDAR 彩色點雲（3DGS 初始化）
//  二進位編排與 COLMAP scripts/python/read_write_model.py 完全一致。
//

import Foundation
import simd

nonisolated extension Data {
    /// 以主機端序（iOS/macOS 皆為 little-endian，即 COLMAP 要求的端序）附加原始 bytes
    mutating func appendLE<T>(_ value: T) {
        Swift.withUnsafeBytes(of: value) { append(contentsOf: $0) }
    }
}

nonisolated enum ExportManager {

    // MARK: - COLMAP sparse model

    /// 世界上方向對齊：ARKit 世界為 +Y up，但 COLMAP/3DGS 生態多沿用 OpenCV 相機慣例、
    /// 隱含假設重力 ≈ +Y（-Y 為 up），ARKit 資料直接匯入會上下顛倒。
    /// flipWorldUp = 繞世界 X 軸轉 180°（(x,y,z)→(x,-y,-z)），對齊 COLMAP 慣例（預設開啟）。
    /// 這是剛體變換，同時作用於相機姿態與點雲、不改變重建品質，只轉正顯示方向。
    static func writeColmapSparse(records: [FrameRecord], points: [CloudPoint],
                                  to sessionDir: URL, flipWorldUp: Bool = true) throws {
        guard !records.isEmpty else { return }
        let sparseDir = sessionDir.appendingPathComponent("sparse/0", isDirectory: true)
        try FileManager.default.createDirectory(at: sparseDir, withIntermediateDirectories: true)
        try writeCamerasBin(records: records, to: sparseDir.appendingPathComponent("cameras.bin"))
        try writeImagesBin(records: records, to: sparseDir.appendingPathComponent("images.bin"),
                           flipWorldUp: flipWorldUp)
        try writePoints3DBin(points: points, to: sparseDir.appendingPathComponent("points3D.bin"),
                             flipWorldUp: flipWorldUp)
    }

    /// ARKit c2w（GL 慣例、row-major 16）→ COLMAP 儲存格式：w2c 的（四元數 wxyz、平移）。
    /// 數學：（可選）先左乘世界翻轉 diag(1,-1,-1,1)；再 c2w_cv = c2w_gl · diag(1,-1,-1)
    ///       右乘翻相機局部 Y/Z 軸；R_w2c = R_cvᵀ，t_w2c = -R_cvᵀ·t（剛體解析逆）。
    static func colmapPose(fromRowMajorC2WGL m0: [Double],
                           flipWorldUp: Bool = false) -> (q: simd_quatd, t: SIMD3<Double>) {
        var m = m0
        if flipWorldUp {          // 世界繞 X 軸 180° = 左乘 diag(1,-1,-1,1) = 負 row 1、row 2
            for i in 4..<12 { m[i] = -m[i] }
        }
        let c0 = SIMD3<Double>(m[0], m[4], m[8])
        let c1 = -SIMD3<Double>(m[1], m[5], m[9])
        let c2 = -SIMD3<Double>(m[2], m[6], m[10])
        let t = SIMD3<Double>(m[3], m[7], m[11])
        let rW2C = simd_double3x3(c0, c1, c2).transpose
        var q = simd_normalize(simd_quatd(rW2C))
        if q.real < 0 { q = simd_quatd(real: -q.real, imag: -q.imag) }   // 正規化 qw ≥ 0
        return (q, -(rW2C * t))
    }

    /// 一張影像一組 PINHOLE 內參（camera_id = image_id）。
    ///
    /// 不用「取中位數做單一相機」：那個做法只在對焦被鎖住時成立，而我們刻意讓對焦保持連續自動
    /// （鎖對焦會把整段掃描凍在起始那一刻的景深裡，離開就糊 —— 見 CameraControls.lockForScan）。
    /// 連續對焦會帶來內參呼吸（focus breathing）：鏡組移動使有效焦長變化，iPhone 主鏡由無限遠
    /// 到近距約 1~2%，在 1920 寬的畫面邊緣就是 10~19 px 的重投影誤差 —— 遠大於可以忽略的程度。
    /// 但這是**可修的**：ARKit 每幀都給了當下的內參，FrameRecord 也早就逐幀存下來了。
    /// COLMAP 格式原生支援一張影像一組相機，msplat 的 loader 也是逐影像查 camId
    /// （load_colmap.cpp: `cameras.find(img.camId)`），所以逐幀輸出的成本是零。
    /// 於是「失焦模糊」（不可修）被換成「內參變動」（完全吸收）。
    private static func writeCamerasBin(records: [FrameRecord], to url: URL) throws {
        var data = Data(capacity: records.count * 48 + 8)
        data.appendLE(UInt64(records.count))                  // num_cameras
        for r in records {
            data.appendLE(Int32(r.id))                        // camera_id = image_id
            data.appendLE(Int32(1))                           // model_id: PINHOLE
            data.appendLE(UInt64(r.intrinsics.width))
            data.appendLE(UInt64(r.intrinsics.height))
            data.appendLE(r.intrinsics.fx)
            data.appendLE(r.intrinsics.fy)
            data.appendLE(r.intrinsics.cx)
            data.appendLE(r.intrinsics.cy)
        }
        try data.write(to: url, options: [.atomic])
    }

    private static func writeImagesBin(records: [FrameRecord], to url: URL,
                                       flipWorldUp: Bool) throws {
        var data = Data(capacity: records.count * 96 + 8)
        data.appendLE(UInt64(records.count))
        for r in records {
            let (q, t) = colmapPose(fromRowMajorC2WGL: r.transform, flipWorldUp: flipWorldUp)
            data.appendLE(Int32(r.id))
            data.appendLE(q.real)
            data.appendLE(q.imag.x)
            data.appendLE(q.imag.y)
            data.appendLE(q.imag.z)
            data.appendLE(t.x)
            data.appendLE(t.y)
            data.appendLE(t.z)
            data.appendLE(Int32(r.id))                        // camera_id = image_id（逐幀內參）
            data.append(r.imageFile.data(using: .utf8)!)
            data.append(0)                                    // name 結尾 \0
            data.appendLE(UInt64(0))                          // num_points2D（無 SfM 觀測）
        }
        try data.write(to: url, options: [.atomic])
    }

    private static func writePoints3DBin(points: [CloudPoint], to url: URL,
                                         flipWorldUp: Bool) throws {
        var data = Data(capacity: points.count * 43 + 8)
        data.appendLE(UInt64(points.count))
        var pointID: UInt64 = 1
        for p in points {
            data.appendLE(pointID)
            pointID += 1
            data.appendLE(Double(p.x))
            data.appendLE(flipWorldUp ? Double(-p.y) : Double(p.y))    // 與姿態同步翻轉
            data.appendLE(flipWorldUp ? Double(-p.z) : Double(p.z))
            data.append(p.r)
            data.append(p.g)
            data.append(p.b)
            data.appendLE(Double(1.0))                        // error（無重投影資訊，設常數）
            data.appendLE(UInt64(0))                          // 空 track
        }
        try data.write(to: url, options: [.atomic])
    }

    // MARK: - PLY（快速檢視 / nerfstudio 種子點用的副本）

    static func writePLY(_ points: [CloudPoint], to url: URL) throws {
        var data = Data(capacity: points.count * 15 + 512)
        var header = "ply\nformat binary_little_endian 1.0\n"
        header += "comment fable ARKit capture (world: ARKit gravity-aligned, Y-up, meters)\n"
        header += "element vertex \(points.count)\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        data.append(header.data(using: .ascii)!)
        for p in points {
            data.appendLE(p.x)
            data.appendLE(p.y)
            data.appendLE(p.z)
            data.append(p.r)
            data.append(p.g)
            data.append(p.b)
        }
        try data.write(to: url, options: [.atomic])
    }

    // MARK: - meta / 修正後姿態 / zip

    /// 錨點修正後的姿態另存一份 jsonl（sidecar）：
    /// poses.jsonl 保留採集當下的原始 VIO 姿態，本檔為 sparse/0 實際使用的版本，
    /// tools/（arkit2gs、validate）偵測到本檔會優先讀取。
    static func writeRefinedPoses(_ records: [FrameRecord], to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        var data = Data()
        for r in records {
            data.append(try enc.encode(r))
            data.append(0x0A)
        }
        try data.write(to: url, options: [.atomic])
    }

    static func writeMeta(_ meta: SessionMeta, to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(meta).write(to: url, options: [.atomic])
    }

    /// 目錄 → .zip。利用 NSFileCoordinator 的 .forUploading 讀取選項：
    /// 系統會在協調讀取時自動把目錄壓成 zip 暫存檔（AirDrop / Files 同款機制），
    /// 免任何第三方相依。同步阻塞，請在背景 Task 呼叫。
    static func zipDirectory(_ dir: URL, to dest: URL) throws {
        var coordinatorError: NSError?
        var innerError: Error?
        NSFileCoordinator().coordinate(readingItemAt: dir, options: [.forUploading],
                                       error: &coordinatorError) { zipURL in
            do {
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: zipURL, to: dest)
            } catch {
                innerError = error
            }
        }
        if let coordinatorError { throw coordinatorError }
        if let innerError { throw innerError }
    }
}
