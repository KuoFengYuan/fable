//
//  MsplatTrainer.swift
//  fable — 手機端 3DGS 訓練 + 檢視（msplat）的 Swift 驅動層
//
//  MsplatSession 擁有 msplat 的 dataset / trainer 生命週期，所有 msplat 呼叫都在自己的
//  序列佇列上執行（msplat 全域狀態非執行緒安全，序列化即安全）：
//    - start(...)：setup + 阻塞式訓練迴圈（背景），逐步回報進度與可選的即時預覽。
//    - renderOrbit(...)：訓練完成後（佇列閒置）從任意環繞視角 render → 供互動檢視。
//    - close()：銷毀 trainer/dataset。
//  訓練完成後刻意「不」銷毀 trainer，模型留在記憶體以便使用者轉動檢視。
//
//  環繞相機以「訓練相機的實際姿態」為基準（同一世界 frame），繞場景質心旋轉，
//  up 取自真實相機 → 方向永遠正確、不會上下顛倒（不需知道絕對世界上方向）。
//

import Foundation
import CoreGraphics
import simd

/// 跨執行緒取消旗標（NSLock 保護，可從背景訓練迴圈輪詢）
nonisolated final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func cancel() { lock.lock(); value = true; lock.unlock() }
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return value }
}

nonisolated final class MsplatSession: @unchecked Sendable {

    /// 一張 render（RGBA8888）
    nonisolated struct PreviewImage: Sendable {
        var rgba: Data
        var width: Int
        var height: Int

        func cgImage() -> CGImage? {
            guard width > 0, height > 0, rgba.count == width * height * 4 else { return nil }
            let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
            guard let provider = CGDataProvider(data: rgba as CFData) else { return nil }
            return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                           bitmapInfo: info, provider: provider, decode: nil,
                           shouldInterpolate: true, intent: .defaultIntent)
        }
    }

    nonisolated struct Progress: Sendable {
        var iteration: Int
        var total: Int
        var splatCount: Int
        var msPerStep: Float
        var preview: PreviewImage?
    }

    enum TrainingError: LocalizedError {
        case datasetEmpty, initFailed
        var errorDescription: String? {
            switch self {
            case .datasetEmpty: return "找不到可訓練的相機/影像（COLMAP 資料為空）"
            case .initFailed: return "msplat 訓練器初始化失敗"
            }
        }
    }

    private let queue = DispatchQueue(label: "fable.msplat.session", qos: .userInitiated)
    private var dataset: UnsafeMutableRawPointer?
    private var trainer: UnsafeMutableRawPointer?
    private var numCameras = 0
    private var previewCam: Int32 = 0

    // Trackball 自由轉動狀態（主緒拖曳更新、render 時讀取；lock 保護 → 訓練中也能轉）
    private let orbitLock = NSLock()
    private var orbitQuat = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))  // 累積旋轉（identity 起始）
    private var orbitCenter = SIMD3<Float>(0, 0, 0)   // 樞軸＝物件中心
    private var baseOffset = SIMD3<Float>(0, 0, 1)    // 初始 eye − center
    private var orbitDist: Float = 1.0                // 縮放：eye→中心距離倍率（pinch 調整，<1 拉近、>1 拉遠）
    private var baseUp = SIMD3<Float>(0, -1, 0)       // 初始 up（世界垂直；若顛倒改 (0,1,0)）
    // 直式 render 內參（填滿直式螢幕、免螢幕旋轉硬湊）
    private var portraitW: Int32 = 720
    private var portraitH: Int32 = 1560
    private var portraitFx: Float = 1300
    private var portraitFy: Float = 1300

    /// 拖曳 → 以「當前視角的螢幕軸」累積旋轉（arcball，無固定 up 的 gimbal，像抓著模型翻）。
    func applyDrag(dx: Float, dy: Float) {
        orbitLock.lock(); defer { orbitLock.unlock() }
        let up = simd_normalize(orbitQuat.act(baseUp))
        let eye = orbitCenter + orbitQuat.act(baseOffset * orbitDist)
        let fwd = simd_normalize(orbitCenter - eye)
        var right = simd_cross(fwd, up)
        right = simd_length(right) > 1e-4 ? simd_normalize(right) : SIMD3<Float>(1, 0, 0)
        let k: Float = 0.007
        let qYaw = simd_quatf(angle: -dx * k, axis: up)       // 水平拖曳＝繞當前垂直軸
        let qPitch = simd_quatf(angle: -dy * k, axis: right)  // 垂直拖曳＝繞當前水平軸
        orbitQuat = simd_normalize(qYaw * qPitch * orbitQuat)
    }

    /// pinch 縮放：scale>1（手指張開）拉近、<1 拉遠。夾在合理範圍避免穿過物件或飛太遠。
    func applyZoom(_ scale: Float) {
        guard scale > 0 else { return }
        orbitLock.lock(); defer { orbitLock.unlock() }
        orbitDist = min(max(orbitDist / scale, 0.15), 6.0)
    }

    /// 雙指平移：沿當前相機的 right/up 軸移動樞軸（eye 隨之同移）→ 模型在畫面中平移。
    /// dx,dy 為螢幕點位移（dy>0 為下）。位移量以樞軸深度換算，接近 1:1 手感。
    func applyPan(dx: Float, dy: Float) {
        orbitLock.lock(); defer { orbitLock.unlock() }
        let up = simd_normalize(orbitQuat.act(baseUp))
        let eye = orbitCenter + orbitQuat.act(baseOffset * orbitDist)
        let fwd = simd_normalize(orbitCenter - eye)
        var right = simd_cross(fwd, up)
        right = simd_length(right) > 1e-4 ? simd_normalize(right) : SIMD3<Float>(1, 0, 0)
        let u = simd_cross(right, fwd)                     // 正交化螢幕上方
        let dist = simd_length(baseOffset) * orbitDist
        // 螢幕點→world：dist/fy 為每 render 像素的世界量，×0.6 補償 render(240 寬) 放大到全螢幕
        let k = 0.6 * dist / max(portraitFy, 1)
        // 手指右(dx>0)→模型右→相機左（-right）；手指下(dy>0)→模型下→相機上（+u）。感覺相反就翻號。
        orbitCenter += right * (-dx * k) + u * (dy * k)
    }

    func resetView() {
        orbitLock.lock()
        orbitQuat = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        orbitDist = 1.0
        orbitLock.unlock()
    }

    // MARK: - 訓練

    /// setup + 阻塞式訓練迴圈。務必透過本 session 佇列執行（內部已 dispatch）。
    func start(colmapDir: String, metallib: String?,
               iterations: Int, shDegree: Int, maxGaussians: Int,
               downscale: Float, maxImageDim: Int, previewEvery: Int, wantPreview: Bool,
               outputPLY: String,
               isCancelled: @escaping @Sendable () -> Bool,
               thermalPaused: @escaping @Sendable () -> Bool,
               onProgress: @escaping @Sendable (Progress) -> Void,
               onError: @escaping @Sendable (Error) -> Void,
               onDone: @escaping @Sendable (_ cancelled: Bool) -> Void) {
        queue.async { [self] in
            do {
                try setup(colmapDir: colmapDir, metallib: metallib,
                          iterations: iterations, shDegree: shDegree,
                          maxGaussians: maxGaussians, downscale: downscale,
                          maxImageDim: maxImageDim)
            } catch {
                onError(error); return
            }

            var step = 0
            while step < iterations {
                if isCancelled() { break }
                while thermalPaused() {
                    if isCancelled() { break }
                    Thread.sleep(forTimeInterval: 0.25)
                }
                if isCancelled() { break }

                let stats = msplat_trainer_step(trainer)
                step = Int(stats.iteration)

                let previewStep = wantPreview && (step % max(1, previewEvery) == 0 || step == iterations)
                if previewStep {
                    onProgress(Progress(iteration: step, total: iterations,
                                        splatCount: Int(stats.splatCount),
                                        msPerStep: stats.msPerStep,
                                        preview: renderCurrent()))   // 直式 + 當前 trackball 角度
                } else if step % 10 == 0 {
                    onProgress(Progress(iteration: step, total: iterations,
                                        splatCount: Int(stats.splatCount),
                                        msPerStep: stats.msPerStep, preview: nil))
                }
            }

            if let t = trainer { msplat_trainer_export_ply(t, outputPLY) }
            onDone(isCancelled())
        }
    }

    private func setup(colmapDir: String, metallib: String?,
                       iterations: Int, shDegree: Int, maxGaussians: Int,
                       downscale: Float, maxImageDim: Int) throws {
        if let metallib { msplat_set_metallib_path(metallib) }
        guard let ds = msplat_dataset_create(colmapDir, downscale, Int32(maxImageDim), false, 0) else {
            throw TrainingError.initFailed
        }
        dataset = ds
        numCameras = Int(msplat_dataset_num_train(ds))
        guard numCameras > 0 else { throw TrainingError.datasetEmpty }
        previewCam = Int32(numCameras / 2)

        var cfg = msplat_default_config()
        cfg.iterations = Int32(iterations)
        cfg.shDegree = Int32(shDegree)
        cfg.maxGaussians = Int32(maxGaussians)
        cfg.bgColor = (0, 0, 0)   // 黑底（取代 msplat 除錯用的洋紅），檢視乾淨
        guard let t = msplat_trainer_create(ds, cfg) else { throw TrainingError.initFailed }
        trainer = t

        computeOrbitBasis()
    }

    // MARK: - 互動檢視（訓練完成後）

    /// 渲染當前 trackball 視角（直式）。完成後在主佇列外回呼。
    func renderView(completion: @escaping @Sendable (PreviewImage?) -> Void) {
        queue.async { [self] in completion(renderCurrent()) }
    }

    /// 診斷用：從「真實訓練相機」的確切視角 render（訓練當下優化的視角）。
    /// 用來區分「模型壞了」還是「環繞相機算錯」。
    func renderTrainingView(completion: @escaping @Sendable (PreviewImage?) -> Void) {
        queue.async { [self] in
            guard let t = trainer else { completion(nil); return }
            let pb = msplat_trainer_render(t, previewCam, false)   // MsplatPixelBuffer：float RGB
            guard let fdata = pb.data, pb.width > 0, pb.height > 0 else { completion(nil); return }
            let w = Int(pb.width), h = Int(pb.height)
            var rgba = [UInt8](repeating: 255, count: w * h * 4)
            for i in 0..<(w * h) {
                rgba[i * 4 + 0] = UInt8(max(0, min(1, fdata[i * 3 + 0])) * 255)
                rgba[i * 4 + 1] = UInt8(max(0, min(1, fdata[i * 3 + 1])) * 255)
                rgba[i * 4 + 2] = UInt8(max(0, min(1, fdata[i * 3 + 2])) * 255)
            }
            free(UnsafeMutableRawPointer(fdata))   // C API：呼叫端負責 free
            completion(PreviewImage(rgba: Data(rgba), width: w, height: h))
        }
    }

    func exportPLY(to path: String) {
        queue.async { [self] in if let t = trainer { msplat_trainer_export_ply(t, path) } }
    }

    func close() {
        queue.async { [self] in
            if let t = trainer { msplat_trainer_destroy(t); trainer = nil }
            if let d = dataset { msplat_dataset_destroy(d); dataset = nil }
        }
    }

    // MARK: - 內部：render / 環繞幾何（皆在 session 佇列上呼叫）

    private func cameraPose(_ index: Int32) -> [Float] {
        var m = [Float](repeating: 0, count: 16)
        msplat_dataset_camera_pose(dataset, index, &m)
        return m
    }

    /// 渲染當前 trackball 視角為直式影像（自訂內參 + portrait 尺寸）。
    private func renderCurrent() -> PreviewImage? {
        guard let t = trainer else { return nil }
        orbitLock.lock()
        let pose = currentPoseLocked()
        orbitLock.unlock()
        let cx = Float(portraitW) / 2, cy = Float(portraitH) / 2
        var w = portraitW, h = portraitH
        var buf = [UInt8](repeating: 0, count: Int(portraitW) * Int(portraitH) * 4)
        buf.withUnsafeMutableBufferPointer { p in
            msplat_trainer_render_pose_intrinsics(t, pose, portraitFx, portraitFy, cx, cy,
                                                  portraitW, portraitH, p.baseAddress, &w, &h)
        }
        return PreviewImage(rgba: Data(buf), width: Int(w), height: Int(h))
    }

    /// 由 orbitQuat 算相機 GL camToWorld（繞 orbitCenter 旋轉、lookAt 中心）。呼叫前需持 orbitLock。
    private func currentPoseLocked() -> [Float] {
        let eye = orbitCenter + orbitQuat.act(baseOffset * orbitDist)
        let up = simd_normalize(orbitQuat.act(baseUp))
        let forward = simd_normalize(orbitCenter - eye)
        var r = simd_cross(forward, up)
        r = simd_length(r) > 1e-4 ? simd_normalize(r) : SIMD3<Float>(1, 0, 0)
        let u = simd_cross(r, forward)
        // GL 慣例：相機 +X=right、+Y=up、+Z=-forward
        return [
            r.x, u.x, -forward.x, eye.x,
            r.y, u.y, -forward.y, eye.y,
            r.z, u.z, -forward.z, eye.z,
            0,   0,   0,          1,
        ]
    }

    private static func pos(_ m: [Float]) -> SIMD3<Float> { SIMD3(m[3], m[7], m[11]) }

    /// 由點雲質心（物件本身）設 trackball 樞軸與初始視角 + 直式內參。
    private func computeOrbitBasis() {
        guard numCameras > 0 else { return }
        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(numCameras)
        for i in 0..<numCameras { positions.append(Self.pos(cameraPose(Int32(i)))) }
        var camCentroid = SIMD3<Float>(0, 0, 0)
        for p in positions { camCentroid += p }
        camCentroid /= Float(numCameras)

        // 樞軸＝點雲質心（物件本身），而非相機群中心（桌面/場景掃描相機不在物件中心）
        var oc = [Float](repeating: 0, count: 3)
        var orad: Float = 1
        msplat_dataset_points_bounds(dataset, &oc, &orad)
        orbitCenter = SIMD3<Float>(oc[0], oc[1], oc[2])
        let radius = max(orad, 0.05)

        // 初始 eye：相機側、距物件中心約 3× 半徑，框住物件
        var dir = camCentroid - orbitCenter
        dir = simd_length(dir) > 1e-4 ? simd_normalize(dir) : SIMD3<Float>(0, 0, 1)
        baseOffset = dir * (radius * 3.0)
        baseUp = SIMD3<Float>(0, -1, 0)   // 世界垂直（重力）；若顛倒改 (0, 1, 0)
        orbitLock.lock()
        orbitQuat = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        orbitDist = 1.0
        orbitLock.unlock()

        // 直式內參：垂直 FOV 60°。解析度刻意小（240×520）—— 訓練中 render 疊在訓練上，
        // 大尺寸會 OOM；小尺寸縮放到螢幕仍夠看。穩定後可再往上調。
        portraitW = 240; portraitH = 520
        let fovY: Float = 60 * .pi / 180
        portraitFy = 0.5 * Float(portraitH) / tan(fovY / 2)
        portraitFx = portraitFy
    }
}
