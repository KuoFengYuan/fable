//
//  CaptureController.swift
//  fable — ARKit session 主控：狀態監聽、智慧快門觸發、背景寫入調度、匯出
//
//  執行緒模型：
//  - ARSession delegate 預設回呼在主執行緒；session(_:didUpdate:) 內只做 O(1) 品質計算，
//    關鍵幀觸發時做一次 buffer memcpy（~2ms、約 2Hz），不會卡 UI。
//  - JPEG 編碼 / 磁碟 I/O 在 FrameWriter actor；點雲反投影在 PointCloudAccumulator actor。
//  - 背壓：pendingWrites 達上限即跳過本幀，快門條件仍成立，下一幀自動重試。
//

import Foundation
import ARKit
import SceneKit
import Combine
import UIKit
import simd

@MainActor
final class CaptureController: NSObject, ObservableObject {

    enum Phase: Equatable {
        case idle          // 尚未開拍（可放置圓頂）
        case scanning
        case processing    // 掃描後：錨點姿態修正 + 點雲重融合（進度條）
        case review        // 3D 檢視優化後點雲，決定匯出 / 續掃 / 捨棄 / 訓練
        case training      // 手機端 3DGS 訓練（msplat）：即時預覽越訓越清晰
        case exporting     // 寫 COLMAP + zip
        case done
    }

    // MARK: - UI 狀態
    @Published var phase: Phase = .idle
    @Published var mode: ScanMode = .object
    @Published var assessment = QualityAssessment()
    @Published var keyframeCount = 0
    @Published var pointCount = 0
    @Published var coverage: Double = 0
    @Published var exportedZip: URL?
    @Published var statusText: String?
    @Published var domePlaced = false
    @Published var trackingReady = false
    @Published var showPointCloud = true
    /// 掃描期間鎖定對焦 / 曝光 / 白平衡（預設開啟）：
    /// 內參穩定、色彩一致、避免 AF 拉風箱造成的模糊與追蹤擾動
    @Published var lockCameraParams = true
    @Published var exportProgress: Double = 0
    /// Review 階段顯示（＝實際將匯出）的重融合點雲與修正後軌跡
    @Published var reviewPoints: [CloudPoint] = []
    @Published var reviewTrajectory: [simd_float4x4] = []

    // MARK: - 手機端 3DGS 訓練（msplat）狀態
    @Published var trainingIteration = 0
    @Published var trainingTotal = 0
    @Published var trainingSplatCount = 0
    @Published var trainingPreview: UIImage?     // 訓練中/完成的即時 render
    @Published var trainingComplete = false      // .training 階段內：進行中 vs 已完成
    @Published var trainedPLY: URL?
    /// 訓練時是否顯示即時預覽：開＝看過程（越訓越清晰），關＝背景訓練、略快。預設開
    @Published var showTrainingProcess = true
    private var trainingCancel = CancelFlag()
    private var trainingTask: Task<Void, Never>?
    private var session: MsplatSession?
    // 訓練後互動檢視（trackball）render 節流
    private var orbitInFlight = false
    private var orbitPending = false

    let config = CaptureConfig()
    let hasLiDAR = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

    // MARK: - 內部元件
    private weak var arView: ARSCNView?
    private var visualizer: CoverageVisualizer?
    private let monitor = QualityMonitor()
    private var shutter = SmartShutter()
    private var writer: FrameWriter?
    private var accumulator: PointCloudAccumulator?
    private var sessionDir: URL?
    private var frameIndex = 0
    private var frameCounter = 0
    private var pendingWrites = 0
    private var previewInFlight = false
    private var pixelBufferPool: CVPixelBufferPool?
    private var lastWarningHaptic: TimeInterval = 0
    private var lastSparseIntegration: TimeInterval = 0
    private let captureHaptic = UIImpactFeedbackGenerator(style: .light)
    private let warningHaptic = UINotificationFeedbackGenerator()
    /// 關鍵幀 → ARAnchor：ARKit 地圖優化（迴環/重定位修正）會回頭調整錨點，
    /// 停止時讀回即得「修正後姿態」—— 免費的輕量級 pose graph 精修
    private var keyframeAnchors: [Int: UUID] = [:]
    private var refinedRecords: [FrameRecord] = []
    /// 點雲空間磚 → ARAnchor：融合/去重在錨點局部系進行，錨點被 ARKit 修正時整磚跟著移動
    private var tileAnchorID: [Int64: UUID] = [:]
    private var tileKeyByAnchor: [UUID: Int64] = [:]
    private var latestTileTransforms: [Int64: simd_float4x4] = [:]

    // MARK: - Session 生命週期

    func attach(arView: ARSCNView) {
        self.arView = arView
        arView.session.delegate = self

        let viz = CoverageVisualizer(config: config)
        viz.attach(to: arView.scene)
        viz.onCoverageChanged = { [weak self] c in self?.coverage = c }
        visualizer = viz

        runSession()
        monitor.start()
    }

    func teardown() {
        trainingCancel.cancel()
        session?.close(); session = nil
        releaseCameraLocks()
        arView?.session.pause()
        monitor.stop()
        UIApplication.shared.isIdleTimerDisabled = false
    }

    private func makeARConfig() -> ARWorldTrackingConfiguration {
        let cfg = ARWorldTrackingConfiguration()
        cfg.worldAlignment = .gravity          // 世界 +Y = 反重力 → 3DGS 場景天然正立
        cfg.planeDetection = [.horizontal]     // 供物件模式 raycast 放置圓頂
        cfg.environmentTexturing = .none       // 省下環境貼圖的 GPU/散熱成本
        if hasLiDAR {
            cfg.frameSemantics.insert(.sceneDepth)          // 原始深度：存檔用
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
                cfg.frameSemantics.insert(.smoothedSceneDepth)  // 平滑深度：點雲融合用
            }
        }
        return cfg
    }

    private func runSession() {
        arView?.session.run(makeARConfig(), options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - 物件模式：點擊放置涵蓋圓頂

    func placeObjectAnchor(at point: CGPoint) {
        guard phase == .idle, mode == .object, let arView else { return }
        guard let query = arView.raycastQuery(from: point, allowing: .estimatedPlane, alignment: .any),
              let hit = arView.session.raycast(query).first else {
            statusText = "找不到表面，請將準心對準物件再點一次"
            return
        }
        let p = hit.worldTransform.columns.3
        // 圓頂中心略高於命中表面，較貼近物件幾何中心
        let center = SIMD3<Float>(p.x, p.y, p.z) + SIMD3<Float>(0, 0.15, 0)
        let camPos = arView.session.currentFrame.map { MatrixUtil.position($0.camera.transform) }
            ?? (center + SIMD3<Float>(1, 0, 0))
        // 把「使用者目前站位」當作標準拍攝半徑
        let radius = max(0.4, simd_distance(camPos, center) * 0.9)
        visualizer?.setObjectCenter(center, radius: radius)
        domePlaced = true
        statusText = "圓頂已放置，按快門開始掃描；繞行讓灰格全部變綠"
    }

    // MARK: - 開始 / 停止

    func startScan() {
        guard phase == .idle else { return }
        do {
            try beginSessionStorage()
        } catch {
            statusText = "無法建立掃描資料夾：\(error.localizedDescription)"
            return
        }
        if lockCameraParams { applyCameraLocks() }
        shutter.reset()
        frameIndex = 0
        keyframeCount = 0
        pointCount = 0
        phase = .scanning
        statusText = nil
        UIApplication.shared.isIdleTimerDisabled = true   // 掃描中不鎖屏
        captureHaptic.prepare()
    }

    /// 停止掃描 → 取回錨點修正後姿態 → 暫停 AR → 背景重融合 → review
    func stopScan() {
        guard phase == .scanning else { return }
        phase = .processing
        exportProgress = 0
        statusText = nil
        UIApplication.shared.isIdleTimerDisabled = false
        let refined = snapshotRefinedTransforms()   // 必須在 pause 前讀 anchors
        arView?.session.pause()                     // review 期間停止追蹤，省電省熱
        monitor.stop()
        Task { await processScan(refinedTransforms: refined) }
    }

    private func processScan(refinedTransforms: [Int: [Double]]) async {
        guard let writer, let accumulator, let dir = sessionDir else {
            phase = .idle
            return
        }
        let raw = await writer.snapshotRecords()
        var corrected = 0
        refinedRecords = raw.map { record in
            var r = record
            if let t = refinedTransforms[r.id] {
                if t != r.transform { corrected += 1 }
                r.transform = t
            }
            return r
        }

        // 重融合：用修正後姿態把所有關鍵幀原始深度重新反投影 + 加權平均
        var points: [CloudPoint] = []
        if hasLiDAR && config.saveDepth && !refinedRecords.isEmpty {
            let records = refinedRecords
            let cfg = config
            let onProg: @Sendable (Double) -> Void = { p in
                Task { @MainActor [weak self] in self?.exportProgress = p }
            }
            points = await Task.detached(priority: .userInitiated) {
                RefusionEngine.refuse(records: records, sessionDir: dir, config: cfg, progress: onProg)
            }.value
        }
        if points.isEmpty {                          // 無 LiDAR / 無深度時退回即時累積雲
            points = await accumulator.bestPoints(target: config.exportMaxPoints)
        }

        reviewPoints = points
        reviewTrajectory = refinedRecords.map { RefusionEngine.float4x4(rowMajor: $0.transform) }
        pointCount = points.count
        statusText = corrected > 0 ? "姿態已修正 \(corrected) 幀（ARKit 地圖優化）" : nil
        phase = .review
    }

    /// review 通過（或訓練完成）→ 寫 COLMAP sparse + PLY + 修正後姿態 + zip。
    /// 若已訓練，dir 內的 gaussians.ply 會一併打包進 zip。
    func exportAndShare() {
        guard phase == .review || (phase == .training && trainingComplete),
              let dir = sessionDir else { return }
        phase = .exporting
        statusText = "打包中…"
        let records = refinedRecords
        let points = reviewPoints
        Task {
            _ = await writer?.finish()
            do {
                try ExportManager.writeColmapSparse(records: records, points: points, to: dir,
                                                    flipWorldUp: config.flipWorldUpForExport)
                if !points.isEmpty {
                    // points.ply 保留 ARKit 原生 +Y up（與 poses.jsonl 一致，供 validate/檢視）；
                    // 訓練用點雲以 sparse/0/points3D.bin 為準（已對齊 COLMAP 慣例）
                    try ExportManager.writePLY(points, to: dir.appendingPathComponent("points.ply"))
                }
                try ExportManager.writeRefinedPoses(records,
                                                    to: dir.appendingPathComponent("poses_refined.jsonl"))
            } catch {
                statusText = "匯出 COLMAP 資料失敗：\(error.localizedDescription)"
            }

            let zipURL = dir.deletingLastPathComponent()
                .appendingPathComponent(dir.lastPathComponent + ".zip")
            let zipOK: Bool = await Task.detached(priority: .userInitiated) {
                do {
                    try ExportManager.zipDirectory(dir, to: zipURL)
                    return true
                } catch {
                    print("[Export] zip 失敗: \(error)")
                    return false
                }
            }.value

            exportedZip = zipOK ? zipURL : nil
            statusText = zipOK
                ? "完成：\(records.count) 幀 / \(points.count) 點（COLMAP 格式，可直接訓練）"
                : "壓縮失敗；原始資料保留在「檔案 App → fable → scans」"
            phase = .done
            writer = nil
            accumulator = nil
        }
    }

    /// review 發現破洞 → 回到掃描續拍（不 reset：保留地圖與錨點，ARKit 自動重新定位）
    func resumeScan() {
        guard phase == .review, let arView else { return }
        reviewPoints = []
        reviewTrajectory = []
        arView.session.run(makeARConfig(), options: [])
        if lockCameraParams { applyCameraLocks() }
        monitor.start()
        shutter.reset()
        phase = .scanning
        statusText = "重新定位中：回到剛才的位置附近即可繼續"
        UIApplication.shared.isIdleTimerDisabled = true
    }

    // MARK: - 手機端訓練

    /// review（或訓練完成後重訓）→ 手機端 3DGS 訓練：
    /// 先寫 COLMAP sparse/0 當訓練輸入（images/ 掃描時已寫），再在 msplat session 逐步訓練。
    /// 訓練完成後 session 保留 → 可拖曳環繞檢視訓練結果。
    func startTraining() {
        guard phase == .review || (phase == .training && trainingComplete),
              let dir = sessionDir, !refinedRecords.isEmpty else { return }
        session?.close()                 // 重訓：先關舊 session
        phase = .training
        trainingComplete = false
        trainingPreview = nil
        trainingIteration = 0
        trainingTotal = config.trainIterations
        trainingSplatCount = 0
        statusText = "準備訓練資料…"
        UIApplication.shared.isIdleTimerDisabled = true

        let cancel = CancelFlag()
        trainingCancel = cancel
        let records = refinedRecords
        let points = reviewPoints
        let cfg = config
        let wantPreview = showTrainingProcess
        let metallib = Bundle.main.path(forResource: "default", ofType: "metallib")
        let plyURL = dir.appendingPathComponent("gaussians.ply")
        let newSession = MsplatSession()
        session = newSession

        // 回呼在 MainActor 層定義、各自弱捕獲 self（避免巢狀 Task 的「捕獲 var self」問題）
        let onProgress: @Sendable (MsplatSession.Progress) -> Void = { p in
            Task { @MainActor [weak self] in self?.applyTrainingProgress(p) }
        }
        let onError: @Sendable (Error) -> Void = { e in
            Task { @MainActor [weak self] in self?.failTraining(e) }
        }
        let onDone: @Sendable (Bool) -> Void = { c in
            Task { @MainActor [weak self] in self?.finishTraining(ply: plyURL, cancelled: c) }
        }
        let thermalPaused: @Sendable () -> Bool = {
            cfg.trainThermalThrottle &&
            ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue
        }

        // trainingTask 本身不捕獲 self：先背景寫 COLMAP 訓練輸入，再啟動 session 訓練
        trainingTask = Task {
            do {
                try await Task.detached(priority: .userInitiated) {
                    try ExportManager.writeColmapSparse(records: records, points: points, to: dir,
                                                        flipWorldUp: cfg.flipWorldUpForExport)
                }.value
            } catch {
                onError(error); return
            }
            newSession.start(
                colmapDir: dir.path, metallib: metallib,
                iterations: cfg.trainIterations, shDegree: cfg.trainSHDegree,
                maxGaussians: cfg.trainMaxGaussians, downscale: cfg.trainDownscale,
                maxImageDim: cfg.trainMaxImageDim,
                previewEvery: cfg.trainPreviewEvery, wantPreview: wantPreview,
                outputPLY: plyURL.path,
                isCancelled: { cancel.isCancelled },
                thermalPaused: thermalPaused,
                onProgress: onProgress, onError: onError, onDone: onDone)
        }
    }

    func cancelTraining() {
        guard phase == .training, !trainingComplete else { return }
        trainingCancel.cancel()
        statusText = "停止訓練中…（保留目前結果供檢視）"
    }

    /// 訓練完成後返回 review（點雲檢視）；zip 匯出已含 gaussians.ply
    func backToReviewFromTraining() {
        guard phase == .training else { return }
        session?.close(); session = nil
        trainingComplete = false
        trainingPreview = nil
        phase = .review
        statusText = nil
    }

    /// 檢視訓練結果：拖曳 → 更新環繞角度並重新 render（節流：一次一張、保留最新請求）
    func orbitTrainedView(deltaX: Float, deltaY: Float) {
        guard phase == .training else { return }
        session?.applyDrag(dx: deltaX, dy: deltaY)   // trackball 自由轉動（訓練中/完成都可）
        if trainingComplete { requestOrbitRender() } // 完成後：即時 on-demand render（訓練中由迴圈定期 render）
    }

    /// pinch 縮放檢視（scale>1 拉近）。與拖曳同樣節流；訓練中由訓練迴圈定期 render。
    func zoomTrainedView(scale: Float) {
        guard phase == .training else { return }
        session?.applyZoom(scale)
        if trainingComplete { requestOrbitRender() }
    }

    private func requestOrbitRender() {
        guard let session, phase == .training, trainingComplete else { return }
        if orbitInFlight { orbitPending = true; return }
        orbitInFlight = true
        session.renderView { img in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let img { self.setPreviewImage(img) }
                self.orbitInFlight = false
                if self.orbitPending { self.orbitPending = false; self.requestOrbitRender() }
            }
        }
    }

    /// 由 render 結果建 UIImage（已是直立直式 render，不需再旋轉）。
    private func setPreviewImage(_ pv: MsplatSession.PreviewImage?) {
        guard let pv, let cg = pv.cgImage() else { return }
        trainingPreview = UIImage(cgImage: cg)
    }

    private func applyTrainingProgress(_ p: MsplatSession.Progress) {
        guard phase == .training, !trainingComplete else { return }
        trainingIteration = p.iteration
        trainingSplatCount = p.splatCount
        setPreviewImage(p.preview)
        statusText = "訓練中　\(p.iteration)/\(p.total)　\(p.splatCount / 1000)k splats"
    }

    private func finishTraining(ply: URL, cancelled: Bool) {
        UIApplication.shared.isIdleTimerDisabled = false
        trainedPLY = FileManager.default.fileExists(atPath: ply.path) ? ply : nil
        trainingComplete = true
        trainingIteration = trainingTotal
        statusText = cancelled
            ? "已停止：\(trainingSplatCount / 1000)k splats — 拖曳可轉動檢視、亦可匯出"
            : "訓練完成：\(trainingSplatCount / 1000)k splats — 拖曳可轉動檢視"
        session?.resetView()
        session?.renderView { img in   // 初始 trackball 視角（直立、直式、框住物件）
            Task { @MainActor [weak self] in
                guard let self else { return }
                if let img { self.setPreviewImage(img) }
            }
        }
    }

    private func failTraining(_ error: Error) {
        UIApplication.shared.isIdleTimerDisabled = false
        statusText = "訓練失敗：\(error.localizedDescription)"
        trainingComplete = false
        session?.close(); session = nil
        phase = .review
    }

    /// 捨棄本次掃描：刪除資料、全新開始
    func discardScan() {
        guard phase == .review || phase == .done || phase == .training else { return }
        trainingCancel.cancel()
        session?.close(); session = nil
        if let dir = sessionDir { try? FileManager.default.removeItem(at: dir) }
        if let zip = exportedZip { try? FileManager.default.removeItem(at: zip) }
        writer = nil
        accumulator = nil
        sessionDir = nil
        cleanupToIdle()
    }

    func togglePointCloud() {
        showPointCloud.toggle()
        visualizer?.setPointCloudHidden(!showPointCloud)
    }

    func resetForNewScan() {
        guard phase == .done || phase == .idle else { return }
        cleanupToIdle()
    }

    private func cleanupToIdle() {
        exportedZip = nil
        statusText = nil
        keyframeCount = 0
        pointCount = 0
        coverage = 0
        exportProgress = 0
        domePlaced = false
        reviewPoints = []
        reviewTrajectory = []
        refinedRecords = []
        session?.close(); session = nil
        trainingComplete = false
        trainingPreview = nil
        trainedPLY = nil
        trainingIteration = 0
        trainingSplatCount = 0
        keyframeAnchors = [:]
        tileAnchorID = [:]
        tileKeyByAnchor = [:]
        latestTileTransforms = [:]
        visualizer?.reset()
        runSession()          // reset tracking（review/done 期間 session 已暫停）
        releaseCameraLocks()  // 回到 idle 恢復自動對焦/曝光，方便取景
        monitor.start()
        phase = .idle
    }

    /// 讀取每個關鍵幀錨點的「目前」變換 —— ARKit 若做過地圖修正，值會與採集當下不同
    private func snapshotRefinedTransforms() -> [Int: [Double]] {
        guard let anchors = arView?.session.currentFrame?.anchors else { return [:] }
        var byID: [UUID: simd_float4x4] = [:]
        for anchor in anchors { byID[anchor.identifier] = anchor.transform }
        var out: [Int: [Double]] = [:]
        for (index, id) in keyframeAnchors {
            if let t = byID[id] { out[index] = MatrixUtil.rowMajor16(t) }
        }
        return out
    }

    private func beginSessionStorage() throws {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let stamp = formatter.string(from: Date())
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("scans/scan_\(stamp)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        sessionDir = dir

        writer = try FrameWriter(sessionDir: dir,
                                 saveDepth: config.saveDepth && hasLiDAR,
                                 jpegQuality: config.jpegQuality)
        accumulator = PointCloudAccumulator(config: config)

        let meta = SessionMeta(device: Self.deviceModel(),
                               osVersion: UIDevice.current.systemVersion,
                               startedAt: ISO8601DateFormatter().string(from: Date()),
                               mode: mode.rawValue,
                               lidarAvailable: hasLiDAR)
        try ExportManager.writeMeta(meta, to: dir.appendingPathComponent("meta.json"))
    }

    /// 相機三鎖：對焦（內參穩定、AF 不拉風箱）、曝光（ISO/快門固定 → 亮度一致、
    /// 模糊可預測）、白平衡（色彩一致 → 融合不閃色、3DGS 訓練色彩乾淨）。
    /// 在使用者取景完成、按下快門的當下鎖定 —— AE/AF 已收斂於目標物。
    private func applyCameraLocks() {
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else { return }
        do {
            try device.lockForConfiguration()
            if device.isFocusModeSupported(.locked) {
                device.focusMode = .locked
            }
            if device.isExposureModeSupported(.locked) {
                device.exposureMode = .locked
            }
            if device.isWhiteBalanceModeSupported(.locked) {
                device.whiteBalanceMode = .locked
            }
            device.unlockForConfiguration()
        } catch {
            print("[Camera] 參數鎖定失敗: \(error)")
        }
    }

    private func releaseCameraLocks() {
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else { return }
        do {
            try device.lockForConfiguration()
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                device.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            device.unlockForConfiguration()
        } catch {
            print("[Camera] 參數解鎖失敗: \(error)")
        }
    }

    nonisolated private static func deviceModel() -> String {
        var sys = utsname()
        uname(&sys)
        return withUnsafeBytes(of: &sys.machine) { buffer in
            String(decoding: buffer.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}

// MARK: - ARSessionDelegate（主執行緒回呼）

extension CaptureController: @preconcurrency ARSessionDelegate {

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameCounter += 1

        let a = monitor.assess(frame: frame, config: config)
        // UI 每 6 幀（~0.1s）更新一次即可，避免 60Hz 重繪
        if frameCounter % 6 == 0 { assessment = a }
        if !trackingReady, case .normal = frame.camera.trackingState { trackingReady = true }

        guard phase == .scanning else { return }

        // 遮斷級警告觸覺回饋（限流 1 次 / 1.5 秒）
        if a.captureBlocked, frame.timestamp - lastWarningHaptic > 1.5 {
            warningHaptic.notificationOccurred(.warning)
            lastWarningHaptic = frame.timestamp
        }

        // 過熱保護：critical 直接停拍並保住已拍資料
        if ProcessInfo.processInfo.thermalState == .critical {
            statusText = "裝置過熱，已自動停止並匯出"
            stopScan()
            return
        }

        // ARKit 修正了磚錨點（漂移校正/重定位）→ 更新快照並讓點雲磚跟著實體表面移動（防殘影）
        if !tileKeyByAnchor.isEmpty {
            var current: [Int64: simd_float4x4] = [:]
            current.reserveCapacity(tileKeyByAnchor.count)
            for anchor in frame.anchors {
                if let key = tileKeyByAnchor[anchor.identifier] { current[key] = anchor.transform }
            }
            latestTileTransforms = current
            visualizer?.syncTileTransforms(current)
        }

        // 點雲連續融合（~10Hz，與快門解耦）：只收姿態可靠且清晰的幀，
        // 模糊幀的姿態-深度錯位是點雲殘影的另一主要來源
        if frameCounter % config.previewFrameInterval == 0,
           !a.captureBlocked, a.blurPixels <= config.previewMaxBlurPixels {
            integratePreview(frame, blurPixels: a.blurPixels)
        }

        guard a.allowCapture else { return }
        guard pendingWrites < config.maxPendingWrites else { return }
        guard shutter.shouldCapture(pose: frame.camera.transform,
                                    time: frame.timestamp,
                                    config: config) else { return }
        captureKeyframe(frame, assessment: a)
    }

    /// 即時點雲融合：主執行緒只複製 buffer（~2ms）→ actor 在錨點局部系反投影/去重/打包 →
    /// 主執行緒建缺席錨點、O(1) 更新磚節點。delegate 回呼零計算 —— 不餓死 ARKit 的 VIO。
    private func integratePreview(_ frame: ARFrame, blurPixels: Float) {
        guard let accumulator, !previewInFlight else { return }
        let anchorSnapshot = latestTileTransforms
        if hasLiDAR {
            if pixelBufferPool == nil {
                let src = frame.capturedImage
                pixelBufferPool = PixelBufferUtil.makePool(
                    width: CVPixelBufferGetWidth(src),
                    height: CVPixelBufferGetHeight(src),
                    pixelFormat: CVPixelBufferGetPixelFormatType(src))
            }
            guard let pool = pixelBufferPool,
                  let packet = PointExtractor.makePacket(frame: frame, pool: pool,
                                                         blurPixels: blurPixels) else { return }
            previewInFlight = true
            Task {
                await accumulator.integrate(packet, anchorTransforms: anchorSnapshot)
                await self.flushTiles()
                self.previewInFlight = false
            }
        } else if let featurePoints = frame.rawFeaturePoints,
                  frame.timestamp - lastSparseIntegration > 1.0 {
            lastSparseIntegration = frame.timestamp
            let points = featurePoints.points
            previewInFlight = true
            Task {
                await accumulator.integrateSparse(points: points, anchorTransforms: anchorSnapshot)
                await self.flushTiles()
                self.previewInFlight = false
            }
        }
    }

    /// 建立新磚的 ARAnchor（局部系原點），再把髒磚渲染資料掛上對應節點
    private func flushTiles() async {
        guard let accumulator else { return }
        let pending = await accumulator.takePendingAnchors()
        for (key, center) in pending where tileAnchorID[key] == nil {
            var t = matrix_identity_float4x4
            t.columns.3 = SIMD4<Float>(center.x, center.y, center.z, 1)
            if let session = arView?.session {
                let anchor = ARAnchor(name: "tile", transform: t)
                session.add(anchor: anchor)
                tileAnchorID[key] = anchor.identifier
                tileKeyByAnchor[anchor.identifier] = key
            }
            latestTileTransforms[key] = t     // 錨點回呼前的初始變換（= 磚中心）
        }
        let tiles = await accumulator.dirtyTileRenderData(limit: 2)
        for tile in tiles {
            let xform = latestTileTransforms[tile.key]
                ?? { var m = matrix_identity_float4x4
                     m.columns.3 = SIMD4<Float>(tile.center.x, tile.center.y, tile.center.z, 1)
                     return m }()
            visualizer?.updateTile(tile, transform: xform)
        }
        pointCount = await accumulator.count
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        statusText = "AR Session 失敗：\(error.localizedDescription)"
    }

    private func captureKeyframe(_ frame: ARFrame, assessment a: QualityAssessment) {
        let src = frame.capturedImage
        let w = CVPixelBufferGetWidth(src)
        let h = CVPixelBufferGetHeight(src)
        if pixelBufferPool == nil {
            pixelBufferPool = PixelBufferUtil.makePool(width: w, height: h,
                                                       pixelFormat: CVPixelBufferGetPixelFormatType(src))
        }
        guard let pool = pixelBufferPool,
              let copy = PixelBufferUtil.clone(src, pool: pool) else { return }

        // 深度 / 信心圖 → 緊湊 Data（float32 / uint8 raw）
        var depthData: Data?
        var confData: Data?
        var dw = 0, dh = 0
        if config.saveDepth, let sceneDepth = frame.sceneDepth {
            dw = CVPixelBufferGetWidth(sceneDepth.depthMap)
            dh = CVPixelBufferGetHeight(sceneDepth.depthMap)
            depthData = PixelBufferUtil.tightData(sceneDepth.depthMap, bytesPerPixel: 4)
            confData = sceneDepth.confidenceMap.map { PixelBufferUtil.tightData($0, bytesPerPixel: 1) }
        }

        let camera = frame.camera
        let K = camera.intrinsics
        frameIndex += 1
        let name = String(format: "frame_%05d", frameIndex)

        let record = FrameRecord(
            id: frameIndex,
            timestamp: frame.timestamp,
            transform: MatrixUtil.rowMajor16(camera.transform),
            intrinsics: CameraIntrinsics(fx: Double(K[0][0]), fy: Double(K[1][1]),
                                         cx: Double(K[2][0]), cy: Double(K[2][1]),
                                         width: w, height: h),
            exposureDuration: camera.exposureDuration,
            exposureOffsetEV: Double(camera.exposureOffset),
            ambientLux: frame.lightEstimate.map { Double($0.ambientIntensity) },
            estimatedBlurPx: Double(a.blurPixels),
            imageFile: name + ".jpg",
            depthFile: depthData != nil ? name + "_depth.bin" : nil,
            confidenceFile: confData != nil ? name + "_conf.bin" : nil,
            depthWidth: depthData != nil ? dw : nil,
            depthHeight: depthData != nil ? dh : nil)

        let keyframe = Keyframe(pixelBuffer: copy,
                                depthData: depthData,
                                confidenceData: confData,
                                depthWidth: dw, depthHeight: dh,
                                c2w: camera.transform,
                                record: record)

        // 掛錨點：ARKit 後續的地圖優化會調整它，停止時讀回 = 修正後姿態
        if let session = arView?.session {
            let anchor = ARAnchor(name: name, transform: camera.transform)
            session.add(anchor: anchor)
            keyframeAnchors[frameIndex] = anchor.identifier
        }

        pendingWrites += 1
        keyframeCount = frameIndex
        visualizer?.addKeyframe(pose: camera.transform)
        captureHaptic.impactOccurred(intensity: 0.6)

        // 點雲融合已由 integratePreview() 連續進行，關鍵幀只負責存檔
        guard let writer else {
            pendingWrites -= 1
            return
        }
        Task {
            await writer.write(keyframe)
            self.pendingWrites -= 1
        }
    }
}
