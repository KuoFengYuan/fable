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
    /// 融合完成度：已被足夠多幀觀測的表面占比。場景模式沒有涵蓋率圓頂，
    /// 這是唯一能回答「掃夠了沒」的訊號；也比幀數有意義（站原地拍 100 幀是沒用的）。
    @Published var fusionCompleteness: Double = 0
    @Published var coverage: Double = 0
    @Published var coverageHint: String?   // 缺角提醒：往哪補掃（物件模式圓頂）
    @Published var exportedZip: URL?
    @Published var statusText: String?
    @Published var domePlaced = false
    @Published var trackingReady = false
    @Published var showPointCloud = true
    /// 預覽點雲上色模式。掃描當下使用者最需要知道的不是顏色對不對，
    /// 而是「這塊融合夠了沒、要不要再繞一次」——熱圖直接把觀測不足的表面標紅。
    @Published var colorMode: PointColorMode = .rgb
    /// 掃描期間鎖定曝光 / 白平衡（預設開啟）：光度一致，3DGS 的光度損失才對齊得起來。
    /// **對焦不鎖**（見 CameraControls.lockForScan：鎖了會把整段掃描凍在起始那一刻的景深，
    /// 鏡頭一離開就糊）。代價是連續對焦會拉焦，拉焦當下那幾幀確實不清晰 ——
    /// 由 QualityMonitor 的清晰度閘門（直接量影像，不是推估）擋掉，不讓它們變成關鍵幀。
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

    /// 相機手動調整（曝光補償/快門/ISO/白平衡/對焦）。預設全自動；
    /// 使用者調整後，startScan 只鎖定「還在自動」的項目，手動值原樣帶進整段掃描 ——
    /// 也就是「預設鎖定，但可以調整完再鎖」。
    let cameraControls = CameraControls()
    /// 平面圖擷取（RoomPlan，共用同一個 ARSession）。掃描時同步收集，匯出時轉成 usdz/json/svg
    let floorPlan = FloorPlanCapture()
    /// 建好的平面圖，於 processing 階段產生 → review 可顯示、匯出時寫檔
    @Published private(set) var floorPlanData: FloorPlanData?
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
    /// 因清晰度不足而放棄抓幀的次數（診斷用：拿來判斷門檻是否過嚴）
    private var sharpnessRejects = 0
    private var frameCounter = 0
    private var pendingWrites = 0
    private var previewInFlight = false
    /// 本幀是否已進過預覽融合（關鍵幀路徑與 10Hz 路徑可能落在同一幀，避免 obs 重複累加）
    private var lastPreviewFrame = -1
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
        viz.onGuidanceChanged = { [weak self] hint in self?.coverageHint = hint }
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
        cfg.isAutoFocusEnabled = true          // 明示：對焦交給 ARKit 連續自動（見 CameraControls.lockForScan）
        if hasLiDAR {
            cfg.frameSemantics.insert(.sceneDepth)          // 原始深度：存檔用
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
                cfg.frameSemantics.insert(.smoothedSceneDepth)  // 平滑深度：點雲融合用
            }
            // 場景重建網格：ARKit 以每一幀（60fps）的深度做 TSDF 融合並隨漂移修正更新。
            // 我們的重融合只吃 ~120 個關鍵幀 → mesh 能補上關鍵幀漏掉的表面（天花板/角落）。
            if config.useSceneMesh,
               ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
                cfg.sceneReconstruction = .mesh
            }
        }
        return cfg
    }

    /// 印出本機可用的 ARKit 影像格式。
    ///
    /// 訂正一個先前寫錯的推論：我原本說「4K 拍再降到 1600，雜訊降 2.4 倍」。
    /// 那對固定 sensor 不成立 —— iPhone 輸出 1920×1440 時本來就已經在 sensor 上做 binning，
    /// 4K 只是少 bin 一點；降採樣回同一尺寸後 SNR 大致打平。
    /// **高解析度買到的是細節，不是低雜訊。** 顆粒感要靠 ISO（見 denoiseISOThreshold）解。
    /// 這份清單留著是為了知道有沒有「更高解析度又維持 60fps」的選項可換細節（不換雜訊），
    /// 以及確認目前跑在哪個格式 —— 只能實機問。
    private func logVideoFormats() {
        let cur = arView?.session.configuration?.videoFormat
        for f in ARWorldTrackingConfiguration.supportedVideoFormats {
            let r = f.imageResolution
            let mark = (f == cur) ? "  ← 目前使用" : ""
            print(String(format: "[VideoFormat] %.0f×%.0f @ %dfps  %@%@",
                         r.width, r.height, f.framesPerSecond,
                         f.captureDeviceType.rawValue, mark))
        }
    }

    private func runSession() {
        arView?.session.run(makeARConfig(), options: [.resetTracking, .removeExistingAnchors])
        logVideoFormats()
        // 曝光上限要在取景階段就設好，AE 才有時間在上限內收斂；
        // session.run 會重設裝置設定，故必須在 run 之後。
        cameraControls.capExposureDuration()
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
        startFloorPlan(fresh: true)
        // 掃描中收合：中途改曝光會讓前後幀成像不一致（外觀校正要修的正是這個）
        cameraControls.expanded = nil
        cameraControls.railExpanded = false
        shutter.reset()
        frameIndex = 0
        keyframeCount = 0
        pointCount = 0
        sharpnessRejects = 0
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
        let meshVerts = snapshotMeshVertices()      // 同上：pause 後 anchors 就讀不到了
        floorPlan.stopCapture()                     // 只是 stop()，最終資料由 delegate 稍後送達
        Task {
            // pause 必須等 RoomPlan 交回 CapturedRoomData 之後 —— session 一 pause
            // 它就收不完那一段了。waitForSegment 自帶逾時，不會卡住匯出流程。
            await floorPlan.waitForSegment()
            arView?.session.pause()                 // review 期間停止追蹤，省電省熱
            monitor.stop()
            await processScan(refinedTransforms: refined, meshVertices: meshVerts)
        }
    }

    private func processScan(refinedTransforms: [Int: [Double]], meshVertices: [SIMD3<Float>]) async {
        guard let writer, let accumulator, let dir = sessionDir else {
            phase = .idle
            return
        }
        let raw = await writer.snapshotRecords()
        if !raw.isEmpty {
            let sharp = raw.map(\.sharpnessRatio).sorted()
            let blur = raw.map(\.estimatedBlurPx).sorted()
            let iso = raw.map(\.iso).sorted()
            let expo = raw.map(\.exposureDuration).sorted()
            let m = raw.count / 2
            print(String(format:
                "清晰度: %d 幀，清晰度比中位數 %.2f / 最差 %.2f；模糊估計中位數 %.1fpx / 最差 %.1fpx；" +
                "另有 %d 次因不夠清晰而放棄抓幀",
                raw.count, sharp[m], sharp[0], blur[m], blur[raw.count - 1], sharpnessRejects))
            print(String(format:
                "曝光: ISO 中位數 %.0f / 最高 %.0f，快門中位數 1/%.0fs / 最長 1/%.0fs（上限 1/60s）",
                iso[m], iso[raw.count - 1], 1 / expo[m], 1 / expo[raw.count - 1]))
        }
        var corrected = 0
        refinedRecords = raw.map { record in
            var r = record
            if let t = refinedTransforms[r.id] {
                if t != r.transform { corrected += 1 }
                r.transform = t
            }
            return r
        }
        // 模糊幀全域複核。必須在姿態修正**之後**：BlurFilter 靠位置/朝向找「看同一片表面」
        // 的鄰居，用未修正的姿態會找錯鄰居。判定寫回紀錄而非直接刪除，
        // poses_refined.jsonl 與 images/ 都保留完整，可回頭檢查判定對不對。
        refinedRecords = BlurFilter.annotate(refinedRecords)
        let dropped = refinedRecords.filter { $0.blurVerdict == .drop }.count
        let demoted = refinedRecords.filter { $0.blurVerdict == .demote }.count
        if dropped + demoted > 0 {
            print("模糊複核: \(refinedRecords.count) 幀 → 排除 \(dropped) 幀（幾何不可信，點雲也不用）"
                  + "、\(demoted) 幀（顏色糊，不進訓練但深度仍以降權併入點雲）")
        }

        // 重融合：用修正後姿態把所有關鍵幀原始深度重新反投影 + 加權平均
        var points: [CloudPoint] = []
        if hasLiDAR && config.saveDepth && !refinedRecords.isEmpty {
            let records = refinedRecords
            let cfg = config
            let onProg: @Sendable (Double) -> Void = { p in
                Task { @MainActor [weak self] in self?.exportProgress = p }
            }
            let mesh = meshVertices
            points = await Task.detached(priority: .userInitiated) {
                RefusionEngine.refuse(records: records, sessionDir: dir, config: cfg,
                                      meshVertices: mesh, progress: onProg)
            }.value
        }
        if points.isEmpty {                          // 無 LiDAR / 無深度時退回即時累積雲
            points = await accumulator.bestPoints(target: config.exportMaxPoints)
        }

        // 平面圖建模（秒級）：放在 processing 階段，review 時就已經有結果可看/可匯出
        if config.captureFloorPlan, FloorPlanCapture.isSupported {
            floorPlanData = await floorPlan.build()
            if let fp = floorPlanData { logFloorPlan(fp) }
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
        // 訓練/匯出只吃 .keep：.drop 幾何不可信、.demote 顏色糊，兩者都不該當訓練影像。
        // （能走到這裡的 .demote 都是「鄰居夠多」才被判的，排除它不會少掉任何視角。）
        let records = refinedRecords.filter { $0.blurVerdict == .keep }
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
                await writeFloorPlan(to: dir)
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
        // 刻意不 reset：續掃的這一段會成為「另一個房間」，最後由 StructureBuilder 合併成整層。
        // 一間一間掃再合併的精度也優於一鏡到底（一鏡到底會在門口累積漂移）。
        startFloorPlan(fresh: false)
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
        let records = refinedRecords.filter { $0.blurVerdict == .keep }
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
                maxImageDim: cfg.trainMaxImageDim, maxTrainFrames: cfg.trainMaxFrames,
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

    /// 雙指平移檢視（沿螢幕平面移動樞軸）。dx,dy 為螢幕點位移。
    func panTrainedView(dx: Float, dy: Float) {
        guard phase == .training else { return }
        session?.applyPan(dx: dx, dy: dy)
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

    /// 切換「真實顏色 / 融合品質熱圖」。切換後必須把所有磚標記重畫，
    /// 否則只有之後才變動的磚會換色、畫面兩種配色混在一起。
    func toggleColorMode() {
        colorMode = (colorMode == .rgb) ? .fusionQuality : .rgb
        Task { await accumulator?.markAllDirty() }
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
        coverageHint = nil
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
        releaseCameraLocks()  // 回到 idle 恢復自動曝光/白平衡，方便取景
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

    /// 讀取 ARKit 場景重建網格的世界座標頂點（停止前呼叫，與姿態快照同一時機）。
    /// 注意：ARGeometrySource 的頂點是 packed float3（stride 12B），
    /// 不可直接 bind 成 SIMD3<Float>（Swift 的 SIMD3<Float> 佔 16B）—— 必須逐分量讀。
    private func snapshotMeshVertices() -> [SIMD3<Float>] {
        guard config.useSceneMesh,
              let anchors = arView?.session.currentFrame?.anchors else { return [] }
        var out: [SIMD3<Float>] = []
        for case let mesh as ARMeshAnchor in anchors {
            let src = mesh.geometry.vertices
            guard src.format == .float3 else { continue }
            let base = src.buffer.contents()
            let m = mesh.transform
            out.reserveCapacity(out.count + src.count)
            for i in 0..<src.count {
                let p = base.advanced(by: src.offset + i * src.stride)
                              .assumingMemoryBound(to: Float.self)
                let w = m * SIMD4<Float>(p[0], p[1], p[2], 1)
                out.append(SIMD3<Float>(w.x, w.y, w.z))
            }
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
                                 jpegQuality: config.jpegQuality,
                                 denoiseISOThreshold: config.denoiseISOThreshold,
                                 denoiseMaxNoiseLevel: config.denoiseMaxNoiseLevel)
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
    /// 開始掃描時鎖定相機參數。委派給 CameraControls —— 它只凍結「還在自動」的項目，
    /// 使用者手動指定過的（快門/ISO/白平衡/對焦）維持自訂值。
    /// 「預設鎖定」與「調整完再鎖」因此是同一條路徑，不會互相覆蓋。
    /// 平面圖摘要 ＋ 一道軸序自我檢查。
    ///
    /// 為什麼需要這道檢查：我們假設 Surface.dimensions = (寬, 高, 厚)、且 transform 的
    /// 局部 X 軸沿牆面寬度方向。這個假設在桌機上無法驗證，而一旦相反，
    /// 每面「牆長」會全部變成樓高（約 2.4m）、平面圖整張報廢 —— 但看起來還是有線條，
    /// 不會有任何錯誤訊息。所以直接檢查樓高是否落在合理區間，錯了就明講。
    private func logFloorPlan(_ fp: FloorPlanData) {
        let w = fp.boundsM.count == 4 ? fp.boundsM[2] - fp.boundsM[0] : 0
        let d = fp.boundsM.count == 4 ? fp.boundsM[3] - fp.boundsM[1] : 0
        print(String(format: "平面圖: %d 房、%d 牆、%d 門、%d 窗、%d 家具，外接 %.2f×%.2fm",
                     fp.roomCount, fp.walls.count, fp.doors.count,
                     fp.windows.count, fp.objects.count, w, d))
        let heights = fp.walls.compactMap { $0.dimensions.count > 1 ? $0.dimensions[1] : nil }.sorted()
        let lengths = fp.walls.map(\.lengthM).sorted()
        guard !heights.isEmpty else { return }
        let mh = heights[heights.count / 2], ml = lengths[lengths.count / 2]
        print(String(format: "  牆長中位數 %.2fm、樓高中位數 %.2fm", ml, mh))
        if !(2.0...3.6).contains(mh) {
            print("  ⚠️ 樓高中位數不在 2.0~3.6m —— RoomPlan 的 dimensions 軸序可能與假設相反，"
                  + "平面圖的牆長會是錯的。請改用 dimensions[0] 當高、[1] 當寬（見 FloorPlanData.segment）")
        }
    }

    /// 平面圖三種輸出，各有各的用途，所以都寫：
    ///   floorplan.usdz — RoomPlan 原生，帶完整 3D 幾何與門窗語意，可直接進 CAD / BIM
    ///   floorplan.json — 參數化資料 ＋ 已投影到水平面的 2D 線段，給程式化後處理用
    ///   floorplan.svg  — 直接看得到的俯視平面圖（含 1m 網格、牆長標註、比例尺）
    /// 座標維持 ARKit 原生（+Y up、公尺），與 points.ply / poses.jsonl 一致；
    /// 匯出 COLMAP 用的世界翻轉**不**套用在這裡 —— 那是 3DGS 生態的慣例，平面圖不需要。
    private func writeFloorPlan(to dir: URL) async {
        guard config.captureFloorPlan, FloorPlanCapture.isSupported else { return }
        await floorPlan.exportUSDZ(to: dir.appendingPathComponent("floorplan.usdz"))
        guard let fp = floorPlanData else { return }
        do {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(fp).write(to: dir.appendingPathComponent("floorplan.json"),
                                     options: [.atomic])
            try Data(fp.svg().utf8).write(to: dir.appendingPathComponent("floorplan.svg"),
                                          options: [.atomic])
        } catch {
            print("[FloorPlan] 寫檔失敗: \(error)")
        }
    }

    /// 開始一段平面圖擷取。fresh = 全新掃描（清空累積）；否則累積成另一個房間。
    private func startFloorPlan(fresh: Bool) {
        guard config.captureFloorPlan, FloorPlanCapture.isSupported,
              let session = arView?.session else { return }
        if fresh {
            floorPlan.reset()
            floorPlanData = nil
        }
        floorPlan.start(on: session)
    }

    private func applyCameraLocks() { cameraControls.lockForScan() }

    private func releaseCameraLocks() { cameraControls.unlock() }

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

        // 點雲連續融合（~10Hz，與快門解耦）：只收「姿態可靠 + 清晰 + 相機夠穩」的幀。
        // 追蹤丟失/模糊（captureBlocked, blurPixels）+ 相機速度閘門（angular/linear）三管齊下：
        // 移動過快時姿態延遲/誤差大 → 深度投影到錯位的世界座標 → 點不貼合表面且出殘影，故直接跳過。
        if frameCounter % config.previewFrameInterval == 0,
           !a.captureBlocked, a.blurPixels <= config.previewMaxBlurPixels,
           a.angularSpeedRadS <= config.previewMaxAngularSpeedRadS,
           a.linearSpeedMS <= config.previewMaxLinearSpeedMS {
            lastPreviewFrame = frameCounter
            integratePreview(frame, blurPixels: a.blurPixels)
        }

        // 缺角提醒（物件模式圓頂）：每 ~12 幀（~5Hz）更新「離你最近的缺角在哪、往哪補」
        if let viz = visualizer, viz.hasDome, frameCounter % 12 == 0 {
            viz.updateGuidance(pose: frame.camera.transform)
        }

        guard a.allowCapture else { return }
        // 速度閘門：blur 擋不住「亮處快速移動」（曝光短 → 模糊低，但 VIO 姿態仍有延遲誤差）。
        // 關鍵幀姿態會同時進入重融合的反投影與 3DGS 訓練的相機外參，錯了兩邊一起壞。
        guard a.angularSpeedRadS <= config.keyframeMaxAngularSpeedRadS,
              a.linearSpeedMS <= config.keyframeMaxLinearSpeedMS else { return }
        // 清晰度閘門：上面兩道都是「由運動推估模糊」，對失焦與 AF 拉焦完全盲目
        // （手機靜止、對焦跑掉 → 推估值 0，照樣存檔）。這道直接量影像本身。
        // 被擋下只是等幾幀：SmartShutter 的觸發條件在拍成之前一直成立。
        guard a.sharpnessRatio >= config.minSharpnessRatio else {
            sharpnessRejects += 1
            return
        }
        guard pendingWrites < config.maxPendingWrites else { return }
        guard shutter.shouldCapture(pose: frame.camera.transform,
                                    time: frame.timestamp,
                                    config: config) else { return }
        // 關鍵幀一律也進預覽融合。預覽的閘門比關鍵幀嚴很多
        // （10Hz 取樣 + 角速度 0.6 vs 1.0、線速度 0.5 vs 0.8、模糊 12 vs 16），
        // 所以會出現「掃了、關鍵幀也存了，但預覽沒點」——停止後重融合才冒出來。
        // 這會讓融合熱圖說謊：把其實會有點的區域標成缺點，害使用者去重掃已經夠的地方。
        // 讓預覽的覆蓋 ⊇ 重融合會用到的，熱圖才有參考價值。
        if frameCounter != lastPreviewFrame {
            lastPreviewFrame = frameCounter
            integratePreview(frame, blurPixels: a.blurPixels)
        }
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
            let camPos = MatrixUtil.position(frame.camera.transform)
            previewInFlight = true
            Task {
                await accumulator.integrateSparse(points: points, anchorTransforms: anchorSnapshot,
                                                  cameraPosition: camPos)
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
        let tiles = await accumulator.dirtyTileRenderData(limit: 2, mode: colorMode)
        for tile in tiles {
            let xform = latestTileTransforms[tile.key]
                ?? { var m = matrix_identity_float4x4
                     m.columns.3 = SIMD4<Float>(tile.center.x, tile.center.y, tile.center.z, 1)
                     return m }()
            visualizer?.updateTile(tile, transform: xform)
        }
        pointCount = await accumulator.count
        fusionCompleteness = await accumulator.fusionCompleteness
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
            iso: cameraControls.currentISO,
            ambientLux: frame.lightEstimate.map { Double($0.ambientIntensity) },
            estimatedBlurPx: Double(a.blurPixels),
            sharpness: Double(a.sharpness),
            sharpnessRatio: Double(a.sharpnessRatio),
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
