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
    /// 即時點雲疊加。**預設關閉** —— 掃描當下最該看的是「哪裡還沒掃到」，
    /// 而滿畫面的點會把真實場景與 RoomPlan 的結構線都蓋掉。要看隨時可以開。
    @Published var showPointCloud = false
    /// RoomPlan 即時結構疊加（牆／門／窗的發光邊框）。預設開啟：
    /// 它直接回答「掃到哪了」，而且面積小、不擋畫面。
    @Published var showRoomPlan = true
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
    /// 特徵追蹤：掃描時同步抽角點並建立跨幀對應，停止後餵給 BundleAdjuster。
    /// 放在掃描期做而非停止後，是因為停止後要重新解碼 120 張 JPEG（~2.4s，比 BA 還貴），
    /// 而此刻影像已經在記憶體裡。
    private let featureTracker = FeatureTracker()
    /// 建好的平面圖，於 processing 階段產生 → review 可顯示、匯出時寫檔
    @Published private(set) var floorPlanData: FloorPlanData?
    /// 是否以上次的 ARWorldMap 開始 —— 讓這次掃描與上次落在**同一個座標系**。
    /// 這是跨 session（關掉 app 再掃下一個房間）唯一的共同參考；
    /// 同一次 session 內的續掃 ARKit 本來就會自動重定位，不需要它。
    @Published private(set) var continueFromLastMap = false
    /// ARKit 正在以舊地圖重定位（尚未接上）。此時姿態不可信，必須擋住開拍。
    @Published private(set) var relocalizing = false
    /// 迴環閉合提示：走遠之後提醒回起點，讓 ARKit 修正整條軌跡的累積漂移
    @Published private(set) var loopHint: String?
    /// RoomPlan 的即時引導 ＋ 牆高不足提示（見 FloorPlanCapture.coachingHint）。
    /// **這是平面圖品質最重要的一條回饋**：每一份實機 log 都是
    /// 「2 牆、樓高 0.80m ⚠️ 掃描不完整」，而那行警告先前只在 review 才印 ——
    /// 大範圍場景到那時已經不可能重走一遍。
    @Published private(set) var floorPlanHint: String?
    /// 近 4 秒因清晰度不足而放棄的抓幀數（0 = 沒在掉幀）
    @Published private(set) var recentRejectCount = 0
    /// 掃描品質摘要，review 階段顯示（原本只印在 log 裡，使用者看不到）
    @Published private(set) var scanSummary: ScanSummary?

    /// review 期間是否疊出平面圖預覽（匯出前先驗證，不要盲匯）
    @Published var showFloorPlan = false
    /// 平面圖是否連活動家具（椅子/沙發/桌子/電視）一起畫。
    /// 預設關：建築製圖只畫固定設備，活動家具會蓋住圖面。
    /// 畫面與匯出共用這個旗標 —— 看到的就是匯出的。
    @Published var showPlanFurniture = false
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
    /// 迴環閉合追蹤：起點、累積行走距離、是否已閉合過
    private var scanStartPosition: SIMD3<Float>?
    private var traveledM: Float = 0
    private var lastTravelPosition: SIMD3<Float>?
    private var loopClosed = false
    private var lastWorldMapMB: Double?
    /// BA 的結果。**注意位姿不一定被套用**（config.baApplyPoses），
    /// 所以摘要要一併記下「有沒有套用」與保留集判定，否則 baAfterPx 會被誤讀成
    /// 「輸出的解析度天花板」，而實際輸出用的是 ARKit 位姿。
    private var baResult: PoseRefineResult?
    /// 因清晰度不足而放棄抓幀的次數（診斷用：拿來判斷門檻是否過嚴）
    private var sharpnessRejects = 0
    /// 近幾秒被放棄的時間戳。用來即時告訴使用者「你正在掉幀」——
    /// 這是**實測結果**，比 blurPixels 那個推估值可靠：實機 log 出現過
    /// 「69% 的幀被清晰度閘門丟掉，但 HUD 全程沒有任何警告」，
    /// 因為推估值(11.4px 中位數)沒到警告線(14px)，而清晰度閘門從 ~11px 就開始擋。
    /// 與其去猜兩個門檻要怎麼對齊，不如直接把發生的事講出來。
    private var recentRejects: [TimeInterval] = []
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
        // 初始狀態必須在這裡套用一次 —— 先前只有 toggle 時才呼叫 setPointCloudHidden，
        // 所以預設值改成 false 之後，第一次進畫面仍然會看到點雲。
        viz.setPointCloudHidden(!showPointCloud)
        viz.setRoomHidden(!showRoomPlan)
        // RoomPlan 的即時面直接進渲染層，不繞 SwiftUI（每 0.4s 一組陣列，
        // 走 @Published 等於每次都讓整個 HUD 重新求值）
        floorPlan.onRoomUpdated = { [weak viz] surfaces in
            viz?.updateRoomSurfaces(surfaces)
            viz?.updateDollhouse(surfaces)
        }
        viz.setDollhouseHidden(!showRoomPlan)
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
        // 帶入上次的地圖 → ARKit 進入 relocalizing，鏡頭對回掃過的區域就會接上，
        // 之後的姿態與上次同座標系。Apple 要求搭配 .resetTracking 執行（見 runSession）。
        if continueFromLastMap, let map = WorldMapStore.loadLatest() {
            cfg.initialWorldMap = map
        }
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

    /// 切換「延續上次座標系」。**立刻重跑 session** 而不是等按快門 ——
    /// 重定位需要使用者把鏡頭對回舊區域、可能要幾秒，
    /// 這件事必須發生在開拍之前，否則等於用不可信的姿態拍了一段。
    func setContinueFromLastMap(_ on: Bool) {
        guard phase == .idle else { return }
        continueFromLastMap = on && WorldMapStore.hasLatest
        relocalizing = false
        trackingReady = false
        runSession()
        statusText = continueFromLastMap
            ? "請把鏡頭對準上次掃描過的區域，等待重新定位"
            : nil
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
        recentRejects = []
        recentRejectCount = 0
        Task { await featureTracker.reset() }
        scanStartPosition = nil
        lastTravelPosition = nil
        traveledM = 0
        loopClosed = false
        loopHint = nil
        floorPlanHint = nil
        lastWorldMapMB = nil
        baResult = nil
        scanSummary = nil
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
        let tStop = Date()
        Task {
            // 世界地圖：只有「取圖」需要活著的 session，序列化不需要。
            // 所以取完就把序列化丟到背景並與 processScan 並行 ——
            // 先前是 await 整個存檔完成才開始後處理，那一整段是使用者的乾等，
            // 而且序列化跑在 main actor 上，連進度條都動不了。
            let box = await captureWorldMap()
            let mapTask = Task { @MainActor [weak self] in
                if let box { await self?.persistWorldMap(box) }
            }

            // **pause 必須等 RoomPlan 交回 CapturedRoomData** —— session 一停它就收不完。
            //
            // 這裡踩過一次坑：先前是「取完地圖立刻 pause」，而註解寫「實測資料照樣送達」。
            // 那個實測成立是因為當時 pause 之前還夾著 saveWorldMap（序列化 10~40MB
            // ＋ 兩次寫檔，1~3 秒），剛好給了 RoomPlan 送資料的時間。
            // 把序列化搬到背景之後那段緩衝消失，平面圖就出不來了 ——
            // 一個「加速」改動意外拿掉了另一件事賴以成立的前提。
            //
            // 正解是不要靠巧合：pause **不在使用者的等待路徑上**（使用者等的是
            // processScan），所以讓它明確地等 RoomPlan，與後處理並行。
            // 代價只是 ARSession 多活幾秒，而幀處理本來就被 phase != .scanning 擋掉了。
            let roomTask = Task { @MainActor [weak self] in
                await self?.floorPlan.waitForSegment(timeout: 12)
                self?.arView?.session.pause()       // review 期間停止追蹤，省電省熱
                self?.monitor.stop()
            }

            await processScan(refinedTransforms: refined, meshVertices: meshVerts, since: tStop)
            // 兩者通常早就完成了；等一下只是確保摘要拿得到地圖大小、
            // 以及 session 一定有被 pause 掉（不然 review 期間會一直吃電）
            await mapTask.value
            await roomTask.value
            if scanSummary != nil { scanSummary?.worldMapMB = lastWorldMapMB }
        }
    }

    private func processScan(refinedTransforms: [Int: [Double]], meshVertices: [SIMD3<Float>],
                             since tStop: Date) async {
        guard let writer, let accumulator, let dir = sessionDir else {
            phase = .idle
            return
        }
        // 分段計時從**按下停止**起算。先前 t0 設在重融合前面，於是「總計 1.81s」
        // 完全不含世界地圖、BA、模糊複核 —— 一個叫「總計」卻不是總計的數字，
        // 正是我在這個專案被誤導過三次的同一類錯誤。
        var seg: [(String, Double)] = []
        var tMark = tStop
        func mark(_ name: String) {
            seg.append((name, Date().timeIntervalSince(tMark)))
            tMark = Date()
        }
        // 每一段都講出來，並且讓進度條涵蓋整條流程。
        //
        // 先前進度條只由重融合的回呼驅動，而重融合是**最後**一段 ——
        // 前面三段使用者看到的是一條靜止在 0% 的進度條，那比沒有進度條更像卡住。
        // 權重是暫定的：真實比例要等新的分段計時（見下方 mark）跑過實機才知道。
        func stage(_ text: String, _ base: Double) {
            statusText = text
            exportProgress = base
        }

        stage("讀取關鍵幀…", 0)
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
        defer { reportDrift(raw: raw, refined: refinedRecords) }
        refinedRecords = raw.map { record in
            var r = record
            if let t = refinedTransforms[r.id] {
                if t != r.transform { corrected += 1 }
                r.transform = t
            }
            return r
        }
        // 局部 BA：以「ARKit ＋ 錨點修正」為初值，用掃描時建好的跨幀對應做微調。
        // 位置在這裡的理由：
        //   · 必須在錨點修正**之後** —— 那是初值，BA 只做微調
        //   · 必須在 BlurFilter 與重融合**之前** —— 它們都吃姿態，晚了就白做
        mark("讀取關鍵幀")
        if config.baRounds > 0 {
            stage("校正相機位姿…", 0.10)
            let obs = await featureTracker.observations()
            print(await featureTracker.stats())
            let ba = BundleAdjuster.refine(records: refinedRecords, observations: obs,
                                           rounds: config.baRounds)
            // ba.poses 只在保留集通過閘門時才非空（見 BundleAdjuster.kHoldoutGate）——
            // 也就是「這次掃描的 BA 確實讓沒參與求解的 track 也變準了」。
            if !ba.poses.isEmpty && config.baApplyPoses {
                refinedRecords = refinedRecords.map { r in
                    guard let m = ba.poses[r.id] else { return r }
                    var out = r
                    out.transform = RefusionEngine.rowMajor(m)
                    return out
                }
            } else if !ba.poses.isEmpty {
                // 一定要講出來。BA 的那幾行 log 照樣印，若不說「沒套用」，
                // 看 log 的人（包括我自己）會以為輸出用的是修正後的位姿。
                print("  ⚠️ 保留集通過但 config.baApplyPoses = false（硬總開關）"
                      + " → 輸出仍用 ARKit＋錨點修正的位姿")
            }
            baResult = ba
        }

        // 模糊幀全域複核。必須在姿態修正**之後**：BlurFilter 靠位置/朝向找「看同一片表面」
        // 的鄰居，用未修正的姿態會找錯鄰居。判定寫回紀錄而非直接刪除，
        // poses_refined.jsonl 與 images/ 都保留完整，可回頭檢查判定對不對。
        mark("位姿校正")
        stage("複核模糊幀…", 0.35)
        refinedRecords = BlurFilter.annotate(refinedRecords)
        let dropped = refinedRecords.filter { $0.blurVerdict == .drop }.count
        let demoted = refinedRecords.filter { $0.blurVerdict == .demote }.count
        if dropped + demoted > 0 {
            print("模糊複核: \(refinedRecords.count) 幀 → 排除 \(dropped) 幀（幾何不可信，點雲也不用）"
                  + "、\(demoted) 幀（顏色糊，不進訓練但深度仍以降權併入點雲）")
        }

        // 平面圖**不擋 review**。
        //
        // 實測 RoomPlan 從 stop() 到 didEndWith 要 8 秒以上（它自己的最終優化），
        // 而重融合只花 1 秒。先前把兩者並行仍然要等較慢的那個 ——
        // 使用者按下停止後乾等 8 秒才看到點雲，而平面圖其實只有
        // 「review 按平面圖」和「匯出」兩個時機才需要。
        // 改成背景建，好了再更新 @Published；review 的平面圖按鈕本來就綁 floorPlanData != nil，
        // 所以它會自己出現。
        mark("模糊複核")
        if config.captureFloorPlan, FloorPlanCapture.isSupported {
            Task { @MainActor [weak self] in
                guard let self else { return }
                let fp = await self.floorPlan.build()
                self.floorPlanData = fp
                if let fp { self.logFloorPlan(fp) }
            }
        }

        var points: [CloudPoint] = []
        if hasLiDAR && config.saveDepth && !refinedRecords.isEmpty {
            stage("融合點雲…", 0.45)
            let records = refinedRecords
            let cfg = config
            // 重融合佔進度條的後 55%（前面三段各自佔一段，見 stage）
            let onProg: @Sendable (Double) -> Void = { p in
                Task { @MainActor [weak self] in self?.exportProgress = 0.45 + p * 0.55 }
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
        mark("重融合")

        // 逐段列出，而且總計就是「按下停止到看到 review」的牆鐘時間 ——
        // 這樣下次要優化才知道該動哪一段，不必再猜。
        let total = seg.reduce(0) { $0 + $1.1 }
        print(String(format: "處理耗時: 總計 %.2fs（按下停止 → review）= ", total)
              + seg.map { String(format: "%@ %.2fs", $0.0, $0.1) }.joined(separator: " + ")
              + "（世界地圖與平面圖在背景，不計入）")

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

    func toggleRoomPlan() {
        showRoomPlan.toggle()
        visualizer?.setRoomHidden(!showRoomPlan)
        visualizer?.setDollhouseHidden(!showRoomPlan)
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
    /// 平面圖摘要 ＋ 掃描完整度判斷。
    ///
    /// 這裡原本放的是「dimensions 軸序自我檢查」，已移除 —— 那個判斷是錯的：
    /// Apple 文件明確定義 Surface.dimensions 為 (width, height, depth)，假設本來就對。
    /// 實機上樓高 1.83m 的成因是牆只被掃到 1.83m 高，不是軸序（同一份程式在另一次
    /// 掃描樓高正常，軸序若相反不可能只錯一次）。那道檢查唯一的效果是對不完整的掃描說謊。
    private func logFloorPlan(_ fp: FloorPlanData) {
        print(String(format: "平面圖: %d 房、%d 牆、%d 門、%d 窗、%d 家具，外接 %.2f×%.2fm",
                     fp.roomCount, fp.walls.count, fp.doors.count,
                     fp.windows.count, fp.objects.count, fp.sizeM.x, fp.sizeM.y))
        guard !fp.walls.isEmpty else { return }
        print(String(format: "  牆長中位數 %.2fm、最長牆 %.2fm、樓高中位數 %.2fm",
                     fp.medianWallLengthM, fp.longestWallM, fp.medianWallHeightM))
        if let reason = fp.incompleteReason {
            print("  ⚠️ 掃描不完整：\(reason)")
        }
    }

    /// 使用者為房間命名。改動會一併進到匯出的 floorplan.json / .svg
    /// （writeFloorPlan 讀的就是 floorPlanData）。
    func renameRoom(at index: Int, to name: String) {
        guard var fp = floorPlanData, fp.rooms.indices.contains(index) else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        fp.rooms[index].customLabel = trimmed.isEmpty ? nil : trimmed
        floorPlanData = fp
    }

    /// 平面圖三種輸出，各有各的用途，所以都寫：
    ///   floorplan.usdz — RoomPlan 原生，帶完整 3D 幾何與門窗語意，可直接進 CAD / BIM
    ///   floorplan.json — 參數化資料 ＋ 已投影到水平面的 2D 線段，給程式化後處理用
    ///   floorplan.svg  — 直接看得到的俯視平面圖（含 1m 網格、牆長標註、比例尺）
    /// 座標維持 ARKit 原生（+Y up、公尺），與 points.ply / poses.jsonl 一致；
    /// 匯出 COLMAP 用的世界翻轉**不**套用在這裡 —— 那是 3DGS 生態的慣例，平面圖不需要。
    private func writeFloorPlan(to dir: URL) async {
        guard config.captureFloorPlan, FloorPlanCapture.isSupported else { return }
        // 平面圖是背景建的（見 processScan），匯出時若還沒好就在這裡等 ——
        // 匯出是使用者明確要求的動作，少一個檔比多等幾秒糟。
        if floorPlanData == nil { floorPlanData = await floorPlan.build() }
        await floorPlan.exportUSDZ(to: dir.appendingPathComponent("floorplan.usdz"))
        guard let fp = floorPlanData else { return }
        do {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(fp).write(to: dir.appendingPathComponent("floorplan.json"),
                                     options: [.atomic])
            let svg = fp.svg(showAllFurniture: showPlanFurniture)
            try Data(svg.utf8).write(to: dir.appendingPathComponent("floorplan.svg"),
                                     options: [.atomic])
        } catch {
            print("[FloorPlan] 寫檔失敗: \(error)")
        }
    }

    /// 取出 ARKit 當下的地圖並存檔。必須在 session 還活著時呼叫。
    /// 追蹤狀態不佳時 ARKit 會拒絕給地圖（回 error）—— 那種地圖本來就不該留，
    /// 帶著它下次會一直重定位失敗。
    /// 非 Sendable 的 ARKit 物件單向交給背景。交出後 main 這邊不再碰它，
    /// 所以沒有共享可變狀態（與 Keyframe 的 pixelBuffer 同一個理由）。
    private struct MapBox: @unchecked Sendable { let map: ARWorldMap }

    /// 從**活著的** session 取出當下地圖。只有這一步需要 session，所以取完就能 pause。
    private func captureWorldMap() async -> MapBox? {
        guard let session = arView?.session else { return nil }
        return await withCheckedContinuation { c in
            session.getCurrentWorldMap { m, error in
                if let error { print("[WorldMap] 取得失敗（追蹤品質不足？）: \(error)") }
                c.resume(returning: m.map(MapBox.init))
            }
        }
    }

    /// 序列化並寫檔。**必須離開 main actor。**
    ///
    /// 這是「按下停止之後的第一段乾等」的真正來源：ARWorldMap 動輒 10~40MB，
    /// NSKeyedArchiver 是 CPU 密集的同步呼叫，先前直接跑在 main actor 上 ——
    /// 不只擋住畫面，連重融合的進度條都動不了，於是使用者看到的是一段完全靜止的等待。
    /// 而它跟後處理沒有任何依賴關係，本來就該平行。
    private func persistWorldMap(_ box: MapBox) async {
        let dir = sessionDir
        let anchors = box.map.anchors.count
        let bytes = await Task.detached(priority: .utility) {
            WorldMapStore.save(box.map, sessionDir: dir)
        }.value
        lastWorldMapMB = bytes.map { Double($0) / 1_048_576 }
        if let bytes {
            print(String(format: "世界地圖已保存 %.1f MB（%d 個錨點）—— 下次可選「延續上次座標系」",
                         Double(bytes) / 1_048_576, anchors))
        }
    }

    /// 迴環閉合追蹤。這是**降低漂移投報率最高**的一件事：ARKit 只有在認出
    /// 「我來過這裡」時才會做全域修正，把累積誤差攤回整條軌跡；
    /// 走一條開放路徑不回頭的話，誤差只會一路累積下去，而且不會有任何警告。
    /// 沒人會主動這樣做，所以必須提示。
    private func updateLoopClosure(_ frame: ARFrame) {
        let p = MatrixUtil.position(frame.camera.transform)
        guard let start = scanStartPosition else {
            scanStartPosition = p
            lastTravelPosition = p
            return
        }
        if let last = lastTravelPosition {
            let step = simd_distance(p, last)
            if step > 0.05 { traveledM += step; lastTravelPosition = p }   // 0.05m 門檻濾掉抖動
        }
        let fromStart = simd_distance(p, start)
        if traveledM >= config.loopHintTravelM, fromStart <= config.loopClosedRadiusM {
            if !loopClosed {
                loopClosed = true
                captureHaptic.impactOccurred()
            }
            loopHint = nil
        } else if !loopClosed, traveledM >= config.loopHintTravelM {
            loopHint = String(format: "已走 %.0f m —— 走回起點閉環，讓 ARKit 修正累積漂移",
                              traveledM)
        }
        // RoomPlan 的引導與牆高檢查。
        //
        // **為什麼在這裡同步而不是讓 HUD 直接讀 floorPlan**：SwiftUI 的
        // @ObservedObject 不會觀察巢狀的 ObservableObject，所以 controller.floorPlan
        // 改變時畫面不會更新。而這個專案沒有其他 Combine 訂閱，為了一個字串
        // 引入一套 publisher 管線不划算 —— 照 loopHint 既有的做法逐幀同步即可。
        // 只在值真的改變時寫入，否則每幀都會觸發一次 SwiftUI 更新。
        let fpHint = floorPlan.coachingHint
        if fpHint != floorPlanHint { floorPlanHint = fpHint }
    }

    /// 量化 ARKit 實際修正了多少漂移。
    /// 先前只數「有幾幀被改動」，但那不分「動了 1mm」和「動了 30cm」——
    /// 後者代表這次掃描漂移嚴重、幾何可信度低，使用者應該知道。
    private func reportDrift(raw: [FrameRecord], refined: [FrameRecord]) {
        var s = ScanSummary()
        s.keyframes = refined.count
        s.traveledM = Double(traveledM)
        s.loopClosed = loopClosed
        s.blurDropped = refined.filter { $0.blurVerdict == .drop }.count
        s.blurDemoted = refined.filter { $0.blurVerdict == .demote }.count
        s.worldMapMB = lastWorldMapMB
        s.baBeforePx = baResult?.residualsPx.first
        s.baAfterPx = baResult?.residualsPx.last
        // 這兩欄是為了讓摘要無法被誤讀：只有 baApplied 為真時，baAfterPx 才描述
        // 實際輸出；否則輸出的天花板是 baBeforePx，而 baHoldoutDelta 是「若套用會怎樣」。
        s.baApplied = config.baApplyPoses && !(baResult?.poses.isEmpty ?? true)
        s.baHoldoutDelta = baResult?.holdoutDelta
        defer { scanSummary = s }
        let byID = Dictionary(uniqueKeysWithValues: raw.map { ($0.id, $0.transform) })
        var deltas: [Double] = []
        for r in refined {
            guard let o = byID[r.id], o.count == 16, r.transform.count == 16 else { continue }
            let dx = r.transform[3] - o[3], dy = r.transform[7] - o[7], dz = r.transform[11] - o[11]
            deltas.append((dx * dx + dy * dy + dz * dz).squareRoot())
        }
        guard !deltas.isEmpty else { return }
        deltas.sort()
        let med = deltas[deltas.count / 2], worst = deltas[deltas.count - 1]
        s.driftMedianCm = med * 100
        s.driftMaxCm = worst * 100

        // 兩種修正的形狀完全不同，混為一談會誤導：
        //   累積漂移      中位數 ≪ 最大值（誤差沿軌跡累積，早期的幀幾乎沒動）
        //   重定位跳變    中位數 ≈ 最大值（整組幀一起位移，把世界對齊到舊地圖）
        // 實機出現過「中位數 34.3cm / 最大 34.4cm、只走了 0.9m」——
        // 那不是 38% 的漂移率，是開了「延續上次座標系」後 ARKit 重定位的全域對齊量。
        let uniform = worst > 0.02 && med / worst > 0.9
        if uniform {
            print(String(format: "全域重定位: 整組位姿一致位移 %.1f cm"
                         + "（中位數≈最大值 ⇒ 世界座標系被對齊到舊地圖，不是累積漂移）",
                         med * 100))
        } else {
            print(String(format: "漂移修正: 中位數 %.1f cm / 最大 %.1f cm（行走 %.1f m）",
                         med * 100, worst * 100, traveledM))
        }
        // 迴環只在「走得夠遠、本來就該閉環」時才值得提。
        // 走 0.9m 也印「未閉合」只是噪音 —— 那種距離根本無所謂閉不閉。
        if traveledM >= config.loopHintTravelM {
            print(loopClosed
                  ? "  迴環已閉合 —— ARKit 有機會做全域修正"
                  : "  ⚠️ 走了 \(Int(traveledM))m 但沒有回到起點，"
                    + "ARKit 沒有機會做全域修正，遠端的累積誤差留在資料裡了")
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
        // 重定位狀態：帶入舊地圖後 ARKit 會處於 relocalizing，此時姿態不可信
        if case .limited(.relocalizing) = frame.camera.trackingState {
            if !relocalizing { relocalizing = true }
        } else if relocalizing {
            relocalizing = false
            if continueFromLastMap { statusText = "已接上上次的座標系" }
        }

        guard phase == .scanning else { return }

        // 遮斷級警告觸覺回饋（限流 1 次 / 1.5 秒）
        if a.captureBlocked, frame.timestamp - lastWarningHaptic > 1.5 {
            warningHaptic.notificationOccurred(.warning)
            lastWarningHaptic = frame.timestamp
        }

        updateLoopClosure(frame)
        // 掉幀率：只留近 4 秒。連續掉幀代表使用者正在流失資料而不自知
        recentRejects.removeAll { frame.timestamp - $0 > 4 }
        if recentRejectCount != recentRejects.count { recentRejectCount = recentRejects.count }

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
        // Dollhouse 擺位：**每幀**都要更新，不能像上面那樣抽幀 ——
        // 它跟著相機走，5Hz 會看起來一頓一頓的。內容只是一次 transform 賦值，
        // 真正的重建是在 onRoomUpdated（節流 0.4s）那裡。
        if showRoomPlan { visualizer?.placeDollhouse(camera: frame.camera.transform) }

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
            recentRejects.append(frame.timestamp)
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
        let cfg = config
        let tracker = featureTracker
        Task {
            await writer.write(keyframe)
            self.pendingWrites -= 1
            // 特徵抽取放在寫檔之後：兩者共用同一份 clone 的 pixel buffer
            //（CVPixelBuffer 是 refcounted，兩個 actor 各自持有沒問題），
            // 而寫檔先做可以早一點釋放背壓（pendingWrites）。
            guard cfg.baRounds > 0, let dData = keyframe.depthData, dw > 0, dh > 0 else { return }
            let confArr = keyframe.confidenceData.map { [UInt8]($0) }
            await tracker.add(frameID: record.id, luma: keyframe.pixelBuffer,
                              depth: dData, conf: confArr, dw: dw, dh: dh,
                              K: record.intrinsics, c2w: keyframe.c2w,
                              minDepth: cfg.pointMinDepthM, maxDepth: cfg.pointMaxDepthM)
        }
    }
}
