//
//  FloorPlanCapture.swift
//  fable — 與 3DGS 採集共用同一個 ARSession 的 RoomPlan 平面圖擷取
//
//  為什麼能共生：RoomCaptureSession 可以吃外部提供的 ARSession（iOS 17+），
//  而 RoomPlan 想要的輸入條件我們本來就已經滿足 ——
//  worldAlignment = .gravity（垂直方向零誤差，牆才站得直）、
//  frameSemantics 含 .sceneDepth、sceneReconstruction = .mesh。
//  於是同一次走動同時產出 3DGS 關鍵幀與平面圖，使用者不必掃兩趟。
//
//  為什麼不自己從 ARMeshAnchor 做平面擬合：牆面偵測、開口/門窗語意、
//  Manhattan 正則化（把累積的 yaw 誤差硬拉回 90°）RoomPlan 都做了，而且做得更好。
//  重寫一遍只會得到一個更差的版本。
//
//  多房間：每一段掃描（含 review 後「繼續掃描」）產出一個 CapturedRoomData，
//  最後由 StructureBuilder 合併成整層。這正好對上既有的 resumeScan 流程 ——
//  而「一間一間掃再合併」也是精度最好的做法（一鏡到底會在門口累積漂移）。
//

import Foundation
import ARKit
import Combine
import RoomPlan
import simd

/// RoomPlan 偵測到的一個元素，已剝離 RoomPlan 型別 —— 渲染層不必認識 CapturedRoom。
/// transform 是它的局部座標系，size 為 (寬, 高, 深)。
nonisolated struct RoomSurface: Sendable {
    enum Kind: Sendable { case wall, door, window, opening, object }
    var transform: simd_float4x4
    var size: SIMD3<Float>
    var kind: Kind

    /// 家具畫成 3D 線框盒（12 條邊），牆／門／窗畫成平面矩形（4 條邊）。
    ///
    /// **家具的盒子才是那個視覺的主角。** RoomPlan 原生畫面裡最醒目的就是沙發、
    /// 桌子、櫃子上那些發光立方體 —— 它們直接告訴使用者「這件東西已經被認出來了」。
    /// 只畫牆的話畫面幾乎是空的，因為牆通常在視野邊緣。
    var isBox: Bool { kind == .object }
}

@MainActor
final class FloorPlanCapture: NSObject, ObservableObject {

    /// RoomPlan 的即時引導提示（「請靠近一點」等），可直接顯示在 HUD
    @Published private(set) var instruction: String?
    /// 已完成的掃描段數（＝將被合併的房間數）
    @Published private(set) var segmentCount = 0

    /// 本段目前偵測到的牆數與最高的一面牆（公尺）。
    ///
    /// **這兩個數字必須在掃描當下就看得到，不能等到掃完。** 每一份實機 log 都是
    /// 「1 房、2 牆、樓高 0.80m ⚠️ 掃描不完整」—— 而那行警告印在 review 階段，
    /// 那時整趟掃描已經結束、大範圍場景不可能重走一遍。
    /// RoomPlan 的 didUpdate 每幀都給出當下的房間幾何，資訊本來就在，只是沒被用。
    @Published private(set) var wallCount = 0
    @Published private(set) var maxWallHeightM: Float = 0

    /// 即時房間面的回呼，交給渲染層疊在相機畫面上。
    /// 用回呼而不是 @Published：這是每 0.4s 一組陣列，走 SwiftUI 的發佈管線
    /// 等於每次都讓整個 HUD 重新求值，而它其實只有 SceneKit 需要。
    var onRoomUpdated: (([RoomSurface]) -> Void)?
    private var lastSurfacePush: CFTimeInterval = 0

    /// 給 HUD 的單一提示字串。優先序：RoomPlan 自己的引導 > 我們的牆高檢查。
    /// RoomPlan 的引導比較急迫（它知道自己正在丟失追蹤），牆高則是慢性問題。
    var coachingHint: String? {
        if let instruction { return instruction }
        // 一面牆都還沒偵測到時不催 —— 剛開始掃本來就沒有
        guard wallCount > 0 else { return nil }
        if maxWallHeightM < Self.kMinWallHeightM {
            return String(format: "牆只掃到 %.1fm 高 —— 請把鏡頭往上帶到牆與天花板的交界",
                          maxWallHeightM)
        }
        return nil
    }

    /// 牆高低於此值就提示往上掃（公尺）。
    /// 一般住宅樓高 2.4~2.8m；掃到 1.8m 以上代表使用者確實有往上帶，
    /// 剩下的由 RoomPlan 自己外推。設太高會在正常掃描時一直跳提示。
    static let kMinWallHeightM: Float = 1.8

    private var session: RoomCaptureSession?
    private var segments: [CapturedRoomData] = []
    private var capturing = false
    /// 多個並行等待者（見 waitForSegment）。用 id 索引，讓各自的逾時只喚醒自己
    private var waiters: [Int: CheckedContinuation<Void, Never>] = [:]
    private var nextWaiterID = 0

    static var isSupported: Bool { RoomCaptureSession.isSupported }

    /// 這個面是不是一面**站得直**的牆（門窗開口同理）。
    ///
    /// **這個檢查在幾何上是硬的，不是啟發式。** ARSession 用
    /// worldAlignment = .gravity，世界 +Y 永遠是反重力方向 ——
    /// 所以真正的牆，法向必然是水平的（與 +Y 垂直）。
    /// 法向明顯偏離水平的「牆」不可能是牆，是 RoomPlan 在非房間場景
    /// （對著桌面、螢幕、隔板堆掃）硬判出來的東西。
    ///
    /// 為什麼要擋：那些面會以歪斜的巨大線框橫跨整個畫面，
    /// 而且第一片判錯之後後續的面會跟著它對齊 ——
    /// 實機回報的「地板一開始歪，牆壁和桌子都會歪」就是這樣連鎖出來的。
    /// Surface 的區域座標系是面躺在 XY 平面、Z 為法向，所以取第 2 欄。
    ///
    /// 15° 的容許量：RoomPlan 自己的牆面擬合有幾度誤差，太嚴會把真牆也擋掉。
    /// 家具（object）不套這個檢查 —— 它們本來就可以是任意朝向。
    static func isUpright(_ transform: simd_float4x4) -> Bool {
        let n = SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        let len = simd_length(n)
        guard len > 1e-5 else { return false }
        return abs(n.y / len) <= 0.26        // sin(15°)
    }

    // MARK: - 生命週期

    /// 掛到既有的 ARSession 上開始一段擷取。arSession 必須已在 run
    /// （由 CaptureController.runSession 負責）—— RoomPlan 只消費幀，不自己配置 session。
    func start(on arSession: ARSession) {
        guard Self.isSupported, !capturing else { return }
        let s = RoomCaptureSession(arSession: arSession)
        s.delegate = self
        var cfg = RoomCaptureSession.Configuration()
        cfg.isCoachingEnabled = false     // 我們有自己的 HUD，不要 RoomPlan 疊 UI
        s.run(configuration: cfg)
        session = s
        capturing = true
    }

    /// 開新掃描前清空累積（對應 startScan；resumeScan 不呼叫這個，才能累積多房間）
    func reset() {
        segments.removeAll()
        segmentCount = 0
        instruction = nil
        wallCount = 0
        maxWallHeightM = 0
        invalidateBuilt()
    }

    /// 停止擷取。**必須在 ARSession.pause() 之前呼叫**，並且要 await
    /// waitForSegment() 讓 RoomPlan 把最終資料送到 delegate —— session 被 pause 掉
    /// 之後它就收不完了。
    func stopCapture() {
        guard capturing else { return }
        session?.stop()
    }

    /// 等 didEndWith 把 CapturedRoomData 交回來。帶逾時，避免 delegate 不觸發時卡住整個流程
    /// （寧可少一張平面圖，不能讓掃描結果匯不出去）。
    ///
    /// **同時可以有多個等待者**：停止流程要等它才敢 pause（session 一停 RoomPlan
    /// 就收不完），背景建模也要等它。先前只存一個 continuation，
    /// 第二個呼叫會直接覆蓋掉第一個 —— 那個 task 就永遠不會恢復。
    /// 這種洩漏的症狀正是「平面圖有時候出不來」，而且不會有任何錯誤訊息。
    func waitForSegment(timeout: TimeInterval = 8) async {
        guard capturing else { return }      // idempotent：已收完就直接返回
        let id = nextWaiterID
        nextWaiterID += 1
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            waiters[id] = c
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(timeout))
                guard let self, let w = self.waiters.removeValue(forKey: id) else { return }
                // 逾時**不**把 capturing 設 false —— 資料可能還在路上，
                // 後面的等待者（build()）應該真的再等一次，而不是立刻返回。
                // 先前逾時也會清掉 capturing，於是 build() 的「再等一次」其實是空的。
                print("[FloorPlan] RoomPlan 最終資料尚未送達（等了 \(Int(timeout))s），"
                      + "先繼續後續處理 —— 建模時會再等")
                w.resume()
            }
        }
    }

    /// 資料到手：喚醒**所有**等待者，並結束擷取狀態
    private func releaseWait() {
        capturing = false
        session = nil
        let all = waiters
        waiters.removeAll()
        for c in all.values { c.resume() }
    }

    // MARK: - 建模

    /// 逐段建模並快取。RoomBuilder 是秒級的重運算，而 build() 與 exportUSDZ() 都需要結果，
    /// 快取避免同一份資料建兩次。StructureBuilder 吃的是 [CapturedRoom]（不是 RoomData），
    /// 所以合併前每一段都得先各自建成 room。
    private var builtRooms: [CapturedRoom]?

    private func rooms() async throws -> [CapturedRoom] {
        if let builtRooms { return builtRooms }
        let builder = RoomBuilder(options: [.beautifyObjects])
        var out: [CapturedRoom] = []
        for data in segments {
            out.append(try await builder.capturedRoom(from: data))
        }
        builtRooms = out
        return out
    }

    /// 把累積的所有段落建成平面圖。單段直接用，多段用 StructureBuilder 合併成整層。
    /// 這一步是重運算（秒級），呼叫端應在 processing 階段做、別擋在 UI 上。
    func build() async -> FloorPlanData? {
        // 先確保最後一段資料已到手。
        //
        // 為什麼這裡要再等一次：stopScan 已經 await 過 waitForSegment，但那一次是為了
        // 「在 ARSession.pause() 之前把資料收完」，有較短的逾時。
        // 而 build() 現在與重融合並行（重融合只花 ~0.6s），若 RoomPlan 的最終資料
        // 稍晚才到，build() 會看到空的 segments 而回 nil ——
        // 實機 log 出現過「逾時 8s、略過平面圖」但後來平面圖又出來了，
        // 就是這個競態剛好賭贏。waitForSegment 是 idempotent 的（capturing 為 false 時直接返回），
        // 所以在這裡再等一次是安全的，也讓結果不再靠運氣。
        await waitForSegment(timeout: 20)
        guard !segments.isEmpty else { return nil }
        do {
            let rs = try await rooms()
            guard let first = rs.first else { return nil }
            if rs.count == 1 { return FloorPlanData(room: first) }
            let structure = try await StructureBuilder(options: [.beautifyObjects])
                .capturedStructure(from: rs)
            return FloorPlanData(structure: structure)
        } catch {
            print("[FloorPlan] 建模失敗: \(error)")
            return nil
        }
    }

    /// 把 RoomPlan 原生的 USDZ 也寫出來（帶完整 3D 幾何與語意，可直接丟進 CAD/BIM 工具）
    func exportUSDZ(to url: URL) async {
        guard !segments.isEmpty else { return }
        do {
            let rs = try await rooms()
            guard let first = rs.first else { return }
            if rs.count == 1 {
                try first.export(to: url)
            } else {
                try await StructureBuilder(options: [.beautifyObjects])
                    .capturedStructure(from: rs).export(to: url)
            }
        } catch {
            print("[FloorPlan] USDZ 匯出失敗: \(error)")
        }
    }

    /// 開新掃描時也要丟掉快取，否則第二次掃描會拿到上一次的房間
    private func invalidateBuilt() { builtRooms = nil }
}

// MARK: - RoomCaptureSessionDelegate

extension FloorPlanCapture: @preconcurrency RoomCaptureSessionDelegate {

    /// 注意：這個回呼**可能在 waitForSegment 逾時之後才到**。
    /// 那不是錯誤 —— 資料照樣收下，build() 會拿到它（build 內也會再等一次）。
    func captureSession(_ session: RoomCaptureSession,
                        didEndWith data: CapturedRoomData, error: Error?) {
        if let error {
            print("[FloorPlan] 本段擷取結束但有錯誤: \(error)")
        } else {
            segments.append(data)
            segmentCount = segments.count
            builtRooms = nil        // 多了一段 → 先前建好的合併結果失效
        }
        releaseWait()
    }

    func captureSession(_ session: RoomCaptureSession,
                        didProvide instruction: RoomCaptureSession.Instruction) {
        self.instruction = Self.text(for: instruction)
    }

    /// 即時房間幾何。兩個用途：
    ///   · 在**掃描當下**就知道牆掃得夠不夠高（dimensions 是 (width, height, depth)，牆高取 .y）
    ///   · 把面交給渲染層疊在相機畫面上，讓使用者看得到「已經掃到哪」
    func captureSession(_ session: RoomCaptureSession, didUpdate room: CapturedRoom) {
        // 牆數與牆高也只算站得直的那些 —— 否則「牆只掃到 0.8m」的提示會被
        // 誤判出來的歪斜面餵飽，該提示的時候反而不提示
        let upright = room.walls.filter { Self.isUpright($0.transform) }
        wallCount = upright.count
        maxWallHeightM = upright.reduce(Float(0)) { max($0, $1.dimensions.y) }

        // 節流：didUpdate 觸發得比畫面更新還密，每次都重建幾何會讓 SceneKit 一直重上傳。
        //
        // 0.4 → 0.2s：0.4s 的更新間隔即使補了動畫仍然看得出一段一段。
        // 這一段的成本是「把 ~30 個元素的節點屬性重設一次」（約 360 次賦值）＋
        // dollhouse 的包圍盒，量級是毫秒；提高到 5Hz 仍遠低於掃描時的預算，
        // 而流暢度是使用者直接感受得到的。
        let now = CACurrentMediaTime()
        guard now - lastSurfacePush > 0.2 else { return }
        lastSurfacePush = now

        var out: [RoomSurface] = []
        out.reserveCapacity(room.walls.count + room.doors.count + room.windows.count
                            + room.openings.count + room.objects.count)
        func add(_ list: [CapturedRoom.Surface], _ kind: RoomSurface.Kind) {
            for s in list where Self.isUpright(s.transform) {
                out.append(RoomSurface(transform: s.transform, size: s.dimensions, kind: kind))
            }
        }
        add(room.walls, .wall)
        add(room.doors, .door)
        add(room.windows, .window)
        add(room.openings, .opening)
        // 家具：AR 疊加**不過濾**活動家具。
        // 平面圖那邊會濾掉椅子沙發（見 FloorPlanData+RoomPlan 的 kFixtureCategories），
        // 但那是製圖的取捨；在掃描當下，「沙發已經被認出來了」正是使用者要的回饋。
        for o in room.objects {
            out.append(RoomSurface(transform: o.transform, size: o.dimensions, kind: .object))
        }
        onRoomUpdated?(out)
    }

    private static func text(for i: RoomCaptureSession.Instruction) -> String? {
        switch i {
        case .moveCloseToWall:   "靠近牆面一點"
        case .moveAwayFromWall:  "離牆面遠一點"
        case .slowDown:          "放慢一點"
        case .turnOnLight:       "光線不足，請開燈"
        case .normal:            nil
        case .lowTexture:        "此處紋理不足，請對準有特徵的區域"
        @unknown default:        nil
        }
    }
}
