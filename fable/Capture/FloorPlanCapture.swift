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

@MainActor
final class FloorPlanCapture: NSObject, ObservableObject {

    /// RoomPlan 的即時引導提示（「請靠近一點」等），可直接顯示在 HUD
    @Published private(set) var instruction: String?
    /// 已完成的掃描段數（＝將被合併的房間數）
    @Published private(set) var segmentCount = 0

    private var session: RoomCaptureSession?
    private var segments: [CapturedRoomData] = []
    private var capturing = false
    private var endContinuation: CheckedContinuation<Void, Never>?

    static var isSupported: Bool { RoomCaptureSession.isSupported }

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
    func waitForSegment(timeout: TimeInterval = 8) async {
        guard capturing else { return }
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            endContinuation = c
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(timeout))
                if self?.endContinuation != nil {
                    print("[FloorPlan] 等待 RoomPlan 最終資料逾時（\(timeout)s），略過平面圖")
                }
                self?.releaseWait()
            }
        }
        session = nil
        capturing = false
    }

    private func releaseWait() {
        guard let c = endContinuation else { return }   // 已被另一方取走 → 不重複 resume
        endContinuation = nil
        c.resume()
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
