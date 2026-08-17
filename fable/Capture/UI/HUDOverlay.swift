//
//  HUDOverlay.swift
//  fable — 掃描 HUD：警告膠囊、速度儀、統計面板、快門與匯出控制
//

import SwiftUI

struct HUDOverlay: View {
    @ObservedObject var controller: CaptureController
    @Environment(\.dismiss) private var dismiss
    @State private var showDiscardConfirm = false

    var body: some View {
        ZStack {
            severeGlow
            VStack(spacing: 10) {
                header
                guidanceBanner
                fusionLegend
                speedGauge
                HStack {
                    Spacer()
                    CameraControlBar(controls: controller.cameraControls,
                                     enabled: controller.phase == .idle)
                }
                Spacer()
                statusLine
                bottomControls
            }
            .padding()
        }
        .animation(.easeInOut(duration: 0.25), value: controller.assessment.worst)
        .animation(.easeInOut(duration: 0.25), value: controller.phase)
        .animation(.easeInOut(duration: 0.25), value: controller.coverageHint)
        .animation(.easeInOut(duration: 0.25), value: controller.loopHint)
        .animation(.easeInOut(duration: 0.25), value: controller.recentRejectCount >= 4)
        .animation(.easeInOut(duration: 0.25), value: controller.relocalizing)
    }

    // MARK: - 全螢幕紅框：遮斷級警告（暫停抓幀中）的強視覺提示

    @ViewBuilder
    private var severeGlow: some View {
        if controller.phase == .scanning, controller.assessment.captureBlocked {
            RoundedRectangle(cornerRadius: 28)
                .strokeBorder(Color.red.opacity(0.65), lineWidth: 5)
                .ignoresSafeArea()
                .allowsHitTesting(false)
        }
    }

    // MARK: - 單一提示插槽（依優先序只顯示最重要的一則）

    /// 四種提示（重定位／品質警告／迴環／缺角）**共用一個位置**。
    ///
    /// 它們原本各佔一條橫幅，最壞情況同時出現四條把取景畫面塞滿 ——
    /// 而使用者在任一時刻只能對一件事做出反應，多餘的那幾條只是雜訊。
    /// 優先序即「現在最該做什麼」：
    ///   1. 重定位中   姿態不可信、快門也被擋住，其他都不重要
    ///   2. 遮斷級警告 正在暫停抓幀，不處理就一直沒有資料
    ///   3. 迴環提示   影響全域精度，且錯過就補不回來
    ///   4. 缺角提醒   局部覆蓋，之後還能補
    ///   5. 提醒級警告 照拍，只是品質差一點
    private struct Guidance {
        let text: String
        let symbol: String
        let background: Color
        let foreground: Color
    }

    private var guidance: Guidance? {
        if controller.relocalizing {
            return Guidance(text: "重新定位中：請把鏡頭對準上次掃描過的區域",
                            symbol: "point.3.connected.trianglepath.dotted",
                            background: .blue.opacity(0.9), foreground: .white)
        }
        guard controller.phase == .scanning else { return nil }
        let a = controller.assessment
        if a.captureBlocked, let w = a.worst {
            return Guidance(text: w.message, symbol: w.symbol,
                            background: .red.opacity(0.88), foreground: .white)
        }
        // 正在掉幀：這是實測結果不是推估，優先於所有「可能會怎樣」的提示
        if controller.recentRejectCount >= 4 {
            return Guidance(text: "畫面不夠清晰，已略過 \(controller.recentRejectCount) 次抓幀 —— 請放慢",
                            symbol: "camera.metering.none",
                            background: .red.opacity(0.85), foreground: .white)
        }
        if let hint = controller.loopHint {
            return Guidance(text: hint, symbol: "arrow.triangle.capsulepath",
                            background: .yellow.opacity(0.92), foreground: .black)
        }
        if let hint = controller.coverageHint {
            return Guidance(text: hint, symbol: "scope",
                            background: .orange.opacity(0.85), foreground: .white)
        }
        if let w = a.worst {
            return Guidance(text: w.message, symbol: w.symbol,
                            background: .orange.opacity(0.88), foreground: .white)
        }
        return nil
    }

    @ViewBuilder
    private var guidanceBanner: some View {
        if let g = guidance {
            Label(g.text, systemImage: g.symbol)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(g.foreground)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(g.background, in: Capsule())
                .transition(.move(edge: .top).combined(with: .opacity))
        }
    }


    /// 熱圖圖例。只在熱圖模式顯示 —— 顏色本身要能自我解釋。
    /// 量的是「從幾個不同方向看過」而非次數：站著不動連拍不會讓顏色前進，
    /// 因為視差為零、幾何沒有被約束。
    @ViewBuilder
    private var fusionLegend: some View {
        if controller.phase == .scanning, controller.showPointCloud,
           controller.colorMode == .fusionQuality {
            HStack(spacing: 8) {
                Text("視角").font(.caption2.weight(.semibold))
                HStack(spacing: 3) {
                    ForEach(0..<7) { i in
                        let q = Double(i) / 6
                        Rectangle()
                            .fill(Color(red: q < 0.5 ? 1 : 2 * (1 - q),
                                        green: q < 0.5 ? 2 * q : 1,
                                        blue: 0.15))
                            .frame(width: 14, height: 8)
                    }
                }
                Text("單一角度 → 多角度").font(.caption2)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
        }
    }

    // MARK: - 頂部：關閉鍵 + 點雲開關 + 統計面板

    private var header: some View {
        HStack(alignment: .top) {
            VStack(spacing: 10) {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark")
                        .font(.headline)
                        .padding(10)
                        .background(.ultraThinMaterial, in: Circle())
                }
                .foregroundStyle(.white)
                .opacity(controller.phase == .scanning ? 0 : 1)

                if controller.phase == .scanning {
                    Button {
                        controller.togglePointCloud()
                    } label: {
                        Image(systemName: controller.showPointCloud
                              ? "circle.grid.3x3.fill" : "circle.grid.3x3")
                            .font(.headline)
                            .padding(10)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .foregroundStyle(controller.showPointCloud ? .cyan : .white)

                    // 融合品質熱圖：直接把「這塊還沒掃夠」畫在表面上，
                    // 比任何數字或文字提示都直觀 —— 使用者看到紅色就知道要再繞一次。
                    if controller.showPointCloud {
                        Button {
                            controller.toggleColorMode()
                        } label: {
                            Image(systemName: controller.colorMode == .fusionQuality
                                  ? "thermometer.medium" : "paintpalette")
                                .font(.headline)
                                .padding(10)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                        .foregroundStyle(controller.colorMode == .fusionQuality ? .orange : .white)
                    }
                }
            }

            Spacer()

            if controller.phase != .training {   // 訓練/檢視時隱藏掃描統計 bar
            VStack(alignment: .trailing, spacing: 3) {
                Label("\(controller.keyframeCount) 幀", systemImage: "camera.viewfinder")
                Label("\(controller.pointCount / 1000)k 點", systemImage: "circle.grid.3x3.fill")
                if controller.mode == .object && controller.domePlaced {
                    Label(String(format: "涵蓋 %.0f%%", controller.coverage * 100),
                          systemImage: "globe.asia.australia.fill")
                }
                // 融合完成度：場景模式沒有涵蓋率圓頂，這是唯一的「掃夠了沒」訊號。
                // 顏色即結論——紅/橘代表大部分表面觀測不足，別急著停。
                if controller.phase == .scanning {
                    let f = controller.fusionCompleteness
                    Label(String(format: "視角 %.0f%%", f * 100),
                          systemImage: "arrow.triangle.turn.up.right.diamond.fill")
                        .foregroundStyle(f < 0.3 ? .red : (f < 0.6 ? .orange : .green))
                }
                Label(storageEstimate, systemImage: "internaldrive")
                if !controller.hasLiDAR {
                    Label("無 LiDAR", systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.yellow)
                }
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.white)
            .padding(10)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    private static func ago(_ d: Date) -> String {
        let s = Int(Date().timeIntervalSince(d))
        if s < 90 { return "剛剛" }
        if s < 3600 { return "\(s / 60) 分鐘前" }
        if s < 86400 { return "\(s / 3600) 小時前" }
        return "\(s / 86400) 天前"
    }

    private var storageEstimate: String {
        let mb = Double(controller.keyframeCount) * 0.62
        return mb < 1000 ? String(format: "%.0f MB", mb) : String(format: "%.1f GB", mb / 1000)
    }

    // MARK: - 移動速度儀（綠＝安全、橘＝輕微模糊、紅＝暫停抓幀）

    @ViewBuilder
    private var speedGauge: some View {
        if controller.phase == .scanning {
            let blur = controller.assessment.blurPixels
            let cfg = controller.config
            HStack(spacing: 8) {
                Image(systemName: "tortoise.fill").font(.caption2)
                ProgressView(value: Double(min(1.0, blur / cfg.blockBlurPixels)))
                    .tint(blur > cfg.blockBlurPixels ? .red
                          : (blur > cfg.maxBlurPixels ? .orange : .green))
                    .frame(width: 130)
                Image(systemName: "hare.fill").font(.caption2)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
        }
    }

    // MARK: - 底部狀態與控制

    @ViewBuilder
    private var statusLine: some View {
        if let text = statusHint {
            Text(text)
                .font(.footnote)
                .foregroundStyle(.white)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
        }
    }

    private var statusHint: String? {
        if let s = controller.statusText { return s }
        switch controller.phase {
        case .idle:
            if !controller.trackingReady { return "初始化中：請緩慢平移手機讓 ARKit 建立追蹤" }
            if controller.mode == .object && !controller.domePlaced { return "點擊畫面中的物件，放置視角涵蓋圓頂" }
            return "按下快門開始掃描（每移動 10cm 或轉動 6° 自動抓幀）"
        case .scanning:
            return nil
        case .processing:
            return "點雲優化中：姿態修正 + 多視角加權融合…"
        case .review:
            return "檢查點雲品質：單指旋轉、雙指縮放；可繼續掃描補拍，或直接訓練成 3DGS"
        case .training:
            return nil
        case .exporting:
            return "打包 COLMAP 資料集…"
        case .done:
            return nil
        }
    }

    private var bottomControls: some View {
        VStack(spacing: 14) {
            if controller.phase == .idle {
                // 延續上次座標系：跨 session 掃下一個房間時，兩份資料才拼得起來。
                // 同一次 session 內的續掃 ARKit 會自動重定位，不需要這個。
                if let info = WorldMapStore.latestInfo() {
                    Toggle(isOn: Binding(
                        get: { controller.continueFromLastMap },
                        set: { controller.setContinueFromLastMap($0) })) {
                        Label(String(format: "延續上次座標系（%.1f MB · %@）",
                                     Double(info.bytes) / 1_048_576,
                                     Self.ago(info.modified)),
                              systemImage: "point.3.filled.connected.trianglepath.dotted")
                            .font(.caption.weight(.medium))
                    }
                    .tint(.blue)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 7)
                    .background(.ultraThinMaterial, in: Capsule())
                    .foregroundStyle(.white)
                }

                // 相機參數鎖定開關（預設開啟）：按快門當下鎖定曝光/白平衡。
                // 對焦刻意不在此列 —— 鎖對焦＝凍結景深，離開起始距離就糊。
                Toggle(isOn: $controller.lockCameraParams) {
                    Label("鎖定曝光 / 白平衡",
                          systemImage: controller.lockCameraParams ? "lock.fill" : "lock.open")
                        .font(.caption.weight(.medium))
                }
                .tint(.green)
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
                .frame(width: 290)
                .background(.ultraThinMaterial, in: Capsule())
                .foregroundStyle(.white)

                Picker("模式", selection: $controller.mode) {
                    ForEach(ScanMode.allCases) { m in
                        Text(m.label).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 180)
            }

            switch controller.phase {
            case .idle, .scanning:
                shutterButton
            case .processing:
                VStack(spacing: 8) {
                    ProgressView(value: controller.exportProgress)
                        .tint(.white)
                        .frame(width: 220)
                    Text("\(Int(controller.exportProgress * 100))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white)
                }
                .padding(.bottom, 18)
            case .review:
                reviewControls
            case .training:
                trainingControls
            case .exporting:
                ProgressView()
                    .controlSize(.large)
                    .tint(.white)
                    .padding(.bottom, 18)
            case .done:
                doneControls
            }
        }
    }

    /// 掃描品質摘要。這些數字原本只印在 log 裡，使用者看不到 ——
    /// 而「漂移多少、迴環有沒有閉合」正是判斷這份資料能不能用的依據。
    /// 只顯示需要注意的項目：一切正常時整張卡不出現，不佔版面。
    @ViewBuilder
    private var scanSummaryCard: some View {
        if let s = controller.scanSummary, !summaryRows(s).isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(summaryRows(s), id: \.text) { row in
                    Label(row.text, systemImage: row.symbol)
                        .font(.caption)
                        .foregroundStyle(row.tint)
                }
            }
            .padding(.horizontal, 14).padding(.vertical, 9)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private func summaryRows(_ s: ScanSummary) -> [(text: String, symbol: String, tint: Color)] {
        var rows: [(String, String, Color)] = []
        // 迴環：未閉合代表 ARKit 沒機會做全域修正，遠端誤差留在資料裡
        if s.traveledM >= 8 && !s.loopClosed {
            rows.append((String(format: "走了 %.0fm 未回起點 —— 遠端可能有累積漂移", s.traveledM),
                         "arrow.triangle.capsulepath", .orange))
        }
        // 漂移修正幅度：大代表這次追蹤本來就飄，全域幾何可信度低
        if s.driftMaxCm >= 10 {
            rows.append((String(format: "姿態修正 中位數 %.0fcm / 最大 %.0fcm",
                                s.driftMedianCm, s.driftMaxCm),
                         "scope", s.driftMaxCm >= 30 ? .red : .orange))
        }
        if s.blurDropped + s.blurDemoted > 0 {
            rows.append((s.blurDropped > 0
                         ? "排除 \(s.blurDropped) 幀（幾何不可信）、\(s.blurDemoted) 幀（顏色糊）"
                         : "\(s.blurDemoted) 幀顏色糊，不進訓練但仍供點雲",
                         "camera.metering.none", .secondary))
        }
        if let mb = s.worldMapMB {
            rows.append((String(format: "世界地圖已存 %.1f MB —— 下次可延續同一座標系", mb),
                         "point.3.filled.connected.trianglepath.dotted", .secondary))
        }
        return rows.map { (text: $0.0, symbol: $0.1, tint: $0.2) }
    }

    private var reviewControls: some View {
        VStack(spacing: 12) {
            scanSummaryCard
            // 平面圖預覽（只有 RoomPlan 真的產出東西時才出現）
            if let fp = controller.floorPlanData, !fp.walls.isEmpty {
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        controller.showFloorPlan.toggle()
                    }
                } label: {
                    Label(controller.showFloorPlan
                          ? "回到點雲"
                          : String(format: "平面圖（%d 牆 · %.1f×%.1fm）",
                                   fp.walls.count, fp.sizeM.x, fp.sizeM.y),
                          systemImage: controller.showFloorPlan ? "cube" : "map")
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 14).padding(.vertical, 7)
                        .background(.ultraThinMaterial, in: Capsule())
                        .foregroundStyle(.white)
                }
            }
            // 訓練時是否即時顯示過程（關＝背景訓練略快，完成後仍可檢視）
            Toggle(isOn: $controller.showTrainingProcess) {
                Label("邊訓練邊看過程", systemImage: "eye")
                    .font(.caption.weight(.medium))
            }
            .tint(.green)
            .padding(.horizontal, 14).padding(.vertical, 6)
            .frame(width: 220)
            .background(.ultraThinMaterial, in: Capsule())
            .foregroundStyle(.white)

            // 主要行動：直接在手機上訓練成 3DGS
            Button {
                controller.startTraining()
            } label: {
                Label("訓練成 3DGS", systemImage: "sparkles")
                    .font(.headline)
                    .padding(.horizontal, 22)
                    .padding(.vertical, 13)
                    .background(Color.accentColor, in: Capsule())
                    .foregroundStyle(.white)
            }
            HStack(spacing: 12) {
                Button {
                    controller.exportAndShare()
                } label: {
                    Label("匯出", systemImage: "square.and.arrow.up")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 16).padding(.vertical, 10)
                        .background(.ultraThinMaterial, in: Capsule())
                        .foregroundStyle(.white)
                }
                Button {
                    controller.resumeScan()
                } label: {
                    Label("續掃", systemImage: "plus.viewfinder")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 16).padding(.vertical, 10)
                        .background(.ultraThinMaterial, in: Capsule())
                        .foregroundStyle(.white)
                }
                Button(role: .destructive) {
                    showDiscardConfirm = true
                } label: {
                    Image(systemName: "trash")
                        .font(.subheadline)
                        .padding(11)
                        .background(.ultraThinMaterial, in: Circle())
                        .foregroundStyle(.red)
                }
            }
        }
        .confirmationDialog("捨棄這次掃描？", isPresented: $showDiscardConfirm, titleVisibility: .visible) {
            Button("刪除掃描資料", role: .destructive) { controller.discardScan() }
            Button("取消", role: .cancel) {}
        }
    }

    // MARK: - 訓練控制（進行中：進度＋停止；完成：匯出／重訓／回檢視）

    @ViewBuilder
    private var trainingControls: some View {
        if controller.trainingComplete {
            HStack(spacing: 12) {
                Button {
                    controller.exportAndShare()
                } label: {
                    Label("匯出並分享", systemImage: "square.and.arrow.up")
                        .font(.headline)
                        .padding(.horizontal, 18).padding(.vertical, 12)
                        .background(Color.accentColor, in: Capsule())
                        .foregroundStyle(.white)
                }
                Button {
                    controller.startTraining()
                } label: {
                    Label("重訓", systemImage: "arrow.clockwise")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 14).padding(.vertical, 11)
                        .background(.ultraThinMaterial, in: Capsule())
                        .foregroundStyle(.white)
                }
                Button {
                    controller.backToReviewFromTraining()
                } label: {
                    Image(systemName: "cube.transparent")
                        .font(.headline)
                        .padding(11)
                        .background(.ultraThinMaterial, in: Circle())
                        .foregroundStyle(.white)
                }
            }
        } else {
            VStack(spacing: 10) {
                ProgressView(value: Double(controller.trainingIteration),
                             total: Double(max(1, controller.trainingTotal)))
                    .tint(.white)
                    .frame(width: 240)
                Button {
                    controller.cancelTraining()
                } label: {
                    Label("停止", systemImage: "stop.fill")
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 18).padding(.vertical, 10)
                        .background(.ultraThinMaterial, in: Capsule())
                        .foregroundStyle(.white)
                }
            }
            .padding(.bottom, 6)
        }
    }

    private var shutterButton: some View {
        Button {
            if controller.phase == .scanning {
                controller.stopScan()
            } else {
                controller.startScan()
            }
        } label: {
            ZStack {
                Circle()
                    .strokeBorder(.white, lineWidth: 4)
                    .frame(width: 78, height: 78)
                if controller.phase == .scanning {
                    RoundedRectangle(cornerRadius: 7)
                        .fill(.red)
                        .frame(width: 32, height: 32)
                } else {
                    Circle()
                        .fill(.red)
                        .frame(width: 62, height: 62)
                }
            }
        }
        // 重定位未完成時姿態不可信，此時開拍等於把錯的外參寫進資料 —— 直接擋住
        .disabled(shutterBlocked)
        .opacity(shutterBlocked ? 0.4 : 1)
    }

    private var shutterBlocked: Bool {
        controller.phase == .idle && (!controller.trackingReady || controller.relocalizing)
    }

    private var doneControls: some View {
        HStack(spacing: 12) {
            if let zip = controller.exportedZip {
                ShareLink(item: zip) {
                    Label("分享 .zip", systemImage: "square.and.arrow.up")
                        .font(.headline)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 12)
                        .background(Color.accentColor, in: Capsule())
                        .foregroundStyle(.white)
                }
            }
            Button {
                controller.resetForNewScan()
            } label: {
                Label("新掃描", systemImage: "arrow.counterclockwise")
                    .font(.headline)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 12)
                    .background(.ultraThinMaterial, in: Capsule())
                    .foregroundStyle(.white)
            }
        }
    }
}
