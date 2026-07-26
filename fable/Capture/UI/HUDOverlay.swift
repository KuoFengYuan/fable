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
                warningBanner
                coverageHintBanner
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

    // MARK: - 缺角提醒：往哪補掃（物件模式圓頂；擋掉遮斷級警告時不搶版面）

    @ViewBuilder
    private var coverageHintBanner: some View {
        if controller.phase == .scanning, !controller.assessment.captureBlocked,
           let hint = controller.coverageHint {
            Label(hint, systemImage: "scope")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(Color.orange.opacity(0.85), in: Capsule())
                .transition(.move(edge: .top).combined(with: .opacity))
        }
    }

    /// 融合品質熱圖的圖例。只在熱圖模式顯示 —— 顏色本身要能自我解釋，
    /// 否則使用者只會看到「畫面變紅色」而不知道那代表要補掃。
    @ViewBuilder
    private var fusionLegend: some View {
        if controller.phase == .scanning, controller.showPointCloud,
           controller.colorMode == .fusionQuality {
            HStack(spacing: 8) {
                Text("融合").font(.caption2.weight(.semibold))
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
                Text("不足 → 充分").font(.caption2)
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
                    Label(String(format: "融合 %.0f%%", f * 100),
                          systemImage: "square.stack.3d.up.fill")
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

    private var storageEstimate: String {
        let mb = Double(controller.keyframeCount) * 0.62
        return mb < 1000 ? String(format: "%.0f MB", mb) : String(format: "%.1f GB", mb / 1000)
    }

    // MARK: - 警告膠囊（只顯示最嚴重一項）

    @ViewBuilder
    private var warningBanner: some View {
        if controller.phase == .scanning, let worst = controller.assessment.worst {
            // 紅 = 遮斷級（暫停抓幀），橘 = 提醒級（照拍）
            Label(worst.message, systemImage: worst.symbol)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(controller.assessment.captureBlocked
                            ? Color.red.opacity(0.88) : Color.orange.opacity(0.88),
                            in: Capsule())
                .transition(.move(edge: .top).combined(with: .opacity))
        }
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
                // 相機參數鎖定開關（預設開啟）：按快門當下鎖定對焦/曝光/白平衡
                Toggle(isOn: $controller.lockCameraParams) {
                    Label("鎖定對焦 / 曝光 / 白平衡",
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

    private var reviewControls: some View {
        VStack(spacing: 12) {
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
        .disabled(controller.phase == .idle && !controller.trackingReady)
        .opacity(controller.phase == .idle && !controller.trackingReady ? 0.4 : 1)
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
