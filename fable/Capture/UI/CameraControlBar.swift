//
//  CameraControlBar.swift
//  fable — 相機手動調整列（右側直立膠囊 + 展開滑桿）
//

import SwiftUI

struct CameraControlBar: View {
    @ObservedObject var controls: CameraControls
    /// 掃描中不給調 —— 中途改曝光會讓前後幀成像不一致，等於自己製造外觀不一致
    let enabled: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if let item = controls.expanded { slider(for: item) }
            iconRail
        }
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.4)
        .animation(.easeInOut(duration: 0.2), value: controls.expanded)
    }

    // MARK: - 圖示列

    private var iconRail: some View {
        VStack(spacing: 6) {
            ForEach(CameraControls.Item.allCases) { item in
                Button {
                    controls.syncFromDevice()
                    controls.expanded = (controls.expanded == item) ? nil : item
                } label: {
                    Image(systemName: item.symbol)
                        .font(.system(size: 16, weight: .semibold))
                        .frame(width: 40, height: 40)
                }
                .foregroundStyle(tint(for: item))
            }
            Divider().frame(width: 24).overlay(.white.opacity(0.3))
            Button {
                controls.resetAll()
                controls.expanded = nil
            } label: {
                Image(systemName: "arrow.uturn.backward")
                    .font(.system(size: 15, weight: .semibold))
                    .frame(width: 40, height: 36)
            }
            .foregroundStyle(controls.hasManualOverride ? .white : .white.opacity(0.45))
        }
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22))
    }

    /// 黃色 = 該項已被手動指定（一眼看出哪些脫離自動）
    private func tint(for item: CameraControls.Item) -> Color {
        let manual: Bool
        switch item {
        case .ev:      manual = controls.ev != 0
        case .shutter: manual = controls.shutterManual
        case .iso:     manual = controls.isoManual
        case .wb:      manual = controls.wbManual
        case .focus:   manual = controls.focusManual
        }
        if controls.expanded == item { return .cyan }
        return manual ? .yellow : .white
    }

    // MARK: - 展開的滑桿

    @ViewBuilder
    private func slider(for item: CameraControls.Item) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(item.label).font(.caption.weight(.semibold))
                Spacer()
                Text(valueText(item)).font(.caption.monospacedDigit())
            }
            switch item {
            case .ev:
                Slider(value: $controls.ev, in: controls.evRange) { _ in controls.applyEV() }
                    .onChange(of: controls.ev) { _, _ in controls.applyEV() }
            case .shutter:
                Slider(value: $controls.shutterSec, in: controls.shutterRange) { editing in
                    if editing { controls.shutterManual = true }
                    controls.applyExposure()
                }
                .onChange(of: controls.shutterSec) { _, _ in
                    controls.shutterManual = true; controls.applyExposure()
                }
            case .iso:
                Slider(value: $controls.iso, in: controls.isoRange) { editing in
                    if editing { controls.isoManual = true }
                    controls.applyExposure()
                }
                .onChange(of: controls.iso) { _, _ in
                    controls.isoManual = true; controls.applyExposure()
                }
            case .wb:
                Slider(value: $controls.kelvin, in: controls.kelvinRange) { editing in
                    if editing { controls.wbManual = true }
                    controls.applyWhiteBalance()
                }
                .onChange(of: controls.kelvin) { _, _ in
                    controls.wbManual = true; controls.applyWhiteBalance()
                }
            case .focus:
                Slider(value: $controls.lensPosition, in: 0...1) { editing in
                    if editing { controls.focusManual = true }
                    controls.applyFocus()
                }
                .onChange(of: controls.lensPosition) { _, _ in
                    controls.focusManual = true; controls.applyFocus()
                }
            }
            if item == .shutter {
                Text("上限 1/60s：再長就必定動態模糊")
                    .font(.caption2).foregroundStyle(.white.opacity(0.6))
            }
        }
        .tint(.cyan)
        .foregroundStyle(.white)
        .frame(width: 190)
        .padding(.horizontal, 12).padding(.vertical, 10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func valueText(_ item: CameraControls.Item) -> String {
        switch item {
        case .ev:      String(format: "%+.1f EV", controls.ev)
        case .shutter: controls.shutterManual
                        ? "1/\(Int((1 / controls.shutterSec).rounded()))s" : "自動"
        case .iso:     controls.isoManual ? "\(Int(controls.iso))" : "自動"
        case .wb:      controls.wbManual ? "\(Int(controls.kelvin))K" : "自動"
        case .focus:   controls.focusManual
                        ? String(format: "%.2f", controls.lensPosition) : "自動"
        }
    }
}
