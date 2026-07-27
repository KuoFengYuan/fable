//
//  CameraControls.swift
//  fable — 掃描前的相機手動調整（調完即鎖）
//
//  為什麼要手動：既有的 lockCameraParams 是「凍結自動曝光當下選到的值」。
//  問題是自動測光會被場景裡最亮的東西主導 —— 房間裡有窗戶或投影幕時，
//  牆面與家具會被壓暗到欠曝，那些區域的紋理沒了，3DGS 的光度損失就沒東西可對齊。
//  讓使用者先把曝光/白平衡/對焦調到適合「要重建的表面」，再鎖定，
//  整段掃描的成像才會一致且資訊量最大。
//
//  與 ARKit 的關係：ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera
//  是 ARKit 允許外部調整的同一顆裝置。曝光時間會直接影響 VIO 與動態模糊，
//  故上限夾在 kMaxShutter（見下）——這對 3DGS 也是對的，長曝光＝糊。
//

import Foundation
import AVFoundation
import Combine
import ARKit

@MainActor
final class CameraControls: ObservableObject {

    /// 可調項目。順序即 UI 由上而下的排列。
    enum Item: String, CaseIterable, Identifiable {
        case ev, shutter, iso, wb, focus
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .ev:      "plusminus.circle"
            case .shutter: "timer"
            case .iso:     "camera.aperture"
            case .wb:      "thermometer.sun"
            case .focus:   "viewfinder"
            }
        }
        var label: String {
            switch self {
            case .ev:      "曝光補償"
            case .shutter: "快門"
            case .iso:     "ISO"
            case .wb:      "白平衡"
            case .focus:   "對焦"
            }
        }
    }

    /// 手持掃描的曝光時間上限。動態模糊 ≈ 角速度 × 焦距 × 曝光時間；
    /// 1/60s 在正常掃描速度下已接近 maxBlurPixels，再長就是必糊。
    static let kMaxShutter: Double = 1.0 / 60

    /// 控制列是否展開。預設收合成一顆 A —— 絕大多數情況全自動就夠，
    /// 平常不該讓五個圖示佔住取景畫面。
    @Published var railExpanded = false
    @Published var expanded: Item?          // 目前展開的滑桿（nil = 收合）
    @Published var ev: Float = 0            // EV 補償（stops）
    @Published var isoManual = false
    @Published var iso: Float = 100
    @Published var shutterManual = false
    @Published var shutterSec: Double = 1.0 / 120
    @Published var wbManual = false
    @Published var kelvin: Float = 5000
    @Published var focusManual = false
    @Published var lensPosition: Float = 0.5

    /// 是否有任何手動覆寫（UI 用來提示「已手動」）
    var hasManualOverride: Bool {
        isoManual || shutterManual || wbManual || focusManual || ev != 0
    }

    private var device: AVCaptureDevice? {
        ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera
    }

    // MARK: - 裝置能力（UI 的滑桿範圍要照實際硬體，不能寫死）

    var evRange: ClosedRange<Float> {
        guard let d = device else { return -2...2 }
        return d.minExposureTargetBias...d.maxExposureTargetBias
    }
    var isoRange: ClosedRange<Float> {
        guard let d = device else { return 32...1600 }
        return d.activeFormat.minISO...d.activeFormat.maxISO
    }
    var shutterRange: ClosedRange<Double> {
        guard let d = device else { return 1.0 / 2000...Self.kMaxShutter }
        let lo = CMTimeGetSeconds(d.activeFormat.minExposureDuration)
        let hi = min(Self.kMaxShutter, CMTimeGetSeconds(d.activeFormat.maxExposureDuration))
        return lo...max(lo, hi)
    }
    var kelvinRange: ClosedRange<Float> { 2800...8000 }

    /// 從裝置目前狀態初始化滑桿值（開啟控制列時呼叫，讓起點就是「現在的樣子」）
    func syncFromDevice() {
        guard let d = device else { return }
        ev = d.exposureTargetBias
        iso = min(max(d.iso, isoRange.lowerBound), isoRange.upperBound)
        let cur = CMTimeGetSeconds(d.exposureDuration)
        if cur.isFinite, cur > 0 { shutterSec = min(max(cur, shutterRange.lowerBound), shutterRange.upperBound) }
        lensPosition = d.lensPosition
        let t = d.temperatureAndTintValues(for: d.deviceWhiteBalanceGains)
        if t.temperature.isFinite { kelvin = min(max(t.temperature, kelvinRange.lowerBound), kelvinRange.upperBound) }
    }

    // MARK: - 套用

    private func withDevice(_ body: (AVCaptureDevice) throws -> Void) {
        guard let d = device else { return }
        do {
            try d.lockForConfiguration()
            defer { d.unlockForConfiguration() }
            try body(d)
        } catch {
            print("[CameraControls] 套用失敗: \(error)")
        }
    }

    func applyEV() {
        withDevice { d in
            let v = min(max(ev, d.minExposureTargetBias), d.maxExposureTargetBias)
            d.setExposureTargetBias(v, completionHandler: nil)
        }
    }

    /// ISO 與快門共用 setExposureModeCustom —— 只手動其中一項時，另一項傳
    /// AVCaptureDevice.currentISO / currentExposureDuration 讓硬體維持現值。
    func applyExposure() {
        withDevice { d in
            guard d.isExposureModeSupported(.custom) else { return }
            if !isoManual && !shutterManual {
                if d.isExposureModeSupported(.continuousAutoExposure) {
                    d.exposureMode = .continuousAutoExposure
                }
                return
            }
            let dur = shutterManual
                ? CMTimeMakeWithSeconds(min(max(shutterSec, shutterRange.lowerBound),
                                            shutterRange.upperBound), preferredTimescale: 1_000_000)
                : AVCaptureDevice.currentExposureDuration
            let sens = isoManual
                ? min(max(iso, d.activeFormat.minISO), d.activeFormat.maxISO)
                : AVCaptureDevice.currentISO
            d.setExposureModeCustom(duration: dur, iso: sens, completionHandler: nil)
        }
    }

    func applyWhiteBalance() {
        withDevice { d in
            guard wbManual else {
                if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                    d.whiteBalanceMode = .continuousAutoWhiteBalance
                }
                return
            }
            guard d.isWhiteBalanceModeSupported(.locked) else { return }
            let tt = AVCaptureDevice.WhiteBalanceTemperatureAndTintValues(temperature: kelvin, tint: 0)
            var g = d.deviceWhiteBalanceGains(for: tt)
            // 增益必須落在 [1, maxWhiteBalanceGain]，超出會直接丟例外
            let hi = d.maxWhiteBalanceGain
            g.redGain = min(max(g.redGain, 1), hi)
            g.greenGain = min(max(g.greenGain, 1), hi)
            g.blueGain = min(max(g.blueGain, 1), hi)
            d.setWhiteBalanceModeLocked(with: g, completionHandler: nil)
        }
    }

    func applyFocus() {
        withDevice { d in
            if focusManual {
                guard d.isLockingFocusWithCustomLensPositionSupported else { return }
                d.setFocusModeLocked(lensPosition: min(max(lensPosition, 0), 1), completionHandler: nil)
            } else if d.isFocusModeSupported(.continuousAutoFocus) {
                d.focusMode = .continuousAutoFocus
            }
        }
    }

    /// 全部回自動（對應參考 UI 的重設鍵）
    func resetAll() {
        ev = 0; isoManual = false; shutterManual = false; wbManual = false; focusManual = false
        applyEV(); applyExposure(); applyWhiteBalance(); applyFocus()
        syncFromDevice()
    }

    /// 讓自動曝光**不准**用超過 kMaxShutter 的曝光時間（改為拉 ISO）。
    /// 室內昏暗處自動測光會直接把快門放長到 1/15~1/30s，此時動態模糊 ≈ 角速度 × 焦距 × 曝光時間
    /// 隨曝光線性上升：0.3 rad/s（很慢的平移）在 1/15s 下就是 30 px 模糊，等於整段掃描沒有一張能用。
    /// 拉 ISO 換來的雜訊對 3DGS 遠比模糊便宜 —— 雜訊在多視角平均下會消掉，模糊不會。
    /// 掃描開始前就要設好，讓 AE 有時間在上限內收斂。
    func capExposureDuration() {
        withDevice { d in
            let lo = CMTimeGetSeconds(d.activeFormat.minExposureDuration)
            d.activeMaxExposureDuration =
                CMTimeMakeWithSeconds(max(Self.kMaxShutter, lo), preferredTimescale: 1_000_000)
        }
    }

    /// 開始掃描時鎖定。已被使用者手動指定的項目維持其自訂值，
    /// 只把「還在自動」的項目凍結在當下 —— 這就是「調整完後再鎖定」。
    ///
    /// **對焦刻意不鎖。** 鎖定＝把鏡頭凍結在「按下快門那一刻」的對焦距離，
    /// 而掃描過程中被攝距離一直在變。iPhone 主鏡（f/1.78、實際焦長 ~6.9mm）的超焦距約 5.3m，
    /// 於是景深是：鎖在 3m → 清晰 1.9~6.9m（還算堪用）；鎖在 0.5m → **清晰 0.46~0.55m**。
    /// 使用者習慣先把物件framing好再按開始，正好落在後者 —— 一旦離開那 10cm，整段掃描全糊。
    /// 曝光/白平衡鎖定是為了「光度一致」（3DGS 的光度損失要對齊），
    /// 但對焦不影響光度、只影響鋭利度，而 3DGS 最怕的就是不鋭利。
    /// 對焦改變帶來的內參呼吸（focus breathing）另有解法：
    /// 每幀內參本來就存在 FrameRecord 裡，COLMAP 匯出改成一張影像一組內參即可完全吸收。
    func lockForScan() {
        capExposureDuration()
        withDevice { d in
            if !isoManual, !shutterManual, d.isExposureModeSupported(.locked) { d.exposureMode = .locked }
            if !wbManual, d.isWhiteBalanceModeSupported(.locked) { d.whiteBalanceMode = .locked }
            // 對焦維持連續自動（除非使用者手動指定了鏡位）
            if !focusManual, d.isFocusModeSupported(.continuousAutoFocus) {
                d.focusMode = .continuousAutoFocus
            }
        }
    }

    func unlock() {
        withDevice { d in
            if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
            if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
            if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                d.whiteBalanceMode = .continuousAutoWhiteBalance
            }
        }
        ev = 0; isoManual = false; shutterManual = false; wbManual = false; focusManual = false
    }
}
