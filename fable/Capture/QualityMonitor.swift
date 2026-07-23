//
//  QualityMonitor.swift
//  fable — 即時品質監控：角速度 / 動態模糊估計 / 光線 / 距離 / 追蹤狀態
//

import Foundation
import ARKit
import CoreMotion
import simd

/// 品質問題種類。rawValue 越小優先度越高（HUD 只顯示最嚴重的一項）。
/// 是否暫停抓幀由 QualityAssessment.captureBlocked 決定（兩級門檻），
/// 一般警告只提醒、不擋拍。
nonisolated enum QualityIssue: Int, CaseIterable, Identifiable, Sendable, Comparable {
    case trackingLost
    case tooFast
    case deviceHot
    case tooDark
    case tooBright
    case tooClose
    case tooFar

    var id: Int { rawValue }

    static func < (lhs: QualityIssue, rhs: QualityIssue) -> Bool { lhs.rawValue < rhs.rawValue }

    var message: String {
        switch self {
        case .trackingLost: "追蹤不穩，請放慢並對準紋理豐富的區域"
        case .tooFast:      "移動太快會產生動態模糊，請放慢"
        case .deviceHot:    "裝置過熱，建議暫停散熱"
        case .tooDark:      "光線不足，請補光或移至較亮處"
        case .tooBright:    "光線過強，注意過曝"
        case .tooClose:     "距離太近，請後退一點"
        case .tooFar:       "距離太遠，請靠近目標"
        }
    }

    var symbol: String {
        switch self {
        case .trackingLost: "wifi.exclamationmark"
        case .tooFast:      "hare.fill"
        case .deviceHot:    "thermometer.high"
        case .tooDark:      "moon.fill"
        case .tooBright:    "sun.max.fill"
        case .tooClose:     "arrow.down.right.and.arrow.up.left"
        case .tooFar:       "arrow.up.left.and.arrow.down.right"
        }
    }
}

nonisolated struct QualityAssessment: Sendable {
    var issues: [QualityIssue] = []
    var blurPixels: Float = 0
    var centerDepthM: Float = -1
    var angularSpeedRadS: Float = 0
    var linearSpeedMS: Float = 0
    /// true = 追蹤丟失或模糊超過遮斷門檻 → 暫停抓幀（紅色警告）
    var captureBlocked = false

    var allowCapture: Bool { !captureBlocked }
    var worst: QualityIssue? { issues.min() }
}

/// 每個 ARFrame 呼叫一次 assess()。角速度優先讀陀螺儀（CoreMotion），
/// 無法取得時退回姿態差分。所有計算皆為 O(1)，可安心跑在 session delegate 熱路徑上。
final class QualityMonitor {

    private let motion = CMMotionManager()
    private var lastPose: simd_float4x4?
    private var lastTime: TimeInterval = -1
    private var smoothedAngular: Float = 0
    private var smoothedLinear: Float = 0

    func start() {
        guard motion.isDeviceMotionAvailable else { return }
        motion.deviceMotionUpdateInterval = 1.0 / 60.0
        motion.startDeviceMotionUpdates()   // 不帶 handler，由 assess() 輪詢最新值
    }

    func stop() {
        motion.stopDeviceMotionUpdates()
    }

    func assess(frame: ARFrame, config: CaptureConfig) -> QualityAssessment {
        var a = QualityAssessment()
        let camera = frame.camera

        // 1. 追蹤狀態
        switch camera.trackingState {
        case .normal: break
        default: a.issues.append(.trackingLost)
        }

        // 2. 角速度 / 線速度（EMA 平滑，避免單幀抖動觸發警告）
        let dt = lastTime > 0 ? Float(frame.timestamp - lastTime) : 0
        var angular: Float = 0
        if let rate = motion.deviceMotion?.rotationRate {
            angular = Float((rate.x * rate.x + rate.y * rate.y + rate.z * rate.z).squareRoot())
        } else if let lp = lastPose, dt > 0 {
            angular = MatrixUtil.rotationAngleDeg(lp, camera.transform) * .pi / 180 / dt
        }
        var linear: Float = 0
        if let lp = lastPose, dt > 0 {
            linear = simd_distance(MatrixUtil.position(lp), MatrixUtil.position(camera.transform)) / dt
        }
        smoothedAngular = smoothedAngular * 0.7 + angular * 0.3
        smoothedLinear = smoothedLinear * 0.7 + linear * 0.3
        a.angularSpeedRadS = smoothedAngular
        a.linearSpeedMS = smoothedLinear

        // 3. 目標距離（LiDAR 中心區域中位數）
        if let depthMap = frame.sceneDepth?.depthMap {
            a.centerDepthM = Self.centerMedianDepth(depthMap)
        }

        // 4. 動態模糊估計：像素位移 ≈ (ω + v/z) × fx × 曝光時間
        //    旋轉是主要殺手（rolling shutter 亦然），平移項以中心景深歸一化
        let fx = Float(camera.intrinsics[0][0])
        var flow = smoothedAngular
        if a.centerDepthM > 0 { flow += smoothedLinear / a.centerDepthM }
        a.blurPixels = flow * fx * Float(camera.exposureDuration)
        if a.blurPixels > config.maxBlurPixels {
            a.issues.append(.tooFast)
        }

        // 5. 光線（lux；1000 為標準室內照度）
        if let lux = frame.lightEstimate?.ambientIntensity {
            if lux < config.minAmbientLux { a.issues.append(.tooDark) }
            else if lux > config.maxAmbientLux { a.issues.append(.tooBright) }
        }

        // 6. 距離
        if a.centerDepthM > 0 {
            if a.centerDepthM < config.minTargetDistanceM { a.issues.append(.tooClose) }
            else if a.centerDepthM > config.maxTargetDistanceM { a.issues.append(.tooFar) }
        }

        // 7. 散熱
        let thermal = ProcessInfo.processInfo.thermalState
        if thermal == .serious || thermal == .critical {
            a.issues.append(.deviceHot)
        }

        // 遮斷判定（兩級制）：只有追蹤丟失或嚴重模糊才暫停抓幀
        a.captureBlocked = a.issues.contains(.trackingLost)
            || a.blurPixels > config.blockBlurPixels

        lastPose = camera.transform
        lastTime = frame.timestamp
        return a
    }

    /// 取中心 5×5 網格深度的中位數（比單點抗噪，比全圖便宜）
    nonisolated private static func centerMedianDepth(_ pb: CVPixelBuffer) -> Float {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return -1 }
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        var samples: [Float] = []
        samples.reserveCapacity(25)
        for dy in -2...2 {
            let row = base + (h / 2 + dy * 8) * stride
            for dx in -2...2 {
                let d = row.load(fromByteOffset: (w / 2 + dx * 8) * 4, as: Float32.self)
                if d.isFinite && d > 0 { samples.append(d) }
            }
        }
        guard !samples.isEmpty else { return -1 }
        return samples.sorted()[samples.count / 2]
    }
}
