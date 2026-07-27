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
    case notSharp
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
        case .notSharp:     "畫面不夠清晰，請稍停讓對焦穩定"
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
        case .notSharp:     "camera.metering.none"
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
    /// 影像清晰度的**直接量測**（歸一化二階差分能量）。負值 = 量不到。
    /// blurPixels 是由「角速度 × 曝光時間」推估的，只涵蓋動態模糊；
    /// 這一項才看得到失焦、對焦來回搜尋（AF hunting）、鏡頭霧氣等推估看不到的原因。
    var sharpness: Float = -1
    /// 清晰度相對於「近 0.5 秒內同一場景達到過的最佳值」的比例（0...1）。
    /// 絕對清晰度與場景紋理量綁死（白牆再清晰也是低值），只有相對值可以設門檻。
    var sharpnessRatio: Float = 1
    /// true = 追蹤丟失或模糊超過遮斷門檻 → 暫停抓幀（紅色警告）
    var captureBlocked = false

    var allowCapture: Bool { !captureBlocked }
    var worst: QualityIssue? { issues.min() }
}

/// 每個 ARFrame 呼叫一次 assess()。角速度優先讀陀螺儀（CoreMotion），
/// 無法取得時退回姿態差分。所有計算皆為 O(1)，可安心跑在 session delegate 熱路徑上。
final class QualityMonitor {

    /// 清晰度基準線的每幀衰減率。0.97 → 半衰期約 23 幀（0.4s）、~0.5s 衰到 1/e。
    /// 刻意讓它比「模糊事件」（手震一下、AF 拉焦，約 0.1~0.3s）長、比「掃到另一片紋理量不同的
    /// 表面」（連續移動下以秒計）短 —— 這樣模糊會被抓到，而把鏡頭轉向白牆不會被誤判成模糊。
    private static let kSharpPeakDecay: Float = 0.97
    /// 連續幾幀不清晰才在 HUD 上示警（60fps → 15 幀 ≈ 0.25s）。
    /// 抓幀閘門是**立即**生效的（該幀不清晰就不存），但示警要遲滯：
    /// 短暫的不清晰（AF 拉一下焦、鏡頭掃過紋理量差很多的兩片表面）由閘門靜靜擋掉就好，
    /// 跳一個紅字出來只會讓使用者以為壞了。持續不清晰（鏡頭有指紋、AF 卡住、太暗）才值得說。
    private static let kNotSharpFrames = 15
    /// 角速度超過此值就凍結清晰度基準線（見 assess 內說明）。
    /// 0.2 rad/s ≈ 11°/s：比「手持自然晃動」高、比任何刻意的轉動低。
    private static let kPeakFreezeRadS: Float = 0.2

    private let motion = CMMotionManager()
    private var lastPose: simd_float4x4?
    private var lastTime: TimeInterval = -1
    private var smoothedAngular: Float = 0
    private var smoothedLinear: Float = 0
    private var sharpPeak: Float = 0
    private var notSharpStreak = 0

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
        // 角速度優先取 ARKit 姿態差分，陀螺儀只當補強 —— 順序與直覺相反，理由是**時間對齊**：
        //
        //   陀螺儀輪詢拿到的是「現在」的角速度，但手上這一幀是 30~50ms 前曝光的
        //   （ARKit 帶 sceneDepth 的管線延遲）。轉彎「結束」時最致命：手已經停了、
        //   陀螺儀讀到接近 0，可是正在送進來的那一幀是轉彎峰值時曝的，而 SmartShutter
        //   的「轉角 ≥ 6°」條件剛好在此刻滿足 → 精準地把整段最糊的那一幀存成關鍵幀。
        //   這就是「視角轉彎時最容易出現模糊照片」的成因。
        //
        //   姿態差分量的是 frame k-1 → k 之間的平均角速度，而 frame.camera.transform
        //   本來就屬於該幀的時刻，天然對齊，量到的正是那個曝光窗實際抹過的角度。
        //   2 rad/s 的轉動在 16.7ms 內是 1.9°，遠高於 ARKit 的旋轉雜訊（~0.05°），SNR 沒問題。
        //   再取 max(姿態差分, 陀螺儀) 補上「正在加速進轉彎」——那時陀螺儀跑在前面。
        var angular: Float = 0
        if let lp = lastPose, dt > 0 {
            angular = MatrixUtil.rotationAngleDeg(lp, camera.transform) * .pi / 180 / dt
        }
        if let rate = motion.deviceMotion?.rotationRate {
            angular = max(angular,
                          Float((rate.x * rate.x + rate.y * rate.y + rate.z * rate.z).squareRoot()))
        }
        var linear: Float = 0
        if let lp = lastPose, dt > 0 {
            linear = simd_distance(MatrixUtil.position(lp), MatrixUtil.position(camera.transform)) / dt
        }
        smoothedAngular = smoothedAngular * 0.7 + angular * 0.3
        smoothedLinear = smoothedLinear * 0.7 + linear * 0.3
        // 回報 max(瞬時, 平滑)。純 EMA 會把單幀尖峰壓到 30%：甩一下 2.0 rad/s 只讀到 0.6，
        // 剛好從 1.0 的閘門底下溜過去 —— 而那一幀確實是糊的。取 max 讓尖峰不被平滑掉，
        // 同時保留 EMA 對「持續快速移動」的記憶（穩定移動時兩者本來就相等，門檻不會變嚴）。
        a.angularSpeedRadS = max(angular, smoothedAngular)
        a.linearSpeedMS = max(linear, smoothedLinear)

        // 3. 目標距離（LiDAR 中心區域中位數）
        if let depthMap = frame.sceneDepth?.depthMap {
            a.centerDepthM = Self.centerMedianDepth(depthMap)
        }

        // 4. 幾何劣化估計：像素位移 ≈ (ω + v/z) × fx × (曝光時間 + 捲簾讀出時間)
        //    平移項以中心景深歸一化。
        //
        //    為什麼要加上捲簾讀出時間 —— 原本只算曝光時間，於是「亮處」被系統性低估：
        //    明亮辦公室 AE 會縮到 1/250s，1 rad/s 的轉動只估出 6px，看起來完全合格；
        //    但 CMOS 是逐列讀出的，整幀跨越約 10ms，這段時間內相機仍在轉 →
        //    畫面上下兩端對應不同的相機姿態，變成剪切變形（skew），
        //    對 3DGS 是直接違反針孔模型，而且**縮短曝光完全救不到**。
        //    1 rad/s + 1/250s 曝光的實際劣化是 6px 模糊 ＋ 14px 剪切 ≈ 20px，不是 6px。
        //    兩者是不同的成因（一個是曝光內抹動、一個是幀內姿態不一致），
        //    但對「這一幀能不能當訓練影像」的影響同向，故合成單一保守指標。
        let fx = Float(camera.intrinsics[0][0])
        var flow = a.angularSpeedRadS
        if a.centerDepthM > 0 { flow += a.linearSpeedMS / a.centerDepthM }
        a.blurPixels = flow * fx * Float(camera.exposureDuration + config.rollingShutterReadoutS)
        if a.blurPixels > config.maxBlurPixels {
            a.issues.append(.tooFast)
        }

        // 4b. 清晰度：直接量影像本身，補上 blurPixels 推估不到的失焦 / AF 搜尋。
        //     絕對值與場景紋理量綁死，故拿它跟「近 0.5s 內同場景的最佳值」比。
        let sharp = Self.sharpness(frame.capturedImage)
        if sharp >= 0 {
            // 相機轉得快時**凍結**基準線，不讓它衰減。
            //
            // 這是這道閘門原本的致命缺口：sharpPeak 只有 ~0.5s 的記憶，
            // 而轉彎掃描是「持續」糊 1~2 秒 —— 峰值會一路降到糊的水準，比值回到 1.0，
            // 於是整段轉彎都被放行。離線模擬（tools/test_sharpness.py）量到的是：
            // 持續 20px 模糊的 90 幀裡，轉彎開始 267ms 後比值就爬回門檻以上，82% 放行。
            // 對照組（手震 0.1s）則 100% 擋掉 —— 它只抓得到「短暫」模糊。
            //
            // 凍結後，基準線保留的是「上次拿穩時這個場景有多清晰」，
            // 整段轉彎都會被拿去跟那個值比，持續模糊再也騙不過去。
            // 代價：轉彎後停在低紋理表面時基準線偏高（誤擋），但只持續到動作停下後 ~0.5s
            // 衰減恢復為止，而且那時本來就該補拍一張清晰的。
            let moving = a.angularSpeedRadS > Self.kPeakFreezeRadS
            sharpPeak = max(sharp, sharpPeak * (moving ? 1.0 : Self.kSharpPeakDecay))
            a.sharpness = sharp
            a.sharpnessRatio = sharpPeak > 1e-6 ? sharp / sharpPeak : 1
            if a.sharpnessRatio < config.minSharpnessRatio {
                notSharpStreak += 1
            } else {
                notSharpStreak = 0
            }
            if notSharpStreak >= Self.kNotSharpFrames { a.issues.append(.notSharp) }
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

    /// 影像清晰度 = 歸一化的二階差分能量（Laplacian 能量的可分離、抽樣版）。
    ///
    /// 直接讀 ARKit capturedImage 的 luma plane（YCbCr 420 biplanar 的 plane 0）——
    /// 免色彩轉換、免複製。中央 80% 區域每 6 px 抽一點：1920×1440 約 5.6 萬取樣點，
    /// 純整數運算 <0.5ms，可以每幀跑在 session delegate 上。
    ///
    /// 差分步距 2 px 是刻意的：步距 1 的響應集中在 Nyquist 附近，遇到 JPEG/去馬賽克雜訊
    /// 容易把雜訊當細節；步距 2 對我們真正在意的 4~20 px 模糊核最敏感。
    /// 最後除以平均亮度 → 變成對比度量測，暗處與亮處的值可以互相比較。
    nonisolated private static func sharpness(_ pb: CVPixelBuffer) -> Float {
        guard CVPixelBufferGetPlaneCount(pb) >= 1 else { return -1 }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pb, 0) else { return -1 }
        let w = CVPixelBufferGetWidthOfPlane(pb, 0)
        let h = CVPixelBufferGetHeightOfPlane(pb, 0)
        let rowBytes = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let p = base.assumingMemoryBound(to: UInt8.self)

        let s = 2               // 差分步距（px）
        let step = 6            // 抽樣間隔（px）
        let mx = max(w / 10, s), my = max(h / 10, s)
        let x0 = mx, x1 = w - mx, y0 = my, y1 = h - my
        guard x1 > x0, y1 > y0 else { return -1 }

        var energy = 0, luma = 0, n = 0
        var y = y0
        while y < y1 {
            let row = p + y * rowBytes
            let up = p + (y - s) * rowBytes
            let dn = p + (y + s) * rowBytes
            var x = x0
            while x < x1 {
                let c = Int(row[x])
                energy += abs(2 * c - Int(row[x - s]) - Int(row[x + s]))
                       +  abs(2 * c - Int(up[x]) - Int(dn[x]))
                luma += c
                n += 1
                x += step
            }
            y += step
        }
        guard n > 0, luma > 0 else { return -1 }
        return Float(energy) / Float(luma)
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
