//
//  DepthDensifier.swift
//  fable — 單目深度補洞（Depth Anything V2 Small, Core ML / ANE）
//
//  用途：LiDAR 在「低信心區 / 深色吸光表面 / 玻璃鏡面 / 超過 pointMaxDepthM」完全沒有回波，
//  這些洞在 3DGS 初始化時就是缺點雲 → 訓練只能靠光度梯度亂長（黑塊/霧感的來源）。
//  單目深度可以填這些洞，但它是「相對」深度 —— 用同一幀的 LiDAR 點做仿射對齊還原度量。
//
//  為什麼不用 Apple Depth Pro：既然一定要用 LiDAR 重新縮放，「zero-shot metric」
//  這個賣點對我們就是白付的成本。
//      Depth Pro           ~1.9GB，桌機 GPU 0.3s/幀
//      DepthAnythingV2-S   49.8MB，ANE ~34ms/幀（Apple 官方 Core ML 轉檔）
//  差兩個數量級，而對齊後的度量精度是由 LiDAR 決定的，與模型是否 metric 無關。
//
//  刻意的設計限制：只在「LiDAR 無效」的像素補點，不取代任何 LiDAR 觀測。
//  理由是 ARKit LiDAR 對 2cm voxel 其實是過取樣的（256×192 → 每像素 5.36mrad，
//  2m 處取樣間距 10.7mm < 20mm voxel），拿推論值去稀釋量測值只有壞處。
//

import Foundation
import CoreML
import CoreVideo

nonisolated final class DepthDensifier: @unchecked Sendable {

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    /// 模型輸入尺寸（執行期由 MLModelDescription 查得，不寫死 —— 換模型/換版本都不會錯）
    let width: Int
    let height: Int

    /// 模型不在 bundle 裡（使用者尚未加入 target）時回 nil → 呼叫端自動退回純 LiDAR 路徑。
    init?(resourceName: String = "DepthAnythingV2SmallF16") {
        // Xcode 會把 .mlpackage 編成 .mlmodelc 放進 bundle
        let url = Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc")
               ?? Bundle.main.url(forResource: resourceName, withExtension: "mlpackage")
        guard let url else { return nil }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all          // 讓 Core ML 優先排到 ANE
        guard let m = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }

        let desc = m.modelDescription
        // 取第一個 image 型別的輸入（本模型是 "image"，但不寫死名稱）
        guard let (inName, inDesc) = desc.inputDescriptionsByName.first(where: {
                  $0.value.imageConstraint != nil }),
              let ic = inDesc.imageConstraint,
              let outName = desc.outputDescriptionsByName.keys.sorted().first
        else { return nil }

        model = m
        inputName = inName
        outputName = outName
        width = ic.pixelsWide
        height = ic.pixelsHigh
        guard width > 0, height > 0 else { return nil }
    }

    /// 推論一幀。輸入為 width×height 的 RGBA8（呼叫端負責縮到 self.width/height）。
    /// 回傳「相對逆深度」(disparity-like)，長度 width*height；數值越大＝越近。
    func inferDisparity(rgba: [UInt8]) -> [Float]? {
        guard rgba.count >= width * height * 4,
              let pb = Self.makeBGRA(rgba: rgba, w: width, h: height),
              let input = try? MLDictionaryFeatureProvider(
                  dictionary: [inputName: MLFeatureValue(pixelBuffer: pb)]),
              let out = try? model.prediction(from: input),
              let fv = out.featureValue(for: outputName)
        else { return nil }
        return Self.toFloats(fv, expected: width * height)
    }

    // MARK: - 緩衝轉換

    private static func makeBGRA(rgba: [UInt8], w: Int, h: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buf = pb else { return nil }
        CVPixelBufferLockBaseAddress(buf, [])
        defer { CVPixelBufferUnlockBaseAddress(buf, []) }
        guard let base = CVPixelBufferGetBaseAddress(buf) else { return nil }
        let dstStride = CVPixelBufferGetBytesPerRow(buf)   // 可能有 padding，不可假設 = w*4
        let dst = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<h {
            let s = y * w * 4, d = y * dstStride
            for x in 0..<w {
                dst[d + x * 4 + 0] = rgba[s + x * 4 + 2]   // B
                dst[d + x * 4 + 1] = rgba[s + x * 4 + 1]   // G
                dst[d + x * 4 + 2] = rgba[s + x * 4 + 0]   // R
                dst[d + x * 4 + 3] = 255
            }
        }
        return buf
    }

    /// 輸出可能是灰階影像（16Half / 32Float）或 MLMultiArray —— 兩種都吃。
    private static func toFloats(_ fv: MLFeatureValue, expected: Int) -> [Float]? {
        if let pb = fv.imageBufferValue {
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
            let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
            let stride = CVPixelBufferGetBytesPerRow(pb)
            let fmt = CVPixelBufferGetPixelFormatType(pb)
            var out = [Float](repeating: 0, count: w * h)
            switch fmt {
            case kCVPixelFormatType_OneComponent16Half:
                for y in 0..<h {
                    let row = base.advanced(by: y * stride).assumingMemoryBound(to: Float16.self)
                    for x in 0..<w { out[y * w + x] = Float(row[x]) }
                }
            case kCVPixelFormatType_OneComponent32Float:
                for y in 0..<h {
                    let row = base.advanced(by: y * stride).assumingMemoryBound(to: Float.self)
                    for x in 0..<w { out[y * w + x] = row[x] }
                }
            case kCVPixelFormatType_OneComponent8:
                for y in 0..<h {
                    let row = base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self)
                    for x in 0..<w { out[y * w + x] = Float(row[x]) }
                }
            default:
                return nil
            }
            return out.count == expected ? out : nil
        }
        if let ma = fv.multiArrayValue {
            guard ma.count == expected else { return nil }
            var out = [Float](repeating: 0, count: ma.count)
            switch ma.dataType {
            case .float32:
                let p = ma.dataPointer.assumingMemoryBound(to: Float.self)
                for i in 0..<ma.count { out[i] = p[i] }
            case .float16:
                let p = ma.dataPointer.assumingMemoryBound(to: Float16.self)
                for i in 0..<ma.count { out[i] = Float(p[i]) }
            case .double:
                let p = ma.dataPointer.assumingMemoryBound(to: Double.self)
                for i in 0..<ma.count { out[i] = Float(p[i]) }
            default:
                return nil
            }
            return out
        }
        return nil
    }
}

// MARK: - LiDAR 仿射對齊

/// 相對逆深度 → 度量深度的仿射還原：`1/z ≈ a·d + b`
///
/// 這是「affine-invariant depth」的標準還原式。用同一幀的高信心 LiDAR 像素當監督，
/// 以 trimmed least squares 疊代（LiDAR 濾過信心後仍有離群，純 LS 會被拉歪）。
nonisolated struct DepthScaleFit: Sendable {
    let a: Float
    let b: Float
    let inliers: Int
    let rmse: Float          // 逆深度空間的殘差（1/m）

    /// - pairs: (相對逆深度 d, 度量深度 z) 配對
    /// - 回傳 nil 表示這一幀不該用 MDE（樣本太少或條件數太差）
    static func fit(pairs: [(d: Float, z: Float)], minSamples: Int = 200) -> DepthScaleFit? {
        guard pairs.count >= minSamples else { return nil }

        // 條件數守衛：若整幀深度變化太小（例如正對一面平牆），a 無法被約束 → 擬合會爆掉。
        // 檢查 d 的分佈跨度，太窄就放棄這一幀（寧可只用 LiDAR，也不要注入亂數幾何）。
        var dMin = Float.greatestFiniteMagnitude, dMax = -Float.greatestFiniteMagnitude
        for p in pairs { dMin = min(dMin, p.d); dMax = max(dMax, p.d) }
        guard dMax - dMin > 1e-3 else { return nil }

        var idx = Array(pairs.indices)
        var a: Float = 0, b: Float = 0, rmse: Float = 0

        // 3 輪 trimmed LS：全樣本 → 保留殘差最小的 80% → 再擬合
        for round in 0..<3 {
            var sd = 0.0, sy = 0.0, sdd = 0.0, sdy = 0.0
            for i in idx {
                let d = Double(pairs[i].d), y = 1.0 / Double(pairs[i].z)
                sd += d; sy += y; sdd += d * d; sdy += d * y
            }
            let n = Double(idx.count)
            let den = n * sdd - sd * sd
            guard abs(den) > 1e-12 else { return nil }
            a = Float((n * sdy - sd * sy) / den)
            b = Float((sdd * sy - sd * sdy) / den)

            var res = idx.map { i -> (Float, Int) in
                let e = abs(a * pairs[i].d + b - 1 / pairs[i].z)
                return (e, i)
            }
            var sq = 0.0
            for r in res { sq += Double(r.0 * r.0) }
            rmse = Float((sq / n).squareRoot())

            if round < 2 {
                res.sort { $0.0 < $1.0 }
                let keep = max(minSamples, Int(Double(res.count) * 0.8))
                idx = res.prefix(keep).map { $0.1 }
            }
        }

        // a 必須為正（相對逆深度與真實逆深度同向）；否則模型輸出方向與假設相反 → 放棄
        guard a > 0, a.isFinite, b.isFinite else { return nil }
        return DepthScaleFit(a: a, b: b, inliers: idx.count, rmse: rmse)
    }

    /// 相對逆深度 → 度量深度。回傳 nil 表示該像素落在無效區（逆深度 ≤ 0）。
    @inline(__always)
    func depth(_ d: Float) -> Float? {
        let inv = a * d + b
        return inv > 1e-4 ? 1 / inv : nil
    }
}
