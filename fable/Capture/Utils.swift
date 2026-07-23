//
//  Utils.swift
//  fable — simd 矩陣工具與 CVPixelBuffer 零依賴複製 / 取樣
//

import Foundation
import CoreVideo
import simd

// MARK: - 矩陣工具

nonisolated enum MatrixUtil {

    /// simd 為 column-major；輸出 row-major 16 元素（序列化用）
    static func rowMajor16(_ m: simd_float4x4) -> [Double] {
        var out = [Double]()
        out.reserveCapacity(16)
        for r in 0..<4 {
            for c in 0..<4 { out.append(Double(m[c][r])) }
        }
        return out
    }

    static func rotation3x3(_ m: simd_float4x4) -> simd_float3x3 {
        simd_float3x3(simd_make_float3(m.columns.0),
                      simd_make_float3(m.columns.1),
                      simd_make_float3(m.columns.2))
    }

    static func position(_ m: simd_float4x4) -> SIMD3<Float> {
        simd_make_float3(m.columns.3)
    }

    /// 兩姿態間的旋轉角（度）
    static func rotationAngleDeg(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        let dq = simd_quatf(rotation3x3(a)).inverse * simd_quatf(rotation3x3(b))
        var angle = dq.angle
        if angle > .pi { angle = 2 * .pi - angle }
        return angle * 180 / .pi
    }
}

// MARK: - CVPixelBuffer 工具

nonisolated enum PixelBufferUtil {

    static func makePool(width: Int, height: Int, pixelFormat: OSType) -> CVPixelBufferPool? {
        let bufferAttrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: pixelFormat,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        ]
        let poolAttrs: [CFString: Any] = [kCVPixelBufferPoolMinimumBufferCountKey: 4]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, bufferAttrs as CFDictionary, &pool)
        return pool
    }

    /// 逐 plane memcpy 複製。ARKit 的 capturedImage 屬於 ARKit 內部 buffer pool，
    /// 保留它會餓死追蹤管線 —— 一定要複製到自有 pool 再丟給背景執行緒。
    static func clone(_ src: CVPixelBuffer, pool: CVPixelBufferPool) -> CVPixelBuffer? {
        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &out) == kCVReturnSuccess,
              let dst = out else { return nil }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        if CVPixelBufferIsPlanar(src) {
            for plane in 0..<CVPixelBufferGetPlaneCount(src) {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, plane),
                      let d = CVPixelBufferGetBaseAddressOfPlane(dst, plane) else { return nil }
                let sStride = CVPixelBufferGetBytesPerRowOfPlane(src, plane)
                let dStride = CVPixelBufferGetBytesPerRowOfPlane(dst, plane)
                let h = CVPixelBufferGetHeightOfPlane(src, plane)
                if sStride == dStride {
                    memcpy(d, s, sStride * h)
                } else {
                    let n = min(sStride, dStride)
                    for r in 0..<h { memcpy(d + r * dStride, s + r * sStride, n) }
                }
            }
        } else {
            guard let s = CVPixelBufferGetBaseAddress(src),
                  let d = CVPixelBufferGetBaseAddress(dst) else { return nil }
            let sStride = CVPixelBufferGetBytesPerRow(src)
            let dStride = CVPixelBufferGetBytesPerRow(dst)
            let h = CVPixelBufferGetHeight(src)
            if sStride == dStride {
                memcpy(d, s, sStride * h)
            } else {
                let n = min(sStride, dStride)
                for r in 0..<h { memcpy(d + r * dStride, s + r * sStride, n) }
            }
        }
        return dst
    }

    /// 將單平面 buffer（深度 Float32 / 信心圖 UInt8）複製成去除 stride padding 的緊湊 Data
    static func tightData(_ pb: CVPixelBuffer, bytesPerPixel: Int) -> Data {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return Data() }
        let rowBytes = w * bytesPerPixel
        if stride == rowBytes {
            return Data(bytes: base, count: rowBytes * h)
        }
        var data = Data(capacity: rowBytes * h)
        for r in 0..<h {
            data.append(Data(bytes: base + r * stride, count: rowBytes))
        }
        return data
    }
}

/// 對 420f (NV12 full-range) YCbCr buffer 做點取樣 → RGB。
/// 用於幫 LiDAR 點雲上色；直接讀 luma/chroma plane，不經 CoreImage，無方向翻轉疑慮
/// （深度圖與 capturedImage 同為 sensor 座標，(u,v) 一一對應）。
nonisolated struct YUVSampler {
    private let pb: CVPixelBuffer
    private let yBase: UnsafeRawPointer?
    private let yStride: Int
    private let cBase: UnsafeRawPointer?
    private let cStride: Int
    private let width: Int
    private let height: Int

    init(_ pb: CVPixelBuffer) {
        self.pb = pb
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        width = CVPixelBufferGetWidth(pb)
        height = CVPixelBufferGetHeight(pb)
        if CVPixelBufferGetPlaneCount(pb) >= 2 {
            yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0).map { UnsafeRawPointer($0) }
            yStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
            cBase = CVPixelBufferGetBaseAddressOfPlane(pb, 1).map { UnsafeRawPointer($0) }
            cStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
        } else {
            yBase = nil; yStride = 0; cBase = nil; cStride = 0
        }
    }

    func unlock() {
        CVPixelBufferUnlockBaseAddress(pb, .readOnly)
    }

    /// BT.601 full-range 轉換（ARKit capturedImage 為 420YpCbCr8BiPlanarFullRange）
    func rgb(atNormalizedU nu: Float, v nv: Float) -> (UInt8, UInt8, UInt8) {
        guard let yBase, let cBase else { return (160, 160, 160) }
        let x = min(width - 1, max(0, Int(nu * Float(width))))
        let y = min(height - 1, max(0, Int(nv * Float(height))))
        let Y = Float(yBase.load(fromByteOffset: y * yStride + x, as: UInt8.self))
        let ci = (y / 2) * cStride + (x / 2) * 2
        let Cb = Float(cBase.load(fromByteOffset: ci, as: UInt8.self)) - 128
        let Cr = Float(cBase.load(fromByteOffset: ci + 1, as: UInt8.self)) - 128
        func clamp8(_ f: Float) -> UInt8 { UInt8(min(255, max(0, f))) }
        return (clamp8(Y + 1.402 * Cr),
                clamp8(Y - 0.344136 * Cb - 0.714136 * Cr),
                clamp8(Y + 1.772 * Cb))
    }
}
