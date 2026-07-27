//
//  Models.swift
//  fable — 資料模型與序列化格式
//

import Foundation
import CoreVideo
import simd

/// 針孔相機內參（像素單位，對應 sensor 座標的原始影像解析度）
nonisolated struct CameraIntrinsics: Codable, Sendable {
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
    var width: Int
    var height: Int

    /// 依目標解析度線性縮放（例如 1920×1440 → 深度圖 256×192）
    func scaled(toWidth w: Int, height h: Int) -> CameraIntrinsics {
        let sx = Double(w) / Double(width)
        let sy = Double(h) / Double(height)
        return CameraIntrinsics(fx: fx * sx, fy: fy * sy, cx: cx * sx, cy: cy * sy, width: w, height: h)
    }
}

/// poses.jsonl 每行一筆。
/// transform 為 row-major 4×4 camera-to-world，ARKit/OpenGL 相機慣例（X右、Y上、-Z 為視線方向），
/// 世界座標為 ARKit gravity 對齊（+Y 為反重力方向），單位公尺。
nonisolated struct FrameRecord: Codable, Sendable {
    var id: Int
    /// ARFrame.timestamp：開機起算的單調時鐘（秒），影像/深度/姿態取自同一 ARFrame，天然同步
    var timestamp: Double
    var transform: [Double]
    var intrinsics: CameraIntrinsics
    var exposureDuration: Double
    var exposureOffsetEV: Double
    var ambientLux: Double?
    var estimatedBlurPx: Double
    /// 影像清晰度的直接量測（歸一化二階差分能量）與其相對基準線的比例。
    /// 離線挑幀用：同一區域拍到多張時，可據此選最鋭利的餵給訓練。
    var sharpness: Double = 0
    var sharpnessRatio: Double = 1
    var imageFile: String
    var depthFile: String?
    var confidenceFile: String?
    var depthWidth: Int?
    var depthHeight: Int?
}

/// meta.json：一次掃描的全域資訊，Python 端據此判斷座標慣例與深度格式
nonisolated struct SessionMeta: Codable, Sendable {
    var app = "fable-gs-capture"
    var version = 1
    var device: String
    var osVersion: String
    var startedAt: String
    var mode: String
    var worldAlignment = "gravity"
    var cameraConvention = "arkit_gl_c2w_row_major"
    var imageOrientation = "sensor_landscape_right"
    var depthFormat = "float32_raw_little_endian"
    var lidarAvailable: Bool
}

/// 世界座標彩色點。score 為採集品質分數（距離近、靠畫面中心、低模糊 → 高分），
/// 供 voxel 內擇優與匯出下採樣使用。
nonisolated struct CloudPoint: Sendable {
    var x: Float, y: Float, z: Float
    var r: UInt8, g: UInt8, b: UInt8
    var score: Float = 1
}

/// 跨 actor 傳遞的關鍵幀封包。
/// pixelBuffer 是從自有 CVPixelBufferPool 複製出的副本（絕不保留 ARFrame 原始 buffer），
/// 所有權單向轉移給 FrameWriter → 標記 @unchecked Sendable 是安全的。
nonisolated struct Keyframe: @unchecked Sendable {
    let pixelBuffer: CVPixelBuffer
    let depthData: Data?
    let confidenceData: Data?
    let depthWidth: Int
    let depthHeight: Int
    let c2w: simd_float4x4
    var record: FrameRecord
}
