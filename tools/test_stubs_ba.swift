import Foundation
import simd
//
//  只有 BA 測試需要：它不編譯真正的 FeatureTracker.swift，所以要有替身。
//  （test_feature_index 反過來 —— 它編譯真的那一份，故不可與本檔同時使用。）
//
struct FeatureObservation { var frameID: Int; var trackID: Int; var u, v, depth: Float }
enum FeatureTracker {
    static func project(_ w: SIMD3<Float>, w2c: simd_float4x4, K: CameraIntrinsics) -> (Float, Float)? {
        let pc = w2c * SIMD4<Float>(w, 1)
        guard pc.z < -1e-4 else { return nil }
        let d = -pc.z
        return (Float(K.cx) + pc.x * Float(K.fx) / d, Float(K.cy) - pc.y * Float(K.fy) / d)
    }
}
