import Foundation
import simd
// 測試用最小定義：BA 只碰到這幾個型別，不必把整個 app module 拉進來
struct CameraIntrinsics { var fx, fy, cx, cy: Double; var width, height: Int }
enum BlurVerdict { case keep, demote, drop }
struct FrameRecord { var id: Int; var transform: [Double]; var intrinsics: CameraIntrinsics
                     var blurVerdict: BlurVerdict = .keep }
struct FeatureObservation { var frameID: Int; var trackID: Int; var u, v, depth: Float }
enum FeatureTracker {
    static func project(_ w: SIMD3<Float>, w2c: simd_float4x4, K: CameraIntrinsics) -> (Float, Float)? {
        let pc = w2c * SIMD4<Float>(w, 1)
        guard pc.z < -1e-4 else { return nil }
        let d = -pc.z
        return (Float(K.cx) + pc.x * Float(K.fx) / d, Float(K.cy) - pc.y * Float(K.fy) / d)
    }
}
enum RefusionEngine {
    static func float4x4(rowMajor m: [Double]) -> simd_float4x4 {
        var o = matrix_identity_float4x4
        for r in 0..<4 { for c in 0..<4 { o[c][r] = Float(m[r * 4 + c]) } }
        return o
    }
}
// PoseRefiner 只用到這幾個常數 + 兩個函式；FusedVoxelGrid 相依的 solveRigid 用不到
struct FusedVoxelGrid { var voxelSize: Float = 0.02
                        struct Cell { var mean = SIMD3<Float>.zero }
                        var cells: [Int64: Cell] = [:] }
enum PointCloudMath { static func voxelKey(_ p: SIMD3<Float>, size: Float) -> Int64? { nil } }
