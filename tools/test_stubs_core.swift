import Foundation
import simd
//
//  測試用最小型別定義（不必把整個 app module 拉進來）。
//  被 test_bundle_adjust 與 test_feature_index 共用。
//
struct CameraIntrinsics { var fx, fy, cx, cy: Double; var width, height: Int }
enum BlurVerdict { case keep, demote, drop }
struct FrameRecord { var id: Int; var transform: [Double]; var intrinsics: CameraIntrinsics
                     var blurVerdict: BlurVerdict = .keep }
enum RefusionEngine {
    static func float4x4(rowMajor m: [Double]) -> simd_float4x4 {
        var o = matrix_identity_float4x4
        for r in 0..<4 { for c in 0..<4 { o[c][r] = Float(m[r * 4 + c]) } }
        return o
    }
}
