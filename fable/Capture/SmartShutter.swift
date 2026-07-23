//
//  SmartShutter.swift
//  fable — 基於「位移距離 + 視角旋轉差」的自動抓幀決策
//
//  設計理由：固定 fps 暴力存檔會產生大量近乎重複的幀（原地不動時最糟），
//  既浪費儲存也讓 3DGS 訓練視角分佈失衡。改為幾何驅動：
//  只有當相機「真的移動出新視角」時才存檔，天然得到均勻的多視角覆蓋。
//

import Foundation
import simd

nonisolated struct SmartShutter {

    private var lastPose: simd_float4x4?
    private var lastTime: TimeInterval = -1

    mutating func reset() {
        lastPose = nil
        lastTime = -1
    }

    /// 平移超過 keyframeTranslationM「或」旋轉超過 keyframeRotationDeg 即觸發，
    /// 並以 minKeyframeInterval 防止手震高頻連拍。品質 gate 由呼叫端把關
    /// （品質不合格時不呼叫本函式，內部狀態不前進，條件保持成立直到拍下為止）。
    mutating func shouldCapture(pose: simd_float4x4, time: TimeInterval, config: CaptureConfig) -> Bool {
        guard let last = lastPose else {
            lastPose = pose
            lastTime = time
            return true    // 第一幀無條件抓
        }
        guard time - lastTime >= config.minKeyframeInterval else { return false }

        let translation = simd_distance(MatrixUtil.position(pose), MatrixUtil.position(last))
        let rotationDeg = MatrixUtil.rotationAngleDeg(last, pose)
        guard translation >= config.keyframeTranslationM || rotationDeg >= config.keyframeRotationDeg else {
            return false
        }
        lastPose = pose
        lastTime = time
        return true
    }
}
