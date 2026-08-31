//
//  PoseRefiner.swift
//  fable — BundleAdjuster 共用的 6×6 位姿求解工具
//
//  為什麼位姿這件事對 3DGS 最關鍵：位姿誤差直接變成 3DGS 的解析度天花板。
//
//      δ = 1cm 的橫向位姿誤差 @ z=2m, fx=1450  →  δ·fx/z = 7.25 px 重投影誤差
//
//  高斯若縮得比這 7px 更小，各視角就會互相矛盾、光度損失反而上升 ——
//  所以優化器會主動把高斯維持在「剛好模糊到能兼容所有視角」的大小。
//  調再多密集化參數都突破不了這條線。
//
//  ── 這個檔案剩下什麼 ──────────────────────────────────────────────
//  只有數值工具：小角度增量的參數化（deltaTransform）、6×6 Cholesky、
//  以及去除全域剛體自由度（removeGlobalDrift）。使用者是 BundleAdjuster。
//
//  原本還有一條「以融合點雲為固定結構、用 voxel 佔用建立對應」的幾何式精修
//  （solveRigid），**已刪除**。離線測試決定性地否決了它：
//    沿平面切向偏 3cm → 只呈現 0.95cm 表觀誤差（落進同平面鄰格）
//    沿平面法向偏 3cm → 0/40000 點有對應（全部落進空格）
//  在最該修正的方向收不到訊號、在不該動的方向給誤導訊號。
//  重投影誤差沒有這個盲點（表面往鏡頭方向移動，投影位置就是實實在在地位移了），
//  所以那條路改走 BundleAdjuster。要復原請查 git 歷史。
//

import Foundation
import simd

nonisolated struct PoseRefineResult: Sendable {
    /// 逐幀修正後的 c2w（未列出者維持原樣）
    var poses: [Int: simd_float4x4] = [:]
    /// 每一輪的重投影殘差（像素，RMS）。第 0 個是修正前 ——
    /// 它才是決定 3DGS 解析度天花板的量（1cm 位姿誤差 @2m ≈ 7px）
    var residualsPx: [Float] = []
    var roundsApplied = 0
    /// 交叉驗證：保留集（未參與求解的 track）的重投影中位數，修正前 → 修正後（像素）。
    /// **這是唯一一個不在目標函數裡的數字** —— 其餘都是 BA 自己在最小化的量，
    /// 下降是必然的。只有這一對能分辨「位姿真的變好」與「把觀測雜訊吸進位姿」。
    /// 見 BundleAdjuster.kHoldoutEvery。保留集太小時為 nil。
    var holdoutMedianPx: (before: Float, after: Float)?

    /// 保留集改善率（負值＝變好）。nil 代表沒量到。
    var holdoutDelta: Double? {
        guard let h = holdoutMedianPx, h.before > 1e-6 else { return nil }
        return Double(h.after / h.before) - 1
    }

}

nonisolated enum PoseRefiner {

    /// 每輪只套用求得修正量的這個比例。這就是「只做微調」的實作 ——
    /// 等效於一個把位姿拉回 ARKit 原值的軟先驗，避免在退化幾何
    /// （例如走廊直線前進、只看到一面白牆）上把位姿推壞。
    static let kDamping: Float = 0.7
    /// 單輪修正上限。求解要求超過這個量，代表對應建立得有問題（幾何退化），
    /// 而不是真的差這麼多 —— 夾住比信任它安全。
    static let kMaxTransM: Float = 0.05
    static let kMaxRotRad: Float = 0.02        // ≈1.15°
    /// 小角度旋轉（繞 about 點）＋平移 → 4×4 世界變換。
    /// 用 Rodrigues 的一階近似即可 —— 這裡的角度上限是 1.15°，二階項可忽略。
    static func deltaTransform(omega: SIMD3<Float>, trans: SIMD3<Float>,
                               about c: SIMD3<Float>) -> simd_float4x4 {
        // R ≈ I + [ω]×
        let r0 = SIMD3<Float>(1, omega.z, -omega.y)          // 第 0 欄
        let r1 = SIMD3<Float>(-omega.z, 1, omega.x)          // 第 1 欄
        let r2 = SIMD3<Float>(omega.y, -omega.x, 1)          // 第 2 欄
        let R = simd_float3x3(r0, r1, r2)
        let t = c + trans - R * c
        return simd_float4x4(SIMD4<Float>(R[0], 0), SIMD4<Float>(R[1], 0),
                             SIMD4<Float>(R[2], 0), SIMD4<Float>(t, 1))
    }

    /// 6×6 對稱正定 Cholesky 解。手寫是為了讓本檔不依賴 Accelerate ——
    /// 這樣製圖層/融合層一樣能在 macOS 上獨立編譯測試。
    static func choleskySolve6(_ a: [Float], _ b: [Float]) -> [Float]? {
        var l = [Float](repeating: 0, count: 36)
        for i in 0..<6 {
            for j in 0...i {
                var s = a[i * 6 + j]
                for k in 0..<j { s -= l[i * 6 + k] * l[j * 6 + k] }
                if i == j {
                    guard s > 1e-12 else { return nil }      // 非正定 → 幾何退化
                    l[i * 6 + i] = s.squareRoot()
                } else {
                    l[i * 6 + j] = s / l[j * 6 + j]
                }
            }
        }
        var y = [Float](repeating: 0, count: 6)
        for i in 0..<6 {
            var s = b[i]
            for k in 0..<i { s -= l[i * 6 + k] * y[k] }
            y[i] = s / l[i * 6 + i]
        }
        var x = [Float](repeating: 0, count: 6)
        for i in stride(from: 5, through: 0, by: -1) {
            var s = y[i]
            for k in (i + 1)..<6 { s -= l[k * 6 + i] * x[k] }
            x[i] = s / l[i * 6 + i]
        }
        return x
    }

    /// 去除全域漂移：把所有幀的平均修正量扣掉。
    ///
    /// 「把每幀對齊到共識、再用這些幀重建共識」是一個共識收斂迭代，
    /// 但它對整體的剛體移動是無感的（全部一起平移不改變任何殘差）。
    /// 不扣掉平均量的話，整份點雲會隨輪次慢慢漂走，
    /// 而 ARWorldMap / 上一次掃描的座標系就對不上了。
    static func removeGlobalDrift(_ deltas: inout [Int: simd_float4x4]) {
        guard !deltas.isEmpty else { return }
        var mean = SIMD3<Float>.zero
        for d in deltas.values { mean += SIMD3(d.columns.3.x, d.columns.3.y, d.columns.3.z) }
        mean /= Float(deltas.count)
        for (k, d) in deltas {
            var m = d
            m.columns.3 -= SIMD4<Float>(mean, 0)
            deltas[k] = m
        }
    }
}
