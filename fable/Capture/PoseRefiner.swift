//
//  PoseRefiner.swift
//  fable — 以 LiDAR 共識點雲為固定結構的位姿微調（不是完整 BA）
//
//  為什麼這件事對 3DGS 最關鍵：位姿誤差直接變成 3DGS 的解析度天花板。
//
//      δ = 1cm 的橫向位姿誤差 @ z=2m, fx=1450  →  δ·fx/z = 7.25 px 重投影誤差
//
//  高斯若縮得比這 7px 更小，各視角就會互相矛盾、光度損失反而上升 ——
//  所以優化器會主動把高斯維持在「剛好模糊到能兼容所有視角」的大小。
//  調再多密集化參數都突破不了這條線；只有把位姿修準才能提高天花板。
//
//  ── 為什麼不做完整 BA ────────────────────────────────────────────
//  完整 SfM/BA 是為「只有照片」設計的：沒有深度、沒有初始位姿，
//  所以必須靠特徵三角化出結構。我們兩個都有 —— LiDAR 給了度量深度、
//  ARKit 給了位姿、而且已經融合出共識點雲。於是：
//    · 不需要特徵提取（那才是 BA 的主要成本，不是解方程）
//    · 不需要三角化（結構就是 LiDAR）
//    · 不需要處理 7-DoF gauge freedom（深度是度量的，尺度天生固定）
//  剩下的問題只是「每一幀相對共識點雲差了一個剛體變換多少」——
//  每幀 6 個未知數，線性化後是一個 6×6 正規方程，微秒級。
//
//  ── 自我驗證 ───────────────────────────────────────────────────
//  每一輪都量修正前後的殘差；**沒有下降就回退並停止**。
//  所以即使預設開啟也不會把資料改壞：最壞情況等於沒做。
//

import Foundation
import simd

nonisolated struct PoseRefineResult: Sendable {
    /// 逐幀修正後的 c2w（未列出者維持原樣）
    var poses: [Int: simd_float4x4] = [:]
    /// 每一輪的殘差（公尺，RMS）。第 0 個是修正前
    var residualsM: [Double] = []
    var roundsApplied = 0

    var improvedPercent: Double {
        guard let first = residualsM.first, let last = residualsM.last, first > 1e-9 else { return 0 }
        return (1 - last / first) * 100
    }

    /// 殘差換算成 2m 處的重投影像素（fx≈1450）——「這是 3DGS 的解析度天花板」的直觀值
    static func pixelsAt2m(_ residualM: Double, fx: Double = 1450) -> Double {
        residualM * fx / 2.0
    }
}

nonisolated enum PoseRefiner {

    /// 對應點最大容許距離（voxel 的倍數）。超過視為不同表面／離群，不參與求解
    static let kMaxCorrespVoxels: Float = 2.5
    /// 每輪只套用求得修正量的這個比例。這就是「只做微調」的實作 ——
    /// 等效於一個把位姿拉回 ARKit 原值的軟先驗，避免在退化幾何
    /// （例如走廊直線前進、只看到一面白牆）上把位姿推壞。
    static let kDamping: Float = 0.7
    /// 單輪修正上限。求解要求超過這個量，代表對應建立得有問題（幾何退化），
    /// 而不是真的差這麼多 —— 夾住比信任它安全。
    static let kMaxTransM: Float = 0.05
    static let kMaxRotRad: Float = 0.02        // ≈1.15°
    /// 每幀至少要這麼多有效對應才求解。太少的話 6 個未知數是欠定的
    static let kMinCorrespondences = 200

    /// 量一幀的深度與共識點雲之間的剛體錯位，並回傳修正後的 c2w。
    ///
    /// - worldPoints: 該幀深度反投影出的世界座標點（已用當前位姿）
    /// - camPos: 該幀相機位置。旋轉繞相機中心參數化 —— 直接用世界原點會讓
    ///   [p]× 變得很大、旋轉與平移嚴重耦合，條件數爆掉。
    /// 回傳 nil 表示對應不足或求解失敗（該幀維持原位姿）。
    static func solveRigid(worldPoints: [SIMD3<Float>], camPos: SIMD3<Float>,
                           grid: FusedVoxelGrid) -> (delta: simd_float4x4, rms: Double)? {
        let vs = grid.voxelSize
        let maxD = vs * kMaxCorrespVoxels
        let maxD2 = maxD * maxD

        // 線性化的點對點剛體對齊：ω×(p−c) + t = q − p
        // Jacobian 每列 = [ -[p−c]× | I ]（3×6），累加 6×6 正規方程
        var ata = [Float](repeating: 0, count: 36)
        var atb = [Float](repeating: 0, count: 6)
        var sumSq: Double = 0
        var n = 0

        for p in worldPoints {
            guard let key = PointCloudMath.voxelKey(p, size: vs),
                  let cell = grid.cells[key] else { continue }
            let e = cell.mean - p
            let d2 = simd_length_squared(e)
            if d2 > maxD2 { continue }                 // 離群／不同表面
            sumSq += Double(d2)
            n += 1

            let r = p - camPos
            // J = [ -[r]× | I ]，逐列展開（避免建矩陣物件）
            //   row0: ( 0,  r.z, -r.y, 1, 0, 0 )
            //   row1: (-r.z, 0,   r.x, 0, 1, 0 )
            //   row2: ( r.y, -r.x, 0,  0, 0, 1 )
            let rows: [[Float]] = [[0, r.z, -r.y, 1, 0, 0],
                                   [-r.z, 0, r.x, 0, 1, 0],
                                   [r.y, -r.x, 0, 0, 0, 1]]
            let ev = [e.x, e.y, e.z]
            for k in 0..<3 {
                let row = rows[k]
                let b = ev[k]
                for i in 0..<6 {
                    let ri = row[i]
                    if ri == 0 { continue }
                    atb[i] += ri * b
                    for j in i..<6 where row[j] != 0 {
                        ata[i * 6 + j] += ri * row[j]
                    }
                }
            }
        }
        guard n >= kMinCorrespondences else { return nil }
        // 補齊對稱下三角
        for i in 0..<6 { for j in 0..<i { ata[i * 6 + j] = ata[j * 6 + i] } }
        // Levenberg 阻尼：低紋理/退化幾何下正規方程接近奇異，加一點對角保證可解
        let trace = (0..<6).reduce(Float(0)) { $0 + ata[$1 * 6 + $1] }
        let lambda = max(1e-9, trace / 6 * 1e-4)
        for i in 0..<6 { ata[i * 6 + i] += lambda }

        guard let x = choleskySolve6(ata, atb) else { return nil }
        var omega = SIMD3<Float>(x[0], x[1], x[2])
        var trans = SIMD3<Float>(x[3], x[4], x[5])

        // 夾住 + 阻尼（軟先驗：只走一部分，讓位姿不會離 ARKit 太遠）
        let rotN = simd_length(omega)
        if rotN > kMaxRotRad { omega *= kMaxRotRad / rotN }
        let trN = simd_length(trans)
        if trN > kMaxTransM { trans *= kMaxTransM / trN }
        omega *= kDamping
        trans *= kDamping

        let rms = (sumSq / Double(n)).squareRoot()
        return (deltaTransform(omega: omega, trans: trans, about: camPos), rms)
    }

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
