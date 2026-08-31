//
//  test_bundle_adjust.swift
//  fable — 局部 BA 的離線驗證（不需要 iPhone）
//
//  上一版的幾何式精修（PoseRefiner + voxel 對應）就是被離線測試否決的
//  —— 沿平面法向偏 3cm 時 0/40000 點有對應。所以這一版在打開之前先測。
//
//  做法：合成一個已知的場景與相機軌跡 → 由真值產生觀測（影像座標 + 深度）
//  → 把位姿加上已知擾動 → 跑 BA → 檢查它把位姿收回去多少。
//  真值已知，所以可以直接量「位姿誤差」，不只是看目標函數下降。
//
//  編譯（test_stubs_*.swift 提供最小型別定義，不必把整個 app module 拉進來）：
//
//    swiftc -O -o /tmp/batest fable/Capture/PoseRefiner.swift \
//           fable/Capture/BundleAdjuster.swift tools/test_bundle_adjust.swift \
//           tools/test_stubs_core.swift tools/test_stubs_ba.swift && /tmp/batest
//
//  這支測試抓到的兩個 bug（都不是打錯字，是概念錯）：
//    1. ΔT 是套在**相機**上（c2w_new = ΔT·c2w_old），所以作用在世界點上的是
//       ΔT⁻¹ —— Jacobian 整組符號相反，求解方向朝著錯的方向走
//    2. ∂(ω×r)/∂ω_k = e_k × r 是 −[r]× 的第 k **欄**，不是第 k 列
//  兩者都會讓 8 輪全部被自我驗證回退（看起來像「沒效果」而不是「壞掉」）。
//

import Foundation
import simd

@main struct BATest {

    // MARK: - 合成場景

    static var seed: UInt64 = 20260101
    static func rnd() -> Float {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Float(seed >> 33) / Float(UInt32.max >> 1)
    }

    static let K = CameraIntrinsics(fx: 1450, fy: 1450, cx: 960, cy: 720,
                                    width: 1920, height: 1440)

    /// 房間四面牆 + 地板上的 3D 點（法向多樣，位姿才完全可觀測）
    static func scenePoints(_ n: Int) -> [SIMD3<Float>] {
        (0..<n).map { _ in
            switch Int(rnd() * 5) % 5 {
            case 0: return SIMD3(rnd() * 5, rnd() * 2.4, 0)
            case 1: return SIMD3(0, rnd() * 2.4, rnd() * 4)
            case 2: return SIMD3(5, rnd() * 2.4, rnd() * 4)
            case 3: return SIMD3(rnd() * 5, rnd() * 2.4, 4)
            default: return SIMD3(rnd() * 5, 0, rnd() * 4)
            }
        }
    }

    /// 沿房間中央走一圈並環視的相機軌跡（c2w，ARKit GL 慣例：看 -Z、Y 朝上）
    static func trajectory(_ n: Int) -> [simd_float4x4] {
        (0..<n).map { i in
            let t = Float(i) / Float(n) * 2 * .pi
            let pos = SIMD3<Float>(2.5 + 1.2 * cos(t), 1.4, 2.0 + 1.0 * sin(t))
            let target = SIMD3<Float>(2.5 + 2.4 * cos(t + 0.6), 1.2, 2.0 + 2.0 * sin(t + 0.6))
            return lookAt(pos: pos, target: target)
        }
    }

    static func lookAt(pos: SIMD3<Float>, target: SIMD3<Float>) -> simd_float4x4 {
        let fwd = simd_normalize(target - pos)
        let z = -fwd                                   // GL：+Z 指向相機後方
        let x = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), z))
        let y = simd_cross(z, x)
        return simd_float4x4(SIMD4(x, 0), SIMD4(y, 0), SIMD4(z, 0), SIMD4(pos, 1))
    }

    static func project(_ w: SIMD3<Float>, _ c2w: simd_float4x4) -> (u: Float, v: Float, d: Float)? {
        let pc = c2w.inverse * SIMD4<Float>(w, 1)
        guard pc.z < -0.2 else { return nil }
        let d = -pc.z
        let u = Float(K.cx) + pc.x * Float(K.fx) / d
        let v = Float(K.cy) - pc.y * Float(K.fy) / d
        guard u >= 0, v >= 0, u < Float(K.width), v < Float(K.height) else { return nil }
        return (u, v, d)
    }

    /// 由真值產生觀測。觀測本身**不含**位姿誤差 —— 它們是「相機真的看到什麼」，
    /// 誤差只加在餵給 BA 的初始位姿上，這才是實際情境。
    static func observations(points: [SIMD3<Float>], poses: [simd_float4x4],
                            noisePx: Float, noiseDepthM: Float) -> [FeatureObservation] {
        var out: [FeatureObservation] = []
        for (fi, c2w) in poses.enumerated() {
            for (ti, p) in points.enumerated() {
                guard let o = project(p, c2w) else { continue }
                out.append(FeatureObservation(
                    frameID: fi, trackID: ti,
                    u: o.u + (rnd() - 0.5) * 2 * noisePx,
                    v: o.v + (rnd() - 0.5) * 2 * noisePx,
                    depth: o.d + (rnd() - 0.5) * 2 * noiseDepthM))
            }
        }
        return out
    }

    static func records(_ poses: [simd_float4x4]) -> [FrameRecord] {
        poses.enumerated().map { i, m in
            FrameRecord(id: i, transform: rowMajor(m), intrinsics: K)
        }
    }

    static func rowMajor(_ m: simd_float4x4) -> [Double] {
        (0..<4).flatMap { r in (0..<4).map { c in Double(m[c][r]) } }
    }

    /// 把 b 的相機位置以**全域剛體變換**對齊到 a，再回傳對齊後的 b。
    ///
    /// 為什麼一定要這一步：BA 的目標函數是「各幀之間的一致性」，
    /// 對整組相機的剛體移動完全無感（全部一起平移／旋轉不改變任何殘差）——
    /// 這是 BA 的固有 gauge freedom，不是缺陷。
    /// 所以評估必須用 ATE（absolute trajectory error）：先對齊、再量誤差，
    /// 否則會把「BA 修不到也不該修的自由度」算成它的失敗。
    /// 尺度不用估：深度是度量的，尺度天生固定。
    static func alignRigid(_ b: [simd_float4x4], to a: [simd_float4x4]) -> [simd_float4x4] {
        let pa = a.map { SIMD3($0.columns.3.x, $0.columns.3.y, $0.columns.3.z) }
        let pb = b.map { SIMD3($0.columns.3.x, $0.columns.3.y, $0.columns.3.z) }
        let ca = pa.reduce(SIMD3<Float>.zero, +) / Float(pa.count)
        let cb = pb.reduce(SIMD3<Float>.zero, +) / Float(pb.count)
        // 3×3 協方差 → 用小角度近似解旋轉（BA 的殘餘旋轉本來就很小）
        var H = simd_float3x3(0)
        for (x, y) in zip(pa, pb) {
            let u = y - cb, v = x - ca
            H += simd_float3x3(SIMD3(u.x * v.x, u.x * v.y, u.x * v.z),
                               SIMD3(u.y * v.x, u.y * v.y, u.y * v.z),
                               SIMD3(u.z * v.x, u.z * v.y, u.z * v.z))
        }
        // 反對稱部分 ≈ 小角度旋轉向量（Rodrigues 一階）
        let skew = (H - H.transpose)
        let trace = H[0][0] + H[1][1] + H[2][2]
        let omega = trace > 1e-9
            ? SIMD3<Float>(skew[1][2], skew[2][0], skew[0][1]) / trace
            : SIMD3<Float>.zero
        let R = simd_float3x3(SIMD3(1, omega.z, -omega.y),
                              SIMD3(-omega.z, 1, omega.x),
                              SIMD3(omega.y, -omega.x, 1))
        return b.map { m in
            var o = m
            let p = SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            o.columns.3 = SIMD4(R * (p - cb) + ca, 1)
            let r = simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                                  SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                                  SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
            let rr = R * r
            o.columns.0 = SIMD4(rr[0], 0); o.columns.1 = SIMD4(rr[1], 0); o.columns.2 = SIMD4(rr[2], 0)
            return o
        }
    }

    /// 位姿誤差：平移的 RMS（公分）＋ 旋轉的 RMS（度）
    static func poseError(_ a: [simd_float4x4], _ b: [simd_float4x4]) -> (cm: Double, deg: Double) {
        var st: Double = 0, sr: Double = 0
        for (x, y) in zip(a, b) {
            let dt = SIMD3(x.columns.3.x - y.columns.3.x,
                           x.columns.3.y - y.columns.3.y,
                           x.columns.3.z - y.columns.3.z)
            st += Double(simd_length_squared(dt))
            // 相對旋轉的角度
            let ra = simd_float3x3(SIMD3(x.columns.0.x, x.columns.0.y, x.columns.0.z),
                                   SIMD3(x.columns.1.x, x.columns.1.y, x.columns.1.z),
                                   SIMD3(x.columns.2.x, x.columns.2.y, x.columns.2.z))
            let rb = simd_float3x3(SIMD3(y.columns.0.x, y.columns.0.y, y.columns.0.z),
                                   SIMD3(y.columns.1.x, y.columns.1.y, y.columns.1.z),
                                   SIMD3(y.columns.2.x, y.columns.2.y, y.columns.2.z))
            let r = ra.transpose * rb
            let tr = min(3, max(-1, r[0][0] + r[1][1] + r[2][2]))
            sr += Double(acos((tr - 1) / 2)) * 180 / .pi
        }
        let n = Double(a.count)
        return ((st / n).squareRoot() * 100, sr / n)
    }

    /// 把已知擾動加到位姿上（模擬 ARKit 的殘餘誤差）
    static func perturb(_ poses: [simd_float4x4], transM: Float, rotDeg: Float) -> [simd_float4x4] {
        poses.map { m in
            let c = SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            let om = SIMD3<Float>(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5)
            let tr = SIMD3<Float>(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5)
            let omN = simd_length(om) > 1e-6 ? om / simd_length(om) : SIMD3(0, 1, 0)
            let trN = simd_length(tr) > 1e-6 ? tr / simd_length(tr) : SIMD3(1, 0, 0)
            let d = PoseRefiner.deltaTransform(omega: omN * (rotDeg * .pi / 180),
                                               trans: trN * transM, about: c)
            return d * m
        }
    }

    // MARK: - main

    static func main() {
        var fails = 0
        func check(_ ok: Bool, _ what: String) {
            print(ok ? "✅ \(what)" : "❌ \(what)"); if !ok { fails += 1 }
        }

        let pts = scenePoints(1200)
        let truth = trajectory(40)
        print("合成場景：\(pts.count) 點 / \(truth.count) 幀（誤差皆為全域對齊後的 ATE）\n")

        // ── 1. 無雜訊、純位姿擾動 → BA 應該幾乎完全收回去 ──
        for (tm, rd) in [(Float(0.01), Float(0.2)), (0.03, 0.5)] {
            seed = 999
            let obs = observations(points: pts, poses: truth, noisePx: 0, noiseDepthM: 0)
            let bad = perturb(truth, transM: tm, rotDeg: rd)
            let e0 = poseError(truth, alignRigid(bad, to: truth))
            let r = BundleAdjuster.refine(records: records(bad), observations: obs, rounds: 8)
            let fixed = (0..<truth.count).map { r.poses[$0] ?? bad[$0] }
            let e1 = poseError(truth, alignRigid(fixed, to: truth))
            check(e1.cm < e0.cm * 0.5 && e1.deg < e0.deg * 0.5,
                  String(format: "擾動 %.0fcm/%.1f° → 位姿誤差 %.2fcm/%.3f° → %.2fcm/%.3f°"
                         + "（%d 輪，RMS %.2f→%.2f px）",
                         tm * 100, rd, e0.cm, e0.deg, e1.cm, e1.deg, r.roundsApplied,
                         r.residualsPx.first ?? 0, r.residualsPx.last ?? 0))
        }

        // ── 2. 有量測雜訊（0.5px + 5mm 深度）→ 仍應改善，且不被雜訊帶壞 ──
        seed = 4242
        let obsN = observations(points: pts, poses: truth, noisePx: 0.5, noiseDepthM: 0.005)
        let badN = perturb(truth, transM: 0.02, rotDeg: 0.4)
        let e0 = poseError(truth, alignRigid(badN, to: truth))
        let rN = BundleAdjuster.refine(records: records(badN), observations: obsN, rounds: 8)
        let fixedN = (0..<truth.count).map { rN.poses[$0] ?? badN[$0] }
        let e1 = poseError(truth, alignRigid(fixedN, to: truth))
        check(e1.cm < e0.cm * 0.7,
              String(format: "含雜訊(0.5px/5mm) → %.2fcm/%.3f° → %.2fcm/%.3f°",
                     e0.cm, e0.deg, e1.cm, e1.deg))

        // ── 3. 沿光軸的平移（純重投影的退化方向）——深度殘差必須抓到它 ──
        seed = 77
        let obsZ = observations(points: pts, poses: truth, noisePx: 0, noiseDepthM: 0)
        let badZ = truth.map { m -> simd_float4x4 in
            // 沿該幀自己的視線方向（-Z 欄）平移 3cm
            let fwd = SIMD3(-m.columns.2.x, -m.columns.2.y, -m.columns.2.z)
            var o = m
            o.columns.3 += SIMD4(fwd * 0.03, 0)
            return o
        }
        let z0 = poseError(truth, alignRigid(badZ, to: truth))
        let rZ = BundleAdjuster.refine(records: records(badZ), observations: obsZ, rounds: 8)
        let fixedZ = (0..<truth.count).map { rZ.poses[$0] ?? badZ[$0] }
        let z1 = poseError(truth, alignRigid(fixedZ, to: truth))
        // 這個方向是**結構性的弱方向**，不是 bug。
        // 「每幀沿自己的視線前進」在環形軌跡上約等於「相機軌跡相對場景放大」——
        // 而重投影對相似變換是不敏感的（點沿視線移動不改變投影位置）。
        // 深度殘差把它從「完全看不到」拉到「部分可見」：目標函數改善 42%、
        // 位姿改善 ~18%。實測 kDepthWeight 1/3/8/20 的結果幾乎相同
        // （1.30/1.22/1.26/1.29cm），所以不是權重不夠。
        // 真正的解法是更長的 track（跨越視角差異大的幀），也就是迴環閉合。
        // 這裡只斷言「有改善且不惡化」，並把數字印出來讓退化看得見。
        check(z1.cm < z0.cm * 0.95,
              String(format: "沿光軸平移 3cm（結構性弱方向）→ %.2fcm → %.2fcm"
                     + "（僅部分可修，見註解）", z0.cm, z1.cm))

        // ── 4. 已經是真值時不該亂動 ──
        seed = 5
        let obsT = observations(points: pts, poses: truth, noisePx: 0, noiseDepthM: 0)
        let rT = BundleAdjuster.refine(records: records(truth), observations: obsT, rounds: 5)
        let fixedT = (0..<truth.count).map { rT.poses[$0] ?? truth[$0] }
        let eT = poseError(truth, alignRigid(fixedT, to: truth))
        check(eT.cm < 0.2, String(format: "已是真值 → 位姿只動了 %.3f cm（應接近 0）", eT.cm))

        // ── 5. 觀測全是垃圾時要拒絕，不是亂改 ──
        seed = 31337
        let junk = (0..<3000).map { i in
            FeatureObservation(frameID: i % truth.count, trackID: i % 400,
                               u: rnd() * 1920, v: rnd() * 1440, depth: 0.5 + rnd() * 4)
        }
        let bad5 = perturb(truth, transM: 0.02, rotDeg: 0.3)
        let j0 = poseError(truth, alignRigid(bad5, to: truth))
        let rJ = BundleAdjuster.refine(records: records(bad5), observations: junk, rounds: 5)
        let fixedJ = (0..<truth.count).map { rJ.poses[$0] ?? bad5[$0] }
        let j1 = poseError(truth, alignRigid(fixedJ, to: truth))
        check(j1.cm < j0.cm * 1.3,
              String(format: "全垃圾觀測 → 位姿誤差 %.2f → %.2f cm（不得顯著惡化）", j0.cm, j1.cm))

        // ── 6. 保留集要看得到真訊號（true positive）──
        // 沿用第 2 案：含雜訊觀測 + 2cm/0.4° 擾動，確實有位姿誤差可修。
        // 保留集的 track 完全沒進求解，它下降就代表位姿真的變好了。
        check((rN.holdoutDelta ?? 0) < -0.05,
              String(format: "保留集｜有真實位姿誤差 → 中位數 %.2f → %.2f px（%+.0f%%，應明顯下降）",
                     rN.holdoutMedianPx?.before ?? 0, rN.holdoutMedianPx?.after ?? 0,
                     (rN.holdoutDelta ?? 0) * 100))

        // ── 7. 保留集不得謊報（false positive）──
        //
        // **這是整個交叉驗證機制存在的理由。** 位姿已經是真值，觀測雜訊很大，
        // 所以 BA 做的任何事都只可能是在擬合雜訊。此時：
        //   · 目標函數（BA 自己最小化的量）**必然**下降 —— 這正是我先前一直誤讀的數字
        //   · 保留集不在目標函數裡，必須看得出來那個下降是假的
        // 若這一項失敗，保留集就跟其他數字一樣沒有判別力，實機 log 會再次誤導我。
        seed = 8888
        let obsNoisy = observations(points: pts, poses: truth, noisePx: 3.0, noiseDepthM: 0.03)
        let rF = BundleAdjuster.refine(records: records(truth), observations: obsNoisy, rounds: 10)
        let objDrop = (rF.residualsPx.first ?? 1) > 1e-6
            ? (1 - (rF.residualsPx.last ?? 0) / (rF.residualsPx.first ?? 1)) * 100 : 0
        check((rF.holdoutDelta ?? 0) > -0.03,
              String(format: "保留集｜無位姿誤差、只有雜訊(3px/3cm) → 目標函數假性改善 %.0f%%，"
                     + "保留集 %+.0f%%（不得聲稱變好）",
                     objDrop, (rF.holdoutDelta ?? 0) * 100))

        print()
        print(fails == 0 ? "全部通過 — 局部 BA 驗證完成" : "\(fails) 項失敗")
        exit(fails == 0 ? 0 : 1)
    }
}
