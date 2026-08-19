//
//  BundleAdjuster.swift
//  fable — 以 ARKit 位姿為初值的局部 BA（重投影誤差，非幾何對齊）
//
//  ── 為什麼是重投影誤差而不是幾何對齊 ──────────────────────────────
//  先前試過「把每幀深度對齊到融合點雲」（PoseRefiner + voxel 對應），
//  被離線測試否決：沿平面法向偏 3cm 時 0/40000 點有對應 ——
//  在最該修正的方向完全收不到訊號（見 tools/test_pose_refine.swift）。
//
//  重投影誤差沒有這個盲點：表面往鏡頭方向移動，投影位置就是實實在在地位移了。
//  而且它**就是**決定 3DGS 解析度天花板的那個量 ——
//  1cm 位姿誤差 @2m ≈ 7px 重投影誤差，高斯縮得比它小就會被各視角的矛盾懲罰。
//  幾何對齊只是它的代理指標。
//
//  ── 為什麼不需要 Ceres ────────────────────────────────────────
//  因為有 LiDAR，每個特徵的 3D 位置是直接讀出來的、不必三角化。
//  結構固定 ⇒ 各相機的位姿**彼此獨立**，正規方程從 (6N+3M)² 退化成每台相機一個 6×6。
//  6×6 的 Cholesky 手寫就好（PoseRefiner 已驗證），Schur complement 用不到。
//  代價是變成 coordinate descent（交替：修位姿 → 重算 track 3D → 再修位姿）
//  而非聯合解 —— 但 3D 有 LiDAR 錨定、不是自由變數，這個交替是收斂的。
//
//  ── 自我驗證 ─────────────────────────────────────────────────
//  每輪都量修正前後的重投影 RMS，沒下降就回退並停止。
//

import Foundation
import simd

nonisolated enum BundleAdjuster {

    /// Huber 損失的轉折點（像素）。超過此殘差的觀測降權，擋掉誤匹配 ——
    /// 引導式搜尋已經過濾掉大部分，但磁磚/格紋這類自相似區域仍會漏一些。
    static let kHuberPx: Float = 2.0
    /// 單幀至少要這麼多觀測才求解（6 個未知數，實務上要遠多於此才穩）
    static let kMinObsPerFrame = 30
    /// 離群觀測的硬上限（像素）：超過就完全不採用。
    /// ARKit 位姿的殘差量級是十幾 px，超過 40px 幾乎確定是誤匹配。
    static let kMaxResidualPx: Float = 40
    /// 深度殘差自己的 Huber 轉折點（像素等效）。
    ///
    /// **不能與重投影共用 kHuberPx。** 深度殘差的雜訊來源是 LiDAR
    /// （σ≈1cm @2m ⇒ 約 7px 等效），而特徵定位雜訊只有 ~0.5px ——
    /// 用 2px 的門檻會把「3cm 的真實深度不一致」判成離群、權重壓到 9%，
    /// 於是深度項幾乎沒有作用（離線測試量到：沿光軸 3cm 的誤差只修回 30%）。
    /// 12px ≈ 1.7cm @2m ≈ 1.7σ，落在合理的 robust 門檻上。
    static let kDepthHuberPx: Float = 12

    /// 深度殘差的權重倍率（0 = 只用重投影）。
    ///
    /// 1.0 是實測掃出來的：1 / 3 / 8 / 20 對「沿光軸平移」的修正力幾乎相同
    /// （1.30 / 1.22 / 1.26 / 1.29 cm），但加大權重會讓一般情況變差
    /// （含雜訊案例 0.63 → 0.87cm）—— 因為深度雜訊(LiDAR σ≈1cm)被放大進來了。
    /// 也就是說那個方向的弱是**結構性的**、不是權重不夠，加權重只有壞處。
    ///
    /// **為什麼一定要有這一項** —— 純重投影對「沿光軸方向的平移」是弱觀測的：
    /// 把一個點沿著它的視線移動，投影位置幾乎不變。這是單目 SfM 的經典退化，
    /// 一般靠場景深度多樣性（近點遠點位移不同）勉強約束。
    /// 但我們有 LiDAR —— 直接比對「track 的 3D 位置投影到本幀的預測深度」與
    /// 「該像素實測的深度」，那個方向就被硬約束住了。
    ///
    /// 權重取 fx/d，讓 1 公尺的深度誤差換算成與重投影同尺度的像素數
    /// （d=2m、fx=1450 時 1cm 深度誤差 ≈ 7.25 px，正好與橫向誤差同量級）——
    /// 兩種殘差因此可以直接相加，不必另外調係數。
    static let kDepthWeight: Float = 1.0

    /// 執行局部 BA。回傳修正後的位姿與每輪的重投影 RMS（像素）。
    ///
    /// - records: 關鍵幀（transform 為初值，來自 ARKit＋錨點修正）
    /// - observations: 掃描時同步建立的跨幀對應（見 FeatureTracker）
    static func refine(records: [FrameRecord], observations: [FeatureObservation],
                       rounds: Int) -> PoseRefineResult {
        var result = PoseRefineResult()
        guard rounds > 0, !observations.isEmpty else { return result }

        // 逐幀索引：id → (內參, 當前 c2w, 該幀的觀測)
        var order: [Int] = []
        var poses: [Int: simd_float4x4] = [:]
        var intr: [Int: CameraIntrinsics] = [:]
        var obsByFrame: [Int: [FeatureObservation]] = [:]
        for r in records where r.blurVerdict != .drop {
            poses[r.id] = RefusionEngine.float4x4(rowMajor: r.transform)
            intr[r.id] = r.intrinsics
            order.append(r.id)
        }
        for o in observations where poses[o.frameID] != nil {
            obsByFrame[o.frameID, default: []].append(o)
        }
        // 只留觀測足夠的幀
        let framesBefore = order.count
        order = order.filter { (obsByFrame[$0]?.count ?? 0) >= kMinObsPerFrame }
        guard order.count >= 3 else {
            // 不要靜默返回 —— 「完全沒有輸出」看起來像功能沒做，而不是條件不足
            print("BA: 略過 —— \(framesBefore) 幀中只有 \(order.count) 幀的觀測數達 "
                  + "\(kMinObsPerFrame)（共 \(observations.count) 個觀測）。"
                  + "特徵匹配產出率不足，原因見上方「匹配」分解")
            return result
        }

        /// track 的 3D 位置＝所有觀測到它的幀給出的 LiDAR 世界座標平均。
        /// 每輪重算一次：位姿改了，同一個 track 的各幀反投影也跟著改。
        func trackPoints() -> [Int: SIMD3<Float>] {
            trackPointsFor(poses, order: order, intr: intr, obsByFrame: obsByFrame)
        }

        var bestPoses = poses
        for round in 0..<rounds {
            let pts = trackPoints()
            let rBefore = residuals(order: order, poses: poses, intr: intr,
                                    obsByFrame: obsByFrame, points: pts)
            if result.residualsPx.isEmpty { result.residualsPx.append(rBefore.total) }

            // 逐幀獨立求解（結構固定 ⇒ 相機之間解耦）
            var deltas: [Int: simd_float4x4] = [:]
            for id in order {
                guard let c2w = poses[id], let K = intr[id], let obs = obsByFrame[id] else { continue }
                if let d = solveFrame(c2w: c2w, K: K, obs: obs, points: pts) {
                    deltas[id] = d
                }
            }
            guard !deltas.isEmpty else { break }

            // 全域剛體自由度：所有相機一起平移不改變任何重投影殘差 ——
            // 不扣掉平均修正量的話整組位姿會慢慢漂走，
            // 而點雲/ARWorldMap/上一次掃描的座標系就對不上了。
            PoseRefiner.removeGlobalDrift(&deltas)

            var candidate = poses
            for (id, d) in deltas { if let c = candidate[id] { candidate[id] = d * c } }
            let rAfter = residuals(order: order, poses: candidate, intr: intr,
                                   obsByFrame: obsByFrame,
                                   points: trackPointsFor(candidate, order: order,
                                                          intr: intr, obsByFrame: obsByFrame))
            // 自我驗證用 **robust 成本**（＝求解實際最小化的量），不用原始 RMS。
            // 原始 RMS 被少數誤匹配主導：一輪可能把 inlier 改善很多、RMS 卻沒降，
            // 於是被誤判為「沒變好」而提早停止（實機 BA 卡在 8% 改善的成因）。
            guard rAfter.robust < rBefore.robust else { break }

            poses = candidate
            bestPoses = candidate
            result.residualsPx.append(rAfter.total)
            result.roundsApplied = round + 1
        }

        guard result.roundsApplied > 0 else { return result }
        result.poses = bestPoses
        if let a = result.residualsPx.first, let b = result.residualsPx.last {
            // 位姿解讀只能用**重投影項**：深度項被 LiDAR 雜訊主導，
            // 把它算進去會把量測雜訊當成位姿誤差
            let rEnd = residuals(order: order, poses: poses, intr: intr,
                                 obsByFrame: obsByFrame, points: trackPoints())
            print(String(format: "BA: %d 幀 / %d 觀測 / %d tracks × %d 輪 → "
                         + "總 RMS %.2f → %.2f px（改善 %.0f%%）",
                         order.count, observations.count,
                         Set(observations.map(\.trackID)).count, result.roundsApplied,
                         a, b, (1 - b / a) * 100))
            // 中位數才是「典型」誤差：RMS 只擋 40px 硬上限，少數 30px 的誤匹配就能主導它。
            // 位姿解讀要看中位數；RMS 與中位數的差距則代表誤匹配的比重。
            print(String(format: "  重投影 RMS %.2f px / 中位數 %.2f px"
                         + "（⇒ 典型位姿誤差 %.2f cm @2m；RMS 高於中位數的部分是誤匹配）",
                         rEnd.reproj, rEnd.medianReproj,
                         Double(rEnd.medianReproj) * 2 / 1450 * 100))
            print(String(format: "  深度殘差 %.2f px（≈ %.1f cm，屬 LiDAR 量測雜訊，非位姿誤差）",
                         rEnd.depth, Double(rEnd.depth) * 2 / 1450 * 100))
        }
        return result
    }

    // MARK: - 單幀求解

    /// 對一幀解出位姿的小修正量。
    ///
    /// 參數化：c2w_new = ΔT · c2w_old，ΔT 以「繞相機中心的小角度旋轉 ＋ 平移」表示
    /// —— 與 PoseRefiner 一致，避免旋轉/平移在世界原點嚴重耦合。
    ///
    /// 相機座標下 Pc(x) = Pc + [Pc]×ω − t（x = (ω, t) 為世界座標的修正量，
    /// 經 R_w2c 轉到相機座標後的等效量），投影 Jacobian：
    ///
    ///     u = cx + X·fx/d,  v = cy − Y·fy/d,  d = −Z
    ///     ∂u/∂(X,Y,Z) = (−fx/Z,  0,      X·fx/Z²)
    ///     ∂v/∂(X,Y,Z) = ( 0,     fy/Z,  −Y·fy/Z²)
    static func solveFrame(c2w: simd_float4x4, K: CameraIntrinsics,
                           obs: [FeatureObservation],
                           points: [Int: SIMD3<Float>]) -> simd_float4x4? {
        let w2c = c2w.inverse
        let camPos = SIMD3<Float>(c2w.columns.3.x, c2w.columns.3.y, c2w.columns.3.z)
        let fx = Float(K.fx), fy = Float(K.fy), cx = Float(K.cx), cy = Float(K.cy)
        // 世界修正量 → 相機座標的旋轉部分
        let Rwc = simd_float3x3(SIMD3(w2c.columns.0.x, w2c.columns.0.y, w2c.columns.0.z),
                                SIMD3(w2c.columns.1.x, w2c.columns.1.y, w2c.columns.1.z),
                                SIMD3(w2c.columns.2.x, w2c.columns.2.y, w2c.columns.2.z))

        var ata = [Float](repeating: 0, count: 36)
        var atb = [Float](repeating: 0, count: 6)
        var used = 0

        for o in obs {
            guard let X = points[o.trackID] else { continue }
            let pc4 = w2c * SIMD4<Float>(X, 1)
            let Z = pc4.z
            guard Z < -1e-4 else { continue }
            let up = cx + pc4.x * fx / (-Z)
            let vp = cy - pc4.y * fy / (-Z)
            let ru = o.u - up, rv = o.v - vp
            let mag = (ru * ru + rv * rv).squareRoot()
            if mag > kMaxResidualPx { continue }                  // 硬離群
            // Huber：轉折點外降權（√w 施加在 Jacobian 與殘差上）
            let wgt = mag <= kHuberPx ? Float(1) : (kHuberPx / mag)
            let sw = wgt.squareRoot()

            // 投影 Jacobian（2×3，對相機座標）
            let iz = 1 / Z
            let ju = SIMD3<Float>(-fx * iz, 0, pc4.x * fx * iz * iz)
            let jv = SIMD3<Float>(0, fy * iz, -pc4.y * fy * iz * iz)

            // ── ∂Pc/∂x 的推導（兩個容易錯的地方都在這裡）──────────────────
            //
            // ΔT 是套在**相機**上：c2w_new = ΔT · c2w_old
            //   ⇒ w2c_new = c2w_old⁻¹ · ΔT⁻¹
            //   ⇒ Pc_new = Rwc·(ΔT⁻¹ X) + twc ≈ Pc_old − Rwc·(ω×r + t)
            //
            // 注意是 ΔT**⁻¹** 作用在世界點上 —— 相機往 +t 移動等於世界點往 −t 移動。
            // 我第一版把它寫成「世界點 X + ω×r + t」，於是整組 Jacobian 符號相反，
            // 求解方向剛好朝著錯的方向走（離線測試量到解出 +0.0092 而正解是 −0.0092）。
            //
            // 另一個坑：∂(ω×r)/∂ω_k = e_k × r，那是 −[r]× 的**第 k 欄**，不是第 k 列。
            //   e_x × r = ( 0,  −r.z,  r.y)
            //   e_y × r = ( r.z,  0,  −r.x)
            //   e_z × r = (−r.y,  r.x,  0 )
            let r = X - camPos
            let dOmega = [Rwc * SIMD3<Float>(0, r.z, -r.y),      // −(e_x × r)
                          Rwc * SIMD3<Float>(-r.z, 0, r.x),      // −(e_y × r)
                          Rwc * SIMD3<Float>(r.y, -r.x, 0)]      // −(e_z × r)
            let dT = [Rwc * SIMD3<Float>(-1, 0, 0),
                      Rwc * SIMD3<Float>(0, -1, 0),
                      Rwc * SIMD3<Float>(0, 0, -1)]

            var rowU = [Float](repeating: 0, count: 6)
            var rowV = [Float](repeating: 0, count: 6)
            for i in 0..<3 {
                rowU[i] = simd_dot(ju, dOmega[i]) * sw
                rowV[i] = simd_dot(jv, dOmega[i]) * sw
                rowU[i + 3] = simd_dot(ju, dT[i]) * sw
                rowV[i + 3] = simd_dot(jv, dT[i]) * sw
            }
            let bu = ru * sw, bv = rv * sw
            for i in 0..<6 {
                atb[i] += rowU[i] * bu + rowV[i] * bv
                for j in i..<6 {
                    ata[i * 6 + j] += rowU[i] * rowU[j] + rowV[i] * rowV[j]
                }
            }

            // 深度殘差：預測深度(−Z) vs 實測深度。約束重投影看不到的「沿光軸平移」。
            // 權重 fx/d 把公尺換算成同尺度的像素，故可與上面兩列直接相加。
            if kDepthWeight > 0 {
                let d = -Z
                let rd = (d - o.depth) * (fx / d) * kDepthWeight
                // ∂(−Z)/∂x = −(∂Pc/∂x 的 z 分量)
                let dwt = min(Float(1), kDepthHuberPx / max(kDepthHuberPx, abs(rd))).squareRoot()
                var rowD = [Float](repeating: 0, count: 6)
                for i in 0..<3 {
                    rowD[i] = -dOmega[i].z * (fx / d) * kDepthWeight * dwt
                    rowD[i + 3] = -dT[i].z * (fx / d) * kDepthWeight * dwt
                }
                let bd = -rd * dwt        // 殘差定義為 (量測 − 預測)，與上面的 ru/rv 一致
                for i in 0..<6 {
                    atb[i] += rowD[i] * bd
                    for j in i..<6 { ata[i * 6 + j] += rowD[i] * rowD[j] }
                }
            }
            used += 1
        }
        guard used >= kMinObsPerFrame else { return nil }
        for i in 0..<6 { for j in 0..<i { ata[i * 6 + j] = ata[j * 6 + i] } }
        let trace = (0..<6).reduce(Float(0)) { $0 + ata[$1 * 6 + $1] }
        for i in 0..<6 { ata[i * 6 + i] += max(1e-9, trace / 6 * 1e-4) }   // Levenberg

        guard let x = PoseRefiner.choleskySolve6(ata, atb) else { return nil }
        var omega = SIMD3<Float>(x[0], x[1], x[2])
        var trans = SIMD3<Float>(x[3], x[4], x[5])
        let rn = simd_length(omega)
        if rn > PoseRefiner.kMaxRotRad { omega *= PoseRefiner.kMaxRotRad / rn }
        let tn = simd_length(trans)
        if tn > PoseRefiner.kMaxTransM { trans *= PoseRefiner.kMaxTransM / tn }
        omega *= PoseRefiner.kDamping
        trans *= PoseRefiner.kDamping
        return PoseRefiner.deltaTransform(omega: omega, trans: trans, about: camPos)
    }

    // MARK: - 工具

    /// 該觀測在**相機座標**下的位置（位姿無關）。
    /// 與 RefusionEngine 的反投影同一組公式：(xc, -yc, -depth)，ARKit GL 慣例。
    @inline(__always)
    static func cameraLocal(of o: FeatureObservation, K: CameraIntrinsics) -> SIMD3<Float> {
        let xc = (o.u - Float(K.cx)) / Float(K.fx) * o.depth
        let yc = (o.v - Float(K.cy)) / Float(K.fy) * o.depth
        return SIMD3<Float>(xc, -yc, -o.depth)
    }

    static func trackPointsFor(_ poses: [Int: simd_float4x4], order: [Int],
                               intr: [Int: CameraIntrinsics],
                               obsByFrame: [Int: [FeatureObservation]]) -> [Int: SIMD3<Float>] {
        var sum: [Int: SIMD3<Float>] = [:]
        var cnt: [Int: Int] = [:]
        for id in order {
            guard let c2w = poses[id], let K = intr[id], let obs = obsByFrame[id] else { continue }
            for o in obs {
                let w = c2w * SIMD4<Float>(cameraLocal(of: o, K: K), 1)
                sum[o.trackID, default: .zero] += SIMD3(w.x, w.y, w.z)
                cnt[o.trackID, default: 0] += 1
            }
        }
        var out: [Int: SIMD3<Float>] = [:]
        for (t, s) in sum { out[t] = s / Float(cnt[t] ?? 1) }
        return out
    }

    /// 目標函數的 RMS（像素）。含深度項 —— 否則自我驗證看不到深度約束帶來的改善，
    /// 會把有效的一輪判定為「沒變好」而停下。
    /// 這個值同時就是「3DGS 解析度天花板」的直接量測。
    static func reprojectionRMS(order: [Int], poses: [Int: simd_float4x4],
                                intr: [Int: CameraIntrinsics],
                                obsByFrame: [Int: [FeatureObservation]],
                                points: [Int: SIMD3<Float>]) -> Float {
        let r = residuals(order: order, poses: poses, intr: intr,
                          obsByFrame: obsByFrame, points: points)
        return r.total
    }

    /// 分開回報重投影與深度殘差。
    ///
    /// **必須分開看。** 深度殘差的雜訊來源是 LiDAR（σ≈1cm @2m ⇒ 約 7px 等效），
    /// 混進總 RMS 之後再換算成「等效位姿誤差」會把量測雜訊算成位姿誤差 ——
    /// 實機出現過「總 RMS 14.8px ⇒ 聲稱位姿誤差 2.0cm」，但同一份 log 的
    /// 漂移修正只有 0.4cm，兩者矛盾。只有重投影項才適合做位姿解讀。
    static func residuals(order: [Int], poses: [Int: simd_float4x4],
                          intr: [Int: CameraIntrinsics],
                          obsByFrame: [Int: [FeatureObservation]],
                          points: [Int: SIMD3<Float>])
        -> (total: Float, reproj: Float, depth: Float, medianReproj: Float, robust: Double) {
        var sum: Double = 0, sumR: Double = 0, sumD: Double = 0
        var n = 0
        var reprojMags: [Float] = []
        // robust：與求解實際最小化的量一致（Huber 加權平方和）。
        // 自我驗證必須用它，不能用原始 RMS —— 兩者不一致時，
        // 一輪明明改善了 inlier 卻因為少數離群值讓 RMS 沒降而被判失敗、提早停止。
        var robust: Double = 0
        for id in order {
            guard let c2w = poses[id], let K = intr[id], let obs = obsByFrame[id] else { continue }
            let w2c = c2w.inverse
            let fx = Float(K.fx)
            for o in obs {
                guard let X = points[o.trackID] else { continue }
                let pc = w2c * SIMD4<Float>(X, 1)
                guard pc.z < -1e-4 else { continue }
                let d = -pc.z
                let up = Float(K.cx) + pc.x * fx / d
                let vp = Float(K.cy) - pc.y * Float(K.fy) / d
                let du = Double(o.u - up), dv = Double(o.v - vp)
                let r2 = du * du + dv * dv
                if r2 > Double(kMaxResidualPx * kMaxResidualPx) { continue }
                var m2 = r2
                var d2: Double = 0
                if kDepthWeight > 0 {
                    let rd = Double((d - o.depth) * (fx / d) * kDepthWeight)
                    d2 = rd * rd
                    m2 += d2
                }
                sum += m2; sumR += r2; sumD += d2
                let mag = Float(r2.squareRoot())
                reprojMags.append(mag)
                // Huber：|r| ≤ δ 用平方、超過改用線性（與 solveFrame 的加權一致）
                robust += mag <= kHuberPx
                    ? Double(mag * mag)
                    : Double(kHuberPx * (2 * mag - kHuberPx))
                if kDepthWeight > 0 {
                    let dm = Float(d2.squareRoot())
                    robust += dm <= kDepthHuberPx
                        ? Double(dm * dm)
                        : Double(kDepthHuberPx * (2 * dm - kDepthHuberPx))
                }
                n += 1
            }
        }
        guard n > 0 else { return (.infinity, .infinity, .infinity, .infinity, .infinity) }
        let k = Double(n)
        reprojMags.sort()
        return (Float((sum / k).squareRoot()),
                Float((sumR / k).squareRoot()),
                Float((sumD / k).squareRoot()),
                reprojMags[reprojMags.count / 2],
                robust / k)
    }
}
