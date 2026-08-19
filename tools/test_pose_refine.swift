//
//  test_pose_refine.swift
//  fable — 位姿微調求解器的離線驗證（不需要 iPhone）
//
//  這支測試的結論導致 CaptureConfig.poseRefineRounds 預設為 0：
//  以「點落在哪個 voxel」建立對應的做法**不成立** ——
//    沿平面切向偏 3cm → 只呈現 0.95cm 表觀誤差（落進同平面鄰格）
//    沿平面法向偏 3cm → 0/40000 點有對應（全部落進空格）
//  在最該修正的方向收不到訊號、在不該動的方向給誤導訊號。
//  正解是鄰域最近點搜尋 ＋ 點到面殘差（standard ICP）。
//
//  已驗證正確的部分：6×6 線性化求解、阻尼、夾住、去除全域漂移。
//
//  編譯（FusedVoxelGrid / PointCloudMath 從 RefusionEngine.swift 抽出，
//  因為該檔還相依 CoreML/ImageIO，測試不需要那些）：
//
//    python3 - <<'EOF'
//    import pathlib
//    s = pathlib.Path("fable/Capture/RefusionEngine.swift").read_text()
//    i, j = s.index("nonisolated enum PointCloudMath"), s.index("// MARK: - 重融合引擎")
//    pathlib.Path("/tmp/grid.swift").write_text(
//        "import Foundation\nimport simd\n"
//        "struct CloudPoint { var x: Float, y: Float, z: Float\n"
//        "                    var r: UInt8, g: UInt8, b: UInt8\n"
//        "                    var score: Float = 1 }\n\n" + s[i:j])
//    EOF
//    swiftc -O -o /tmp/posetest fable/Capture/PoseRefiner.swift \
//           tools/test_pose_refine.swift /tmp/grid.swift && /tmp/posetest
//

import Foundation
import simd

@main struct PoseTest {
    /// 合成一個房間的三面牆＋地板上的點（法向多樣，剛體約束才完整）
    static func surface(_ n: Int) -> [SIMD3<Float>] {
        var pts: [SIMD3<Float>] = []
        var seed: UInt64 = 12345
        func rnd() -> Float {           // 可重現的 LCG，不用 Math.random
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Float(seed >> 33) / Float(UInt32.max >> 1)
        }
        for _ in 0..<n {
            switch Int(rnd() * 4) % 4 {
            case 0: pts.append(SIMD3(rnd() * 4, rnd() * 2.4, 0))        // 牆 z=0
            case 1: pts.append(SIMD3(0, rnd() * 2.4, rnd() * 3))        // 牆 x=0
            case 2: pts.append(SIMD3(rnd() * 4, 0, rnd() * 3))          // 地板
            default: pts.append(SIMD3(4, rnd() * 2.4, rnd() * 3))       // 牆 x=4
            }
        }
        return pts
    }

    static func main() {
        var fails = 0
        func check(_ ok: Bool, _ what: String) {
            print(ok ? "✅ \(what)" : "❌ \(what)"); if !ok { fails += 1 }
        }

        let truth = surface(40000)
        var grid = FusedVoxelGrid(voxelSize: 0.02, maxCells: 2_000_000)
        grid.insert(truth.map { CloudPoint(x: $0.x, y: $0.y, z: $0.z, r: 0, g: 0, b: 0, score: 1) })
        print("共識點雲 \(grid.count) 格 @ 2cm\n")

        let camPos = SIMD3<Float>(2, 1.2, 1.5)

        // ── 純平移擾動 ──
        for mag: Float in [0.005, 0.02, 0.04] {
            let off = SIMD3<Float>(mag, 0, 0)
            let moved = truth.map { $0 + off }
            guard let (delta, rms) = PoseRefiner.solveRigid(worldPoints: moved,
                                                           camPos: camPos + off, grid: grid) else {
                check(false, "平移 \(mag*100)cm：求解失敗"); continue
            }
            // 套用修正後再量殘差
            let fixed = moved.map { p -> SIMD3<Float> in
                let q = delta * SIMD4<Float>(p, 1); return SIMD3(q.x, q.y, q.z)
            }
            let after = residual(fixed, grid)
            check(after < rms * 0.6,
                  String(format: "平移 %.0fcm → 殘差 %.2f → %.2f cm（降 %.0f%%）",
                         mag*100, rms*100, after*100, (1 - after/rms)*100))
        }

        // ── 繞相機中心的小角度旋轉 ──
        for deg: Float in [0.3, 0.8] {
            let a = deg * .pi / 180
            let R = simd_float3x3(SIMD3(cos(a), 0, -sin(a)), SIMD3(0, 1, 0), SIMD3(sin(a), 0, cos(a)))
            let moved = truth.map { R * ($0 - camPos) + camPos }
            guard let (delta, rms) = PoseRefiner.solveRigid(worldPoints: moved,
                                                           camPos: camPos, grid: grid) else {
                check(false, "旋轉 \(deg)°：求解失敗"); continue
            }
            let fixed = moved.map { p -> SIMD3<Float> in
                let q = delta * SIMD4<Float>(p, 1); return SIMD3(q.x, q.y, q.z)
            }
            let after = residual(fixed, grid)
            check(after < rms * 0.7,
                  String(format: "旋轉 %.1f° → 殘差 %.2f → %.2f cm（降 %.0f%%）",
                         deg, rms*100, after*100, (1 - after/rms)*100))
        }

        // ── 沒有擾動時不該亂動（阻尼與夾住不能產生假修正）──
        if let (delta, _) = PoseRefiner.solveRigid(worldPoints: truth, camPos: camPos, grid: grid) {
            let t = simd_length(SIMD3(delta.columns.3.x, delta.columns.3.y, delta.columns.3.z))
            check(t < 0.004, String(format: "無擾動時修正量 %.1f mm（應接近 0）", t*1000))
        } else { check(false, "無擾動：求解失敗") }

        // ── 全域漂移去除 ──
        var ds: [Int: simd_float4x4] = [:]
        for i in 0..<5 {
            var m = matrix_identity_float4x4
            m.columns.3 = SIMD4<Float>(Float(i) * 0.01 + 0.1, 0, 0, 1)   // 共同偏移 + 遞增
            ds[i] = m
        }
        PoseRefiner.removeGlobalDrift(&ds)
        var mean = SIMD3<Float>.zero
        for d in ds.values { mean += SIMD3(d.columns.3.x, d.columns.3.y, d.columns.3.z) }
        check(simd_length(mean / 5) < 1e-5, "去除全域漂移後平均修正量為 0")

        print()
        print(fails == 0 ? "全部通過 — 位姿微調求解器驗證完成" : "\(fails) 項失敗")
        exit(fails == 0 ? 0 : 1)
    }

    static func residual(_ pts: [SIMD3<Float>], _ grid: FusedVoxelGrid) -> Double {
        var sum: Double = 0; var n = 0
        let maxD2 = pow(grid.voxelSize * PoseRefiner.kMaxCorrespVoxels, 2)
        for p in pts {
            guard let k = PointCloudMath.voxelKey(p, size: grid.voxelSize),
                  let c = grid.cells[k] else { continue }
            let d2 = simd_length_squared(c.mean - p)
            if d2 > maxD2 { continue }
            sum += Double(d2); n += 1
        }
        return n > 0 ? (sum / Double(n)).squareRoot() : .infinity
    }
}
