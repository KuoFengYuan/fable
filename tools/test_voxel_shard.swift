//
//  test_voxel_shard.swift
//  fable — 分片 voxel 融合格的離線驗證（不需要 iPhone）
//
//  為什麼這支測試存在：插入是重融合唯一不能平行的一段，而它隨幀數線性成長
//  （實測 54 幀 1.81s，整層掃描 ~600 幀線性外推是 20 秒）。分片之後 N 條 lane
//  可以同時寫，但**分片改變了跨 cell 的插入順序**。
//
//  這件事有沒有影響，取決於加權平均是否順序相依 —— 而它是的：
//  weight 在 weightCap 封頂之後，「先到」與「後到」的觀測拿到的權重不同。
//  所以若分片沒有嚴格保序，同一份掃描資料會融出**不同的點雲**，
//  而且差異小到不會有人發現，只會表現為「品質有時候好一點有時候差一點」。
//
//  做法：把真正的 FusedVoxelGrid 從 RefusionEngine.swift 切出來，
//  與一份單片參考實作餵完全相同的點，逐 cell 比對到位元。
//
//  編譯（grid.swift 由下方 python 從 RefusionEngine.swift 切出，
//  這樣測的一定是正在用的那份程式碼，不是複製品）：
//
//    python3 - <<'EOF'
//    import pathlib
//    s = pathlib.Path("fable/Capture/RefusionEngine.swift").read_text()
//    i, j = s.index("nonisolated enum PointCloudMath"), s.index("// MARK: - 重融合引擎")
//    pathlib.Path("/tmp/grid.swift").write_text(
//        "import Foundation\nimport Dispatch\nimport simd\n"
//        "struct CloudPoint { var x: Float, y: Float, z: Float\n"
//        "                    var r: UInt8, g: UInt8, b: UInt8\n"
//        "                    var score: Float = 1 }\n\n" + s[i:j])
//    EOF
//    swiftc -O -o /tmp/vstest tools/test_voxel_shard.swift /tmp/grid.swift && /tmp/vstest
//

import Foundation
import simd

/// 單片參考實作：分片之前的原始語意，逐字保留（含 weightCap 的封頂行為）
struct ReferenceGrid {
    struct Cell {
        var mean: SIMD3<Float>
        var color: SIMD3<Float>
        var weight: Float
        var bestScore: Float
        var measured: Bool = true
    }
    var cells: [Int64: Cell] = [:]
    var voxelSize: Float
    let weightCap: Float = 8

    mutating func insert(_ candidates: [CloudPoint], measured: Bool = true) {
        for pt in candidates {
            let pos = SIMD3<Float>(pt.x, pt.y, pt.z)
            guard let key = PointCloudMath.voxelKey(pos, size: voxelSize) else { continue }
            let rgb = SIMD3<Float>(Float(pt.r), Float(pt.g), Float(pt.b))
            if var cell = cells[key] {
                let w = max(0.01, pt.score)
                let total = cell.weight + w
                cell.mean += (pos - cell.mean) * (w / total)
                cell.color += (rgb - cell.color) * (w / total)
                cell.weight = min(total, weightCap)
                cell.bestScore = max(cell.bestScore, pt.score)
                cell.measured = cell.measured || measured
                cells[key] = cell
            } else {
                cells[key] = Cell(mean: pos, color: rgb,
                                  weight: max(0.01, pt.score), bestScore: pt.score,
                                  measured: measured)
            }
        }
    }
}

@main struct VoxelShardTest {

    static var seed: UInt64 = 20260821
    static func rnd() -> Float {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Float(seed >> 33) / Float(UInt32.max >> 1)
    }

    /// 一「幀」的點。刻意讓它們高度重疊（同一片表面被多幀看到）——
    /// 那正是加權平均與封頂行為會被觸發的情況，也是實機的常態
    /// （實測 2.65M 個點融成 152,927 格 ⇒ 平均每格 17 個觀測）。
    static func frame(_ n: Int, spread: Float) -> [CloudPoint] {
        (0..<n).map { _ in
            CloudPoint(x: rnd() * spread, y: rnd() * spread, z: rnd() * spread,
                       r: UInt8(rnd() * 255), g: UInt8(rnd() * 255), b: UInt8(rnd() * 255),
                       score: 0.05 + rnd() * 1.5)
        }
    }

    static func main() {
        var fails = 0
        func check(_ ok: Bool, _ what: String) {
            print(ok ? "✅ \(what)" : "❌ \(what)"); if !ok { fails += 1 }
        }

        let voxel: Float = 0.02

        // ── 1. 逐位元一致：多「幀」重疊插入，每格平均收到十幾個觀測 ──
        seed = 4242
        var sharded = FusedVoxelGrid(voxelSize: voxel, maxCells: 100_000_000)
        var reference = ReferenceGrid(voxelSize: voxel)
        var totalPts = 0
        for _ in 0..<40 {
            // spread 0.6m / 2cm 格 → 約 27000 個可能格位，每幀 20000 點 ⇒ 大量重疊
            let pts = frame(20_000, spread: 0.6)
            totalPts += pts.count
            sharded.insert(pts)
            reference.insert(pts)
        }
        check(sharded.count == reference.cells.count,
              "格數一致：分片 \(sharded.count) vs 單片 \(reference.cells.count)"
              + "（\(totalPts) 個點，平均每格 \(totalPts / max(1, reference.cells.count)) 個觀測）")

        // 逐格比對。用 exportPoints 取出分片版的內容，對照參考版同一個 key。
        // minNeighbors: 0 → 不做孤立點移除，才是純粹的融合結果比對
        let out = sharded.exportPoints(target: Int.max, minNeighbors: 0)
        var worstPos: Float = 0, worstCol: Float = 0, missing = 0
        for p in out {
            let pos = SIMD3<Float>(p.x, p.y, p.z)
            // 分片版的 mean 已經被加權平均移動過，用它反查格子仍會落在同一格
            guard let k = PointCloudMath.voxelKey(pos, size: voxel),
                  let ref = reference.cells[k] else { missing += 1; continue }
            worstPos = max(worstPos, simd_length(pos - ref.mean))
            worstCol = max(worstCol, abs(Float(p.r) - ref.color.x))
            worstCol = max(worstCol, abs(Float(p.g) - ref.color.y))
            worstCol = max(worstCol, abs(Float(p.b) - ref.color.z))
        }
        check(missing == 0 && worstPos == 0 && worstCol <= 1,
              String(format: "逐格逐位元一致：%d 格比對，位置最大差 %.9f m、"
                     + "顏色最大差 %.1f（量化到 UInt8 允許 1）、查無對應 %d 格",
                     out.count, Double(worstPos), Double(worstCol), missing))

        // ── 2. 分片必須散得開，否則平行化沒有意義 ──
        //      voxel key 是 (ix<<42)|(iy<<21)|iz，直接取低位等於只用 z 分片；
        //      掃描面接近水平時（地板、天花板）會全擠在同一片。
        seed = 77
        var planar = FusedVoxelGrid(voxelSize: voxel, maxCells: 100_000_000)
        // 刻意造一個近乎水平的面：y 只有 2cm 的變化，x/z 各 4m
        let flat = (0..<200_000).map { _ in
            CloudPoint(x: rnd() * 4, y: rnd() * 0.02, z: rnd() * 4,
                       r: 128, g: 128, b: 128, score: 1)
        }
        planar.insert(flat)
        let occ = planar.shardOccupancy()
        let maxShare = Double(occ.max() ?? 0) / Double(max(1, occ.reduce(0, +)))
        check(maxShare < 2.0 / Double(occ.count),
              String(format: "水平面 20 萬點：最大分片佔 %.1f%%（%d 片，平均 %.1f%%，"
                     + "需 < 2 倍平均）", maxShare * 100, occ.count, 100.0 / Double(occ.count)))

        // ── 3. 觸頂粗化：分片重建之後仍必須有界，且格距確實加倍 ──
        seed = 5150
        var small = FusedVoxelGrid(voxelSize: voxel, maxCells: 20_000)
        for _ in 0..<10 { small.insert(frame(20_000, spread: 3.0)) }
        check(small.count < 20_000 * 2 && small.voxelSize > voxel,
              String(format: "觸頂粗化有效：上限 20000 → 實際 %d 格，格距 %.0fcm → %.0fcm",
                     small.count, Double(voxel * 100), Double(small.voxelSize * 100)))

        // ── 4. 空輸入不得炸掉（分片版多了一段分堆，邊界要顧）──
        var empty = FusedVoxelGrid(voxelSize: voxel, maxCells: 1000)
        empty.insert([])
        check(empty.count == 0 && empty.exportPoints(target: 100, minNeighbors: 3).isEmpty,
              "空輸入安全：0 格、匯出 0 點")

        print()
        print(fails == 0 ? "全部通過 — 分片 voxel 融合格驗證完成" : "\(fails) 項失敗")
        exit(fails == 0 ? 0 : 1)
    }
}
