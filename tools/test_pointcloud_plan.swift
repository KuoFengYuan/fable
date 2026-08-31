//
//  test_pointcloud_plan.swift
//  fable — 點雲 → 2D 平面圖的離線驗證（不需要 iPhone）
//
//  合成一個已知尺寸的房間點雲（牆、地板、天花板、家具），跑抽取，
//  再對照真值檢查牆的位置與長度。真值已知，所以量得到的是「錯多少公分」，
//  而不是只看「有沒有輸出東西」。
//
//  這支測試在寫的過程中就抓到一個會讓整個方法失效的 bug：
//  房間裡每個水平格子都同時有地板點與天花板點，垂直跨度等於樓高 ——
//  空曠的地板中央因此會跟牆一樣被判成「牆」。修法是先切掉地板/天花板帶。
//
//  編譯：
//    swiftc -O -o /tmp/pctest fable/Capture/FloorPlanData.swift \
//           fable/Capture/PointCloudFloorPlan.swift fable/Capture/FloorPlanDXF.swift \
//           tools/test_pointcloud_plan.swift && /tmp/pctest
//

import Foundation
import simd

@main struct PointCloudPlanTest {

    static var seed: UInt64 = 20260901
    static func rnd() -> Float {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Float(seed >> 33) / Float(UInt32.max >> 1)
    }
    /// LiDAR 級雜訊：σ≈1cm。用兩個均勻分佈相加近似常態，夠用且可重現
    static func noise(_ s: Float) -> Float { (rnd() + rnd() - 1) * s }

    // MARK: - 合成場景

    /// 一個 W×D、樓高 H 的長方形房間。牆掃在**內側面**上（LiDAR 看得到的那一面）。
    static func room(w: Float, d: Float, h: Float, step: Float = 0.03,
                     rotateDeg: Float = 0) -> [SIMD3<Float>] {
        var pts: [SIMD3<Float>] = []
        func add(_ x: Float, _ y: Float, _ z: Float) {
            pts.append(SIMD3(x + noise(0.01), y + noise(0.01), z + noise(0.01)))
        }
        var y: Float = 0
        while y <= h {
            var t: Float = 0
            while t <= w { add(t, y, 0); add(t, y, d); t += step }   // z=0 與 z=d 兩面牆
            t = 0
            while t <= d { add(0, y, t); add(w, y, t); t += step }   // x=0 與 x=w 兩面牆
            y += step
        }
        // 地板與天花板
        var x: Float = 0
        while x <= w {
            var z: Float = 0
            while z <= d { add(x, 0, z); add(x, h, z); z += step }
            x += step
        }
        guard rotateDeg != 0 else { return pts }
        let r = rotateDeg * .pi / 180
        let c = cos(r), s = sin(r)
        return pts.map { SIMD3($0.x * c - $0.z * s, $0.y, $0.x * s + $0.z * c) }
    }

    /// 辦公桌：0.75m 高的水平桌面 ＋ 四支細腳。垂直跨度只有桌面那一層，
    /// 所以**不該**被判成牆 —— 這是這個方法最重要的一條分辨線。
    static func desk(at p: SIMD2<Float>, w: Float = 1.4, d: Float = 0.7) -> [SIMD3<Float>] {
        var pts: [SIMD3<Float>] = []
        var x: Float = 0
        while x <= w {
            var z: Float = 0
            while z <= d { pts.append(SIMD3(p.x + x, 0.75, p.y + z)); z += 0.02 }
            x += 0.02
        }
        return pts
    }

    /// 及腰的紙箱（0.6m 高）：同樣不該成為牆
    static func box(at p: SIMD2<Float>, size: Float = 0.5, h: Float = 0.6) -> [SIMD3<Float>] {
        var pts: [SIMD3<Float>] = []
        var y: Float = 0
        while y <= h {
            var t: Float = 0
            while t <= size {
                pts.append(SIMD3(p.x + t, y, p.y))
                pts.append(SIMD3(p.x + t, y, p.y + size))
                pts.append(SIMD3(p.x, y, p.y + t))
                pts.append(SIMD3(p.x + size, y, p.y + t))
                t += 0.02
            }
            y += 0.02
        }
        return pts
    }

    // MARK: - 評分

    /// 一段牆到真值線段的最大端點距離（公尺）
    static func fits(_ w: FloorPlanSurface, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        let p = SIMD2(w.segment2D[0], w.segment2D[1])
        let q = SIMD2(w.segment2D[2], w.segment2D[3])
        return min(max(simd_distance(p, a), simd_distance(q, b)),
                   max(simd_distance(p, b), simd_distance(q, a)))
    }

    static func main() {
        var fails = 0
        func check(_ ok: Bool, _ what: String) {
            print(ok ? "✅ \(what)" : "❌ \(what)"); if !ok { fails += 1 }
        }

        // ── 1. 軸對齊的空房間：四面牆都要抽出來，長度誤差在格距量級 ──
        seed = 1
        let W: Float = 5, D: Float = 4, H: Float = 2.6
        guard let r1 = PointCloudFloorPlan.extract(points: room(w: W, d: D, h: H)) else {
            print("❌ 空房間：完全沒有輸出"); exit(1)
        }
        print("   \(r1.summary)")
        let truth: [(SIMD2<Float>, SIMD2<Float>, Float)] = [
            (SIMD2(0, 0), SIMD2(W, 0), W), (SIMD2(0, D), SIMD2(W, D), W),
            (SIMD2(0, 0), SIMD2(0, D), D), (SIMD2(W, 0), SIMD2(W, D), D),
        ]
        var worstEnd: Float = 0, worstLen: Float = 0, matched = 0
        for (a, b, len) in truth {
            guard let best = r1.plan.walls.min(by: { fits($0, a, b) < fits($1, a, b) }),
                  fits(best, a, b) < 0.25 else { continue }
            matched += 1
            worstEnd = max(worstEnd, fits(best, a, b))
            worstLen = max(worstLen, abs(best.lengthM - len))
        }
        check(matched == 4,
              "空房間 5×4×2.6m：四面牆全部抽出（實得 \(matched)/4，共 \(r1.plan.walls.count) 段）")
        check(worstEnd < 0.15 && worstLen < 0.15,
              String(format: "牆位精度：端點最大誤差 %.1f cm、長度最大誤差 %.1f cm",
                     worstEnd * 100, worstLen * 100))

        // ── 2. 空曠地板不得被判成牆 ──
        //      這是「先切掉地板/天花板帶」那一步真正要擋的東西。
        //      房間中央 (2.5, 2) 附近若冒出牆段，代表那一步失效了。
        let center = SIMD2<Float>(W / 2, D / 2)
        let strays = r1.plan.walls.filter { w in
            let mid = SIMD2((w.segment2D[0] + w.segment2D[2]) / 2,
                            (w.segment2D[1] + w.segment2D[3]) / 2)
            return simd_distance(mid, center) < 1.2
        }
        check(strays.isEmpty,
              "空曠地板中央沒有假牆（半徑 1.2m 內找到 \(strays.count) 段）")

        // ── 3. 家具不得變成牆 ──
        seed = 2
        var furnished = room(w: W, d: D, h: H)
        furnished += desk(at: SIMD2(1.5, 1.5))
        furnished += box(at: SIMD2(3.2, 2.4))
        guard let r3 = PointCloudFloorPlan.extract(points: furnished) else {
            print("❌ 有家具的房間：完全沒有輸出"); exit(1)
        }
        let nearFurniture = r3.plan.walls.filter { w in
            let mid = SIMD2((w.segment2D[0] + w.segment2D[2]) / 2,
                            (w.segment2D[1] + w.segment2D[3]) / 2)
            return simd_distance(mid, SIMD2(2.2, 1.85)) < 0.9
                || simd_distance(mid, SIMD2(3.45, 2.65)) < 0.7
        }
        check(nearFurniture.isEmpty && r3.plan.walls.count >= 4,
              "0.75m 桌面與 0.6m 紙箱不會變成牆（附近 \(nearFurniture.count) 段，"
              + "全圖 \(r3.plan.walls.count) 段）")

        // ── 4. 房間轉 31° 也要抽得出來 ──
        //      Manhattan 主方向若估錯，牆會沿著錯的軸被切成一堆短段。
        seed = 3
        let deg: Float = 31
        guard let r4 = PointCloudFloorPlan.extract(points:
                room(w: W, d: D, h: H, rotateDeg: deg)) else {
            print("❌ 旋轉房間：完全沒有輸出"); exit(1)
        }
        print("   \(r4.summary)")
        let rr = deg * .pi / 180
        let rc = cos(rr), rs = sin(rr)
        func rot(_ p: SIMD2<Float>) -> SIMD2<Float> {
            SIMD2(p.x * rc - p.y * rs, p.x * rs + p.y * rc)
        }
        var rotMatched = 0, rotWorst: Float = 0
        for (a, b, _) in truth {
            let ra = rot(a), rb = rot(b)
            guard let best = r4.plan.walls.min(by: { fits($0, ra, rb) < fits($1, ra, rb) }),
                  fits(best, ra, rb) < 0.25 else { continue }
            rotMatched += 1
            rotWorst = max(rotWorst, fits(best, ra, rb))
        }
        check(rotMatched == 4,
              String(format: "房間轉 31°：四面牆全部抽出（實得 %d/4，端點最大誤差 %.1f cm）",
                     rotMatched, rotWorst * 100))

        // ── 5. 樓地板面積 ──
        let expect = W * D
        let got = r1.plan.floorAreaM2
        check(abs(got - expect) / expect < 0.15,
              String(format: "樓地板面積 %.1f m²（真值 %.1f，誤差 %.0f%%）",
                     got, expect, abs(got - expect) / expect * 100))

        // ── 6. DXF：結構完整、小數用點號、Y 有翻轉 ──
        let dxf = FloorPlanDXF.make(r1.plan)
        let sections = ["HEADER", "TABLES", "ENTITIES"].allSatisfy { dxf.contains("2\n\($0)\n") }
        let polylines = dxf.components(separatedBy: "\nPOLYLINE\n").count - 1
        let balanced = dxf.components(separatedBy: "\nSEQEND\n").count - 1 == polylines
        check(sections && dxf.hasSuffix("0\nEOF\n") && polylines == r1.plan.walls.count && balanced,
              "DXF 結構完整：三個區段 ＋ EOF ＋ \(polylines) 條 POLYLINE 與同數 SEQEND")
        // 逗號小數是實務上最常見的 DXF 損壞來源（本地化格式化寫出來的），
        // 而 CAD 開起來不會報錯，只會少一半圖元
        check(!dxf.contains(",") , "DXF 不含逗號（逗號小數會讓 CAD 靜默丟棄圖元）")
        // Y 翻轉：ARKit 俯視時 +Z 朝畫面下方，CAD 的 +Y 朝上，不翻會上下鏡射。
        //
        // 不能去比對某條中心線的座標 —— DXF 匯出的是**牆體輪廓**（沿法向各偏半個牆厚），
        // 中心線的值根本不會出現在檔案裡。要驗的是不變式本身：
        // 圖元的 y 範圍必須等於平面圖 z 範圍取負。
        var ys: [Float] = []
        let lines = dxf.components(separatedBy: "\n")
        for (i, l) in lines.enumerated() where l == "20" && i + 1 < lines.count {
            if let v = Float(lines[i + 1]) { ys.append(v) }
        }
        // 標頭的 $EXTMIN/$EXTMAX 也是 20 群組碼，會多出 1m 邊界 —— 只看圖元段
        let entIdx = dxf.range(of: "2\nENTITIES\n").map {
            dxf.distance(from: dxf.startIndex, to: $0.upperBound) } ?? 0
        let entLines = String(dxf.dropFirst(entIdx)).components(separatedBy: "\n")
        var entY: [Float] = []
        for (i, l) in entLines.enumerated() where l == "20" && i + 1 < entLines.count {
            if let v = Float(entLines[i + 1]) { entY.append(v) }
        }
        let zLo = r1.plan.boundsM[1], zHi = r1.plan.boundsM[3]
        let okFlip = !entY.isEmpty
            && abs((entY.min() ?? 0) - (-zHi)) < 0.2
            && abs((entY.max() ?? 0) - (-zLo)) < 0.2
        check(okFlip,
              String(format: "DXF 的 Y 已翻轉：圖元 y ∈ [%.2f, %.2f]，平面圖 z ∈ [%.2f, %.2f]",
                     entY.min() ?? 0, entY.max() ?? 0, zLo, zHi))
        _ = ys

        // ── 7. 大場景：必須在合理時間內跑完 ──
        //
        // **這一項存在的理由是一個真的爆炸。** 線段合併原本寫成「反覆全掃到收斂，
        // 每成功合併一次就整個重來」——那是 O(G³)：30 個 run 沒感覺、
        // 300 個要 14M 次比較、3000 個就是 135 億次。
        // 上面那些測試房間只有十幾個 run，完全測不出來；
        // 而 20×20m 的場景在 5cm 格下輕易有數千個 run，於是大場景直接卡死。
        //
        // 所以規模本身就是要測的東西 —— 小房間全過不代表演算法可用。
        seed = 9
        var big: [SIMD3<Float>] = []
        // 5×4 的房間鋪成 4×4 格（20×16m），中間留走道 —— 隔間牆會產生大量 run
        for gx in 0..<4 {
            for gz in 0..<4 {
                let off = SIMD2<Float>(Float(gx) * 5.5, Float(gz) * 4.5)
                big += room(w: W, d: D, h: H, step: 0.06).map {
                    SIMD3($0.x + off.x, $0.y, $0.z + off.y)
                }
            }
        }
        let t0 = Date()
        let r7 = PointCloudFloorPlan.extract(points: big)
        let secs = Date().timeIntervalSince(t0)
        print(String(format: "   大場景 20×16m：%d 點、%.2fs", big.count, secs))
        if let r7 { print("   \(r7.summary)") }
        check(r7 != nil && secs < 8,
              String(format: "大場景 %d 點在 %.2fs 內完成（上限 8s；先前的 O(G³) 合併會卡死）",
                     big.count, secs))

        print()
        print(fails == 0 ? "全部通過 — 點雲平面圖驗證完成" : "\(fails) 項失敗")
        exit(fails == 0 ? 0 : 1)
    }
}
