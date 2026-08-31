//
//  PointCloudFloorPlan.swift
//  fable — 由 LiDAR 點雲直接產生 2D 平面圖（不經過 RoomPlan）
//
//  ── 為什麼需要這條路 ────────────────────────────────────────────
//  RoomPlan 是**房間**掃描器：它要有地板、成面的牆、牆與天花板的交界才給得出
//  正確結構。實機在辦公室隔間、桌面、貨架前掃描時它會把螢幕邊、桌緣硬判成「牆」，
//  而且第一片判錯之後後續的面會跟著它對齊 —— 產出的是 2 面牆、樓高 0.8m 的東西。
//
//  點雲沒有這個前提。它不需要場景「是個房間」，只需要表面被掃到。
//  而牆在點雲裡有一個非常好認的特徵：**同一個水平格子裡跨越很大的垂直範圍**。
//  桌面、箱子、地板都只佔薄薄一層；只有牆（與書櫃正面）會從膝蓋一路長到頭頂。
//
//  ── 輸出接到哪 ──────────────────────────────────────────────────
//  產出 FloorPlanData，所以現有的整條製圖鏈（牆體填充、尺寸鏈、網格、比例尺、
//  SVG / JSON 匯出）完全不用改。另外提供 DXF 匯出給 AutoCAD 類工具。
//
//  ── 這條路做不到什麼（要講清楚）────────────────────────────────
//  · **不會有門窗。** 牆上的缺口在點雲裡跟「沒掃到的那一段」長得一模一樣，
//    把缺口當門會憑空生出不存在的門。缺口就留成缺口。
//  · **獨立的高櫃會被當成牆。** 它在幾何上確實是一片垂直的面，
//    只靠點雲分不出來 —— 那需要語意，而語意正是 RoomPlan 的強項。
//  兩條路是互補的：RoomPlan 給語意，點雲給幾何可靠度。
//
//  本檔不依賴 ARKit / RoomPlan，可在 macOS 編譯（見 tools/test_pointcloud_plan.swift）。
//

import Foundation
import simd

nonisolated enum PointCloudFloorPlan {

    // MARK: - 參數

    /// 水平佔用格的邊長（公尺）。5cm 對應牆厚的量級，
    /// 太細會讓一面牆散成兩三條平行線，太粗會把 60cm 的走道塞死。
    static let kCellM: Float = 0.05

    /// 判定為「牆」所需的垂直跨度（公尺）。
    ///
    /// **這是整個方法的核心判別式。** 一面牆會從掃描者的膝蓋一路被掃到頭頂以上，
    /// 同一個水平格子裡的點因此橫跨 1m 以上；而桌面、箱子、地板只佔薄薄一層。
    /// 1.0m 保守：辦公桌（0.75m）、矮櫃（1.0m 但通常掃不到頂）都在門檻之下。
    static let kWallMinVerticalM: Float = 1.0

    /// 一個格子至少要有幾個點才算被佔用 —— 擋掉飛點與單次觀測的雜訊
    static let kMinPointsPerCell = 4

    /// 判定牆時要離地板／天花板多遠才算數（公尺）。見 extract() 第 2 步：
    /// 不排除的話空曠地板的垂直跨度就等於樓高，整片地板都會被判成牆。
    static let kFloorCeilMarginM: Float = 0.15
    /// 計算樓地板面積時，貼近地板多少距離內的點算「地板」（公尺）
    static let kFloorBandM: Float = 0.25

    /// 牆段的最短長度（公尺）。短於此的多半是家具邊緣或雜訊
    static let kMinWallLengthM: Float = 0.6

    /// 允許牆段中間有幾格空隙仍視為同一段（公尺）。
    /// 掃描不可能不留洞（家具擋住、視角掃不到），2 格的容忍讓一面牆不會被切成碎片；
    /// 再大就會把真正的門口也接起來。
    static let kBridgeGapM: Float = 0.10

    /// Manhattan 角度搜尋的解析度（度）
    static let kAngleStepDeg: Float = 0.5

    // MARK: - 主流程

    struct Result: Sendable {
        var plan: FloorPlanData
        /// 診斷：讓呼叫端知道每一關剩下多少，卡住時才查得出是哪一步
        var summary: String
    }

    /// 由世界座標點雲抽出平面圖。points 為 ARKit 世界座標（+Y 向上）。
    static func extract(points: [SIMD3<Float>]) -> Result? {
        guard points.count >= 500 else { return nil }

        // ── 1. 樓地板與天花板高度 ──
        // 用百分位而不是「最小/最大值」：點雲一定有飛點，
        // 極值會被單一個雜訊點決定，而樓高是後面每一步的尺度基準。
        var ys = points.map(\.y)
        ys.sort()
        let floorY = ys[ys.count * 2 / 100]
        let ceilY = ys[min(ys.count - 1, ys.count * 98 / 100)]
        let roomHeight = ceilY - floorY
        guard roomHeight > 1.2 else { return nil }   // 連 1.2m 都不到，不是室內空間

        // ── 2. 逐格記錄垂直跨度，挑出牆格 ──
        let bandLo = floorY + kFloorCeilMarginM
        let bandHi = ceilY - kFloorCeilMarginM
        let bandH = bandHi - bandLo
        let pass1 = wallCells(points, bandLo: bandLo, bandHi: bandHi)
        // 門檻由實際觀測到的跨度分佈決定，不是從樓高推（見 spanThreshold）
        let needSpan = spanThreshold(pass1.cells, bandH: bandH)
        let cells1 = pass1.cells.filter { $0.span >= needSpan }
        guard cells1.count >= 20 else { return nil }

        // ── 3. Manhattan 主方向 ──
        let theta = dominantAngle(cells1)

        // ── 4. **把點雲轉正之後重新建格**，再抽線段 ──
        //
        // 不能拿第 2 步已經格點化的格心去旋轉再重新格點化。
        // 規則網格轉一個非 90° 的角度、再對齊到同樣間距的網格，必然同時產生
        // 碰撞（兩個來源格擠進同一格）與空洞（某些格沒人落進去）——
        // 一條連續的牆會變成有洞的虛線，然後被切成一堆低於最短長度的碎片。
        // 實測：房間轉 31° 時，兩面 4m 的牆碎成 1.55/1.55/1.15/0.85/0.75 五段。
        //
        // 從原始點重新建格就完全沒有這個問題 —— 點是稠密的，怎麼轉都還是稠密的。
        let ct = cos(-theta), st = sin(-theta)
        let rotated = points.map {
            SIMD3<Float>($0.x * ct - $0.z * st, $0.y, $0.x * st + $0.z * ct)
        }
        let pass2 = wallCells(rotated, bandLo: bandLo, bandHi: bandHi)
        let cells2 = pass2.cells.filter { $0.span >= needSpan }
        guard cells2.count >= 20 else { return nil }

        let segs = segments(from: cells2)
        let grid = pass2.occupied
        let floorCells = pass1.floorCells
        // 線段是在轉正的座標系抽出來的，要轉回世界座標
        let bt = cos(theta), bst = sin(theta)
        var walls: [FloorPlanSurface] = []
        for s in segs {
            let a = SIMD2<Float>(s.a.x * bt - s.a.y * bst, s.a.x * bst + s.a.y * bt)
            let b = SIMD2<Float>(s.b.x * bt - s.b.y * bst, s.b.x * bst + s.b.y * bt)
            let len = simd_distance(a, b)
            guard len >= kMinWallLengthM else { continue }
            // **牆高填實際觀測到的跨度，不是樓高。**
            // 填樓高的話 medianWallHeightM 永遠等於整個 Y 範圍，
            // 「牆只掃到多高」這個檢查就完全失效 —— 而那正是點雲版最常見的問題。
            walls.append(surface(a: a, b: b, thickness: s.thickness, height: s.spanM))
        }
        guard !walls.isEmpty else { return nil }

        // ── 5. 組出 FloorPlanData ──
        var lo = SIMD2<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD2<Float>(repeating: -.greatestFiniteMagnitude)
        for w in walls {
            for i in stride(from: 0, to: 4, by: 2) {
                let p = SIMD2<Float>(w.segment2D[i], w.segment2D[i + 1])
                lo = simd_min(lo, p); hi = simd_max(hi, p)
            }
        }
        // 樓地板面積：貼近地板高度的佔用格數 × 格面積。
        // 這比「牆的外接矩形」誠實得多 —— L 形或有隔間的空間差很多。

        var plan = FloorPlanData()
        plan.generator = "fable-pointcloud"
        plan.walls = walls
        plan.boundsM = [lo.x, lo.y, hi.x, hi.y]
        plan.boundingAreaM2 = (hi.x - lo.x) * (hi.y - lo.y)
        plan.floorAreaM2 = Float(floorCells) * kCellM * kCellM
        plan.roomCount = 1

        // 跨度分佈：卡在「牆太少」時，這一行直接分辨得出是門檻太嚴還是根本沒掃到牆。
        // 沒有它只能猜，而猜錯就是白改一輪。
        var sp = pass2.cells.map(\.span)
        sp.sort()
        func pct(_ q: Double) -> Float {
            sp.isEmpty ? 0 : sp[min(sp.count - 1, Int(Double(sp.count - 1) * q))]
        }
        let summary = String(
            format: "點雲平面圖: %d 點 → %d 佔用格 → %d 牆格（跨度 ≥%.2fm）→ %d 段牆；"
                    + "樓高 %.2fm、主方向 %.1f°、地板 %.1f m²\n"
                    + "  格跨度分佈 p50 %.2f / p75 %.2f / p90 %.2f / p99 %.2fm"
                    + "（門檻 %.2fm；牆掃到的高度中位數 %.2fm）",
            points.count, grid, cells2.count, needSpan, walls.count,
            roomHeight, theta * 180 / .pi, plan.floorAreaM2,
            pct(0.5), pct(0.75), pct(0.9), pct(0.99), needSpan, plan.medianWallHeightM)
        return Result(plan: plan, summary: summary)
    }

    /// 建水平佔用格並挑出「牆格」。
    ///
    /// **必須先排除地板與天花板帶。** 房間裡每一個水平格子都同時有地板點與天花板點，
    /// 垂直跨度就是整個樓高 —— 空曠的地板中央會跟牆一樣被判成「牆」。
    /// 只看中間那一段，空地就真的是空的，而牆仍然橫跨整段。
    struct WallCell { var at: SIMD2<Float>; var span: Float }

    private static func wallCells(_ points: [SIMD3<Float>],
                                  bandLo: Float, bandHi: Float)
        -> (cells: [WallCell], occupied: Int, floorCells: Int) {
        struct Column { var minY: Float; var maxY: Float; var count: Int }
        var grid: [Int64: Column] = [:]
        var floor = Set<Int64>()
        grid.reserveCapacity(points.count / 8)
        let floorTop = bandLo - kFloorCeilMarginM + kFloorBandM
        for p in points {
            let k = cellKey(p.x, p.z)
            if p.y <= floorTop { floor.insert(k) }
            guard p.y >= bandLo, p.y <= bandHi else { continue }
            if var c = grid[k] {
                c.minY = min(c.minY, p.y); c.maxY = max(c.maxY, p.y); c.count += 1
                grid[k] = c
            } else {
                grid[k] = Column(minY: p.y, maxY: p.y, count: 1)
            }
        }
        var out: [WallCell] = []
        for (k, c) in grid where c.count >= kMinPointsPerCell {
            out.append(WallCell(at: cellCenter(k), span: c.maxY - c.minY))
        }
        return (out, grid.count, floor.count)
    }

    /// 由**實際觀測到的跨度分佈**決定「多高才算牆」。
    ///
    /// 不能只從樓高推。實機那次：樓高 2.87m 推出門檻 1.28m，
    /// 但整份點雲的跨度 p90 才 1.16m、p99 也只有 1.88m ——
    /// 門檻直接訂在分佈的尾巴之外，4160 格只剩 235 格過關。
    ///
    /// 為什麼掃到的跨度上不去：**受限於掃描距離**。垂直視角約 60°，
    /// 單幀能看到牆面的垂直範圍 = 2·d·tan(30°) —— 距離 1m 只有 1.15m，
    /// 要 2.3m 得站到 2m 外。貼著牆掃的人永遠達不到由樓高推出來的門檻。
    ///
    /// 取分佈的 p90：不管掃描距離多近，永遠留下最「立」的那 10% 的格子。
    /// 再夾在 [kWallMinVerticalM, 帶高一半] 之間 ——
    /// 下限擋住「整個場景根本沒有垂直結構」時把櫃檯桌面當牆，
    /// 上限避免掃得很完整時門檻反而被拉高、把矮一點的真牆濾掉。
    static func spanThreshold(_ cells: [WallCell], bandH: Float) -> Float {
        guard !cells.isEmpty else { return kWallMinVerticalM }
        var sp = cells.map(\.span)
        sp.sort()
        let p90 = sp[min(sp.count - 1, Int(Double(sp.count - 1) * 0.9))]
        return min(max(p90, kWallMinVerticalM), max(kWallMinVerticalM, bandH * 0.5))
    }

    // MARK: - 主方向

    /// 找出讓牆格「最對齊座標軸」的旋轉角。
    ///
    /// 目標函數是投影直方圖的平方和：牆一旦與軸平行，它上面的格子會全部落進
    /// 同一個 bin，平方和因此暴增。這比「找最長的線再取其方向」穩健得多 ——
    /// 後者只看一面牆，一段誤判就把整張圖轉歪。
    ///
    /// 只需搜 0~90°：矩形格局在 90° 下自我重複。
    static func dominantAngle(_ cells: [WallCell]) -> Float {
        var best: Float = 0, bestScore: Float = -1
        var a: Float = 0
        while a < 90 {
            let r = a * .pi / 180
            let c = cos(-r), s = sin(-r)
            var hx: [Int: Int] = [:], hz: [Int: Int] = [:]
            for w in cells {
                let p = w.at
                let x = p.x * c - p.y * s, z = p.x * s + p.y * c
                hx[Int((x / kCellM).rounded()), default: 0] += 1
                hz[Int((z / kCellM).rounded()), default: 0] += 1
            }
            var score: Float = 0
            for v in hx.values { score += Float(v * v) }
            for v in hz.values { score += Float(v * v) }
            if score > bestScore { bestScore = score; best = r }
            a += kAngleStepDeg
        }
        return best
    }

    // MARK: - 線段抽取

    struct Seg { var a: SIMD2<Float>; var b: SIMD2<Float>; var thickness: Float
             /// 這段牆實際被掃到的垂直跨度中位數（公尺）——
             /// 不是樓高。它才回答得了「牆掃到多高」。
             var spanM: Float }

    /// 在已對齊的座標系裡抽出軸向線段。
    ///
    /// 每個格子先決定自己屬於哪個方向：比較它所在的橫向連續長度與縱向連續長度，
    /// 取較長者。**不這樣做的話牆角會同時被算進兩條線**，而且薄牆會沿著兩軸各長出
    /// 一條重疊的線段。這一步讓角落的格子歸給比較長的那面牆 ——
    /// 誤差最多一格（5cm），可以忽略。
    static func segments(from cells: [WallCell]) -> [Seg] {
        var occupied = Set<Int64>()
        var spanOf: [Int64: Float] = [:]
        for w in cells {
            let k = key(Int((w.at.x / kCellM).rounded()), Int((w.at.y / kCellM).rounded()))
            occupied.insert(k)
            spanOf[k] = max(spanOf[k] ?? 0, w.span)
        }
        func has(_ i: Int, _ j: Int) -> Bool { occupied.contains(key(i, j)) }

        /// 沿某方向數連續長度（含自己）
        func run(_ i: Int, _ j: Int, dx: Int, dy: Int) -> Int {
            var n = 1
            var a = i + dx, b = j + dy
            while has(a, b) { n += 1; a += dx; b += dy }
            a = i - dx; b = j - dy
            while has(a, b) { n += 1; a -= dx; b -= dy }
            return n
        }

        var horiz = Set<Int64>(), vert = Set<Int64>()
        for w in cells {
            let p = w.at
            let i = Int((p.x / kCellM).rounded()), j = Int((p.y / kCellM).rounded())
            if run(i, j, dx: 1, dy: 0) >= run(i, j, dx: 0, dy: 1) {
                horiz.insert(key(i, j))
            } else {
                vert.insert(key(i, j))
            }
        }

        var out: [Seg] = []
        out += runs(in: horiz, all: occupied, spanOf: spanOf, horizontal: true)
        out += runs(in: vert, all: occupied, spanOf: spanOf, horizontal: false)
        return out
    }

    /// 把同一列（或同一行）上的連續格子接成線段，**再把相鄰列的線段併起來**。
    ///
    /// 跨列合併不是修飾，是必要的：
    ///   · 牆本身有厚度，一面 10cm 的牆會佔兩列 —— 不合併就會輸出兩條平行線，
    ///     製圖層畫出來是兩道牆。
    ///   · 主方向不可能估到完全精確（解析度 0.5°），而且點雲有雜訊，
    ///     所以一面牆在旋轉後的格子上常常在兩列之間來回 ——
    ///     不合併的話每一列都只剩斷斷續續的短段，全都低於最短長度而被丟掉。
    ///     實測：房間轉 31° 時四面牆只認得出兩面。
    ///
    /// 合併後的厚度直接由「跨了幾列」得出，比另外去探鄰格更準也更簡單。
    private static func runs(in set: Set<Int64>, all: Set<Int64>,
                             spanOf: [Int64: Float], horizontal: Bool) -> [Seg] {
        // 依「列」分組：橫向牆同 j、縱向牆同 i
        var lines: [Int: [Int]] = [:]
        for k in set {
            let (i, j) = unkey(k)
            lines[horizontal ? j : i, default: []].append(horizontal ? i : j)
        }
        let bridge = Int((kBridgeGapM / kCellM).rounded())

        // 1) 逐列抽連續段
        struct Run { var line: Int; var lo: Int; var hi: Int }
        var raw: [Run] = []
        for (fixed, var moving) in lines {
            moving.sort()
            var start = moving[0], prev = moving[0]
            for v in moving.dropFirst() {
                if v - prev > bridge + 1 {
                    raw.append(Run(line: fixed, lo: start, hi: prev))
                    start = v
                }
                prev = v
            }
            raw.append(Run(line: fixed, lo: start, hi: prev))
        }

        // 2) 相鄰列且沿軸重疊者併為同一面牆（union-find 過頭，直接反覆掃到收斂即可 ——
        //    一面牆最多跨幾列，迭代次數是個位數）
        var groups = raw.map { [$0] }
        var merged = true
        while merged {
            merged = false
            outer: for i in 0..<groups.count {
                for j in (i + 1)..<groups.count {
                    if adjacent(groups[i], groups[j]) {
                        groups[i] += groups[j]
                        groups.remove(at: j)
                        merged = true
                        break outer
                    }
                }
            }
        }

        var out: [Seg] = []
        for g in groups {
            let lo = g.map(\.lo).min()!, hi = g.map(\.hi).max()!
            guard Float(hi - lo + 1) * kCellM >= kMinWallLengthM else { continue }
            let lineLo = g.map(\.line).min()!, lineHi = g.map(\.line).max()!
            // 中心線取各列的中點，厚度取跨的列數
            let center = Float(lineLo + lineHi) / 2 * kCellM
            let thickness = Float(lineHi - lineLo + 1) * kCellM
            let a: SIMD2<Float>, b: SIMD2<Float>
            if horizontal {
                a = SIMD2(Float(lo) * kCellM, center)
                b = SIMD2(Float(hi) * kCellM, center)
            } else {
                a = SIMD2(center, Float(lo) * kCellM)
                b = SIMD2(center, Float(hi) * kCellM)
            }
            // 這段牆的跨度取所屬格子的中位數 —— 平均會被單一格的極值拉走
            var spans: [Float] = []
            for r in g {
                for v in r.lo...r.hi {
                    let k = horizontal ? key(v, r.line) : key(r.line, v)
                    if let sp = spanOf[k] { spans.append(sp) }
                }
            }
            spans.sort()
            let span = spans.isEmpty ? 0 : spans[spans.count / 2]
            out.append(Seg(a: a, b: b, thickness: thickness, spanM: span))
        }
        return out

        /// 兩群是否該合併：任一對 run 相鄰列（差 ≤1）且沿軸區間重疊
        func adjacent(_ x: [Run], _ y: [Run]) -> Bool {
            for p in x {
                for q in y where abs(p.line - q.line) <= 1 {
                    if p.lo <= q.hi + bridge && q.lo <= p.hi + bridge { return true }
                }
            }
            return false
        }
    }

    // MARK: - 小工具

    @inline(__always)
    private static func key(_ i: Int, _ j: Int) -> Int64 {
        (Int64(i) &+ (1 << 20)) << 21 | (Int64(j) &+ (1 << 20))
    }
    @inline(__always)
    private static func unkey(_ k: Int64) -> (Int, Int) {
        let mask: Int64 = (1 << 21) - 1
        return (Int((k >> 21) - (1 << 20)), Int((k & mask) - (1 << 20)))
    }
    @inline(__always)
    private static func cellKey(_ x: Float, _ z: Float) -> Int64 {
        key(Int((x / kCellM).rounded(.down)), Int((z / kCellM).rounded(.down)))
    }
    @inline(__always)
    private static func cellCenter(_ k: Int64) -> SIMD2<Float> {
        let (i, j) = unkey(k)
        return SIMD2((Float(i) + 0.5) * kCellM, (Float(j) + 0.5) * kCellM)
    }

    /// 由兩端點組出 FloorPlanSurface。製圖層只吃 segment2D 與 dimensions[2]，
    /// 但 transform 仍照著填 —— JSON 匯出的下游（以及 RoomPlan 版本的資料）都預期它存在。
    private static func surface(a: SIMD2<Float>, b: SIMD2<Float>,
                                thickness: Float, height: Float) -> FloorPlanSurface {
        let d = b - a
        let len = simd_length(d)
        let ang = atan2(d.y, d.x)
        let mid = (a + b) / 2
        let c = cos(-ang), s = sin(-ang)
        // 世界 XZ 平面上的旋轉，寫成 row-major 4×4（+Y 為上）
        let m: [Float] = [c, 0, s, mid.x,
                          0, 1, 0, height / 2,
                          -s, 0, c, mid.y,
                          0, 0, 0, 1]
        return FloorPlanSurface(category: "wall", confidence: "high",
                                transform: m,
                                dimensions: [len, height, thickness],
                                segment2D: [a.x, a.y, b.x, b.y],
                                lengthM: len)
    }
}
