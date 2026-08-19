//
//  render_floorplan.swift
//  fable — 在 macOS 上直接跑製圖層，產出 SVG 以檢視畫法（不需要 iPhone）
//
//  製圖層（FloorPlanData.swift + FloorPlanDrawing.swift）刻意不依賴 RoomPlan，
//  所以可以在桌機上編譯執行 —— 改了畫法之後不必上機就能看到結果。
//
//      swiftc -o /tmp/render fable/Capture/FloorPlanData.swift \
//              fable/Capture/FloorPlanDrawing.swift tools/render_floorplan.swift
//      /tmp/render out.svg                       # 合成的兩房格局
//      /tmp/render out.svg scan/floorplan.json   # 實機匯出的資料
//
//  合成資料是一個 5.2×4.0m 的兩房格局（客廳＋臥室、隔間牆、門、窗、開口、家具），
//  用來檢查牆實體、門弧線、窗三線、房間名稱/面積、尺寸標註是否都正確。
//

import Foundation
import simd

// MARK: - 合成一個格局

/// 依中心、朝向、長度做出一面牆（與 FloorPlanData+RoomPlan.segment 相同的公式）
func wall(_ cx: Float, _ cz: Float, yaw: Float, length: Float,
          height: Float = 2.5, thickness: Float = 0.12,
          category: String = "wall") -> FloorPlanSurface {
    let c = cos(yaw), s = sin(yaw)
    let t: [Float] = [c, 0, s, cx,
                      0, 1, 0, height / 2,
                      -s, 0, c, cz,
                      0, 0, 0, 1]
    let h = length / 2
    return FloorPlanSurface(category: category, confidence: "high", transform: t,
                            dimensions: [length, height, thickness],
                            segment2D: [cx - c * h, cz - s * h, cx + c * h, cz + s * h],
                            lengthM: length)
}

func room(_ label: String?, _ corners: [(Float, Float)]) -> FloorPlanRoom {
    let pts = corners.map { SIMD2<Float>($0.0, $0.1) }
    var area: Float = 0, cx: Float = 0, cz: Float = 0
    for i in pts.indices {
        let q = pts[(i + 1) % pts.count]
        let cr = pts[i].x * q.y - q.x * pts[i].y
        area += cr
        cx += (pts[i].x + q.x) * cr
        cz += (pts[i].y + q.y) * cr
    }
    area /= 2
    let c = abs(area) > 1e-6 ? SIMD2(cx / (6 * area), cz / (6 * area))
                             : SIMD2<Float>(0, 0)
    return FloorPlanRoom(label: label, polygon2D: corners.flatMap { [$0.0, $0.1] },
                         areaM2: abs(area), labelAt: [c.x, c.y])
}

func synthetic() -> FloorPlanData {
    var d = FloorPlanData()
    d.roomCount = 2
    let W: Float = 5.2, D: Float = 4.0, half = Float.pi / 2
    d.walls = [
        wall(W / 2, 0, yaw: 0, length: W),                  // 上
        wall(W, D / 2, yaw: half, length: D),               // 右
        wall(W / 2, D, yaw: 0, length: W),                  // 下
        wall(0, D / 2, yaw: half, length: D),               // 左
        wall(3.1, D / 2, yaw: half, length: D, thickness: 0.09),   // 隔間牆
    ]
    // 門：外門在下牆、房門在隔間牆
    d.doors = [
        wall(1.2, D, yaw: 0, length: 0.9, thickness: 0.12, category: "door"),
        wall(3.1, 2.6, yaw: half, length: 0.8, thickness: 0.09, category: "door"),
    ]
    // 窗：上牆兩扇、右牆一扇
    d.windows = [
        wall(1.4, 0, yaw: 0, length: 1.3, thickness: 0.12, category: "window"),
        wall(4.2, 0, yaw: 0, length: 1.0, thickness: 0.12, category: "window"),
        wall(W, 1.2, yaw: half, length: 1.1, thickness: 0.12, category: "window"),
    ]
    // 開口（無門扇的通道）
    d.openings = [wall(3.1, 0.9, yaw: half, length: 0.95, thickness: 0.09,
                       category: "opening")]
    d.rooms = [
        room("livingRoom", [(0.06, 0.06), (3.05, 0.06), (3.05, 3.94), (0.06, 3.94)]),
        room("bedroom", [(3.15, 0.06), (5.14, 0.06), (5.14, 3.94), (3.15, 3.94)]),
    ]
    // 家具佔地
    func obj(_ cat: String, _ cx: Float, _ cz: Float, _ w: Float, _ h: Float) -> FloorPlanObject {
        FloorPlanObject(category: cat, transform: [], dimensions: [w, 0.8, h],
                        footprint2D: [cx - w/2, cz - h/2, cx + w/2, cz - h/2,
                                      cx + w/2, cz + h/2, cx - w/2, cz + h/2])
    }
    d.objects = [obj("sofa", 1.5, 2.9, 1.9, 0.8), obj("table", 1.5, 1.6, 1.1, 0.7),
                 obj("bed", 4.1, 1.4, 1.5, 2.0), obj("storage", 4.6, 3.4, 1.0, 0.6)]
    d.finalize()
    return d
}

/// 合成資料需要自己算 bounds / 面積合計（實機路徑由 computeBounds 做）
extension FloorPlanData {
    mutating func finalize() {
        var lo = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
        var hi = SIMD2<Float>(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
        for w in walls {
            for k in stride(from: 0, to: 4, by: 2) {
                let p = SIMD2<Float>(w.segment2D[k], w.segment2D[k + 1])
                lo = simd_min(lo, p); hi = simd_max(hi, p)
            }
        }
        boundsM = [lo.x, lo.y, hi.x, hi.y]
        boundingAreaM2 = (hi.x - lo.x) * (hi.y - lo.y)
        floorAreaM2 = rooms.reduce(0) { $0 + $1.areaM2 }
    }
}

/// 把整個格局繞原點轉一個角度 —— 用來驗證 planRotationRad 會不會把它轉回正。
/// 實機上這個角度就是使用者按下開始掃描時手機的朝向（ARKit 只固定重力軸）。
func rotateAll(_ d: FloorPlanData, byDeg deg: Float) -> FloorPlanData {
    var d = d
    let a = deg * .pi / 180, c = cos(a), s = sin(a)
    func R(_ x: Float, _ z: Float) -> (Float, Float) { (x * c - z * s, x * s + z * c) }
    func rf(_ flat: [Float]) -> [Float] {
        var out: [Float] = []
        for i in stride(from: 0, to: flat.count - 1, by: 2) {
            let (x, z) = R(flat[i], flat[i + 1]); out += [x, z]
        }
        return out
    }
    func rs(_ v: [FloorPlanSurface]) -> [FloorPlanSurface] {
        v.map { var w = $0; w.segment2D = rf($0.segment2D); return w }
    }
    d.walls = rs(d.walls); d.doors = rs(d.doors)
    d.windows = rs(d.windows); d.openings = rs(d.openings)
    d.objects = d.objects.map { var o = $0; o.footprint2D = rf($0.footprint2D); return o }
    d.rooms = d.rooms.map {
        var r = $0
        r.polygon2D = rf($0.polygon2D)
        let (x, z) = R($0.labelAt[0], $0.labelAt[1]); r.labelAt = [x, z]
        return r
    }
    d.finalize()
    return d
}

// MARK: - main

@main
struct Render {
    static func main() throws {
        let args = CommandLine.arguments
        let outPath = args.dropFirst(1).first { $0.hasSuffix(".svg") } ?? "floorplan.svg"
        // args[2] 可以是 json 路徑，也可以是「先轉幾度」的數字（測試用）
        var jsonPath: String?
        var rotateDeg: Float?
        var selfTest = false
        for a in args.dropFirst(1) {
            if a == "--selftest" { selfTest = true }
            else if let v = Float(a) { rotateDeg = v }
            else if a.hasSuffix(".json") { jsonPath = a }
        }
        var data: FloorPlanData
        if let jsonPath {
            let raw = try Data(contentsOf: URL(fileURLWithPath: jsonPath))
            data = try JSONDecoder().decode(FloorPlanData.self, from: raw)
            print("讀入 \(jsonPath)")
        } else {
            data = synthetic()
            print("使用合成的兩房格局")
        }
        if let deg = rotateDeg {
            data = rotateAll(data, byDeg: deg)
            print(String(format: "先轉 %.0f° → planRotationRad 回報 %.2f°（應為 %.0f°）",
                         deg, data.planRotationRad * 180 / .pi, -deg))
        }

        if selfTest {
            try runSelfTest()
            return
        }

        let svg = data.svg()
        try Data(svg.utf8).write(to: URL(fileURLWithPath: outPath))

        print(String(format: "%d 房 · %d 牆 · %d 門 · %d 窗 · %d 開口 · %d 家具",
                     data.roomCount, data.walls.count, data.doors.count,
                     data.windows.count, data.openings.count, data.objects.count))
        print(String(format: "地板面積 %.2f m²（外接 %.2f × %.2f m）",
                     data.floorAreaM2, data.sizeM.x, data.sizeM.y))
        print("圖元 \(data.drawing().count) 個 → \(outPath)（\(svg.count) bytes）")
        // 驗證圖面範圍真的把所有內容包住（含尺寸標註），不會被裁掉
        let b = data.drawingBoundsM
        var lo = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
        var hi = -lo
        for prim in data.drawing() {
            if case let .path(pts, _, _) = prim {
                for v in pts { lo = simd_min(lo, v); hi = simd_max(hi, v) }
            }
        }
        print(String(format: "圖面範圍 x[%.2f, %.2f] y[%.2f, %.2f]",
                     b[0], b[2], b[1], b[3]))
        print(String(format: "內容範圍 x[%.2f, %.2f] y[%.2f, %.2f]",
                     lo.x, hi.x, lo.y, hi.y))
        let ok = lo.x >= b[0] && hi.x <= b[2] && lo.y >= b[1] && hi.y <= b[3]
        print(ok ? "✅ 內容完全在圖面範圍內" : "❌ 內容超出圖面範圍 → 會被裁掉")

    }

    // MARK: - 回歸測試（swiftc 後執行 `render --selftest`）

    /// 製圖層的關鍵不變量。改了旋轉、命中測試或命名邏輯就重跑。
    static func runSelfTest() throws {
        var fails = 0
        func check(_ ok: Bool, _ what: String) {
            print(ok ? "✅ \(what)" : "❌ \(what)")
            if !ok { fails += 1 }
        }

        // 1. 轉正：任意角度的輸入都要被轉回水平
        for deg in [Float(0), 12, 45, -17, 31, 88] {
            let d = rotateAll(synthetic(), byDeg: deg)
            let got = d.planRotationRad * 180 / .pi
            // 牆向對 90° 取模，故 -deg 與 -deg±90 等價
            let err = abs((got + deg).truncatingRemainder(dividingBy: 90))
            check(min(err, 90 - err) < 0.5,
                  String(format: "轉正 %.0f° → 回報 %.2f°（模 90° 誤差 %.2f°）",
                         deg, got, min(err, 90 - err)))
        }

        // 2. 圖面範圍要包住所有內容（含尺寸標註），否則會被裁掉
        for deg in [Float(0), 37] {
            let d = rotateAll(synthetic(), byDeg: deg)
            let b = d.drawingBoundsM
            var lo = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
            var hi = -lo
            for p in d.drawing() {
                if case let .path(pts, _, _) = p {
                    for v in pts { lo = simd_min(lo, v); hi = simd_max(hi, v) }
                }
            }
            check(lo.x >= b[0] && hi.x <= b[2] && lo.y >= b[1] && hi.y <= b[3],
                  "內容完全在圖面範圍內（轉 \(Int(deg))°）")
        }

        // 3. 命中測試：形心命中該房間、屋外不命中
        let d = synthetic()
        let liv = SIMD2<Float>(d.rooms[0].labelAt[0], d.rooms[0].labelAt[1])
        let bed = SIMD2<Float>(d.rooms[1].labelAt[0], d.rooms[1].labelAt[1])
        check(d.roomIndex(containing: liv) == 0, "客廳形心命中 index 0")
        check(d.roomIndex(containing: bed) == 1, "臥室形心命中 index 1")
        check(d.roomIndex(containing: SIMD2(-3, -3)) == nil, "屋外不命中任何房間")

        // 4. 自訂名稱：優先於 RoomPlan 判定，且會進到 SVG；全空白要退回判定值
        var e = synthetic()
        check(e.rooms[0].displayName == "客廳", "預設用 RoomPlan 判定的名稱")
        e.rooms[0].customLabel = "主臥室"
        check(e.rooms[0].displayName == "主臥室", "自訂名稱優先")
        let svg = e.svg()
        check(svg.contains("主臥室") && !svg.contains(">客廳<"), "自訂名稱寫進 SVG")
        e.rooms[0].customLabel = "   "
        check(e.rooms[0].displayName == "客廳", "只有空白時退回判定值")

        // 5. 活動家具預設不畫；固定設備一律要畫
        var f = synthetic()
        f.objects = [
            FloorPlanObject(category: "chair", transform: [], dimensions: [0.5, 0.9, 0.5],
                            footprint2D: [1, 1, 1.5, 1, 1.5, 1.5, 1, 1.5]),
            FloorPlanObject(category: "sofa", transform: [], dimensions: [1.9, 0.8, 0.8],
                            footprint2D: [1, 2, 2.9, 2, 2.9, 2.8, 1, 2.8]),
            FloorPlanObject(category: "toilet", transform: [], dimensions: [0.4, 0.8, 0.6],
                            footprint2D: [4, 1, 4.4, 1, 4.4, 1.6, 4, 1.6]),
        ]
        func objectPaths(_ d: FloorPlanData, all: Bool) -> Int {
            d.drawing(showAllFurniture: all).filter {
                if case let .path(_, _, st) = $0 { return st.stroke == .object }
                return false
            }.count
        }
        check(objectPaths(f, all: false) == 1, "預設只畫固定設備（3 件中只有馬桶被畫）")
        check(objectPaths(f, all: true) == 3, "開啟後椅子/沙發也畫出來")

        // 6. 標頭顯示的外接尺寸要等於圖上尺寸標註的數字（外緣到外緣）
        let g = synthetic()
        let drawn = g.drawnSizeM
        let dimTexts = g.drawing().compactMap { p -> String? in
            if case let .text(t, _, _, c, _, _) = p, c == .dim { return t }
            return nil
        }
        let wantW = String(format: "%.2f m", drawn.x)
        let wantD = String(format: "%.2f m", drawn.y)
        check(dimTexts.contains(wantW) && dimTexts.contains(wantD),
              "標頭尺寸(\(wantW) × \(wantD)) 與圖上標註一致")

        print()
        print(fails == 0 ? "全部通過 — 製圖層不變量驗證完成" : "\(fails) 項失敗")
        if fails > 0 { exit(1) }
    }
}
