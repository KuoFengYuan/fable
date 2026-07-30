//
//  FloorPlanData.swift
//  fable — RoomPlan 的參數化結果 → 可用的平面圖（JSON + SVG）
//
//  RoomPlan 給的是 3D 參數化元素（牆/門/窗/開口，各帶 transform 與 dimensions）。
//  平面圖要的是 2D：把每面牆的中心線投影到水平面，得到線段集合。
//
//  投影的正確性依賴 worldAlignment = .gravity —— 世界 +Y 是反重力方向，
//  所以水平面就是 XZ 平面，牆的局部 X 軸（寬度方向）天然是水平的，直接取 (x, z) 即可。
//
//  俯視方向：觀察者在 +Y 往 -Y 看。取螢幕右 = 世界 +X，則螢幕上方必須是 -Z
//  （因為 (+X) × (-Z) = +Y，才是朝向觀察者的右手系）。SVG 的 y 軸向下增長，
//  故 svg_y = world_z 正好給出**不鏡像**的俯視圖 —— 這件事搞錯會讓整張平面圖左右翻轉。
//
//  座標保持 ARKit 原生（+Y up、公尺），與 points.ply / poses.jsonl 一致；
//  匯出 COLMAP 用的世界翻轉不套用在這裡（那是為了 3DGS 生態的慣例，平面圖不需要）。
//

import Foundation
import RoomPlan
import simd

// MARK: - 資料模型

nonisolated struct FloorPlanSurface: Codable, Sendable {
    var category: String
    var confidence: String
    /// row-major 4×4，ARKit 世界座標
    var transform: [Float]
    /// (寬, 高, 厚)，公尺
    var dimensions: [Float]
    /// 中心線投影到水平面的兩端點 [x0, z0, x1, z1]，公尺
    var segment2D: [Float]
    /// 水平長度（公尺）＝ dimensions.x，另外列出方便直接讀
    var lengthM: Float
}

nonisolated struct FloorPlanObject: Codable, Sendable {
    var category: String
    var transform: [Float]
    var dimensions: [Float]
    /// 水平佔地的四個角 [x0,z0, x1,z1, x2,z2, x3,z3]
    var footprint2D: [Float]
}

nonisolated struct FloorPlanData: Codable, Sendable {
    var generator = "fable-roomplan"
    var version = 1
    var coordinateSystem = "arkit_world_y_up_meters"
    var planeMapping = "svg_x = world_x, svg_y = world_z（俯視、非鏡像）"
    var roomCount = 1
    var walls: [FloorPlanSurface] = []
    var doors: [FloorPlanSurface] = []
    var windows: [FloorPlanSurface] = []
    var openings: [FloorPlanSurface] = []
    var objects: [FloorPlanObject] = []
    /// 牆體外接矩形（公尺）[minX, minZ, maxX, maxZ]，粗估整體尺寸用
    var boundsM: [Float] = []
    /// 牆體外接矩形面積（m²）。**不是**實際地板面積 ——
    /// 非矩形格局會高估，僅供快速對照，正式面積請用 USDZ 的地板幾何。
    var boundingAreaM2: Float = 0
}

// MARK: - 從 RoomPlan 轉換

extension FloorPlanData {

    init(room: CapturedRoom) {
        self.init()
        roomCount = 1
        walls = room.walls.map(Self.surface)
        doors = room.doors.map(Self.surface)
        windows = room.windows.map(Self.surface)
        openings = room.openings.map(Self.surface)
        objects = room.objects.map(Self.object)
        computeBounds()
    }

    init(structure: CapturedStructure) {
        self.init()
        roomCount = structure.rooms.count
        walls = structure.walls.map(Self.surface)
        doors = structure.doors.map(Self.surface)
        windows = structure.windows.map(Self.surface)
        openings = structure.openings.map(Self.surface)
        objects = structure.objects.map(Self.object)
        computeBounds()
    }

    private mutating func computeBounds() {
        var lo = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
        var hi = SIMD2<Float>(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
        for w in walls {
            for k in stride(from: 0, to: 4, by: 2) {
                let p = SIMD2<Float>(w.segment2D[k], w.segment2D[k + 1])
                lo = simd_min(lo, p); hi = simd_max(hi, p)
            }
        }
        guard hi.x > lo.x else { return }
        boundsM = [lo.x, lo.y, hi.x, hi.y]
        boundingAreaM2 = (hi.x - lo.x) * (hi.y - lo.y)
    }

    private static func surface(_ s: CapturedRoom.Surface) -> FloorPlanSurface {
        let (seg, len) = segment(transform: s.transform, width: s.dimensions.x)
        return FloorPlanSurface(category: name(s.category), confidence: name(s.confidence),
                                transform: rowMajor(s.transform),
                                dimensions: [s.dimensions.x, s.dimensions.y, s.dimensions.z],
                                segment2D: seg, lengthM: len)
    }

    private static func object(_ o: CapturedRoom.Object) -> FloorPlanObject {
        FloorPlanObject(category: name(o.category), transform: rowMajor(o.transform),
                        dimensions: [o.dimensions.x, o.dimensions.y, o.dimensions.z],
                        footprint2D: footprint(transform: o.transform,
                                               width: o.dimensions.x, depth: o.dimensions.z))
    }

    /// 中心 ± (局部 X 軸 × 半寬)，投影到 XZ 平面
    private static func segment(transform t: simd_float4x4, width: Float) -> ([Float], Float) {
        let c = t.columns.3
        let ax = SIMD2<Float>(t.columns.0.x, t.columns.0.z)
        let n = simd_length(ax)
        let dir = n > 1e-6 ? ax / n : SIMD2<Float>(1, 0)
        let h = width * 0.5
        let p0 = SIMD2<Float>(c.x, c.z) - dir * h
        let p1 = SIMD2<Float>(c.x, c.z) + dir * h
        return ([p0.x, p0.y, p1.x, p1.y], width)
    }

    /// 家具佔地：中心 ± 局部 X/Z 兩軸的半長，四個角
    private static func footprint(transform t: simd_float4x4,
                                  width: Float, depth: Float) -> [Float] {
        let c = SIMD2<Float>(t.columns.3.x, t.columns.3.z)
        let ex = SIMD2<Float>(t.columns.0.x, t.columns.0.z) * (width * 0.5)
        let ez = SIMD2<Float>(t.columns.2.x, t.columns.2.z) * (depth * 0.5)
        // 逐個具名，不要寫成一行陣列字面值 —— SIMD2 的運算子重載組合起來
        // 會讓 Swift 型別檢查器爆掉（"unable to type-check in reasonable time"）
        let a: SIMD2<Float> = c - ex - ez
        let b: SIMD2<Float> = c + ex - ez
        let d: SIMD2<Float> = c + ex + ez
        let e: SIMD2<Float> = c - ex + ez
        return [a.x, a.y, b.x, b.y, d.x, d.y, e.x, e.y]
    }

    private static func rowMajor(_ m: simd_float4x4) -> [Float] {
        (0..<4).flatMap { r in (0..<4).map { c in m[c][r] } }
    }

    private static func name(_ c: CapturedRoom.Surface.Category) -> String {
        switch c {
        case .wall:            "wall"
        case .door:            "door"
        case .window:          "window"
        case .opening:         "opening"
        case .floor:           "floor"
        @unknown default:      "unknown"
        }
    }

    private static func name(_ c: CapturedRoom.Object.Category) -> String {
        String(describing: c)
    }

    private static func name(_ c: CapturedRoom.Confidence) -> String {
        switch c {
        case .high:   "high"
        case .medium: "medium"
        case .low:    "low"
        @unknown default: "unknown"
        }
    }
}

// MARK: - SVG（直接可看的平面圖）

extension FloorPlanData {

    // 色碼一律走常數再插值。直接把 "#111111" 寫進 #"..."# 原始字串會壞掉 ——
    // 色碼前面那個引號加上 # 剛好構成 "# ，會提前終止原始字串的結束分隔符。
    private enum Ink {
        static let wall = "#111111", door = "#f59e0b", window = "#3b82f6"
        static let opening = "#888888", grid = "#e8e8e8", paper = "#ffffff"
        static let objFill = "#f2f2f2", objLine = "#cccccc"
        static let label = "#444444", muted = "#666666"
        static let font = "-apple-system,Helvetica,sans-serif"
    }

    /// 產生俯視平面圖 SVG。1 公尺 = pxPerMeter 像素，含公尺網格與比例尺。
    /// 牆粗實線、門橘、窗藍、開口虛線、家具淡灰框 —— 一般平面圖的畫法。
    func svg(pxPerMeter: Float = 100, margin: Float = 40) -> String {
        guard boundsM.count == 4 else {
            return #"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60"><text x="8" y="34">無牆面資料</text></svg>"#
        }
        let lo = SIMD2<Float>(boundsM[0], boundsM[1])
        let hi = SIMD2<Float>(boundsM[2], boundsM[3])
        let w = (hi.x - lo.x) * pxPerMeter + margin * 2
        let h = (hi.y - lo.y) * pxPerMeter + margin * 2
        // world (x,z) → svg (x,y)：y 不翻轉，故為非鏡像俯視（見檔頭說明）
        func X(_ v: Float) -> Float { (v - lo.x) * pxPerMeter + margin }
        func Y(_ v: Float) -> Float { (v - lo.y) * pxPerMeter + margin }
        func f(_ v: Float) -> String { String(format: "%.1f", v) }

        var s = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"\(f(w))\" height=\"\(f(h))\""
        s += " viewBox=\"0 0 \(f(w)) \(f(h))\">\n"
        s += "<rect width=\"100%\" height=\"100%\" fill=\"\(Ink.paper)\"/>\n"

        // 1m 網格
        s += "<g stroke=\"\(Ink.grid)\" stroke-width=\"1\">"
        var gx = lo.x.rounded(.up)
        while gx <= hi.x {
            s += line(X(gx), margin, X(gx), h - margin, f)
            gx += 1
        }
        var gz = lo.y.rounded(.up)
        while gz <= hi.y {
            s += line(margin, Y(gz), w - margin, Y(gz), f)
            gz += 1
        }
        s += "</g>\n"

        // 家具佔地（先畫，壓在牆下面）
        if !objects.isEmpty {
            s += "<g fill=\"\(Ink.objFill)\" stroke=\"\(Ink.objLine)\" stroke-width=\"1\">"
            for o in objects where o.footprint2D.count == 8 {
                let pts = stride(from: 0, to: 8, by: 2)
                    .map { "\(f(X(o.footprint2D[$0]))),\(f(Y(o.footprint2D[$0 + 1])))" }
                    .joined(separator: " ")
                s += "<polygon points=\"\(pts)\"/>"
            }
            s += "</g>\n"
        }

        // 由淡到重疊上去：開口（虛線）→ 窗 → 門 → 牆
        s += group(openings, stroke: Ink.opening, width: 3, dash: "6,4", X: X, Y: Y, f: f)
        s += group(windows, stroke: Ink.window, width: 5, dash: nil, X: X, Y: Y, f: f)
        s += group(doors, stroke: Ink.door, width: 5, dash: nil, X: X, Y: Y, f: f)
        s += group(walls, stroke: Ink.wall, width: 7, dash: nil, X: X, Y: Y, f: f)

        // 牆長標註（只標 ≥1m 的，避免小段擠成一團）
        s += "<g font-family=\"\(Ink.font)\" font-size=\"11\" fill=\"\(Ink.label)\">"
        for wl in walls where wl.lengthM >= 1.0 && wl.segment2D.count == 4 {
            let mx = X((wl.segment2D[0] + wl.segment2D[2]) / 2)
            let my = Y((wl.segment2D[1] + wl.segment2D[3]) / 2)
            s += "<text x=\"\(f(mx))\" y=\"\(f(my - 4))\" text-anchor=\"middle\">\(f(wl.lengthM))m</text>"
        }
        s += "</g>\n"

        // 比例尺 + 摘要
        let by = h - margin * 0.45
        s += "<g stroke=\"\(Ink.wall)\" stroke-width=\"2\">"
        s += line(margin, by, margin + pxPerMeter, by, f) + "</g>"
        s += "<text x=\"\(f(margin))\" y=\"\(f(by - 6))\" font-size=\"11\""
        s += " font-family=\"\(Ink.font)\" fill=\"\(Ink.wall)\">1 m</text>"
        let summary = "\(roomCount) 房 · \(walls.count) 牆 · \(doors.count) 門 · \(windows.count) 窗"
        s += "<text x=\"\(f(w - margin))\" y=\"\(f(by - 6))\" font-size=\"11\" text-anchor=\"end\""
        s += " font-family=\"\(Ink.font)\" fill=\"\(Ink.muted)\">\(summary)</text>"
        s += "\n</svg>\n"
        return s
    }

    private func line(_ x1: Float, _ y1: Float, _ x2: Float, _ y2: Float,
                      _ f: (Float) -> String) -> String {
        "<line x1=\"\(f(x1))\" y1=\"\(f(y1))\" x2=\"\(f(x2))\" y2=\"\(f(y2))\"/>"
    }

    private func group(_ items: [FloorPlanSurface], stroke: String, width: Float,
                       dash: String?, X: (Float) -> Float, Y: (Float) -> Float,
                       f: (Float) -> String) -> String {
        guard !items.isEmpty else { return "" }
        let d = dash.map { " stroke-dasharray=\"\($0)\"" } ?? ""
        var s = "<g stroke=\"\(stroke)\" stroke-width=\"\(f(width))\" stroke-linecap=\"round\"\(d)>"
        for i in items where i.segment2D.count == 4 {
            s += line(X(i.segment2D[0]), Y(i.segment2D[1]),
                      X(i.segment2D[2]), Y(i.segment2D[3]), f)
        }
        return s + "</g>\n"
    }
}
