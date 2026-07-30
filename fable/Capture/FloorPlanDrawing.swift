//
//  FloorPlanDrawing.swift
//  fable — 平面圖的「製圖層」：把 RoomPlan 參數轉成建築平面圖的畫法
//
//  為什麼要有中間層：SVG 匯出與螢幕預覽先前各畫一遍，兩份繪圖邏輯必然漂走
//  （已經發生過：SVG 有牆長標註、畫面沒有）。改成單一來源 ——
//  drawing() 產出世界座標的圖元清單，兩個後端只負責座標轉換與上色。
//  於是「螢幕上看到的」與「匯出檔裡的」在結構上不可能不同。
//
//  畫法沿用建築平面圖慣例，而不是把線段畫粗：
//    牆    → 依實際厚度畫成實體（poché），不是中心線
//    門    → 牆上開缺口 ＋ 90° 開啟弧線 ＋ 門扇
//    窗    → 牆上開缺口 ＋ 三道平行細線
//    開口  → 只開缺口，不畫符號
//    房間  → 地板多邊形淡色填充 ＋ 名稱 ＋ 面積
//  「開缺口」是靠繪製順序做的：先畫牆實體，再用紙色實心蓋掉門窗位置。
//

import Foundation
import simd

/// 語意色，不綁定具體色值 —— 由各後端各自對應（SVG 用十六進位、SwiftUI 用 Color）
nonisolated enum PlanColor: String, Sendable {
    case ink, paper, wall, door, window, opening, roomFill, roomText, dim, object
}

nonisolated enum PlanAlign: String, Sendable { case center, left, right }

nonisolated struct PlanStyle: Sendable {
    var stroke: PlanColor?
    var fill: PlanColor?
    /// 線寬以「像素」計：縮放時線寬不變，才符合製圖慣例（線重代表語意，不代表尺寸）
    var widthPx: Float = 1
    var dash: [Float]?
}

/// 世界座標（公尺，XZ 平面）的繪圖圖元。弧線一律先離散成折線，
/// 讓後端只需要實作「填多邊形／畫折線／畫字」三件事。
nonisolated enum PlanPrimitive: Sendable {
    case path(points: [SIMD2<Float>], closed: Bool, style: PlanStyle)
    case text(String, at: SIMD2<Float>, sizePx: Float, color: PlanColor,
              align: PlanAlign, bold: Bool)
}

extension FloorPlanData {

    /// 牆厚缺值時的替代值（公尺）。RoomPlan 偶爾給 0，畫成 0 厚會整面牆消失
    private static let kFallbackWallThickness: Float = 0.10

    /// 產生整張平面圖的圖元清單，順序即繪製順序（後畫的蓋前面的）。
    ///
    /// 內容先在 ARKit 世界座標建好，再**整批**旋轉到水平（見 planRotationRad）——
    /// 旋轉放在最後一步而不是散在各個 builder 裡，是為了不可能漏掉某一種圖元。
    /// 尺寸標註鏈在旋轉之後才依「旋轉後的範圍」產生，否則標的會是歪的外接框。
    func drawing() -> [PlanPrimitive] {
        let a = planRotationRad
        var content = content()
        if abs(a) > 0.005 {
            let c = cos(a), s = sin(a)
            content = content.map { Self.rotate($0, cos: c, sin: s) }
        }
        return content + dimensionChain(bounds: Self.bounds(of: content))
    }

    /// 旋轉後的圖面範圍（含尺寸標註留白）。兩個後端都靠它算縮放。
    var drawingBoundsM: [Float] {
        guard !walls.isEmpty else { return [] }
        let b = Self.bounds(of: {
            let a = planRotationRad
            let c = cos(a), s = sin(a)
            return content().map { Self.rotate($0, cos: c, sin: s) }
        }())
        guard b.count == 4 else { return [] }
        // 左側留白較大：垂直尺寸的文字是靠右對齊、往左延伸，留太少會被裁掉（實機出現過）
        return [b[0] - 1.7, b[1] - 1.0, b[2] + 1.0, b[3] + 1.0]
    }

    private static func rotate(_ p: PlanPrimitive, cos c: Float, sin s: Float) -> PlanPrimitive {
        func R(_ v: SIMD2<Float>) -> SIMD2<Float> {
            SIMD2(v.x * c - v.y * s, v.x * s + v.y * c)
        }
        switch p {
        case let .path(pts, closed, style):
            return .path(points: pts.map(R), closed: closed, style: style)
        case let .text(t, at, size, color, align, bold):
            return .text(t, at: R(at), sizePx: size, color: color, align: align, bold: bold)
        }
    }

    private static func bounds(of prims: [PlanPrimitive]) -> [Float] {
        var lo = SIMD2<Float>(.greatestFiniteMagnitude, .greatestFiniteMagnitude)
        var hi = SIMD2<Float>(-.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
        for p in prims {
            // 只用實體幾何定範圍：文字位置不算（它不是圖面內容），
            // 也避免門的開啟弧線把範圍撐大
            guard case let .path(pts, _, style) = p, style.fill != nil || style.stroke == .wall
            else { continue }
            for v in pts { lo = simd_min(lo, v); hi = simd_max(hi, v) }
        }
        guard hi.x > lo.x else { return [] }
        return [lo.x, lo.y, hi.x, hi.y]
    }

    /// 圖面內容（未旋轉）
    private func content() -> [PlanPrimitive] {
        var out: [PlanPrimitive] = []

        // 1. 房間地板：淡色填充 ＋ 細邊界
        for r in rooms {
            let poly = points(r.polygon2D)
            guard poly.count >= 3 else { continue }
            out.append(.path(points: poly, closed: true,
                             style: PlanStyle(stroke: nil, fill: .roomFill)))
        }

        // 2. 家具佔地：只描邊，淡 —— 家具不該搶走格局的視覺重量
        for o in objects {
            let poly = points(o.footprint2D)
            guard poly.count >= 3 else { continue }
            out.append(.path(points: poly, closed: true,
                             style: PlanStyle(stroke: .object, fill: nil, widthPx: 1)))
        }

        // 3. 牆實體（poché）：依實際厚度畫矩形，這是「像平面圖」最關鍵的一步
        for w in walls {
            guard let quad = Self.slab(w) else { continue }
            out.append(.path(points: quad, closed: true,
                             style: PlanStyle(stroke: .wall, fill: .wall, widthPx: 1)))
        }

        // 4. 在牆上開缺口：用紙色實心蓋掉門/窗/開口的位置。
        //    開孔寬度用「所在牆的厚度」而非門窗自身的 thickness —— 後者常較小，
        //    用它會蓋不乾淨、沿牆面留下白邊；而門窗符號也用同一寬度，
        //    符號的外緣才會剛好落在牆的兩個面上（這是製圖上該有的樣子）。
        let cut = cutThickness
        for s in doors + windows + openings {
            guard let quad = Self.slab(s, thickness: cut) else { continue }
            out.append(.path(points: quad, closed: true,
                             style: PlanStyle(stroke: nil, fill: .paper)))
        }

        // 5. 門窗符號
        for d in doors { out += doorSymbol(d) }
        for w in windows { out += Self.windowSymbol(w, thickness: cut) }
        for o in openings {
            // 開口只畫兩側牆端的封口線，不畫符號
            guard let quad = Self.slab(o, thickness: cut) else { continue }
            out.append(.path(points: [quad[0], quad[3]], closed: false,
                             style: PlanStyle(stroke: .opening, widthPx: 1.5, dash: [4, 3])))
            out.append(.path(points: [quad[1], quad[2]], closed: false,
                             style: PlanStyle(stroke: .opening, widthPx: 1.5, dash: [4, 3])))
        }

        // 6. 房間名稱與面積
        for r in rooms where r.areaM2 > 0.5 {
            let at = SIMD2<Float>(r.labelAt.first ?? 0, r.labelAt.count > 1 ? r.labelAt[1] : 0)
            let name = r.label.map(Self.roomName) ?? "房間"
            out.append(.text(name, at: at, sizePx: 13, color: .roomText,
                             align: .center, bold: true))
            out.append(.text(String(format: "%.1f m²", r.areaM2),
                             at: at + SIMD2(0, 0.22), sizePx: 11, color: .roomText,
                             align: .center, bold: false))
        }

        return out
    }

    // MARK: - 幾何

    private func points(_ flat: [Float]) -> [SIMD2<Float>] {
        stride(from: 0, to: flat.count - 1, by: 2).map { SIMD2(flat[$0], flat[$0 + 1]) }
    }

    /// 門窗開孔（與其符號）採用的厚度：取牆厚中位數，比門窗自身的 thickness 可靠。
    /// ×1.02 是為了確保完全蓋掉牆、不留一條白邊。
    private var cutThickness: Float {
        let t = medianWallThicknessM
        return (t > 0.02 ? t : Self.kFallbackWallThickness) * 1.02
    }

    /// 把一個表面畫成帶厚度的矩形：中心 ± 半長沿軸向、± 半厚沿法向。
    /// 回傳 4 個角（沿軸向的兩端各兩點），順序為 [起-左, 起-右, 終-右, 終-左]。
    /// thickness 傳 nil 時用該表面自己的 dimensions[2]。
    private static func slab(_ s: FloorPlanSurface,
                            thickness: Float? = nil) -> [SIMD2<Float>]? {
        guard s.segment2D.count == 4 else { return nil }
        let p0 = SIMD2<Float>(s.segment2D[0], s.segment2D[1])
        let p1 = SIMD2<Float>(s.segment2D[2], s.segment2D[3])
        let d = p1 - p0
        let len = simd_length(d)
        guard len > 1e-5 else { return nil }
        let axis = d / len
        let normal = SIMD2<Float>(-axis.y, axis.x)
        var t = thickness ?? (s.dimensions.count > 2 ? s.dimensions[2] : 0)
        if t < 0.02 { t = kFallbackWallThickness }
        let h = t / 2
        return [p0 + normal * h, p0 - normal * h, p1 - normal * h, p1 + normal * h]
    }

    /// 門：90° 開啟弧線 ＋ 門扇。
    ///
    /// 開向要往**室內**掃，不是隨便挑一側。RoomPlan 不提供開啟方向，
    /// 所以用房間多邊形反推：往牆的兩側各試探一個點，落在房間內的那一側就是室內。
    /// 沒這一步的話外門會往屋外開 —— 建築圖上一眼就看得出是錯的，
    /// 而且弧線會伸到尺寸標註線外面去。
    private func doorSymbol(_ s: FloorPlanSurface) -> [PlanPrimitive] {
        guard s.segment2D.count == 4 else { return [] }
        let p0 = SIMD2<Float>(s.segment2D[0], s.segment2D[1])
        let p1 = SIMD2<Float>(s.segment2D[2], s.segment2D[3])
        let d = p1 - p0
        let r = simd_length(d)
        guard r > 0.05 else { return [] }
        let axis = d / r
        var normal = SIMD2<Float>(-axis.y, axis.x)

        // 探測點取牆中點往兩側各偏 0.35m（大於半個牆厚、小於任何合理房間的半寬）
        let mid = (p0 + p1) / 2
        let probe: Float = 0.35
        let insidePos = roomsContain(mid + normal * probe)
        let insideNeg = roomsContain(mid - normal * probe)
        if insideNeg && !insidePos { normal = -normal }

        let style = PlanStyle(stroke: .door, widthPx: 1.5)
        // 弧線：從 p1（全開）掃到 p0 + normal*r（全閉），16 段折線足夠平滑
        var arc: [SIMD2<Float>] = []
        for i in 0...16 {
            let a = Float(i) / 16 * (.pi / 2)
            arc.append(p0 + axis * (r * cos(a)) + normal * (r * sin(a)))
        }
        return [
            .path(points: arc, closed: false, style: style),
            // 門扇：從鉸點指向全閉位置
            .path(points: [p0, p0 + normal * r], closed: false,
                  style: PlanStyle(stroke: .door, widthPx: 3)),
        ]
    }

    /// 該點是否落在任一房間多邊形內（ray casting）
    private func roomsContain(_ pt: SIMD2<Float>) -> Bool {
        for r in rooms {
            let poly = points(r.polygon2D)
            guard poly.count >= 3 else { continue }
            var inside = false
            var j = poly.count - 1
            for i in poly.indices {
                let a = poly[i], b = poly[j]
                if (a.y > pt.y) != (b.y > pt.y),
                   pt.x < (b.x - a.x) * (pt.y - a.y) / (b.y - a.y) + a.x {
                    inside.toggle()
                }
                j = i
            }
            if inside { return true }
        }
        return false
    }

    /// 窗：沿窗寬三道平行線（外框兩條 ＋ 中線一條），建築圖慣例畫法。
    /// 寬度用牆厚，外緣才會剛好貼在牆的兩個面上。
    private static func windowSymbol(_ s: FloorPlanSurface,
                                     thickness: Float) -> [PlanPrimitive] {
        guard let quad = slab(s, thickness: thickness) else { return [] }
        let style = PlanStyle(stroke: .window, widthPx: 1.5)
        let midA = (quad[0] + quad[1]) / 2
        let midB = (quad[3] + quad[2]) / 2
        return [
            .path(points: [quad[0], quad[3]], closed: false, style: style),
            .path(points: [quad[1], quad[2]], closed: false, style: style),
            .path(points: [midA, midB], closed: false, style: style),
        ]
    }

    /// 外圍總尺寸：下緣標寬、左緣標深，各含端點短刻度線。
    /// bounds 必須是**旋轉後**的範圍，否則標到的是歪掉的外接框、數字沒有意義。
    private func dimensionChain(bounds: [Float]) -> [PlanPrimitive] {
        guard bounds.count == 4 else { return [] }
        let lo = SIMD2<Float>(bounds[0], bounds[1])
        let hi = SIMD2<Float>(bounds[2], bounds[3])
        let gap: Float = 0.45           // 標註線離牆的距離（公尺）
        let tick: Float = 0.10
        let style = PlanStyle(stroke: .dim, widthPx: 1)
        var out: [PlanPrimitive] = []

        // 下緣（寬）
        let yb = hi.y + gap
        out.append(.path(points: [SIMD2(lo.x, yb), SIMD2(hi.x, yb)], closed: false, style: style))
        for x in [lo.x, hi.x] {
            out.append(.path(points: [SIMD2(x, yb - tick), SIMD2(x, yb + tick)],
                             closed: false, style: style))
        }
        out.append(.text(String(format: "%.2f m", hi.x - lo.x),
                         at: SIMD2((lo.x + hi.x) / 2, yb + 0.30),
                         sizePx: 11, color: .dim, align: .center, bold: false))

        // 左緣（深）
        let xl = lo.x - gap
        out.append(.path(points: [SIMD2(xl, lo.y), SIMD2(xl, hi.y)], closed: false, style: style))
        for y in [lo.y, hi.y] {
            out.append(.path(points: [SIMD2(xl - tick, y), SIMD2(xl + tick, y)],
                             closed: false, style: style))
        }
        out.append(.text(String(format: "%.2f m", hi.y - lo.y),
                         at: SIMD2(xl - 0.12, (lo.y + hi.y) / 2),
                         sizePx: 11, color: .dim, align: .right, bold: false))
        return out
    }

    /// RoomPlan 的 section label（camelCase 英文）→ 中文
    static func roomName(_ raw: String) -> String {
        switch raw {
        case "bathroom":    "衛浴"
        case "bedroom":     "臥室"
        case "diningRoom":  "餐廳"
        case "kitchen":     "廚房"
        case "livingRoom":  "客廳"
        case "office":      "書房"
        case "storage":     "儲藏"
        case "laundryRoom": "洗衣間"
        case "hallway":     "走廊"
        case "closet":      "衣櫃間"
        // RoomPlan 判不出用途時會回 unidentified；顯示原字串只會讓人困惑，
        // 而「這是哪種房間」對平面圖來說不是必要資訊，退回通稱即可
        case "unidentified": "房間"
        default:            raw.isEmpty ? "房間" : raw
        }
    }
}
