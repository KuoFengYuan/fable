//
//  FloorPlanDXF.swift
//  fable — 平面圖的 DXF 匯出（AutoCAD / BricsCAD / QCAD / Rhino 等直接開）
//
//  為什麼要 DXF 而不是只有 SVG：SVG 是「一張圖」，DXF 是「一組帶座標與圖層的圖元」。
//  丟進 CAD 之後可以量測、對齊、接著畫 —— 那才是「AutoCAD 那樣」的意思。
//
//  格式選擇：**ASCII DXF R12**。它是最保守的版本，幾乎所有 CAD 都吃，
//  而且不需要 HANDSEED / 物件字典那些後期版本才有的必填區段 ——
//  平面圖只有線與文字，用不到新版的任何東西，選新版只會多出可能寫錯的欄位。
//
//  單位：公尺。CAD 端若預設是 mm，匯入時設比例 1000 即可（DXF 本身不帶單位語意，
//  $INSUNITS 只是提示，所以標頭有寫但不能只靠它）。
//

import Foundation
import simd

nonisolated enum FloorPlanDXF {

    /// 圖層。分層是 DXF 相對 SVG 的主要價值 —— CAD 端可以單獨關掉家具、只留牆線。
    private enum Layer: String {
        case wall = "WALL", opening = "OPENING", object = "FURNITURE"
        case dimension = "DIM", text = "TEXT"
        /// AutoCAD 標準顏色索引（1 紅 2 黃 3 綠 4 青 5 藍 6 洋紅 7 白/黑 8 灰）
        var color: Int {
            switch self {
            case .wall: 7
            case .opening: 4
            case .object: 8
            case .dimension: 3
            case .text: 2
            }
        }
    }

    /// 產生 DXF 文字。座標為公尺，X = 世界 X、Y = **負的**世界 Z。
    ///
    /// 為什麼 Y 取負：ARKit 是右手系、俯視時 +Z 指向畫面下方，而 CAD 的 Y 慣例朝上。
    /// 直接照抄會得到一張上下鏡射的平面圖 —— 那在 CAD 裡不會報錯，
    /// 只會讓所有人看到一個左右顛倒的格局，而且很晚才會被發現。
    static func make(_ plan: FloorPlanData) -> String {
        var out = ""
        out += header(plan)
        out += tables()
        out += "0\nSECTION\n2\nENTITIES\n"

        for w in plan.walls { out += wallSlab(w) }
        for s in plan.doors + plan.windows + plan.openings {
            out += centerLine(s, layer: .opening)
        }
        for o in plan.objects { out += footprint(o) }
        for r in plan.rooms {
            guard r.labelAt.count == 2 else { continue }
            out += text(r.displayName, at: SIMD2(r.labelAt[0], r.labelAt[1]),
                        height: 0.22, layer: .text)
            out += text(String(format: "%.1f m²", r.areaM2),
                        at: SIMD2(r.labelAt[0], r.labelAt[1] + 0.30),
                        height: 0.16, layer: .text)
        }
        out += "0\nENDSEC\n0\nEOF\n"
        return out
    }

    // MARK: - 區段

    private static func header(_ plan: FloorPlanData) -> String {
        // 圖面範圍留 1m 邊界，CAD 開檔時 zoom extents 才不會貼邊
        let b = plan.boundsM.count == 4 ? plan.boundsM : [0, 0, 1, 1]
        return """
        0
        SECTION
        2
        HEADER
        9
        $ACADVER
        1
        AC1009
        9
        $INSUNITS
        70
        6
        9
        $EXTMIN
        10
        \(f(b[0] - 1))
        20
        \(f(-b[3] - 1))
        9
        $EXTMAX
        10
        \(f(b[2] + 1))
        20
        \(f(-b[1] + 1))
        0
        ENDSEC

        """
    }

    private static func tables() -> String {
        let layers: [Layer] = [.wall, .opening, .object, .dimension, .text]
        var s = "0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n\(layers.count)\n"
        for l in layers {
            s += "0\nLAYER\n2\n\(l.rawValue)\n70\n0\n62\n\(l.color)\n6\nCONTINUOUS\n"
        }
        s += "0\nENDTAB\n0\nENDSEC\n"
        return s
    }

    // MARK: - 圖元

    /// 牆畫成**閉合的四邊形**（實際厚度），不是單一條中心線。
    /// CAD 端要的是可以量、可以偏移、可以填充的牆體輪廓；
    /// 中心線只在示意圖有用，一放進施工圖就得重畫。
    private static func wallSlab(_ w: FloorPlanSurface) -> String {
        guard w.segment2D.count == 4 else { return "" }
        let a = SIMD2<Float>(w.segment2D[0], w.segment2D[1])
        let b = SIMD2<Float>(w.segment2D[2], w.segment2D[3])
        let d = b - a
        let len = simd_length(d)
        guard len > 1e-4 else { return "" }
        let axis = d / len
        let n = SIMD2<Float>(-axis.y, axis.x)
        var t = w.dimensions.count > 2 ? w.dimensions[2] : 0.12
        if t < 0.02 { t = 0.12 }
        let h = t / 2
        return polyline([a + n * h, b + n * h, b - n * h, a - n * h],
                        closed: true, layer: .wall)
    }

    private static func centerLine(_ s: FloorPlanSurface, layer: Layer) -> String {
        guard s.segment2D.count == 4 else { return "" }
        return polyline([SIMD2(s.segment2D[0], s.segment2D[1]),
                         SIMD2(s.segment2D[2], s.segment2D[3])],
                        closed: false, layer: layer)
    }

    private static func footprint(_ o: FloorPlanObject) -> String {
        let p = o.footprint2D
        guard p.count >= 6, p.count % 2 == 0 else { return "" }
        var pts: [SIMD2<Float>] = []
        for i in stride(from: 0, to: p.count, by: 2) { pts.append(SIMD2(p[i], p[i + 1])) }
        return polyline(pts, closed: true, layer: .object)
    }

    /// R12 沒有 LWPOLYLINE，要用 POLYLINE ＋ VERTEX ＋ SEQEND。
    /// 多打幾行換到「什麼都開得起來」是划算的。
    private static func polyline(_ pts: [SIMD2<Float>], closed: Bool,
                                 layer: Layer) -> String {
        guard pts.count >= 2 else { return "" }
        var s = "0\nPOLYLINE\n8\n\(layer.rawValue)\n66\n1\n70\n\(closed ? 1 : 0)\n"
        for p in pts {
            s += "0\nVERTEX\n8\n\(layer.rawValue)\n10\n\(f(p.x))\n20\n\(f(-p.y))\n30\n0.0\n"
        }
        s += "0\nSEQEND\n8\n\(layer.rawValue)\n"
        return s
    }

    private static func text(_ str: String, at p: SIMD2<Float>,
                             height: Float, layer: Layer) -> String {
        // 群組碼 1 是文字內容；DXF 不接受內嵌換行，先攤平
        let flat = str.replacingOccurrences(of: "\n", with: " ")
        return "0\nTEXT\n8\n\(layer.rawValue)\n10\n\(f(p.x))\n20\n\(f(-p.y))\n30\n0.0\n"
             + "40\n\(f(height))\n1\n\(flat)\n"
    }

    /// DXF 是純文字格式，數字必須用**點號小數**且不能有千分位 ——
    /// 用 String(describing:) 或本地化格式化在部分語系會寫出逗號小數，
    /// 那種檔案 CAD 開起來會少一半的圖元而且不報錯。
    private static func f(_ v: Float) -> String {
        String(format: "%.6f", v)
    }
}
