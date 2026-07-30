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
//  本檔**刻意不 import RoomPlan**：資料模型與製圖層與平台無關，
//  RoomPlan → FloorPlanData 的轉換放在 FloorPlanData+RoomPlan.swift。
//  好處是製圖層可以在 macOS 上獨立編譯執行（見 tools/render_floorplan.swift），
//  匯出的 SVG 在交給實機之前就能先看過。
//
//  座標保持 ARKit 原生（+Y up、公尺），與 points.ply / poses.jsonl 一致；
//  匯出 COLMAP 用的世界翻轉不套用在這裡（那是為了 3DGS 生態的慣例，平面圖不需要）。
//

import Foundation
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

/// 一個房間：地板多邊形 ＋ 名稱 ＋ 面積。
/// 這是「像平面圖」與「像線框圖」的分界 —— 有封閉多邊形才畫得出填色、標得出名稱與面積。
/// 資料來自 RoomPlan 的 floors（polygonCorners）與 sections（label），
/// 兩者本來就在 CapturedRoom 裡，先前完全沒用到。
nonisolated struct FloorPlanRoom: Codable, Sendable {
    /// 房間名稱（RoomPlan section label，如 kitchen / bedroom；無法判定時為 nil）
    var label: String?
    /// 地板多邊形，投影到水平面的頂點序列 [x0,z0, x1,z1, ...]，公尺
    var polygon2D: [Float]
    /// 多邊形面積（m²），以 shoelace 公式計算 —— 這才是實際地板面積，非外接矩形
    var areaM2: Float
    /// 標註文字要放的位置（多邊形形心）
    var labelAt: [Float]
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
    /// 房間（地板多邊形 ＋ 名稱 ＋ 面積）
    var rooms: [FloorPlanRoom] = []
    /// 所有房間地板多邊形的面積合計（m²）＝ 實際可用面積
    var floorAreaM2: Float = 0
    /// 牆體外接矩形（公尺）[minX, minZ, maxX, maxZ]，粗估整體尺寸用
    var boundsM: [Float] = []
    /// 牆體外接矩形面積（m²）。**不是**實際地板面積 ——
    /// 非矩形格局會高估，僅供快速對照，正式面積請用 USDZ 的地板幾何。
    var boundingAreaM2: Float = 0
}

// MARK: - 驗收用的摘要數字（畫面預覽、log、離線檢查三處共用，不各算一遍）

extension FloorPlanData {

    /// 外接尺寸（公尺）。使用者拿捲尺對照時最先看的就是這兩個數
    var sizeM: SIMD2<Float> {
        guard boundsM.count == 4 else { return .zero }
        return SIMD2(boundsM[2] - boundsM[0], boundsM[3] - boundsM[1])
    }

    var medianWallLengthM: Float { median(walls.map(\.lengthM)) }

    /// 牆厚中位數。門窗開孔與符號都用它，而不是門窗自身的 thickness
    /// （後者常較小，會在牆面留下白邊）
    var medianWallThicknessM: Float {
        median(walls.compactMap { $0.dimensions.count > 2 ? $0.dimensions[2] : nil }
                    .filter { $0 > 0.02 })
    }

    /// 樓高中位數。這是「軸序有沒有搞錯」的判斷依據（見 axisOrderLooksWrong）
    var medianWallHeightM: Float {
        median(walls.compactMap { $0.dimensions.count > 1 ? $0.dimensions[1] : nil })
    }

    var longestWallM: Float { walls.map(\.lengthM).max() ?? 0 }

    /// 程式假設 Surface.dimensions = (寬, 高, 厚)。若 RoomPlan 實際是相反的，
    /// 每面「牆長」會全部變成樓高（~2.4m）→ 平面圖整張報廢，
    /// 但畫面上仍然有線條、不會有任何錯誤訊息。用樓高是否落在合理區間反推。
    var axisOrderLooksWrong: Bool {
        guard !walls.isEmpty else { return false }
        return !(2.0...3.6).contains(medianWallHeightM)
    }

    private func median(_ v: [Float]) -> Float {
        guard !v.isEmpty else { return 0 }
        return v.sorted()[v.count / 2]
    }
}

// MARK: - SVG 後端（把 drawing() 的圖元寫成 SVG）

extension FloorPlanData {

    /// 語意色 → 十六進位。紙白底、墨黑線，牆用深灰實體 —— 印出來也對。
    private static func hex(_ c: PlanColor) -> String {
        switch c {
        case .ink:      "#1a1a1a"
        case .paper:    "#ffffff"
        case .wall:     "#2b2b2b"
        case .door:     "#c2410c"
        case .window:   "#1d4ed8"
        case .opening:  "#78716c"
        case .roomFill: "#f4f1ec"
        case .roomText: "#44403c"
        case .dim:      "#78716c"
        case .object:   "#dcd7cf"
        }
    }

    /// 產生平面圖 SVG。1 公尺 = pxPerMeter 像素。
    func svg(pxPerMeter: Float = 110) -> String {
        let b = drawingBoundsM
        guard b.count == 4, b[2] > b[0], b[3] > b[1] else {
            return #"<svg xmlns="http://www.w3.org/2000/svg" width="240" height="60"><text x="8" y="34" font-size="13">無牆面資料</text></svg>"#
        }
        let w = (b[2] - b[0]) * pxPerMeter
        let h = (b[3] - b[1]) * pxPerMeter
        func f(_ v: Float) -> String { String(format: "%.1f", v) }
        // world (x, z) → svg (x, y)：y 不翻轉 ⇒ 非鏡像俯視（推導見檔頭）
        func X(_ v: Float) -> Float { (v - b[0]) * pxPerMeter }
        func Y(_ v: Float) -> Float { (v - b[1]) * pxPerMeter }

        var s = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"\(f(w))\" height=\"\(f(h))\""
        s += " viewBox=\"0 0 \(f(w)) \(f(h))\">\n"
        s += "<rect width=\"100%\" height=\"100%\" fill=\"\(Self.hex(.paper))\"/>\n"

        for prim in drawing() {
            switch prim {
            case let .path(pts, closed, style):
                guard pts.count >= 2 else { continue }
                let d = pts.enumerated()
                    .map { "\($0.offset == 0 ? "M" : "L")\(f(X($0.element.x))) \(f(Y($0.element.y)))" }
                    .joined(separator: " ") + (closed ? " Z" : "")
                var attrs = "fill=\"\(style.fill.map(Self.hex) ?? "none")\""
                if let st = style.stroke {
                    attrs += " stroke=\"\(Self.hex(st))\" stroke-width=\"\(f(style.widthPx))\""
                    attrs += " stroke-linejoin=\"round\" stroke-linecap=\"round\""
                    if let dash = style.dash {
                        attrs += " stroke-dasharray=\"\(dash.map { f($0) }.joined(separator: ","))\""
                    }
                } else {
                    attrs += " stroke=\"none\""
                }
                s += "<path d=\"\(d)\" \(attrs)/>\n"

            case let .text(str, at, size, color, align, bold):
                let anchor = align == .center ? "middle" : (align == .right ? "end" : "start")
                s += "<text x=\"\(f(X(at.x)))\" y=\"\(f(Y(at.y)))\" font-size=\"\(f(size))\""
                s += " text-anchor=\"\(anchor)\" fill=\"\(Self.hex(color))\""
                s += " font-family=\"-apple-system,Helvetica,sans-serif\""
                s += bold ? " font-weight=\"600\"" : ""
                s += ">\(Self.escape(str))</text>\n"
            }
        }

        // 比例尺（畫在圖外側左下，用像素座標即可）
        let by = h - 10
        s += "<path d=\"M10 \(f(by)) L\(f(10 + pxPerMeter)) \(f(by))\" stroke=\"\(Self.hex(.ink))\""
        s += " stroke-width=\"2\" fill=\"none\"/>\n"
        s += "<text x=\"10\" y=\"\(f(by - 5))\" font-size=\"10\" fill=\"\(Self.hex(.ink))\""
        s += " font-family=\"-apple-system,Helvetica,sans-serif\">1 m</text>\n"
        s += "</svg>\n"
        return s
    }

    private static func escape(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }
}
