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
    /// 使用者自己輸入的房間名稱。優先於 RoomPlan 的判定 ——
    /// RoomPlan 判不出用途（回 unidentified）在單房間掃描時很常見，
    /// 而使用者當然知道那是什麼房間。會一併寫進匯出的 json / svg。
    var customLabel: String?

    /// 圖面與 UI 一律用這個，不要各自決定要顯示哪個名稱
    var displayName: String {
        if let c = customLabel, !c.trimmingCharacters(in: .whitespaces).isEmpty { return c }
        return label.map(FloorPlanData.roomName) ?? "房間"
    }
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

    /// 掃描是否明顯不完整。
    ///
    /// 這裡原本放的是「軸序檢查」（樓高不在 2.0~3.6m 就判定 dimensions 軸序相反），
    /// 那個判斷是錯的，已移除：Apple 文件明確定義 Surface.dimensions 為
    /// (width, height, depth)，假設本來就對。實機上樓高 1.83m 的成因是
    /// **牆只被掃到 1.83m 高**，不是軸序 —— 而同一份程式在另一次掃描樓高正常，
    /// 軸序若真的相反不可能只錯一次。那道檢查唯一的效果是對不完整的掃描說謊。
    ///
    /// 改判真正的成因。三個徵兆任一成立即視為不完整：
    ///   樓高 < 2.0m       牆沒被掃到頂（RoomPlan 只報告實際觀測到的高度）
    ///   最長牆 < 半個外接邊  牆被切成碎片、房間沒閉合
    ///   完全沒有門窗       沿牆掃過的話至少會抓到門
    var scanLooksIncomplete: Bool {
        guard !walls.isEmpty else { return false }
        if medianWallHeightM < 2.0 { return true }
        if longestWallM < sizeM.max() * 0.5 { return true }
        if doors.isEmpty && windows.isEmpty { return true }
        return false
    }

    /// 不完整的具體原因，用來給使用者可行動的說明（無問題時回 nil）
    var incompleteReason: String? {
        guard !walls.isEmpty else { return nil }
        if medianWallHeightM < 2.0 {
            return String(format: "牆只掃到 %.2fm 高：請把鏡頭往上帶到牆與天花板的交界",
                          medianWallHeightM)
        }
        if longestWallM < sizeM.max() * 0.5 {
            return "牆面破碎、房間未閉合：請沿著牆面走一圈並回到起點"
        }
        if doors.isEmpty && windows.isEmpty {
            return "沒有偵測到任何門窗：沿牆掃過時請讓門窗完整入鏡"
        }
        return nil
    }

    /// 讓主要牆向對齊水平所需的旋轉角（弧度，逆時針為正）。
    ///
    /// ARKit 的世界 +X/+Z 是**任意**水平軸 —— gravity 對齊只固定 Y 軸，
    /// 水平朝向取決於使用者按下開始掃描時手機指向哪裡。於是房間會歪著畫，
    /// 實機實測歪了約 45°。建築平面圖一律是正的，所以要把主要牆向轉回水平。
    ///
    /// 做法：牆向對 90° 取模（牆只有兩個正交方向，差 90° 視為同一組），
    /// 再以牆長為權重取圓形平均。角度先 ×4 映射到整個圓、平均完再 ÷4，
    /// 這是在模數空間取平均的標準做法 —— 直接平均角度會在 0/90° 邊界爆掉
    /// （例如 1° 與 89° 的算術平均是 45°，但正確答案是 0°）。
    var planRotationRad: Float {
        var sx: Float = 0, sy: Float = 0
        for w in walls where w.segment2D.count == 4 && w.lengthM > 0.2 {
            let dx = w.segment2D[2] - w.segment2D[0]
            let dz = w.segment2D[3] - w.segment2D[1]
            let a4 = atan2(dz, dx) * 4          // ×4 ⇒ 90° 的模數變成 360°
            sx += cos(a4) * w.lengthM
            sy += sin(a4) * w.lengthM
        }
        guard sx * sx + sy * sy > 1e-9 else { return 0 }
        return -atan2(sy, sx) / 4
    }

    /// 哪個房間包含這個點（未旋轉的世界座標）。點擊命名用。
    /// 多個房間重疊時取面積最小的 —— 小房間更可能是使用者想點的那個。
    func roomIndex(containing pt: SIMD2<Float>) -> Int? {
        var best: (Int, Float)?
        for (i, r) in rooms.enumerated() {
            guard Self.polygonContains(r.polygon2D, pt) else { continue }
            if best == nil || r.areaM2 < best!.1 { best = (i, r.areaM2) }
        }
        return best?.0
    }

    /// ray casting point-in-polygon。多邊形以扁平化的 [x0,z0, x1,z1, ...] 表示。
    /// 這段邏輯原本在 FloorPlanData+RoomPlan（配對房間名稱）與 FloorPlanDrawing
    /// （判斷門的開向）各有一份，抽出來共用。
    static func polygonContains(_ flat: [Float], _ pt: SIMD2<Float>) -> Bool {
        let n = flat.count / 2
        guard n >= 3 else { return false }
        var inside = false
        var j = n - 1
        for i in 0..<n {
            let ax = flat[i * 2], ay = flat[i * 2 + 1]
            let bx = flat[j * 2], by = flat[j * 2 + 1]
            if (ay > pt.y) != (by > pt.y),
               pt.x < (bx - ax) * (pt.y - ay) / (by - ay) + ax {
                inside.toggle()
            }
            j = i
        }
        return inside
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
