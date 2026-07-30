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
        let outPath = args.count > 1 ? args[1] : "floorplan.svg"
        // args[2] 可以是 json 路徑，也可以是「先轉幾度」的數字（測試用）
        var jsonPath: String?
        var rotateDeg: Float?
        for a in args.dropFirst(2) {
            if let v = Float(a) { rotateDeg = v } else { jsonPath = a }
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
}
