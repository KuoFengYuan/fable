//
//  FloorPlanData+RoomPlan.swift
//  fable — RoomPlan（CapturedRoom / CapturedStructure）→ FloorPlanData
//
//  只有這一檔碰 RoomPlan。資料模型與製圖層留在 FloorPlanData.swift / FloorPlanDrawing.swift，
//  兩者與平台無關、可在 macOS 上獨立編譯 —— 於是 SVG 輸出在上機前就能先驗。
//

import Foundation
import RoomPlan
import simd

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
        rooms = Self.rooms(floors: room.floors, sections: room.sections)
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
        rooms = structure.rooms.flatMap { Self.rooms(floors: $0.floors, sections: $0.sections) }
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
        floorAreaM2 = rooms.reduce(0) { $0 + $1.areaM2 }
    }

    /// floors 給多邊形、sections 給名稱，兩者沒有 id 對應關係 ——
    /// 以「section 中心落在哪個多邊形內」配對（point-in-polygon），配不到就留 nil。
    private static func rooms(floors: [CapturedRoom.Surface],
                              sections: [CapturedRoom.Section]) -> [FloorPlanRoom] {
        floors.compactMap { floor in
            let pts = polygon(of: floor)
            guard pts.count >= 3 else { return nil }
            let flat = pts.flatMap { [$0.x, $0.y] }
            let c = centroid(pts)
            let label = sections.first { s in
                contains(pts, SIMD2(s.center.x, s.center.z))
            }.map { String(describing: $0.label) }
            return FloorPlanRoom(label: label, polygon2D: flat,
                                 areaM2: abs(shoelace(pts)), labelAt: [c.x, c.y])
        }
    }

    /// 地板多邊形頂點。polygonCorners 是該表面的局部座標，需以 transform 轉到世界後投影。
    private static func polygon(of s: CapturedRoom.Surface) -> [SIMD2<Float>] {
        s.polygonCorners.map { p in
            let w = s.transform * SIMD4<Float>(p.x, p.y, p.z, 1)
            return SIMD2<Float>(w.x, w.z)
        }
    }

    /// 帶號面積（shoelace）。取絕對值即面積，正負號代表頂點繞向
    private static func shoelace(_ p: [SIMD2<Float>]) -> Float {
        var a: Float = 0
        for i in p.indices {
            let q = p[(i + 1) % p.count]
            a += p[i].x * q.y - q.x * p[i].y
        }
        return a / 2
    }

    /// 多邊形形心（面積加權，非頂點平均 —— L 形房間的頂點平均會跑到牆外）
    private static func centroid(_ p: [SIMD2<Float>]) -> SIMD2<Float> {
        let a = shoelace(p)
        guard abs(a) > 1e-6 else {
            return p.reduce(SIMD2<Float>.zero, +) / Float(max(1, p.count))
        }
        var c = SIMD2<Float>.zero
        for i in p.indices {
            let q = p[(i + 1) % p.count]
            let cross = p[i].x * q.y - q.x * p[i].y
            c += (p[i] + q) * cross
        }
        return c / (6 * a)
    }

    /// ray casting point-in-polygon
    private static func contains(_ poly: [SIMD2<Float>], _ pt: SIMD2<Float>) -> Bool {
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
        return inside
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
