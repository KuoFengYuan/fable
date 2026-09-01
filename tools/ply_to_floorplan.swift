//
//  ply_to_floorplan.swift
//  fable — 從既有的 points.ply 重新產生 2D 平面圖（svg / dxf / json）
//
//  用途：App 端已經寫過一份，但演算法改過之後想拿舊掃描重跑；
//  或想在桌機上調參數看效果，不必每次重掃。
//  跑的是**與 App 完全同一份** PointCloudFloorPlan，所以結果一致。
//
//    swiftc -O -o /tmp/ply2plan fable/Capture/FloorPlanData.swift \
//           fable/Capture/FloorPlanDrawing.swift \
//           fable/Capture/PointCloudFloorPlan.swift fable/Capture/FloorPlanDXF.swift \
//           tools/ply_to_floorplan.swift
//    /tmp/ply2plan <scan_dir>          # 就地覆寫 floorplan_pc.{svg,dxf,json}
//

import Foundation
import simd

@main struct PlyToFloorPlan {

    /// 讀二進位 PLY 的 xyz。只認 fable 自己寫出來的排列
    /// （float x,y,z ＋ uchar r,g,b，共 15 bytes/點）—— 這支工具不是通用 PLY 讀取器。
    static func readPLY(_ url: URL) -> [SIMD3<Float>]? {
        guard let raw = try? Data(contentsOf: url) else { return nil }
        let marker = Array("end_header\n".utf8)
        var off = -1
        outer: for i in 0..<max(0, raw.count - marker.count) {
            for j in 0..<marker.count where raw[i + j] != marker[j] { continue outer }
            off = i + marker.count
            break
        }
        guard off > 0 else { return nil }
        let header = String(decoding: raw[0..<off], as: UTF8.self)
        guard let line = header.split(separator: "\n")
                .first(where: { $0.hasPrefix("element vertex") }),
              let n = Int(line.split(separator: " ").last ?? "") else { return nil }
        guard raw.count >= off + n * 15 else { return nil }
        var out = [SIMD3<Float>]()
        out.reserveCapacity(n)
        raw.withUnsafeBytes { buf in
            let base = buf.baseAddress!.advanced(by: off)
            for i in 0..<n {
                let p = base.advanced(by: i * 15)
                out.append(SIMD3(p.loadUnaligned(fromByteOffset: 0, as: Float.self),
                                 p.loadUnaligned(fromByteOffset: 4, as: Float.self),
                                 p.loadUnaligned(fromByteOffset: 8, as: Float.self)))
            }
        }
        return out
    }

    static func main() {
        guard CommandLine.arguments.count > 1 else {
            print("用法: ply_to_floorplan <scan_dir>")
            exit(1)
        }
        let dir = URL(fileURLWithPath: CommandLine.arguments[1])
        guard let pts = readPLY(dir.appendingPathComponent("points.ply")) else {
            print("讀不到 points.ply"); exit(1)
        }
        print("讀入 \(pts.count) 點")
        guard let r = PointCloudFloorPlan.extract(points: pts) else {
            print("抽不出牆 —— 點太少，或沒有垂直跨度足夠的表面"); exit(1)
        }
        print(r.summary)

        var lens = r.plan.walls.map(\.lengthM)
        lens.sort()
        print(String(format: "牆 %d 段：中位數 %.2fm / p90 %.2fm / 最長 %.2fm；總長 %.0fm",
                     lens.count, lens[lens.count / 2],
                     lens[Int(Double(lens.count - 1) * 0.9)], lens.last ?? 0,
                     lens.reduce(0, +)))
        print(String(format: "外接 %.1f × %.1f m", r.plan.drawnSizeM.x, r.plan.drawnSizeM.y))

        do {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(r.plan)
                .write(to: dir.appendingPathComponent("floorplan_pc.json"), options: [.atomic])
            try Data(r.plan.svg(showAllFurniture: false).utf8)
                .write(to: dir.appendingPathComponent("floorplan_pc.svg"), options: [.atomic])
            try Data(FloorPlanDXF.make(r.plan).utf8)
                .write(to: dir.appendingPathComponent("floorplan_pc.dxf"), options: [.atomic])
            print("已寫出 floorplan_pc.{json,svg,dxf} 到 \(dir.path)")
        } catch {
            print("寫檔失敗: \(error)"); exit(1)
        }
    }
}
