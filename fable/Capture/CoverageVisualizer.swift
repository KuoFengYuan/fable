//
//  CoverageVisualizer.swift
//  fable — AR 實景中的「拍過哪裡」視覺化
//
//  兩層線索疊加在 ARSCNView 上：
//  1. 軌跡折線 + 每個關鍵幀的視向箭錐（全模式）：看得出走過的路徑與拍攝方向
//  2. 視角圓頂（物件模式）：以物件為中心、azimuth × elevation 分格的半球，
//     從某方向拍到物件該格即轉綠 —— 灰格就是尚未涵蓋的死角，
//     直接引導使用者「往灰的地方走」，避免 3DGS 訓練出現破洞。
//

import Foundation
import ARKit
import SceneKit
import UIKit
import simd

/// 點雲 SceneKit 渲染共用工具（AR 即時疊加與 Review 檢視器共用）
nonisolated enum PointCloudRendering {

    /// 由 actor 端打包好的原始 Data 直接組 geometry —— 主執行緒 O(1)，無逐點迴圈
    static func geometry(positions: Data, colors: Data, indices: Data, count: Int,
                         minScreenRadius: CGFloat, maxScreenRadius: CGFloat,
                         pointSize: CGFloat) -> SCNGeometry {
        let vertexSource = SCNGeometrySource(data: positions, semantic: .vertex,
                                             vectorCount: count, usesFloatComponents: true,
                                             componentsPerVector: 3, bytesPerComponent: 4,
                                             dataOffset: 0, dataStride: 12)
        let colorSource = SCNGeometrySource(data: colors, semantic: .color,
                                            vectorCount: count, usesFloatComponents: true,
                                            componentsPerVector: 3, bytesPerComponent: 4,
                                            dataOffset: 0, dataStride: 12)
        let element = SCNGeometryElement(data: indices, primitiveType: .point,
                                         primitiveCount: count, bytesPerIndex: 4)
        element.pointSize = pointSize
        element.minimumPointScreenSpaceRadius = minScreenRadius
        element.maximumPointScreenSpaceRadius = maxScreenRadius
        let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        geometry.materials = [mat]
        return geometry
    }

    /// 頂點色 point primitive geometry（一次性建構用，如 Review 檢視器）
    static func geometry(for points: [CloudPoint],
                         minScreenRadius: CGFloat = 2,
                         maxScreenRadius: CGFloat = 7,
                         pointSize: CGFloat = 0.008) -> SCNGeometry {
        let vertices = points.map { SCNVector3($0.x, $0.y, $0.z) }
        let vertexSource = SCNGeometrySource(vertices: vertices)

        var colors = [Float]()
        colors.reserveCapacity(points.count * 3)
        for p in points {
            colors.append(Float(p.r) / 255)
            colors.append(Float(p.g) / 255)
            colors.append(Float(p.b) / 255)
        }
        let colorData = colors.withUnsafeBufferPointer { Data(buffer: $0) }
        let colorSource = SCNGeometrySource(data: colorData,
                                            semantic: .color,
                                            vectorCount: points.count,
                                            usesFloatComponents: true,
                                            componentsPerVector: 3,
                                            bytesPerComponent: 4,
                                            dataOffset: 0,
                                            dataStride: 12)

        let element = SCNGeometryElement(indices: Array(0..<Int32(points.count)),
                                         primitiveType: .point)
        element.pointSize = pointSize
        element.minimumPointScreenSpaceRadius = minScreenRadius
        element.maximumPointScreenSpaceRadius = maxScreenRadius

        let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
        let mat = SCNMaterial()
        mat.lightingModel = .constant
        geometry.materials = [mat]
        return geometry
    }

    /// 折線（軌跡用）
    static func polyline(_ points: [SCNVector3], color: UIColor) -> SCNGeometry {
        let source = SCNGeometrySource(vertices: points)
        var indices: [Int32] = []
        indices.reserveCapacity((points.count - 1) * 2)
        for i in 0..<(points.count - 1) {
            indices.append(Int32(i))
            indices.append(Int32(i + 1))
        }
        let element = SCNGeometryElement(indices: indices, primitiveType: .line)
        let geometry = SCNGeometry(sources: [source], elements: [element])
        let mat = SCNMaterial()
        mat.diffuse.contents = color
        mat.lightingModel = .constant
        geometry.materials = [mat]
        return geometry
    }
}

final class CoverageVisualizer {

    private let config: CaptureConfig
    private let root = SCNNode()
    private let pathNode = SCNNode()
    private let pointsRoot = SCNNode()             // 即時點雲（空間磚節點掛載處）
    private let roomRoot = SCNNode()               // RoomPlan 即時面（發光邊框）
    private var roomNodes: [SCNNode] = []          // 每個面一個節點，底下四條邊
    private let dollRoot = SCNNode()               // Dollhouse：擺位（相機前下方）
    private let dollContent = SCNNode()            // Dollhouse：內容（房間縮放到原點）
    private var dollNodes: [SCNNode] = []          // [0] 地板板，其後每個元素一個方塊
    private var dollPlaced = false
    private var tileNodes: [Int64: SCNNode] = [:]  // tileKey → 節點（幾何為錨點局部座標）
    private var pathPoints: [SCNVector3] = []

    private var domeRoot: SCNNode?
    private var cellNodes: [SCNNode] = []
    private var cellCovered: [Bool] = []
    private var objectCenter: SIMD3<Float>?
    private var domeRadius: Float = 1
    private var guidanceCellIdx: Int = -1   // 目前高亮的「最近缺角」格

    /// 涵蓋率變動回呼（0...1），供 HUD 顯示
    var onCoverageChanged: ((Double) -> Void)?
    /// 缺角提醒回呼：往哪補掃的一句提示（nil＝無圓頂或已全涵蓋，隱藏提示）
    var onGuidanceChanged: ((String?) -> Void)?

    var hasDome: Bool { domeRoot != nil }

    init(config: CaptureConfig) {
        self.config = config
        root.addChildNode(pathNode)
        root.addChildNode(pointsRoot)
        root.addChildNode(roomRoot)
        dollRoot.addChildNode(dollContent)
        root.addChildNode(dollRoot)
    }

    func attach(to scene: SCNScene) {
        scene.rootNode.addChildNode(root)
    }

    func reset() {
        pathPoints.removeAll()
        pathNode.geometry = nil
        for child in root.childNodes
        where child !== pathNode && child !== pointsRoot && child !== roomRoot
              && child !== dollRoot {
            child.removeFromParentNode()
        }
        for n in roomNodes { n.removeFromParentNode() }
        roomNodes.removeAll()
        for n in dollNodes { n.removeFromParentNode() }
        dollNodes.removeAll()
        dollPlaced = false
        dollContent.isHidden = true
        for tile in pointsRoot.childNodes {
            tile.removeFromParentNode()
        }
        tileNodes = [:]
        domeRoot = nil
        cellNodes = []
        cellCovered = []
        objectCenter = nil
        domeRadius = 1
        guidanceCellIdx = -1
        onCoverageChanged?(0)
        onGuidanceChanged?(nil)
    }

    // MARK: - 即時點雲疊加（空間磚 + ARAnchor 錨定，防漂移殘影）

    func setPointCloudHidden(_ hidden: Bool) {
        pointsRoot.isHidden = hidden
    }

    // MARK: - RoomPlan 即時疊加（掃到哪就看到哪）

    func setRoomHidden(_ hidden: Bool) {
        roomRoot.isHidden = hidden
    }

    /// 發光核心的半徑（公尺）。6mm 直徑。
    private static let kEdgeRadius: CGFloat = 0.003
    /// 外暈半徑。1.8cm 直徑。
    ///
    /// **視覺粗細是由這一層決定的，不是核心。** 外暈是等 alpha 的圓柱（不是真的
    /// 有衰減的光暈），而且是加法混色 —— 在明亮的室內牆面上，0.16 的 alpha 疊上去
    /// 就已經接近飽和，於是整條讀起來就是外暈的直徑。
    /// 先前兩次調細都只動核心，所以看起來沒什麼變化；這次外暈一起收，
    /// 並把 alpha 再降一階讓它真的只是暈。
    private static let kGlowRadius: CGFloat = 0.009

    /// 用 RoomPlan 當下偵測到的面疊出**發光白色邊框**。
    ///
    /// 為什麼是邊框不是填色面板：填色會把整面牆蓋掉，使用者反而看不到真實畫面
    /// 與點雲；而 RoomPlan 要傳達的資訊是「這個面已經被認出來了、範圍到哪」，
    /// 邊框剛好只講這件事。這也是 Apple 自己的 RoomCaptureView 的做法。
    ///
    /// **重用節點而不是每次重建。** didUpdate 觸發得比畫面更新還密，
    /// 每次重建整組節點會讓 SceneKit 不斷重新上傳幾何 —— 掃描中最不能做的就是
    /// 在主執行緒上製造這種尖峰（會餓死 ARKit 的 VIO）。
    /// 面數變動時只增減差額，其餘就地改尺寸與變換。
    /// 生長動畫的時間。**略長於 didUpdate 的節流間隔（0.2s）** ——
    /// 這樣每一次更新的補間都還沒走完就接上下一次，看起來是連續長出來的；
    /// 短於節流間隔的話會變成「動一下、停一下」，比直接跳還難看。
    private static let kGrowDuration: CFTimeInterval = 0.24

    func updateRoomSurfaces(_ surfaces: [RoomSurface]) {
        while roomNodes.count > surfaces.count {
            roomNodes.removeLast().removeFromParentNode()
        }
        // 新面：先以「長度趨近 0、全透明」就位，下面的補間才會是**從零長出來**，
        // 而不是整個矩形憑空出現。
        var fresh: [Int] = []
        while roomNodes.count < surfaces.count {
            let n = Self.makeFrameNode(parent: roomRoot)
            n.opacity = 0
            fresh.append(roomNodes.count)
            roomNodes.append(n)
        }
        if !fresh.isEmpty {
            // 初始狀態要自己一個 duration 0 的交易 —— 否則它會跟目標狀態
            // 在同一次提交裡被合併掉，等於沒有起點可以補間
            SCNTransaction.begin()
            SCNTransaction.animationDuration = 0
            for i in fresh {
                Self.layoutEdges(roomNodes[i], size: .zero, box: surfaces[i].isBox,
                                 color: Self.roomColor(surfaces[i].kind))
                roomNodes[i].simdTransform = surfaces[i].transform
            }
            SCNTransaction.commit()
        }

        // **新面與既有面用不同的緩動，而且必須分成兩個交易。**
        //
        // 新面要的是「畫出來」的手感 → easeOut，起步快、末段收慢。
        // 既有面要的是連續延伸 → 必須線性：每 0.2s 就有一個新目標，
        // 而 easeOut 每一段末尾都在減速、下一段開頭又衝出去 ——
        // 疊起來就是規律的脈動，正是「不夠流暢」的來源。
        // 線性讓速度跨段一致，看起來就是一條穩定往前長的線。
        let existing = Set(0..<surfaces.count).subtracting(fresh)
        func apply(_ indices: some Sequence<Int>, timing: CAMediaTimingFunctionName) {
            SCNTransaction.begin()
            SCNTransaction.animationDuration = Self.kGrowDuration
            SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: timing)
            for i in indices {
                let s = surfaces[i]
                Self.layoutEdges(roomNodes[i], size: s.size, box: s.isBox,
                                 color: Self.roomColor(s.kind))
                roomNodes[i].simdTransform = s.transform
                roomNodes[i].opacity = 1
            }
            SCNTransaction.commit()
        }
        apply(fresh, timing: .easeOut)
        apply(existing.sorted(), timing: .linear)
    }

    /// 一個面的邊框 = 四條**圓管**（上下左右），每條再套一層外暈。
    ///
    /// **為什麼是圓管不是平面細條。** 邊框躺在牆面的平面上，而掃描時使用者
    /// 大多是沿著牆走、以掠射角看它 —— 平面細條在那個角度會縮成一條看不見的線，
    /// 於是「掃到哪了」這件事在最需要的時候剛好消失。
    /// 圓管的截面從任何方向看都一樣，這也是 RoomPlan 原生視覺的做法。
    /// 一律配 12 條邊（3D 盒的上限）。平面矩形只用前 4 條，其餘隱藏 ——
    /// 這樣同一個節點可以在「牆」與「家具」之間重用，不必因為種類變了就重建。
    private static let kMaxEdges = 12

    private static func makeFrameNode(parent: SCNNode) -> SCNNode {
        let node = SCNNode()
        for _ in 0..<kMaxEdges {
            let edge = SCNNode()
            // 外暈掛成子節點：跟著同一個位置與旋轉，只是更粗更淡。
            // 它不再自己縮放 —— 長度由父節點的 scale 帶著走。
            edge.addChildNode(SCNNode())
            node.addChildNode(edge)
        }
        node.renderingOrder = 10
        parent.addChildNode(node)
        return node
    }

    /// 共用的單位圓柱（高度 1），長度改由節點的 Y 縮放決定。
    ///
    /// **不要每條邊各自持有一份幾何。** 一個房間可能有 30 個元素 × 最多 12 條邊
    /// × 2 層（核心＋外暈）＝ 720 份幾何與 draw call ——
    /// 而這是疊在 ARKit 之上、每幀都要畫的東西，主執行緒與 GPU 都吃不消。
    /// 依「顏色 × 層」快取之後只剩 8 份，SceneKit 就能把它們批次掉。
    private static var tubeCache: [String: SCNCylinder] = [:]

    private static func tube(color: UIColor, glow: Bool) -> SCNCylinder {
        let key = "\(color.hashValue)-\(glow)"
        if let c = tubeCache[key] { return c }
        let c = SCNCylinder(radius: glow ? kGlowRadius : kEdgeRadius, height: 1)
        // 圓周分段壓到 8：這是 2cm 粗的管子，再細分也看不出來
        c.radialSegmentCount = 8
        let m = SCNMaterial()
        m.lightingModel = .constant     // 不吃場景光照：這是 HUD 疊加不是實體
        m.writesToDepthBuffer = false   // 不遮住點雲與真實畫面的深度關係
        m.blendMode = .add              // 加法混色 → 亮處發光，暗處不會變成灰塊
        m.diffuse.contents = color
        m.transparency = glow ? 0.10 : 1.0
        c.materials = [m]
        tubeCache[key] = c
        return c
    }

    // MARK: - Dollhouse：掃到哪就長到哪的房間縮圖

    /// 縮圖的邊長上限（公尺）。25cm 在手臂距離看起來與 RoomPlan 原生接近
    private static let kDollSize: Float = 0.25
    /// 擺放位置：相機前方 / 下方（公尺）。低一點才不會擋住正在掃的牆面
    private static let kDollForward: Float = 0.55
    private static let kDollDown: Float = 0.30

    /// 用 RoomPlan 當下的房間長出白色實體縮圖：地板板 + 半透明牆 + 家具方塊。
    ///
    /// 為什麼要有它：線框告訴使用者「這個面被認出來了」，但看不出**整體**掃到多少。
    /// 縮圖把已辨識的房間整個攤在眼前，缺一面牆、少一塊角落一眼就看得到 ——
    /// 那正是大範圍掃描最需要、而掃完才發現就來不及的資訊。
    func updateDollhouse(_ surfaces: [RoomSurface]) {
        guard !surfaces.isEmpty else {
            for n in dollNodes { n.removeFromParentNode() }
            dollNodes.removeAll()
            dollContent.isHidden = true
            return
        }
        dollContent.isHidden = false

        // 房間的世界包圍盒：逐元素轉換 8 個角點（元素不多，精確算比估算省事）
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for s in surfaces {
            let h = s.size / 2
            for sx in [-h.x, h.x] { for sy in [-h.y, h.y] { for sz in [-h.z, h.z] {
                let p = s.transform * SIMD4<Float>(sx, sy, sz, 1)
                lo = simd_min(lo, SIMD3(p.x, p.y, p.z))
                hi = simd_max(hi, SIMD3(p.x, p.y, p.z))
            } } }
        }
        let extent = hi - lo
        let center = (hi + lo) / 2
        let scale = Self.kDollSize / max(0.1, max(extent.x, max(extent.y, extent.z)))

        // 節點增減必須在交易**之外**：新節點要以 opacity 0 起步，
        // 若在動畫交易裡設 0 會先淡出再淡入，變成閃一下
        let need = surfaces.count + 1
        while dollNodes.count > need { dollNodes.removeLast().removeFromParentNode() }
        while dollNodes.count < need {
            let n = SCNNode()
            n.geometry = SCNBox(width: 1, height: 1, length: 1, chamferRadius: 0)
            n.opacity = 0                 // 新元素淡入，與線框的生長動畫同一個節奏
            dollContent.addChildNode(n)
            dollNodes.append(n)
        }

        // 以下全部走補間。**比例尺也要在裡面** —— 房間長大時 scale 每 0.4s 就換一次，
        // 那是整個縮圖一跳一跳最主要的來源，比單一元素的尺寸變化明顯得多。
        // 線性：縮圖的比例尺與各元素尺寸每 0.2s 就換一次目標，
        // 緩動會讓整個房間規律地漲一下停一下。等速才看得出是穩定長大。
        SCNTransaction.begin()
        SCNTransaction.animationDuration = Self.kGrowDuration
        SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .linear)
        defer { SCNTransaction.commit() }

        // 內容節點：把世界座標的房間搬到原點並縮小；擺放由 dollRoot 負責
        dollContent.simdScale = SIMD3<Float>(repeating: scale)
        dollContent.simdPosition = -center * scale
        for n in dollNodes { n.opacity = 1 }

        // 地板：房間 XZ 範圍的一片薄板，落在最低點
        let plate = dollNodes[0]
        if let b = plate.geometry as? SCNBox {
            b.width = CGFloat(max(0.1, extent.x))
            b.length = CGFloat(max(0.1, extent.z))
            b.height = CGFloat(max(0.02, extent.y) * 0.02)
            b.firstMaterial = Self.dollMaterial(.floor)
        }
        plate.simdPosition = SIMD3<Float>(center.x, lo.y, center.z)
        plate.simdOrientation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

        for (i, s) in surfaces.enumerated() {
            let n = dollNodes[i + 1]
            guard let b = n.geometry as? SCNBox else { continue }
            b.width = CGFloat(max(0.01, s.size.x))
            b.height = CGFloat(max(0.01, s.size.y))
            // 牆的 depth 常常是 0 —— 給它一點厚度才看得出是一片牆而不是一條線
            b.length = CGFloat(max(s.isBox ? 0.01 : 0.04, s.size.z))
            b.firstMaterial = Self.dollMaterial(s.isBox ? .object : .wall)
            n.simdTransform = s.transform
        }
    }

    private enum DollPart { case floor, wall, object }

    /// 材質共用（三種），理由同 tube()：這是每幀都要畫的疊加，不能一個節點一份材質。
    private static var dollMaterials: [String: SCNMaterial] = [:]
    private static func dollMaterial(_ part: DollPart) -> SCNMaterial {
        let key = "\(part)"
        if let m = dollMaterials[key] { return m }
        let m = SCNMaterial()
        m.lightingModel = .constant
        switch part {
        case .floor:  m.diffuse.contents = UIColor.white;                    m.transparency = 0.95
        case .wall:   m.diffuse.contents = UIColor(white: 0.88, alpha: 1);   m.transparency = 0.55
        case .object: m.diffuse.contents = UIColor(white: 0.97, alpha: 1);   m.transparency = 0.92
        }
        dollMaterials[key] = m
        return m
    }

    /// 每幀擺位：相機前下方，但**維持世界朝向** —— 這樣它讀起來像一張攤在眼前的地圖，
    /// 而不是跟著頭轉的貼紙。位置做指數平滑，否則手震會讓它抖。
    func placeDollhouse(camera: simd_float4x4) {
        guard !dollContent.isHidden else { return }
        let pos = SIMD3<Float>(camera.columns.3.x, camera.columns.3.y, camera.columns.3.z)
        let fwd = -SIMD3<Float>(camera.columns.2.x, camera.columns.2.y, camera.columns.2.z)
        // 前方向投影到水平面：相機仰俯時縮圖不該跟著上下飄
        var flat = SIMD3<Float>(fwd.x, 0, fwd.z)
        flat = simd_length(flat) > 1e-4 ? simd_normalize(flat) : SIMD3<Float>(0, 0, -1)
        let target = pos + flat * Self.kDollForward + SIMD3<Float>(0, -Self.kDollDown, 0)
        if dollPlaced {
            dollRoot.simdPosition += (target - dollRoot.simdPosition) * 0.12
        } else {
            dollRoot.simdPosition = target
            dollPlaced = true
        }
    }

    func setDollhouseHidden(_ hidden: Bool) { dollRoot.isHidden = hidden }

    /// 一條邊：長度、中心位置、沿哪個軸
    private enum EdgeAxis { case x, y, z }
    private struct Edge { var length: Float; var at: SIMD3<Float>; var axis: EdgeAxis }

    /// 平面矩形取 4 條邊（XY 平面上），3D 盒取 12 條。
    private static func edges(size: SIMD3<Float>, box: Bool) -> [Edge] {
        let hx = max(0.01, size.x) / 2, hy = max(0.01, size.y) / 2
        let w = hx * 2, h = hy * 2
        guard box else {
            return [Edge(length: w, at: [0,  hy, 0], axis: .x),
                    Edge(length: w, at: [0, -hy, 0], axis: .x),
                    Edge(length: h, at: [-hx, 0, 0], axis: .y),
                    Edge(length: h, at: [ hx, 0, 0], axis: .y)]
        }
        let hz = max(0.01, size.z) / 2
        let d = hz * 2
        var out: [Edge] = []
        for sy in [hy, -hy] { for sz in [hz, -hz] {
            out.append(Edge(length: w, at: [0, sy, sz], axis: .x))
        } }
        for sx in [hx, -hx] { for sz in [hz, -hz] {
            out.append(Edge(length: h, at: [sx, 0, sz], axis: .y))
        } }
        for sx in [hx, -hx] { for sy in [hy, -hy] {
            out.append(Edge(length: d, at: [sx, sy, 0], axis: .z))
        } }
        return out
    }

    /// SCNCylinder 的軸是 +Y：沿 X 的邊繞 Z 轉 90°，沿 Z 的邊繞 X 轉 90°，沿 Y 的不動。
    private static func layoutEdges(_ node: SCNNode, size: SIMD3<Float>, box: Bool,
                                    color: UIColor) {
        let list = edges(size: size, box: box)
        for (i, child) in node.childNodes.enumerated() {
            guard i < list.count else { child.isHidden = true; continue }
            child.isHidden = false
            let e = list[i]
            // 幾何是共用的單位圓柱，長度用 Y 縮放做出來（子節點的外暈跟著一起拉）
            child.geometry = tube(color: color, glow: false)
            child.childNodes.first?.geometry = tube(color: color, glow: true)
            child.simdScale = SIMD3<Float>(1, max(0.01, e.length), 1)
            child.simdPosition = e.at
            switch e.axis {
            case .x: child.simdEulerAngles = SIMD3<Float>(0, 0, .pi / 2)
            case .y: child.simdEulerAngles = .zero
            case .z: child.simdEulerAngles = SIMD3<Float>(.pi / 2, 0, 0)
            }
        }
    }

    /// 牆用白色 —— 那是 RoomPlan 原生的視覺語言，使用者一眼認得出「這是房間結構」。
    /// 門窗開口給顏色，因為那正是最該檢查有沒有被誤判的地方。
    /// 疊加混色下顏色會被相機畫面加亮，所以不必再調高 alpha。
    private static func roomColor(_ kind: RoomSurface.Kind) -> UIColor {
        switch kind {
        case .wall, .object: UIColor.white
        case .door:          UIColor.systemGreen
        case .window:        UIColor.systemTeal
        case .opening:       UIColor.systemOrange
        }
    }

    /// 刷新一塊空間磚：geometry 的頂點是「錨點局部座標」，節點變換 = 該磚錨點當下變換。
    /// ARKit 漂移修正 / 重定位調整錨點 → syncTileTransforms 把節點移到新位置 →
    /// 整磚點雲跟著實體表面走，回原視角不錯位（Scaniverse 同款行為）。
    /// 渲染資料已在 actor 端打包完成，這裡只做 O(1) 包裝 —— 不佔主執行緒時間。
    func updateTile(_ tile: TileRenderData, transform: simd_float4x4) {
        guard tile.count > 0 else { return }
        let node: SCNNode
        if let existing = tileNodes[tile.key] {
            node = existing
        } else {
            node = SCNNode()
            pointsRoot.addChildNode(node)
            tileNodes[tile.key] = node
        }
        node.simdTransform = transform
        node.geometry = PointCloudRendering.geometry(positions: tile.positions,
                                                     colors: tile.colors,
                                                     indices: tile.indices,
                                                     count: tile.count,
                                                     minScreenRadius: 1.5,
                                                     maxScreenRadius: 6,
                                                     pointSize: 0.01)
    }

    /// 每幀同步：ARKit 修正了哪些磚錨點，對應節點就移到哪（防殘影核心）
    func syncTileTransforms(_ transforms: [Int64: simd_float4x4]) {
        for (key, xform) in transforms {
            tileNodes[key]?.simdTransform = xform
        }
    }

    // MARK: - 圓頂

    /// 以物件為中心建立視角圓頂。radius 通常取「使用者目前站位到中心的距離」，
    /// 讓圓頂剛好罩在使用者的拍攝球面上。
    func setObjectCenter(_ center: SIMD3<Float>, radius: Float) {
        domeRoot?.removeFromParentNode()
        let dome = SCNNode()
        dome.simdPosition = center

        let azN = config.domeAzimuthBins
        let elN = config.domeElevationBins
        cellNodes = []
        cellNodes.reserveCapacity(azN * elN)
        cellCovered = Array(repeating: false, count: azN * elN)
        let elMax = config.domeElevationMaxDeg * .pi / 180

        for e in 0..<elN {
            let el0 = elMax * Float(e) / Float(elN)
            let el1 = elMax * Float(e + 1) / Float(elN)
            for a in 0..<azN {
                let az0 = 2 * .pi * Float(a) / Float(azN)
                let az1 = 2 * .pi * Float(a + 1) / Float(azN)
                let node = SCNNode(geometry: Self.cellGeometry(radius: radius,
                                                               az0: az0, az1: az1,
                                                               el0: el0, el1: el1))
                dome.addChildNode(node)
                cellNodes.append(node)
            }
        }
        root.addChildNode(dome)
        domeRoot = dome
        objectCenter = center
        domeRadius = radius
        guidanceCellIdx = -1
        onCoverageChanged?(0)
    }

    // MARK: - 缺角提醒（往哪補掃）

    /// 每幀（可節流）呼叫：找出離目前相機最近的未涵蓋格、以橘色高亮，並回報一句「往哪補」的視角相對提示。
    func updateGuidance(pose: simd_float4x4) {
        guard let center = objectCenter, !cellNodes.isEmpty else { onGuidanceChanged?(nil); return }
        let camPos = MatrixUtil.position(pose)
        let toCam = camPos - center
        let dist = simd_length(toCam)
        guard dist > 0.05 else { return }
        let camDir = toCam / dist
        let camAz = atan2(camDir.z, camDir.x)
        let camEl = asin(max(-1, min(1, camDir.y)))

        let azN = config.domeAzimuthBins, elN = config.domeElevationBins
        let elMax = config.domeElevationMaxDeg * .pi / 180

        // 最近的未涵蓋格（az/el 角距最小）
        var bestIdx = -1
        var bestScore = Float.greatestFiniteMagnitude
        for e in 0..<elN {
            let cellEl = elMax * (Float(e) + 0.5) / Float(elN)
            for a in 0..<azN {
                let idx = e * azN + a
                if cellCovered[idx] { continue }
                let cellAz = 2 * .pi * (Float(a) + 0.5) / Float(azN)
                var dAz = cellAz - camAz
                while dAz > .pi { dAz -= 2 * .pi }
                while dAz < -.pi { dAz += 2 * .pi }
                let dEl = cellEl - camEl
                let score = dAz * dAz + dEl * dEl
                if score < bestScore { bestScore = score; bestIdx = idx }
            }
        }
        if bestIdx < 0 { setGuidanceCell(-1); onGuidanceChanged?(nil); return }  // 已全涵蓋
        setGuidanceCell(bestIdx)

        // 目標格中心的世界座標 → 相機視角相對方向（不需螢幕投影即給明確提示）
        let e = bestIdx / azN, a = bestIdx % azN
        let cellEl = elMax * (Float(e) + 0.5) / Float(elN)
        let cellAz = 2 * .pi * (Float(a) + 0.5) / Float(azN)
        let tdir = SIMD3<Float>(cos(cellEl) * cos(cellAz), sin(cellEl), cos(cellEl) * sin(cellAz))
        let target = center + tdir * domeRadius
        let right = SIMD3<Float>(pose[0][0], pose[0][1], pose[0][2])
        let up    = SIMD3<Float>(pose[1][0], pose[1][1], pose[1][2])
        let back  = SIMD3<Float>(pose[2][0], pose[2][1], pose[2][2])   // 相機 +z（背向；視線為 -back）
        let v = target - camPos
        let cx = simd_dot(v, right), cy = simd_dot(v, up), cz = simd_dot(v, back)
        let missing = cellCovered.lazy.filter { !$0 }.count

        let dir: String
        if cz > 0 { dir = "缺角在你後方 — 轉過去補掃" }
        else if abs(cx) >= abs(cy) { dir = cx > 0 ? "往右邊繞去補掃" : "往左邊繞去補掃" }
        else { dir = cy > 0 ? "把視角抬高補掃" : "把視角放低補掃" }
        onGuidanceChanged?("還缺 \(missing) 個視角 · \(dir)")
    }

    /// 高亮「最近缺角」格為橘色；還原前一個（若仍未涵蓋）為原本淡色。
    private func setGuidanceCell(_ idx: Int) {
        if guidanceCellIdx == idx { return }
        if guidanceCellIdx >= 0, guidanceCellIdx < cellNodes.count, !cellCovered[guidanceCellIdx] {
            cellNodes[guidanceCellIdx].geometry?.firstMaterial?.diffuse.contents = UIColor.white.withAlphaComponent(0.14)
        }
        guidanceCellIdx = idx
        if idx >= 0 {
            cellNodes[idx].geometry?.firstMaterial?.diffuse.contents = UIColor.systemOrange.withAlphaComponent(0.7)
        }
    }

    // MARK: - 關鍵幀

    func addKeyframe(pose: simd_float4x4) {
        let p = MatrixUtil.position(pose)
        pathPoints.append(SCNVector3(p.x, p.y, p.z))
        if pathPoints.count >= 2 {
            pathNode.geometry = Self.polyline(pathPoints)
        }
        addDirectionMarker(pose: pose)
        updateDome(pose: pose)
    }

    /// 小箭錐標記已拍視角（線框，底面貼相機、尖端指向拍攝方向）
    private func addDirectionMarker(pose: simd_float4x4) {
        let pyramid = SCNPyramid(width: 0.045, height: 0.05, length: 0.036)
        let mat = SCNMaterial()
        mat.diffuse.contents = UIColor.systemCyan.withAlphaComponent(0.6)
        mat.lightingModel = .constant
        mat.fillMode = .lines
        mat.isDoubleSided = true
        pyramid.materials = [mat]

        let node = SCNNode(geometry: pyramid)
        // SCNPyramid 尖端朝 +Y；繞 X 轉 -90° 使尖端指向相機 -Z（視線方向）
        let tilt = simd_float4x4(simd_quatf(angle: -.pi / 2, axis: SIMD3<Float>(1, 0, 0)))
        node.simdTransform = pose * tilt
        root.addChildNode(node)
    }

    private func updateDome(pose: simd_float4x4) {
        guard let center = objectCenter, !cellNodes.isEmpty else { return }
        let camPos = MatrixUtil.position(pose)
        let toCam = camPos - center
        let dist = simd_length(toCam)
        guard dist > 0.05 else { return }

        // 相機必須大致朝向物件中心，該視角格才算「拍到」
        let forward = -SIMD3<Float>(pose[2][0], pose[2][1], pose[2][2])
        let toCenter = simd_normalize(center - camPos)
        let cosLimit = cos(config.domeViewAngleDeg * .pi / 180)
        guard simd_dot(simd_normalize(forward), toCenter) > cosLimit else { return }

        let dir = toCam / dist
        let elevation = asin(max(-1, min(1, dir.y)))
        guard elevation >= 0 else { return }          // 圓頂僅涵蓋水平面以上
        var azimuth = atan2(dir.z, dir.x)
        if azimuth < 0 { azimuth += 2 * .pi }

        let elMax = config.domeElevationMaxDeg * .pi / 180
        let e = min(config.domeElevationBins - 1, Int(elevation / elMax * Float(config.domeElevationBins)))
        let a = min(config.domeAzimuthBins - 1, Int(azimuth / (2 * .pi) * Float(config.domeAzimuthBins)))
        let idx = e * config.domeAzimuthBins + a

        guard !cellCovered[idx] else { return }
        cellCovered[idx] = true
        cellNodes[idx].geometry?.firstMaterial?.diffuse.contents =
            UIColor.systemGreen.withAlphaComponent(0.45)
        let covered = cellCovered.lazy.filter { $0 }.count
        onCoverageChanged?(Double(covered) / Double(cellCovered.count))
    }

    // MARK: - 幾何生成

    private static func polyline(_ points: [SCNVector3]) -> SCNGeometry {
        PointCloudRendering.polyline(points, color: .systemGreen)
    }

    /// 球面上的一格四邊形（az/el 各內縮 8% 形成格線間隙）
    private static func cellGeometry(radius: Float, az0: Float, az1: Float,
                                     el0: Float, el1: Float) -> SCNGeometry {
        let azInset = (az1 - az0) * 0.08
        let elInset = (el1 - el0) * 0.08
        let a0 = az0 + azInset, a1 = az1 - azInset
        let e0 = el0 + elInset, e1 = el1 - elInset

        func vertex(_ az: Float, _ el: Float) -> SCNVector3 {
            SCNVector3(cos(el) * cos(az) * radius,
                       sin(el) * radius,
                       cos(el) * sin(az) * radius)
        }
        let vertices = [vertex(a0, e0), vertex(a1, e0), vertex(a1, e1), vertex(a0, e1)]
        let indices: [Int32] = [0, 1, 2, 0, 2, 3]
        let source = SCNGeometrySource(vertices: vertices)
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
        let geometry = SCNGeometry(sources: [source], elements: [element])

        let mat = SCNMaterial()
        mat.diffuse.contents = UIColor.white.withAlphaComponent(0.14)
        mat.lightingModel = .constant
        mat.isDoubleSided = true
        mat.writesToDepthBuffer = false
        geometry.materials = [mat]
        return geometry
    }
}
