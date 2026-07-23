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
    private var tileNodes: [Int64: SCNNode] = [:]  // tileKey → 節點（幾何為錨點局部座標）
    private var pathPoints: [SCNVector3] = []

    private var domeRoot: SCNNode?
    private var cellNodes: [SCNNode] = []
    private var cellCovered: [Bool] = []
    private var objectCenter: SIMD3<Float>?

    /// 涵蓋率變動回呼（0...1），供 HUD 顯示
    var onCoverageChanged: ((Double) -> Void)?

    var hasDome: Bool { domeRoot != nil }

    init(config: CaptureConfig) {
        self.config = config
        root.addChildNode(pathNode)
        root.addChildNode(pointsRoot)
    }

    func attach(to scene: SCNScene) {
        scene.rootNode.addChildNode(root)
    }

    func reset() {
        pathPoints.removeAll()
        pathNode.geometry = nil
        for child in root.childNodes where child !== pathNode && child !== pointsRoot {
            child.removeFromParentNode()
        }
        for tile in pointsRoot.childNodes {
            tile.removeFromParentNode()
        }
        tileNodes = [:]
        domeRoot = nil
        cellNodes = []
        cellCovered = []
        objectCenter = nil
        onCoverageChanged?(0)
    }

    // MARK: - 即時點雲疊加（空間磚 + ARAnchor 錨定，防漂移殘影）

    func setPointCloudHidden(_ hidden: Bool) {
        pointsRoot.isHidden = hidden
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
        onCoverageChanged?(0)
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
