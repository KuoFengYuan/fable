//
//  ReviewView.swift
//  fable — 掃描後的點雲驗收檢視器（單指旋轉 / 雙指縮放平移）
//
//  顯示的是「重融合 + 姿態修正後」的點雲 —— 即實際會寫進 points3D.bin 的內容，
//  所見即所得；同時疊上修正後的相機軌跡供檢查追蹤品質。
//

import SwiftUI
import SceneKit
import simd

struct ReviewPointCloudView: UIViewRepresentable {
    let points: [CloudPoint]
    let trajectory: [simd_float4x4]

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView(frame: .zero)
        view.scene = Self.buildScene(points: points, trajectory: trajectory)
        view.allowsCameraControl = true          // 內建軌道相機：旋轉/縮放/平移
        view.autoenablesDefaultLighting = false
        view.antialiasingMode = .none
        view.backgroundColor = UIColor(white: 0.07, alpha: 1)
        view.pointOfView = Self.fittedCamera(for: points)
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}

    private static func buildScene(points: [CloudPoint], trajectory: [simd_float4x4]) -> SCNScene {
        let scene = SCNScene()

        // 點雲分塊掛載（≤250k 點 → ~16 個 draw call）
        var index = 0
        let chunk = 16_384
        while index < points.count {
            let end = min(index + chunk, points.count)
            let node = SCNNode(geometry: PointCloudRendering.geometry(
                for: Array(points[index..<end]), minScreenRadius: 2.5, maxScreenRadius: 9))
            scene.rootNode.addChildNode(node)
            index = end
        }

        // 修正後相機軌跡 + 起點/終點標記
        if trajectory.count >= 2 {
            let positions = trajectory.map { t in
                SCNVector3(t.columns.3.x, t.columns.3.y, t.columns.3.z)
            }
            scene.rootNode.addChildNode(
                SCNNode(geometry: PointCloudRendering.polyline(positions, color: .systemGreen)))
            scene.rootNode.addChildNode(marker(at: positions.first!, color: .systemGreen))
            scene.rootNode.addChildNode(marker(at: positions.last!, color: .systemRed))
        }
        return scene
    }

    private static func marker(at position: SCNVector3, color: UIColor) -> SCNNode {
        let sphere = SCNSphere(radius: 0.02)
        let mat = SCNMaterial()
        mat.diffuse.contents = color
        mat.lightingModel = .constant
        sphere.materials = [mat]
        let node = SCNNode(geometry: sphere)
        node.position = position
        return node
    }

    /// 依點雲包圍盒放置初始相機（斜上方 45°，剛好框住整個場景）
    private static func fittedCamera(for points: [CloudPoint]) -> SCNNode {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        let stride = max(1, points.count / 5000)
        var i = 0
        while i < points.count {
            let p = SIMD3<Float>(points[i].x, points[i].y, points[i].z)
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
            i += stride
        }
        let center = points.isEmpty ? SIMD3<Float>(0, 0, 0) : (lo + hi) * 0.5
        let radius = points.isEmpty ? 2 : max(0.5, simd_length(hi - lo) * 0.5)

        let camera = SCNCamera()
        camera.zNear = 0.01
        camera.zFar = 200
        let node = SCNNode()
        node.camera = camera
        let offset = simd_normalize(SIMD3<Float>(1, 0.7, 1)) * (radius * 2.4)
        node.simdPosition = center + offset
        node.look(at: SCNVector3(center.x, center.y, center.z))
        return node
    }
}
