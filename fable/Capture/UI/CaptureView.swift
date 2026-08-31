//
//  CaptureView.swift
//  fable — AR 掃描主畫面（ARSCNView + HUD 疊層）
//

import SwiftUI
import SceneKit
import ARKit

struct CaptureView: View {
    @StateObject private var controller = CaptureController()

    var body: some View {
        ZStack {
            ARViewContainer(controller: controller)
                .ignoresSafeArea()
            // review / exporting / done 期間以 3D 檢視器覆蓋 AR 畫面（AR view 保持存活以便續掃）
            if showReview {
                ReviewPointCloudView(points: controller.reviewPoints,
                                     trajectory: controller.reviewTrajectory)
                    .ignoresSafeArea()
                    .id(controller.reviewPoints.count)   // 續掃後重新處理 → 重建場景
            }
            // 訓練期間覆蓋 msplat render：訓練中＝即時預覽（越訓越清晰）；
            // 完成後＝可拖曳環繞檢視訓練結果
            if controller.phase == .training {
                TrainingPreviewView(
                    image: controller.trainingPreview,
                    interactive: true,   // 訓練中/完成都可拖曳轉動
                    onOrbit: { d in
                        controller.orbitTrainedView(deltaX: Float(d.width),
                                                    deltaY: Float(d.height))
                    },
                    onPan: { d in
                        controller.panTrainedView(dx: Float(d.width), dy: Float(d.height))
                    },
                    onZoom: { s in controller.zoomTrainedView(scale: Float(s)) })
                    .ignoresSafeArea()
            }
            HUDOverlay(controller: controller)
        }
        .statusBarHidden()
        // 平面圖用獨立頁面而非疊層：它的資訊與操作跟點雲檢視完全不同一組，
        // 疊在 HUD 上兩邊會互相打架（標頭被統計面板夾住、圖例被訓練按鈕壓掉）。
        // fullScreenCover 也順便讓 HUD 整個退場，不必逐項判斷該不該隱藏。
        .fullScreenCover(isPresented: $controller.showFloorPlan) {
            if let fp = controller.floorPlanData {
                FloorPlanView(data: fp,
                              onClose: { controller.showFloorPlan = false },
                              onRename: { i, name in controller.renameRoom(at: i, to: name) },
                              showFurniture: $controller.showPlanFurniture)
            }
        }
        .onDisappear { controller.teardown() }
    }

    private var showReview: Bool {
        switch controller.phase {
        case .review, .exporting, .done: return !controller.reviewPoints.isEmpty
        default: return false
        }
    }
}

private struct ARViewContainer: UIViewRepresentable {
    let controller: CaptureController

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.autoenablesDefaultLighting = true
        view.automaticallyUpdatesLighting = true
        let tap = UITapGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handleTap(_:)))
        view.addGestureRecognizer(tap)
        controller.attach(arView: view)
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(controller: controller) }

    final class Coordinator: NSObject {
        let controller: CaptureController
        init(controller: CaptureController) { self.controller = controller }

        // 物件模式移除後，點擊畫面已無作用（原本是放置涵蓋圓頂）。
        // 保留 Coordinator 骨架 —— 之後若要加「點選重掃某區」之類的互動會用到。
        @objc func handleTap(_ gesture: UITapGestureRecognizer) {}
    }
}

/// 訓練覆蓋層：顯示 msplat render。訓練中＝即時預覽；完成後 interactive＝可拖曳環繞。
/// 尚無首張 render 前顯示轉圈。
private struct TrainingPreviewView: View {
    let image: UIImage?
    var interactive: Bool = false
    var onOrbit: (CGSize) -> Void = { _ in }
    var onPan: (CGSize) -> Void = { _ in }
    var onZoom: (CGFloat) -> Void = { _ in }

    var body: some View {
        ZStack {
            Color.black
            if let image {
                // 影像已轉正為直式（orientation .right）→ aspectFill 填滿螢幕
                Image(uiImage: image)
                    .resizable()
                    .interpolation(.medium)
                    .aspectRatio(contentMode: .fill)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipped()
            } else {
                VStack(spacing: 14) {
                    ProgressView().controlSize(.large).tint(.white)
                    Text("初始化訓練器…")
                        .font(.footnote)
                        .foregroundStyle(.white.opacity(0.8))
                }
            }
            // 手勢分流：單指=旋轉、雙指拖=平移、pinch=縮放。用 UIKit 辨識器（靠觸點數區分），
            // SwiftUI 的 DragGesture 無法分辨單/雙指，會與平移互相打架。
            if interactive {
                GestureCatcher(
                    onRotate: { dx, dy in onOrbit(CGSize(width: dx, height: dy)) },
                    onPan:    { dx, dy in onPan(CGSize(width: dx, height: dy)) },
                    onZoom:   { s in onZoom(s) })
            }
        }
    }
}

/// UIKit 多點觸控分流：單指 pan→旋轉、雙指 pan→平移、pinch→縮放。
/// 用 UIGestureRecognizer 的 min/maxNumberOfTouches 精準區分觸點數（SwiftUI 內建手勢做不到）。
private struct GestureCatcher: UIViewRepresentable {
    var onRotate: (CGFloat, CGFloat) -> Void
    var onPan: (CGFloat, CGFloat) -> Void
    var onZoom: (CGFloat) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UIView {
        let v = UIView()
        v.backgroundColor = .clear
        v.isMultipleTouchEnabled = true

        let rotate = UIPanGestureRecognizer(target: context.coordinator,
                                            action: #selector(Coordinator.handleRotate(_:)))
        rotate.minimumNumberOfTouches = 1
        rotate.maximumNumberOfTouches = 1

        let pan = UIPanGestureRecognizer(target: context.coordinator,
                                         action: #selector(Coordinator.handlePan(_:)))
        pan.minimumNumberOfTouches = 2
        pan.maximumNumberOfTouches = 2

        let pinch = UIPinchGestureRecognizer(target: context.coordinator,
                                             action: #selector(Coordinator.handlePinch(_:)))

        rotate.delegate = context.coordinator
        pan.delegate = context.coordinator
        pinch.delegate = context.coordinator
        v.addGestureRecognizer(rotate)
        v.addGestureRecognizer(pan)
        v.addGestureRecognizer(pinch)
        context.coordinator.panGR = pan
        context.coordinator.pinchGR = pinch
        return v
    }

    func updateUIView(_ uiView: UIView, context: Context) { context.coordinator.parent = self }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var parent: GestureCatcher
        weak var panGR: UIPanGestureRecognizer?
        weak var pinchGR: UIPinchGestureRecognizer?
        private var lastRotate: CGPoint = .zero
        private var lastPan: CGPoint = .zero

        init(_ p: GestureCatcher) { parent = p }

        @objc func handleRotate(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            if g.state == .began { lastRotate = .zero }
            let dx = t.x - lastRotate.x, dy = t.y - lastRotate.y
            lastRotate = t
            if g.state == .changed { parent.onRotate(dx, dy) }
        }

        @objc func handlePan(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            if g.state == .began { lastPan = .zero }
            let dx = t.x - lastPan.x, dy = t.y - lastPan.y
            lastPan = t
            if g.state == .changed { parent.onPan(dx, dy) }
        }

        @objc func handlePinch(_ g: UIPinchGestureRecognizer) {
            guard g.state == .changed else { return }
            let s = g.scale
            g.scale = 1                       // 取增量比例（>1 拉近）
            if s.isFinite, s > 0 { parent.onZoom(s) }
        }

        // 雙指平移與 pinch 需同時作用；單指旋轉維持獨佔
        func gestureRecognizer(_ g: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
            let a = (g === panGR || g === pinchGR)
            let b = (other === panGR || other === pinchGR)
            return a && b
        }
    }
}
