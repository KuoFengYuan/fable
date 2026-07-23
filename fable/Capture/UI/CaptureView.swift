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
                    })
                    .ignoresSafeArea()
            }
            HUDOverlay(controller: controller)
        }
        .statusBarHidden()
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

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            controller.placeObjectAnchor(at: gesture.location(in: gesture.view))
        }
    }
}

/// 訓練覆蓋層：顯示 msplat render。訓練中＝即時預覽；完成後 interactive＝可拖曳環繞。
/// 尚無首張 render 前顯示轉圈。
private struct TrainingPreviewView: View {
    let image: UIImage?
    var interactive: Bool = false
    var onOrbit: (CGSize) -> Void = { _ in }

    @State private var lastDrag: CGSize = .zero

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
        }
        .contentShape(Rectangle())
        .gesture(interactive ? dragGesture : nil)
    }

    private var dragGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let dx = value.translation.width - lastDrag.width
                let dy = value.translation.height - lastDrag.height
                lastDrag = value.translation
                onOrbit(CGSize(width: dx, height: dy))
            }
            .onEnded { _ in lastDrag = .zero }
    }
}
