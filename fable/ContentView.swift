//
//  ContentView.swift
//  fable
//
//  Created by 吳欣怡 on 2026/7/16.
//

import SwiftUI
import ARKit

struct ContentView: View {
    @State private var showCapture = false

    private var hasLiDAR: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    var body: some View {
        VStack(spacing: 28) {
            Spacer()

            Image(systemName: "camera.metering.matrix")
                .font(.system(size: 60, weight: .light))
                .foregroundStyle(.tint)

            VStack(spacing: 6) {
                Text("fable")
                    .font(.largeTitle.bold())
                Text("COLMAP-free 3DGS 訓練資料採集")
                    .font(.headline)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 12) {
                bullet("camera.viewfinder", "ARKit 姿態直出：跳過 SfM，拍完即得相機內外參")
                bullet("cube.transparent", "LiDAR 彩色點雲：作為 3DGS 初始化 Gaussians")
                bullet("gauge.with.needle", "即時品質導引：速度 / 光線 / 距離 / 涵蓋率")
                bullet("doc.zipper", "一鍵匯出 transforms.json + points.ply + 影像 zip")
            }
            .padding(20)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 16))
            .padding(.horizontal)

            Button {
                showCapture = true
            } label: {
                Text("開始掃描")
                    .font(.headline)
                    .frame(maxWidth: 240)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            if !hasLiDAR {
                Label("此裝置無 LiDAR：仍可掃描，點雲將退回稀疏特徵點",
                      systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }

            Spacer()

            Text("拍攝訣竅：多平移、少原地旋轉，繞目標走出弧線")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding()
        .fullScreenCover(isPresented: $showCapture) {
            CaptureView()
        }
    }

    private func bullet(_ symbol: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: symbol)
                .frame(width: 24)
                .foregroundStyle(.tint)
            Text(text)
                .font(.subheadline)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

#Preview {
    ContentView()
}
