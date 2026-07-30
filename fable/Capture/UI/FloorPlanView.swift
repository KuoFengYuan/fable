//
//  FloorPlanView.swift
//  fable — review 階段的平面圖預覽（匯出前先看，不要盲匯）
//
//  直接用 Canvas 畫 FloorPlanData 的 2D 線段，不解析 SVG ——
//  投影數學只有 FloorPlanData.segment2D 一份，畫面與 SVG 共用同一組座標，
//  所以螢幕上看到的形狀就是匯出檔的形狀。
//
//  平面方向與 SVG 一致：view_x ∝ world_x、view_y ∝ world_z，非鏡像俯視（推導見 FloorPlanData）。
//

import SwiftUI

struct FloorPlanView: View {

    let data: FloorPlanData
    let onClose: () -> Void

    @State private var zoom: CGFloat = 1
    @State private var pinch: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var drag: CGSize = .zero

    private var scaleNow: CGFloat { max(0.4, min(8, zoom * pinch)) }
    private var panNow: CGSize {
        CGSize(width: offset.width + drag.width, height: offset.height + drag.height)
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 0) {
                topBar
                summary
                if data.walls.isEmpty {
                    Spacer()
                    VStack(spacing: 8) {
                        Image(systemName: "square.dashed").font(.largeTitle)
                        Text("沒有偵測到牆面").font(.subheadline)
                        Text("RoomPlan 需要沿著牆面掃過去；\n只繞著物件拍不會產生牆")
                            .font(.caption)
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.white.opacity(0.5))
                    }
                    .foregroundStyle(.white.opacity(0.7))
                    Spacer()
                } else {
                    plan
                }
                legend
            }
        }
    }

    // MARK: - 自己的頂部列（獨立頁面，不再借用 HUD 的空間）

    private var topBar: some View {
        HStack {
            Button(action: onClose) {
                Image(systemName: "chevron.left")
                    .font(.headline)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .foregroundStyle(.white)
            Spacer()
            Text("平面圖").font(.headline).foregroundStyle(.white)
            Spacer()
            // 與左側等寬的佔位，讓標題真的居中
            Image(systemName: "chevron.left").font(.headline).padding(10).opacity(0)
        }
        .padding(.horizontal, 12)
        .padding(.top, 6)
    }

    // MARK: - 驗收數字：一行講完，不要排成大字

    private var summary: some View {
        VStack(spacing: 6) {
            Text("\(data.roomCount) 房 · \(data.walls.count) 牆 · "
                 + "\(data.doors.count) 門 · \(data.windows.count) 窗")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.white)
            // 拿雷射測距儀對照時看的就是這三個數
            Text(String(format: "外接 %.2f × %.2f m　·　最長牆 %.2f m　·　樓高 %.2f m",
                        data.sizeM.x, data.sizeM.y,
                        data.longestWallM, data.medianWallHeightM))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white.opacity(0.6))
            if data.axisOrderLooksWrong {
                warn("樓高不合理 → RoomPlan 的 dimensions 軸序與程式假設相反，牆長是錯的")
            } else if data.longestWallM < data.sizeM.max() * 0.5 {
                // 最長牆遠短於整體尺寸 ⇒ 牆被切成碎片、房間沒閉合。
                // 這不是 UI 問題也不是 bug，是掃描路徑沒有沿牆走 —— 講清楚比畫得漂亮有用。
                warn("牆面破碎、房間未閉合：請沿著牆面走一圈並回到起點，"
                     + "3DGS 那種繞著物件拍的路徑產生不出完整的牆")
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 10)
        .padding(.bottom, 8)
    }

    private func warn(_ text: String) -> some View {
        Label(text, systemImage: "exclamationmark.triangle.fill")
            .font(.caption2)
            .foregroundStyle(.orange)
            .multilineTextAlignment(.leading)
            .padding(.top, 2)
    }

    // MARK: - 平面圖本體

    private var plan: some View {
        Canvas { ctx, size in draw(ctx, size) }
            .gesture(
                SimultaneousGesture(
                    MagnifyGesture()
                        .onChanged { pinch = $0.magnification }
                        .onEnded { _ in zoom = scaleNow; pinch = 1 },
                    DragGesture()
                        .onChanged { drag = $0.translation }
                        .onEnded { _ in offset = panNow; drag = .zero }
                )
            )
            .onTapGesture(count: 2) {          // 雙擊回到自動置中
                withAnimation(.easeOut(duration: 0.2)) {
                    zoom = 1; pinch = 1; offset = .zero; drag = .zero
                }
            }
    }

    private func draw(_ ctx: GraphicsContext, _ size: CGSize) {
        let b = data.boundsM
        guard b.count == 4, data.sizeM.x > 0.01, data.sizeM.y > 0.01 else { return }
        let pad: CGFloat = 24
        let fit = min((size.width - pad * 2) / CGFloat(data.sizeM.x),
                      (size.height - pad * 2) / CGFloat(data.sizeM.y))
        let s = fit * scaleNow
        let midX = CGFloat(b[0] + b[2]) / 2, midZ = CGFloat(b[1] + b[3]) / 2
        // world (x, z) → view：y 不翻轉 ⇒ 非鏡像俯視
        func P(_ x: Float, _ z: Float) -> CGPoint {
            CGPoint(x: (CGFloat(x) - midX) * s + size.width / 2 + panNow.width,
                    y: (CGFloat(z) - midZ) * s + size.height / 2 + panNow.height)
        }
        func segments(_ items: [FloorPlanSurface]) -> Path {
            var p = Path()
            for i in items where i.segment2D.count == 4 {
                p.move(to: P(i.segment2D[0], i.segment2D[1]))
                p.addLine(to: P(i.segment2D[2], i.segment2D[3]))
            }
            return p
        }

        // 1m 網格
        var grid = Path()
        var gx = b[0].rounded(.up)
        while gx <= b[2] { grid.move(to: P(gx, b[1])); grid.addLine(to: P(gx, b[3])); gx += 1 }
        var gz = b[1].rounded(.up)
        while gz <= b[3] { grid.move(to: P(b[0], gz)); grid.addLine(to: P(b[2], gz)); gz += 1 }
        ctx.stroke(grid, with: .color(.white.opacity(0.10)), lineWidth: 1)

        // 家具佔地（壓在最底）
        var foot = Path()
        for o in data.objects where o.footprint2D.count == 8 {
            foot.move(to: P(o.footprint2D[0], o.footprint2D[1]))
            for k in stride(from: 2, to: 8, by: 2) {
                foot.addLine(to: P(o.footprint2D[k], o.footprint2D[k + 1]))
            }
            foot.closeSubpath()
        }
        ctx.fill(foot, with: .color(.white.opacity(0.07)))
        ctx.stroke(foot, with: .color(.white.opacity(0.18)), lineWidth: 1)

        // 由淡到重
        ctx.stroke(segments(data.openings), with: .color(.gray),
                   style: StrokeStyle(lineWidth: 3, lineCap: .round, dash: [6, 4]))
        ctx.stroke(segments(data.windows), with: .color(.blue),
                   style: StrokeStyle(lineWidth: 5, lineCap: .round))
        ctx.stroke(segments(data.doors), with: .color(.orange),
                   style: StrokeStyle(lineWidth: 5, lineCap: .round))
        ctx.stroke(segments(data.walls), with: .color(.white),
                   style: StrokeStyle(lineWidth: 6, lineCap: .round))

        // 牆長標註：只標 ≥1m 的，且縮太小就不標（會擠成一團看不清）
        if s > 40 {
            for w in data.walls where w.lengthM >= 1 && w.segment2D.count == 4 {
                let mid = P((w.segment2D[0] + w.segment2D[2]) / 2,
                            (w.segment2D[1] + w.segment2D[3]) / 2)
                ctx.draw(Text(String(format: "%.2fm", w.lengthM))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.75)),
                         at: CGPoint(x: mid.x, y: mid.y - 8))
            }
        }

        // 比例尺
        let barY = size.height - 16
        var bar = Path()
        bar.move(to: CGPoint(x: 20, y: barY))
        bar.addLine(to: CGPoint(x: 20 + s, y: barY))
        ctx.stroke(bar, with: .color(.white), lineWidth: 2)
        ctx.draw(Text("1 m").font(.system(size: 9)).foregroundStyle(.white),
                 at: CGPoint(x: 20 + s / 2, y: barY - 9))
    }

    // MARK: - 圖例

    private var legend: some View {
        HStack(spacing: 14) {
            key(.white, "牆"); key(.orange, "門"); key(.blue, "窗"); key(.gray, "開口")
            Spacer()
            Text("雙指縮放 · 雙擊置中").foregroundStyle(.white.opacity(0.45))
        }
        .font(.caption2)
        .foregroundStyle(.white.opacity(0.8))
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private func key(_ c: Color, _ label: String) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 1).fill(c).frame(width: 14, height: 3)
            Text(label)
        }
    }
}
