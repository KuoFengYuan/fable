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
            Text(data.floorAreaM2 > 0.5
                 ? String(format: "地板 %.1f m²　·　外接 %.2f × %.2f m　·　樓高 %.2f m",
                          data.floorAreaM2, data.sizeM.x, data.sizeM.y, data.medianWallHeightM)
                 : String(format: "外接 %.2f × %.2f m　·　最長牆 %.2f m　·　樓高 %.2f m",
                          data.sizeM.x, data.sizeM.y, data.longestWallM, data.medianWallHeightM))
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

    /// 語意色 → SwiftUI Color。與 SVG 後端的十六進位一一對應（見 FloorPlanData.hex），
    /// 兩邊都是紙白底、墨黑線的製圖配色 —— 螢幕上看到的就是匯出檔的樣子。
    private static func color(_ c: PlanColor) -> Color {
        switch c {
        case .ink:      Color(red: 0.10, green: 0.10, blue: 0.10)
        case .paper:    Color(red: 1.00, green: 1.00, blue: 1.00)
        case .wall:     Color(red: 0.17, green: 0.17, blue: 0.17)
        case .door:     Color(red: 0.76, green: 0.25, blue: 0.05)
        case .window:   Color(red: 0.11, green: 0.31, blue: 0.85)
        case .opening:  Color(red: 0.47, green: 0.44, blue: 0.42)
        case .roomFill: Color(red: 0.96, green: 0.95, blue: 0.93)
        case .roomText: Color(red: 0.27, green: 0.25, blue: 0.24)
        case .dim:      Color(red: 0.47, green: 0.44, blue: 0.42)
        case .object:   Color(red: 0.86, green: 0.84, blue: 0.81)
        }
    }

    /// 只做座標轉換與上色 —— 畫什麼完全由 FloorPlanData.drawing() 決定，
    /// 與 SVG 後端共用同一份圖元清單。
    private func draw(_ ctx: GraphicsContext, _ size: CGSize) {
        let b = data.drawingBoundsM
        guard b.count == 4, b[2] > b[0], b[3] > b[1] else { return }
        let wm = CGFloat(b[2] - b[0]), hm = CGFloat(b[3] - b[1])
        let pad: CGFloat = 12
        let fit = min((size.width - pad * 2) / wm, (size.height - pad * 2) / hm)
        let s = fit * scaleNow
        let midX = CGFloat(b[0] + b[2]) / 2, midZ = CGFloat(b[1] + b[3]) / 2
        // world (x, z) → view：y 不翻轉 ⇒ 非鏡像俯視
        func P(_ p: SIMD2<Float>) -> CGPoint {
            CGPoint(x: (CGFloat(p.x) - midX) * s + size.width / 2 + panNow.width,
                    y: (CGFloat(p.y) - midZ) * s + size.height / 2 + panNow.height)
        }

        // 紙張：只鋪在圖面範圍內，四周留深色，看起來像一張圖而不是換了底色的畫面
        let paper = CGRect(x: P(SIMD2(b[0], b[1])).x, y: P(SIMD2(b[0], b[1])).y,
                           width: wm * s, height: hm * s)
        ctx.fill(Path(roundedRect: paper, cornerRadius: 4),
                 with: .color(Self.color(.paper)))

        for prim in data.drawing() {
            switch prim {
            case let .path(pts, closed, style):
                guard pts.count >= 2 else { continue }
                var path = Path()
                path.move(to: P(pts[0]))
                for p in pts.dropFirst() { path.addLine(to: P(p)) }
                if closed { path.closeSubpath() }
                if let fill = style.fill {
                    ctx.fill(path, with: .color(Self.color(fill)))
                }
                if let stroke = style.stroke {
                    ctx.stroke(path, with: .color(Self.color(stroke)),
                               style: StrokeStyle(lineWidth: CGFloat(style.widthPx),
                                                  lineCap: .round, lineJoin: .round,
                                                  dash: style.dash?.map { CGFloat($0) } ?? []))
                }

            case let .text(str, at, sizePx, color, align, bold):
                // 縮太小就不畫字：擠成一團比沒有更難看
                guard s > 45 else { continue }
                var t = Text(str).font(.system(size: CGFloat(sizePx),
                                               weight: bold ? .semibold : .regular))
                t = t.foregroundStyle(Self.color(color))
                let anchor: UnitPoint = align == .center ? .center
                    : (align == .right ? .trailing : .leading)
                ctx.draw(t, at: P(at), anchor: anchor)
            }
        }
    }

    // MARK: - 圖例

    private var legend: some View {
        HStack(spacing: 14) {
            key(Self.color(.wall), "牆"); key(Self.color(.door), "門")
            key(Self.color(.window), "窗"); key(Self.color(.opening), "開口")
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
