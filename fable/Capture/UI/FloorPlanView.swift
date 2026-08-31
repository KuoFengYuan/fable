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
    /// (房間索引, 新名稱)。RoomPlan 常判不出用途，讓使用者自己命名比顯示 unidentified 有用
    var onRename: (Int, String) -> Void = { _, _ in }
    /// 是否連活動家具一起畫（預設關；建築製圖只畫固定設備）
    @Binding var showFurniture: Bool

    @State private var renamingIndex: Int?
    @State private var draftName = ""

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
        .alert("房間名稱", isPresented: Binding(
            get: { renamingIndex != nil },
            set: { if !$0 { renamingIndex = nil } })) {
            TextField("例如：客廳、主臥、書房", text: $draftName)
            Button("清除") {
                if let i = renamingIndex { onRename(i, "") }
                renamingIndex = nil
            }
            Button("取消", role: .cancel) { renamingIndex = nil }
            Button("儲存") {
                if let i = renamingIndex { onRename(i, draftName) }
                renamingIndex = nil
            }
        } message: {
            Text("會一併寫進匯出的 floorplan.json 與 .svg")
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
            Button { showFurniture.toggle() } label: {
                Image(systemName: showFurniture ? "chair.lounge.fill" : "chair.lounge")
                    .font(.headline)
                    .padding(10)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .foregroundStyle(showFurniture ? .orange : .white)
        }
        .padding(.horizontal, 12)
        .padding(.top, 6)
    }

    // MARK: - 驗收數字：一行講完，不要排成大字

    private var summary: some View {
        VStack(spacing: 6) {
            // 點雲版不列門窗：它是**刻意**不推論門窗的（牆上的缺口跟「沒掃到」
            // 在點雲裡長得一樣），列一個永遠是 0 的欄位只會讓人以為掃壞了。
            Text(data.isFromPointCloud
                 ? "\(data.walls.count) 牆 · 由點雲抽出"
                 : "\(data.roomCount) 房 · \(data.walls.count) 牆 · "
                   + "\(data.doors.count) 門 · \(data.windows.count) 窗")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.white)
            // 拿雷射測距儀對照時看的就是這三個數
            // 外接尺寸用 drawnSizeM（外緣到外緣）而非 sizeM（牆中心線）——
            // 圖上的尺寸標註標的就是前者，用後者會出現「標頭 4.48 / 圖上 4.71」的矛盾
            Text(data.floorAreaM2 > 0.5
                 ? String(format: "地板 %.1f m²　·　外接 %.2f × %.2f m　·　樓高 %.2f m",
                          data.floorAreaM2, data.drawnSizeM.x, data.drawnSizeM.y,
                          data.medianWallHeightM)
                 : String(format: "外接 %.2f × %.2f m　·　最長牆 %.2f m　·　樓高 %.2f m",
                          data.drawnSizeM.x, data.drawnSizeM.y, data.longestWallM,
                          data.medianWallHeightM))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white.opacity(0.6))
            // 資料本身不完整時直接說明原因與該怎麼補救 ——
            // 這不是 UI 問題也不是 bug，是掃描路徑沒有沿牆走：
            // 平面圖要沿牆掃，3DGS 要繞著物件多視角拍，兩者不是同一條路徑。
            //
            // 尾註要跟著來源走。這裡原本寫死「RoomPlan 需要…」，
            // 而平面圖現在也可能是點雲抽出來的 —— 對點雲版講 RoomPlan 的注意事項
            // 是保證誤導的建議，使用者照著做也不會變好。
            if let reason = data.incompleteReason {
                warn(reason + (data.isFromPointCloud
                     ? "（3DGS 那種繞著物件拍的路徑產生不出完整的牆）"
                     : "（RoomPlan 需要沿牆掃一圈，"
                       + "3DGS 那種繞著物件拍的路徑產生不出完整的牆）"))
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

    /// 世界(公尺, XZ) ↔ 畫面座標。繪製與命中測試**共用同一個**轉換 ——
    /// 各寫一份是這個檔案已經踩過兩次的錯誤（SVG 與畫面漂走、標註被裁），不再重犯。
    private struct PlanTransform {
        let scale: CGFloat
        let mid: CGPoint          // 圖面中心（世界座標）
        let center: CGPoint       // 畫面中心 ＋ 平移
        /// drawing() 對內容整批套用的旋轉角；命中測試要反轉它才能對上原始多邊形
        let rotation: Float

        func view(_ p: SIMD2<Float>) -> CGPoint {
            CGPoint(x: (CGFloat(p.x) - mid.x) * scale + center.x,
                    y: (CGFloat(p.y) - mid.y) * scale + center.y)
        }

        /// 畫面點 → **未旋轉**的世界座標（即 FloorPlanData.rooms 裡多邊形所在的座標系）
        func world(_ p: CGPoint) -> SIMD2<Float> {
            let rx = Float((p.x - center.x) / scale + mid.x)
            let ry = Float((p.y - center.y) / scale + mid.y)
            let c = cos(-rotation), s = sin(-rotation)
            return SIMD2(rx * c - ry * s, rx * s + ry * c)
        }
    }

    private func transform(for size: CGSize) -> PlanTransform? {
        let b = data.drawingBoundsM
        guard b.count == 4, b[2] > b[0], b[3] > b[1] else { return nil }
        let wm = CGFloat(b[2] - b[0]), hm = CGFloat(b[3] - b[1])
        let pad: CGFloat = 12
        let fit = min((size.width - pad * 2) / wm, (size.height - pad * 2) / hm)
        return PlanTransform(
            scale: fit * scaleNow,
            mid: CGPoint(x: CGFloat(b[0] + b[2]) / 2, y: CGFloat(b[1] + b[3]) / 2),
            center: CGPoint(x: size.width / 2 + panNow.width,
                            y: size.height / 2 + panNow.height),
            rotation: data.planRotationRad)
    }

    private var plan: some View {
        GeometryReader { geo in
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
                // count:2 必須宣告在 count:1 之前，否則單擊會先吃掉雙擊
                .onTapGesture(count: 2) {          // 雙擊回到自動置中
                    withAnimation(.easeOut(duration: 0.2)) {
                        zoom = 1; pinch = 1; offset = .zero; drag = .zero
                    }
                }
                .onTapGesture { pt in              // 單擊房間 → 命名
                    guard let t = transform(for: geo.size),
                          let i = data.roomIndex(containing: t.world(pt)) else { return }
                    renamingIndex = i
                    draftName = data.rooms[i].customLabel ?? ""
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
        guard b.count == 4, let t = transform(for: size) else { return }
        let s = t.scale
        // world (x, z) → view：y 不翻轉 ⇒ 非鏡像俯視
        func P(_ p: SIMD2<Float>) -> CGPoint { t.view(p) }

        // 紙張：只鋪在圖面範圍內，四周留深色，看起來像一張圖而不是換了底色的畫面
        let origin = P(SIMD2(b[0], b[1]))
        let paper = CGRect(x: origin.x, y: origin.y,
                           width: CGFloat(b[2] - b[0]) * s, height: CGFloat(b[3] - b[1]) * s)
        ctx.fill(Path(roundedRect: paper, cornerRadius: 4),
                 with: .color(Self.color(.paper)))

        for prim in data.drawing(showAllFurniture: showFurniture) {
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
            Text("點房間可命名 · 雙擊置中").foregroundStyle(.white.opacity(0.45))
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
