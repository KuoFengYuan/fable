//
//  HUDStyle.swift
//  fable — 疊在相機畫面上的 HUD 視覺語言
//
//  對齊 RoomPlan 原生那套：深色玻璃 ＋ 細字 ＋ 靠色調而非面積表達嚴重度。
//
//  **抽成共用 modifier 而不是每處各寫一份。** 這種「看起來一致」的東西，
//  只要有兩個地方各自維護就一定會漂移，而漂移出來的畫面正是「不像官方」的來源。
//  HUDOverlay 與 CameraControlBar 都疊在相機上，必須共用同一份。
//

import SwiftUI

extension View {

    /// 深色玻璃底。
    ///
    /// **不用 .ultraThinMaterial 的預設淺色版本。** 材質會跟隨環境明暗，
    /// 而 HUD 疊在相機畫面上 —— 對著白牆或窗戶時淺色材質會整片泛白，
    /// 上面的白字直接消失。強制深色變體才有官方那種「一直看得清楚」的穩定感。
    ///
    /// - tint: 有值時額外疊一層很淡的色底 ＋ 同色描邊，用來表達嚴重度。
    ///   整片實色橫幅在相機畫面上太搶，官方是靠色調而不是靠面積。
    func hudGlass<S: Shape>(_ shape: S, tint: Color? = nil) -> some View {
        background {
            shape
                .fill(.ultraThinMaterial)
                .environment(\.colorScheme, .dark)
                .overlay(shape.fill(tint?.opacity(0.30) ?? .clear))
                .overlay(shape.stroke((tint ?? .white).opacity(tint == nil ? 0.14 : 0.45),
                                      lineWidth: 0.5))
        }
    }

    /// 白字 ＋ 淡陰影。
    ///
    /// 陰影是必要的：疊在相機畫面上，純白字在亮處會糊掉 ——
    /// 而用加粗字重去換可讀性就會失去官方那種輕盈感。用陰影換，字才能維持細。
    func hudText() -> some View {
        foregroundStyle(.white)
            .shadow(color: .black.opacity(0.45), radius: 3, y: 1)
    }
}
