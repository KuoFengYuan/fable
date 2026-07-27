#!/usr/bin/env python3
"""清晰度閘門的離線校準與回歸測試（對應 QualityMonitor.sharpness / minSharpnessRatio）。

把 Swift 那段算式逐字搬到 numpy，餵已知模糊核的合成影像，回答三個問題：
  1. 門檻該設多少？（真模糊與「內容自然變化」各落在哪個區間）
  2. 絕對門檻為什麼不行？
  3. 基準線凍結（kPeakFreezeRadS）真的必要嗎？

合成影像用 1/f 粉紅雜訊 —— 自然影像的功率頻譜就是 1/f，比棋盤格更能代表真實場景。
改動 QualityMonitor 的取樣參數或門檻時請重跑：assert 會擋住讓兩群重疊的改動。

    python test_sharpness.py            # 回歸測試（安靜，只看通過與否）
    python test_sharpness.py -v         # 印出完整校準表
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# ---- 與 QualityMonitor.swift 同步的常數 -------------------------------------
W, H = 1920, 1440
DIFF_STEP = 2       # 二階差分步距（px）
SAMPLE_STEP = 6     # 抽樣間隔（px）
PEAK_DECAY = 0.97   # kSharpPeakDecay
MIN_RATIO = 0.4     # CaptureConfig.minSharpnessRatio


def sharpness(a: np.ndarray) -> float:
    """逐字對應 QualityMonitor.sharpness()。a: (H,W) int32 luma。"""
    h, w = a.shape
    s = DIFF_STEP
    mx, my = max(w // 10, s), max(h // 10, s)
    ys = np.arange(my, h - my, SAMPLE_STEP)
    xs = np.arange(mx, w - mx, SAMPLE_STEP)
    c = a[np.ix_(ys, xs)]
    dx = np.abs(2 * c - a[np.ix_(ys, xs - s)] - a[np.ix_(ys, xs + s)])
    dy = np.abs(2 * c - a[np.ix_(ys - s, xs)] - a[np.ix_(ys + s, xs)])
    return float((dx.sum() + dy.sum()) / c.sum())


def pink_field(w, h, beta, rng):
    """功率頻譜 ~ 1/f^beta 的隨機場（已正規化為零均值、單位標準差）。"""
    fx = np.fft.fftfreq(w)[None, :]
    fy = np.fft.fftfreq(h)[:, None]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1e-6
    ph = rng.uniform(0, 2 * np.pi, (h, w))
    im = np.real(np.fft.ifft2(f ** (-beta / 2) * np.exp(1j * ph)))
    return (im - im.mean()) / im.std()


def gaussian_blur(img, sigma):
    """FFT 域高斯模糊（失焦的模型）。sigma 單位 px。"""
    if sigma <= 0:
        return img
    h, w = img.shape
    fx = np.fft.fftfreq(w)[None, :]
    fy = np.fft.fftfreq(h)[:, None]
    otf = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (fx ** 2 + fy ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * otf))


def motion_blur(img, length_px, angle_deg=15.0):
    """方向性運動模糊（線性 PSF）——轉彎/手震的模型。length = 曝光內位移量。"""
    if length_px <= 0.5:
        return img
    h, w = img.shape
    n = int(round(length_px))
    psf = np.zeros((h, w))
    th = np.deg2rad(angle_deg)
    for i in range(n):
        d = i - (n - 1) / 2.0
        psf[int(round(-d * np.sin(th))) % h, int(round(d * np.cos(th))) % w] += 1.0
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(psf / psf.sum())))


def u8(x):
    return np.clip(x, 0, 255).astype(np.int32)


def ratios(frames, freeze_flags=None):
    """照 Swift 的 sharpPeak 遞推算比值。freeze_flags[i]=True 表示該幀凍結衰減。"""
    peak, out = 0.0, []
    for i, a in enumerate(frames):
        v = sharpness(a)
        frozen = freeze_flags[i] if freeze_flags else False
        peak = max(v, peak * (1.0 if frozen else PEAK_DECAY))
        out.append(v / peak if peak > 1e-6 else 1.0)
    return np.array(out)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    V = args.verbose
    rng = np.random.default_rng(0)
    fails = []

    def check(cond, msg):
        (print(f"✅ {msg}") if cond else fails.append(msg))
        if not cond:
            print(f"❌ {msg}")

    field = pink_field(W * 2, H, 2.2, rng)
    base = 120 + 35 * field[:, :W]

    # --- 1. 絕對值跨場景差 40 倍 → 只能用相對門檻 -------------------------
    scenes = {
        "書架/紋理豐富": (1.8, 45.0, 120.0),
        "一般室內": (2.2, 35.0, 120.0),
        "低紋理白牆": (2.6, 6.0, 190.0),
    }
    absolutes = {}
    for name, (beta, con, mean) in scenes.items():
        f = pink_field(W, H, beta, rng)
        absolutes[name] = sharpness(u8(mean + con * f))
    if V:
        print("\n[1] 全清晰時的絕對清晰度")
        for k, v in absolutes.items():
            print(f"    {k:<14} {v:.4f}")
    spread = max(absolutes.values()) / min(absolutes.values())
    check(spread > 10,
          f"絕對清晰度跨場景差 {spread:.0f}×（>10）→ 絕對門檻不可用，必須用相對比值")

    # --- 2. 真模糊 vs 內容自然變化，兩群要分開 ---------------------------
    # 真模糊：失焦 σ=8（AF 拉焦峰值）、手震 σ=4、轉彎運動模糊 20px
    real_blur = {
        "失焦 σ=8px": sharpness(u8(gaussian_blur(base, 8))) / sharpness(u8(base)),
        "手震 σ=4px": sharpness(u8(gaussian_blur(base, 4))) / sharpness(u8(base)),
        "轉彎運動模糊 20px": sharpness(u8(motion_blur(base, 20))) / sharpness(u8(base)),
    }
    # 內容自然變化：1.5s 內從高紋理平移到低紋理白牆，全程零模糊
    n = 90
    frames = [u8((120 + 70 * i / (n - 1))
                 + (40 * (1 - i / (n - 1)) + 4 * i / (n - 1)) * field[:, 20 * i:20 * i + W])
              for i in range(n)]
    natural_floor = float(ratios(frames)[1:].min())

    if V:
        print("\n[2] 真模糊的比值（該擋）")
        for k, v in real_blur.items():
            print(f"    {k:<20} {v:.3f}")
        print(f"    內容自然變化的最低比值（不該擋）: {natural_floor:.3f}")
    worst_real = max(real_blur.values())
    check(worst_real < MIN_RATIO,
          f"所有真模糊都在門檻下（最寬鬆者 {worst_real:.2f} < {MIN_RATIO}）")
    check(natural_floor > MIN_RATIO,
          f"內容自然變化不會誤擋（最低 {natural_floor:.2f} > {MIN_RATIO}）")

    # --- 3. 基準線凍結：持續模糊會不會被「學會」-------------------------
    # 10 幀清晰 → 90 幀持續 20px 運動模糊（轉彎 1.5s）→ 10 幀清晰
    sharp_f, blur_f = u8(base), u8(motion_blur(base, 20))
    seq = [sharp_f] * 10 + [blur_f] * 90 + [sharp_f] * 10
    turning = [False] * 10 + [True] * 90 + [False] * 10   # 轉彎中角速度超過凍結門檻

    no_freeze = ratios(seq)[10:100]
    with_freeze = ratios(seq, freeze_flags=turning)[10:100]
    leak_no = int((no_freeze >= MIN_RATIO).sum())
    leak_yes = int((with_freeze >= MIN_RATIO).sum())
    if V:
        print("\n[3] 持續 20px 模糊的 90 幀中，被放行的幀數")
        print(f"    不凍結基準線: {leak_no}/90（{100*leak_no/90:.0f}%）"
              f"  ← 第 {int(np.argmax(no_freeze >= MIN_RATIO))} 幀就爬回門檻上"
              f"（{int(np.argmax(no_freeze >= MIN_RATIO))/60*1000:.0f}ms）")
        print(f"    凍結基準線:   {leak_yes}/90（{100*leak_yes/90:.0f}%）")
    check(leak_no > 45,
          f"不凍結時持續模糊確實會被學會（放行 {leak_no}/90）→ 凍結是必要的，非多餘防護")
    check(leak_yes == 0,
          f"凍結後持續模糊 100% 擋下（放行 {leak_yes}/90）")

    # --- 4. 亮度不變性（除以平均亮度的那一步） ---------------------------
    vals = [sharpness(u8(base * k)) for k in (1.0, 0.5, 0.25)]
    dev = max(abs(v / vals[0] - 1) for v in vals)
    if V:
        print("\n[4] 曝光 ×1 / ×0.5 / ×0.25 的清晰度: "
              + ", ".join(f"{v:.4f}" for v in vals))
    check(dev < 0.05, f"對曝光變化不敏感（最大偏差 {dev*100:.1f}% < 5%）")

    print()
    if fails:
        print(f"❌ {len(fails)} 項失敗")
        return 1
    print("全部 6 項通過 — 清晰度閘門校準驗證完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
