#!/usr/bin/env python3
"""驗收一次實機掃描：曝光上限、清晰度閘門、逐幀內參是否都如預期生效。

給定 App 匯出的掃描資料夾（或解壓後的 .zip），逐項檢查採集端的品質修正，
並把「最不清晰的幾幀」列出來讓你直接開圖對照 —— 那是判斷門檻鬆緊的唯一可靠方式。

    python check_capture_quality.py scan_20260727_101530
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import colmap_io

MAX_SHUTTER = 1.0 / 60      # CameraControls.kMaxShutter
MIN_RATIO = 0.4             # CaptureConfig.minSharpnessRatio
OK, WARN, FAIL = "✅", "⚠️ ", "❌"


def load_records(scan: Path):
    p = scan / "poses_refined.jsonl"
    if not p.exists():
        p = scan / "poses.jsonl"
    if not p.exists():
        sys.exit(f"找不到 {p}")
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return sorted(out, key=lambda r: r["id"]), p.name


def pct(v, n):
    return f"{v}/{n}（{100*v/n:.0f}%）" if n else "0/0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir", type=Path)
    args = ap.parse_args()
    recs, src = load_records(args.scan_dir)
    n = len(recs)
    print(f"讀取 {src}：{n} 個關鍵幀\n")
    problems = 0

    # --- 1. 曝光上限（activeMaxExposureDuration = 1/60s）------------------
    exp = [r["exposureDuration"] for r in recs]
    over = [e for e in exp if e > MAX_SHUTTER * 1.05]     # 5% 容差給時基誤差
    print("[1] 曝光時間上限（動態模糊 ∝ 曝光時間）")
    print(f"    中位數 1/{1/sorted(exp)[n//2]:.0f}s、最長 1/{1/max(exp):.0f}s")
    if over:
        problems += 1
        print(f"    {FAIL} {pct(len(over), n)} 幀超過 1/60s —— activeMaxExposureDuration 沒生效")
        print("        （ARKit 可能在 session.run 之後重設了裝置設定；檢查 runSession 的呼叫順序）")
    else:
        print(f"    {OK} 全部 ≤ 1/60s")

    # --- 2. 清晰度閘門 ---------------------------------------------------
    print("\n[2] 清晰度（sharpnessRatio：本幀 ÷ 同場景近期最佳）")
    if "sharpnessRatio" not in recs[0]:
        print(f"    {WARN} 此掃描是舊版 App 匯出的（沒有 sharpness 欄位），跳過")
    else:
        sr = sorted(recs, key=lambda r: r["sharpnessRatio"])
        vals = [r["sharpnessRatio"] for r in sr]
        below = [r for r in sr if r["sharpnessRatio"] < MIN_RATIO]
        print(f"    中位數 {vals[n//2]:.2f}、最差 {vals[0]:.2f}")
        if below:
            problems += 1
            print(f"    {FAIL} {pct(len(below), n)} 幀低於門檻 {MIN_RATIO} 卻仍被存檔 —— 閘門漏了")
        else:
            print(f"    {OK} 沒有任何幀低於門檻 {MIN_RATIO}")
        print("    最不清晰的 5 幀（開圖對照：若它們看起來其實很清晰 → 門檻偏嚴，可調低）：")
        for r in sr[:5]:
            print(f"      {r['imageFile']}  ratio {r['sharpnessRatio']:.2f}"
                  f"  絕對值 {r['sharpness']:.3f}  估計劣化 {r['estimatedBlurPx']:.1f}px")

    # --- 3. 幾何劣化估計（含捲簾快門）-----------------------------------
    print("\n[3] 幾何劣化估計 estimatedBlurPx（運動模糊 ＋ 捲簾剪切）")
    bl = sorted(r["estimatedBlurPx"] for r in recs)
    print(f"    中位數 {bl[n//2]:.1f}px、最差 {bl[-1]:.1f}px")
    if bl[-1] > 16:
        problems += 1
        print(f"    {FAIL} 最差 {bl[-1]:.1f}px 超過 blockBlurPixels(16) 卻仍被存檔")
    elif bl[n//2] > 8:
        print(f"    {WARN} 中位數已超過警告線 8px —— 整段掃描都偏快，放慢會明顯變好")
    else:
        print(f"    {OK} 中位數在警告線 8px 以下")

    # --- 4. 逐幀內參（解除鎖對焦的配套）---------------------------------
    print("\n[4] COLMAP 逐幀內參（吸收連續對焦的內參呼吸）")
    sparse = args.scan_dir / "sparse" / "0"
    if not (sparse / "cameras.bin").exists():
        print(f"    {WARN} 沒有 sparse/0/cameras.bin（尚未按匯出？）跳過")
    else:
        cams = colmap_io.read_cameras_bin(sparse / "cameras.bin")
        imgs = colmap_io.read_images_bin(sparse / "images.bin")
        ids = {c.id for c in cams}
        unmapped = [im for im in imgs if im.camera_id not in ids]
        print(f"    {len(cams)} 台相機 / {len(imgs)} 張影像")
        if len(cams) == 1:
            problems += 1
            print(f"    {FAIL} 只有 1 台相機 —— 逐幀內參沒生效，對焦呼吸會變成重投影誤差")
        elif unmapped:
            problems += 1
            print(f"    {FAIL} {len(unmapped)} 張影像的 camera_id 找不到對應相機")
        else:
            print(f"    {OK} 一張影像一台相機，camera_id 全部對得上")
        fxs = [c.params[0] for c in cams]
        spread = (max(fxs) - min(fxs)) / (sum(fxs) / len(fxs)) * 100
        edge_px = spread / 100 * cams[0].width / 2
        print(f"    實測對焦呼吸：fx {min(fxs):.1f} ~ {max(fxs):.1f}"
              f"（振幅 {spread:.2f}%，畫面邊緣 ≈ {edge_px:.1f}px）")
        print("        ← 這就是逐幀內參吸收掉的量。若 <0.1% 表示這台機器幾乎不呼吸，"
              "單一相機本來也夠；若 >1% 則逐幀內參是必要的。")

    print()
    print(f"{FAIL} {problems} 項未通過" if problems else f"{OK} 全部通過")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
