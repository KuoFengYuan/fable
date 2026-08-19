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

    # --- 1b. 感光度（顆粒感的直接成因）-----------------------------------
    print("\n[1b] 感光度 ISO（顆粒感的直接成因）")
    isos = sorted(r.get("iso", 0) for r in recs)
    if isos[-1] <= 0:
        print(f"    {WARN} 此掃描沒有 iso 欄位（舊版 App），跳過")
    else:
        med, hi = isos[n // 2], isos[-1]
        over = sum(1 for i in isos if i > 400)
        print(f"    中位數 {med:.0f}、最高 {hi:.0f}、超過 400 的有 {pct(over, n)}")
        if med <= 200:
            print(f"    {OK} ISO 很低 —— 顆粒感**不是**感光度造成的，降噪不會生效也不該生效。")
            print("        剩下的顆粒是 ARKit video 幀本身的特性（無多幀降噪），")
            print("        唯一的解法是改走照片管線 captureHighResolutionFrame()。")
        elif med <= 800:
            print(f"    {OK} ISO 中等 —— 自適應降噪會在超過 400 的那 {pct(over, n)} 幀上輕度介入")
        else:
            print(f"    {WARN} ISO 偏高 —— 現場光線不足。補光的效果會遠大於任何後處理；")
            print("        或考慮放寬 CameraControls.kMaxShutter（代價是動態模糊變多）")

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

    # --- 2b. 掃描後的全域模糊複核（BlurFilter）---------------------------
    print("\n[2b] 掃描後全域複核 blurVerdict")
    verdicts = [r.get("blurVerdict", "keep") for r in recs]
    if "blurVerdict" not in recs[0]:
        print(f"    {WARN} 此掃描沒有 blurVerdict 欄位（舊版 App），跳過")
    else:
        drop = [r for r in recs if r.get("blurVerdict") == "drop"]
        demo = [r for r in recs if r.get("blurVerdict") == "demote"]
        keep = n - len(drop) - len(demo)
        print(f"    keep {keep} / demote {len(demo)} / drop {len(drop)}")
        print("      demote = 顏色糊，不進訓練，但深度仍以降權併入點雲（幾何來自 LiDAR，丟了只會開洞）")
        print("      drop   = 幾何不可信（轉太快 → 姿態錯位＋捲簾剪切），點雲也不用（殘影比破洞更糟）")
        if (len(drop) + len(demo)) / n > 0.28:
            print(f"    {WARN} 排除比例 {(len(drop)+len(demo))/n*100:.0f}% 已接近 30% 上限 ——")
            print("        代表整段掃描普遍偏糊，補拍會比調參有效")
        for r in (drop + demo)[:5]:
            print(f"      {r['imageFile']}  {r['blurVerdict']:<6}"
                  f"  sharpness {r.get('sharpness', 0):.3f}"
                  f"  劣化 {r['estimatedBlurPx']:.1f}px")

    # --- 3. 幾何劣化估計（含捲簾快門）-----------------------------------
    # 只看實際會進訓練的幀 —— 被 BlurFilter 排除的本來就是要丟的，拿它們來判失敗是錯的
    print("\n[3] 幾何劣化估計 estimatedBlurPx（運動模糊 ＋ 捲簾剪切，僅計 keep 的幀）")
    kept = [r for r in recs if r.get("blurVerdict", "keep") == "keep"]
    if not kept:
        print(f"    {FAIL} 沒有任何幀通過複核")
        return 1
    bl = sorted(r["estimatedBlurPx"] for r in kept)
    n = len(kept)
    print(f"    {n} 幀，中位數 {bl[n//2]:.1f}px、最差 {bl[-1]:.1f}px")
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

    # --- 5. 平面圖（RoomPlan）--------------------------------------------
    print("\n[5] 平面圖 floorplan.{usdz,json,svg}")
    fp_path = args.scan_dir / "floorplan.json"
    if not fp_path.exists():
        print(f"    {WARN} 沒有 floorplan.json（未開啟 captureFloorPlan、機型不支援，或尚未匯出）")
    else:
        fp = json.loads(fp_path.read_text())
        walls = fp.get("walls", [])
        print(f"    {fp.get('roomCount', 1)} 房 · {len(walls)} 牆 · {len(fp.get('doors', []))} 門"
              f" · {len(fp.get('windows', []))} 窗 · {len(fp.get('objects', []))} 家具")
        for name in ("floorplan.usdz", "floorplan.svg"):
            mark = OK if (args.scan_dir / name).exists() else WARN
            print(f"    {mark} {name}")
        if walls:
            lens = sorted(w["lengthM"] for w in walls)
            hts = sorted(w["dimensions"][1] for w in walls if len(w["dimensions"]) > 1)
            b = fp.get("boundsM", [])
            print(f"    牆長中位數 {lens[len(lens)//2]:.2f}m、最長 {lens[-1]:.2f}m")
            if b and len(b) == 4:
                print(f"    外接尺寸 {b[2]-b[0]:.2f} × {b[3]-b[1]:.2f} m"
                      f"（外接面積 {fp.get('boundingAreaM2', 0):.1f} m²，非實際地板面積）")
            # 掃描完整度。這裡原本判定的是「dimensions 軸序相反」，那是錯的 ——
            # Apple 文件定義 Surface.dimensions 為 (width, height, depth)，假設本來就對；
            # 樓高偏低的真正成因是牆沒被掃到頂。舊判斷只會對不完整的掃描說謊。
            incomplete = []
            if hts and hts[len(hts) // 2] < 2.0:
                incomplete.append(f"牆只掃到 {hts[len(hts)//2]:.2f}m 高（鏡頭要帶到牆與天花板的交界）")
            if b and len(b) == 4 and lens[-1] < max(b[2] - b[0], b[3] - b[1]) * 0.5:
                incomplete.append("牆面破碎、房間未閉合（沿牆走一圈並回到起點）")
            if not fp.get("doors") and not fp.get("windows"):
                incomplete.append("沒有偵測到任何門窗（沿牆掃時讓門窗完整入鏡）")
            if incomplete:
                print(f"    {WARN} 掃描不完整：")
                for r in incomplete:
                    print(f"        · {r}")
                print("        RoomPlan 要沿牆掃一圈，3DGS 要繞著物件多視角拍 ——")
                print("        共用 session 是免費的，共用掃描路徑不是")
            else:
                hm = hts[len(hts) // 2] if hts else 0
                print(f"    {OK} 樓高 {hm:.2f}m、牆完整、有門窗 → 掃描涵蓋看起來足夠")
            print("    ↓ 拿雷射測距儀量最長那道牆與最長對角線，跟上面對一次 ——")
            print("      這是唯一能確認平面圖精度的方法（VIO 漂移不會自己報錯）")

    print()
    print(f"{FAIL} {problems} 項未通過" if problems else f"{OK} 全部通過")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
