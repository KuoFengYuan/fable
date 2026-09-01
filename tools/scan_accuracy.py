#!/usr/bin/env python3
"""
scan_accuracy.py — 量掃描出來的幾何精度（不需要外部真值）

    python3 tools/scan_accuracy.py <scan_dir>

## 為什麼可以在沒有真值的情況下量精度

因為建築物本身就是真值。牆是平的、地板是水平的、房間多半是方的 ——
這些都是**已知的物理約束**，而點雲偏離約束多少，就是誤差多少。

四個指標，各自對應一種失效：

  1. 牆面平面殘差   一面「應該是平的」牆，點散開多厚
                    → 位姿誤差 ＋ 深度雜訊的總和，這是最直接的精度數字
  2. 地板平面度     地板應該是一個水平面。它的殘差只含深度雜訊與**傾斜漂移**
                    （重力對齊很準，所以地板不平＝Y 方向漂移）
  3. 重複面（疊影） 同一面牆出現兩層以上相隔數公分的殼
                    → 同一表面被不同位姿投影到不同位置，是位姿誤差的直接證據
  4. 閉環誤差       走回起點時的位置落差 ÷ 走過的路徑長 = 漂移率
                    → 這是唯一能量到「全域」誤差的指標，前三個都只看局部
  5. 誤差預算       拿「沒有疊影的牆」當對照組，把 (1) 的殘差拆成
                    疊影貢獻 ＋ 其他（深度雜訊、單幀位姿抖動）
                    → 回答「疊影修好之後精度會落在哪」，以及該不該先修疊影

**沒有真值時最可信的是 (1) 與 (3)。** (1) 給你「表面有多厚」，
(3) 告訴你那個厚度是雜訊還是位姿錯位 —— 兩者的處理方式完全不同。
(5) 再把兩者的比例算出來，決定先修哪一個才划算。

想要外部驗證：拿雷射測距儀量一面牆的長度，跟 floorplan.json 的 lengthM 對照。
本工具刻意不猜那個 —— 它量的是內部一致性，那才是它有辦法誠實回答的問題。
"""

import json
import math
import pathlib
import sys

import numpy as np


def read_ply(path):
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace")
    n = next(int(l.split()[-1]) for l in header.splitlines()
             if l.startswith("element vertex"))
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    a = np.frombuffer(raw, dtype=dt, count=n, offset=end)
    return np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)


def floor_ceiling(y):
    """由 Y 直方圖的兩個主峰定出地板與天花板高度。

    **不能用百分位。** 一開始寫成 p2/p98，結果在這份實機資料上算出
    「地板平面殘差 62cm、傾斜 2.86°」—— 而真正的原因是點雲有低到地板下 3m 的
    離群點，p2 落在那條尾巴上，選出來的「地板點」其實是一堆雜訊。
    地板與天花板在直方圖上是**兩個非常明顯的峰**（這份資料：地板 19084 點、
    天花板 30607 點，其他 bin 都在 4000 以下），取峰值穩健得多。
    順便回報離群比例 —— 那本身就是一個品質指標。
    """
    bins = np.arange(y.min(), y.max() + 0.05, 0.05)
    h, _ = np.histogram(y, bins=bins)
    mid = len(h) // 2
    lo_i = int(np.argmax(h[:mid]))
    hi_i = int(np.argmax(h[mid:])) + mid
    fy, cy = float(bins[lo_i]), float(bins[hi_i])
    outside = ((y < fy - 0.30) | (y > cy + 0.30)).mean()
    return fy, cy, float(outside)


def plane_rms(pts):
    """點到最佳擬合平面的 RMS 距離（公尺）。用 PCA 最小特徵向量當法向。"""
    if len(pts) < 20:
        return None
    c = pts - pts.mean(axis=0)
    # SVD 比明確算共變異數穩定，而且點數不大時成本無所謂
    _, s, vh = np.linalg.svd(c, full_matrices=False)
    normal = vh[-1]
    return float(np.sqrt(np.mean((c @ normal) ** 2))), normal


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    d = pathlib.Path(sys.argv[1])
    pts = read_ply(d / "points.ply")
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    floor_y, ceil_y, outlier_frac = floor_ceiling(y)
    span_x = np.percentile(x, 99) - np.percentile(x, 1)
    span_z = np.percentile(z, 99) - np.percentile(z, 1)
    area = span_x * span_z
    density = len(pts) / max(area, 1e-6)

    print(f"掃描範圍  {span_x:.1f} × {span_z:.1f} m（樓高 {ceil_y - floor_y:.2f} m）")
    print(f"點數      {len(pts)}  →  每 m² {density:.0f} 點、平均點距 "
          f"{100 / math.sqrt(max(density, 1e-9)):.1f} cm")
    print(f"離群點    {outlier_frac * 100:.1f}%（落在地板下方或天花板上方 30cm 之外）")
    print()

    # ── 1. 牆面平面殘差 ──────────────────────────────────────────
    # 牆格＝同一水平格內垂直跨度大的格；再依「主方向的兩軸」把它們切成一片片牆。
    band = (y > floor_y + 0.15) & (y < ceil_y - 0.15)
    bx, by, bz = x[band], y[band], z[band]
    cell = 0.10                       # 這裡刻意用 10cm：只是要圈出牆的位置，不必細
    ij = np.stack([np.floor(bx / cell), np.floor(bz / cell)], axis=1).astype(np.int64)
    key = ij[:, 0] * (1 << 21) + ij[:, 1]
    order = np.argsort(key)
    key_s, by_s = key[order], by[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    groups = np.split(np.arange(len(key_s)), bounds)
    wall_cells = []
    for g in groups:
        if len(g) >= 2 and by_s[g].max() - by_s[g].min() >= 1.0:
            wall_cells.append(key_s[g[0]])
    wall_set = set(wall_cells)
    is_wall = np.array([k in wall_set for k in key])

    wx, wy, wz = bx[is_wall], by[is_wall], bz[is_wall]
    print(f"牆點      {len(wx)} 個（佔中段帶的 {len(wx) / max(len(bx), 1) * 100:.0f}%）")

    # 主方向：把牆點投影到旋轉後的軸上，取直方圖平方和最大的角度
    best_a, best_score = 0.0, -1.0
    for deg in np.arange(0, 90, 0.5):
        r = math.radians(deg)
        u = wx * math.cos(r) + wz * math.sin(r)
        v = -wx * math.sin(r) + wz * math.cos(r)
        s = 0.0
        for arr in (u, v):
            h = np.bincount(((arr - arr.min()) / 0.10).astype(int))
            s += float((h.astype(float) ** 2).sum())
        if s > best_score:
            best_score, best_a = s, deg
    r = math.radians(best_a)
    u = wx * math.cos(r) + wz * math.sin(r)
    v = -wx * math.sin(r) + wz * math.cos(r)
    print(f"主方向    {best_a:.1f}°")

    # 沿兩軸各自找「密集的一條線」＝一面牆，量它的平面殘差與厚度
    # 每面牆記 (殘差, 牆帶厚度, 殼間距)；殼間距 None 代表單峰（沒有疊影）
    resid, thick, seps = [], [], []
    for axis, (along, across) in enumerate(((v, u), (u, v))):
        lo, hi = across.min(), across.max()
        bins = np.arange(lo, hi + 0.05, 0.05)
        h, _ = np.histogram(across, bins=bins)
        # 只看點數夠多的 bin —— 那才是牆而不是零星家具
        strong = np.flatnonzero(h > max(30, np.percentile(h[h > 0], 90)))
        merged, i = [], 0
        while i < len(strong):
            j = i
            while j + 1 < len(strong) and strong[j + 1] - strong[j] <= 2:
                j += 1
            merged.append((strong[i], strong[j]))
            i = j + 1
        for a0, a1 in merged:
            c0, c1 = bins[a0], bins[min(a1 + 1, len(bins) - 1)]
            sel = (across >= c0) & (across <= c1)
            if sel.sum() < 200:
                continue
            sub = np.stack([wx[sel], wy[sel], wz[sel]], axis=1)
            out = plane_rms(sub)
            if out is None:
                continue
            rms, _ = out
            resid.append(rms)
            thick.append(c1 - c0)
            # 疊影：同一條牆帶內，跨牆方向的分佈若是雙峰就是兩層殼。
            # 除了「有沒有」，也要量「相隔多遠」—— 那個距離才換算得出誤差貢獻。
            hh, _ = np.histogram(across[sel], bins=np.arange(c0, c1 + 0.02, 0.02))
            peaks = [k for k in range(1, len(hh) - 1)
                     if hh[k] > hh[k - 1] and hh[k] > hh[k + 1] and hh[k] > hh.max() * 0.35]
            if len(peaks) >= 2:
                # 取最高的兩個峰＝兩層主要的殼（三峰以上時，其餘是零星雜訊）
                top = sorted(peaks, key=lambda k: hh[k], reverse=True)[:2]
                seps.append(abs(top[0] - top[1]) * 0.02)
            else:
                seps.append(None)

    if resid:
        resid = np.array(resid)
        thick = np.array(thick)
        dbl = np.array([s is not None for s in seps])
        sep = np.array([s for s in seps if s is not None])
        print()
        print("── 1. 牆面平面殘差（表面該是平的，散開多厚就是誤差）──")
        print(f"   {len(resid)} 面牆：中位數 {np.median(resid) * 100:.1f} cm、"
              f"p90 {np.percentile(resid, 90) * 100:.1f} cm、最差 {resid.max() * 100:.1f} cm")
        print(f"   牆帶厚度 中位數 {np.median(thick) * 100:.0f} cm"
              f"（真實牆厚約 10~15cm，超出的部分就是位置散開量）")
        print(f"   判讀：<1.5cm 良好 ／ 1.5~4cm 可用 ／ >4cm 位姿或深度有明顯問題")
        print()
        print("── 3. 重複面（疊影）──")
        print(f"   {int(dbl.sum())}/{len(resid)} 面牆在跨牆方向上呈現雙峰以上"
              f" → 同一表面被投影到不只一個位置")
        if len(sep):
            print(f"   殼間距 中位數 {np.median(sep) * 100:.1f} cm、"
                  f"p90 {np.percentile(sep, 90) * 100:.1f} cm")

        # ── 5. 誤差預算 ──────────────────────────────────────────
        # 把 (1) 的殘差拆成「疊影」與「其他」兩份。做法不是套公式，而是
        # **拿單峰牆當對照組** —— 它們身上沒有疊影，所以殘差就是深度雜訊
        # ＋局部位姿抖動的總和，那正好是「疊影修好之後會落在哪」的答案。
        # 疊影的貢獻則由兩組相減（平方差開根號）得到，不假設殼有幾層。
        #
        # 另外印出「最強兩峰間距」當**形狀**的診斷，不當誤差來源用：
        #   間距 ≈ 疊影貢獻 → 乾淨的兩層殼，成因是兩趟之間有固定偏移
        #                     （掠射角深度偏差、或兩段軌跡沒對齊）
        #   間距 ≪ 疊影貢獻 → 不是兩層而是多層／糊開，成因是每幀抖動
        #                     在多趟之間各自累積，要治的是單幀品質不是對齊
        # 這兩種的處理方式完全不同，所以值得分辨。
        if len(sep) >= 3 and (~dbl).sum() >= 3:
            base = float(np.median(resid[~dbl]))
            meas = float(np.median(resid[dbl]))
            ghost = math.sqrt(max(meas ** 2 - base ** 2, 0.0))
            gap = float(np.median(sep))
            print()
            print("── 5. 誤差預算（疊影拿掉之後會落在哪）──")
            print(f"   單峰牆 {int((~dbl).sum())} 面：殘差中位數 {base * 100:.1f} cm"
                  f"  ← 深度雜訊＋位姿抖動，這是修好疊影後的地板")
            print(f"   雙峰牆 {int(dbl.sum())} 面：殘差中位數 {meas * 100:.1f} cm")
            print(f"   疊影貢獻 σ = √({meas * 100:.1f}² − {base * 100:.1f}²)"
                  f" = {ghost * 100:.1f} cm")
            print()
            if gap / 2 < ghost * 0.6:
                print(f"   形狀診斷：最強兩峰只隔 {gap * 100:.1f}cm（σ 僅 {gap * 50:.1f}cm），"
                      f"遠小於疊影貢獻 {ghost * 100:.1f}cm")
                print(f"   → **不是乾淨的兩層殼，而是多層／糊開的分佈**。"
                      f"成因偏向每幀深度與位姿抖動，")
                print(f"      而不是兩趟之間的固定偏移 —— 對齊類的修法（掠射角、BA）"
                      f"幫助有限，要先治單幀品質。")
            else:
                print(f"   形狀診斷：最強兩峰隔 {gap * 100:.1f}cm（σ {gap * 50:.1f}cm），"
                      f"與疊影貢獻 {ghost * 100:.1f}cm 同量級")
                print(f"   → **是乾淨的兩層殼**：兩趟之間有固定偏移。"
                      f"掠射角過濾／BA／重新對齊會直接見效。")
        elif len(sep):
            print("   （單峰或雙峰牆不足 3 面，樣本太少，不做誤差預算拆解）")
    else:
        print("   找不到足夠密集的牆帶 —— 點太稀或牆沒掃到")

    # ── 2. 地板平面度 ────────────────────────────────────────────
    fsel = np.abs(y - floor_y) < 0.10
    if fsel.sum() > 500:
        out = plane_rms(pts[fsel])
        if out:
            rms, n = out
            tilt = math.degrees(math.acos(min(1.0, abs(n[1]))))
            print()
            print("── 2. 地板平面度（重力對齊很準，所以不平＝Y 方向漂移）──")
            print(f"   殘差 {rms * 100:.1f} cm、法向偏離鉛直 {tilt:.2f}°"
                  f"（{fsel.sum()} 點）")
            print(f"   判讀：殘差 <2cm 且傾角 <0.5° 表示垂直方向沒有累積漂移")

    # ── 4. 閉環誤差 ──────────────────────────────────────────────
    pj = d / "poses_refined.jsonl"
    if pj.exists():
        P = []
        for line in pj.read_text().splitlines():
            t = json.loads(line).get("transform")
            if t and len(t) == 16:
                P.append([t[3], t[7], t[11]])
        if len(P) > 10:
            P = np.array(P)
            path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
            gap = float(np.linalg.norm(P[-1] - P[0]))
            near = np.linalg.norm(P - P[0], axis=1)
            closed = near[len(P) // 2:].min()
            print()
            print("── 4. 閉環誤差（唯一能量到全域誤差的指標）──")
            print(f"   軌跡 {len(P)} 幀、路徑長 {path:.1f} m")
            print(f"   終點離起點 {gap:.2f} m；後半段最接近起點時差 {closed:.2f} m")
            if closed < 2.0:
                print(f"   → 有回到起點附近，漂移率約 {closed / path * 100:.2f}%"
                      f"（室內 VIO 常見 0.5~2%）")
            else:
                print("   → **沒有走回起點**，所以量不到全域漂移。"
                      "遠端的累積誤差留在資料裡而且不會有任何警告 —— "
                      "下次掃描請走回起點閉環。")


if __name__ == "__main__":
    main()
