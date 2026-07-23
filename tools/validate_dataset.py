#!/usr/bin/env python3
"""掃描資料健檢：時間戳、姿態連續性、內參合理性、點雲-相機幾何自洽。

支援兩種輸入：
  - 手機原始掃描資料夾（poses.jsonl）
  - 轉換後的 nerfstudio 資料夾（transforms.json）

    python validate_dataset.py scan_20260721_103000
    python validate_dataset.py dataset/nerfstudio --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import geometry as G
import ply_io

WARN = "⚠️ "
OK = "✅"
FAIL = "❌"


def load_any(path: Path):
    """回傳 (c2ws_gl, Ks, (w,h), timestamps or None, image_names, ply_path or None)。"""
    poses = path / "poses_refined.jsonl"   # 優先使用錨點修正後姿態
    if not poses.exists():
        poses = path / "poses.jsonl"
    if poses.exists():
        records = [json.loads(l) for l in open(poses) if l.strip()]
        records.sort(key=lambda r: r["id"])
        c2ws = np.stack([np.array(r["transform"]).reshape(4, 4) for r in records])
        Ks = [(r["intrinsics"]["fx"], r["intrinsics"]["fy"],
               r["intrinsics"]["cx"], r["intrinsics"]["cy"]) for r in records]
        wh = (records[0]["intrinsics"]["width"], records[0]["intrinsics"]["height"])
        ts = np.array([r["timestamp"] for r in records])
        names = [r["imageFile"] for r in records]
        ply = path / "points.ply"
        return c2ws, Ks, wh, ts, names, (ply if ply.exists() else None), path / "images"
    if (path / "transforms.json").exists():
        root = json.loads((path / "transforms.json").read_text())
        frames = root["frames"]
        c2ws = np.stack([np.array(f["transform_matrix"]) for f in frames])
        Ks = [(f.get("fl_x", root.get("fl_x")), f.get("fl_y", root.get("fl_y")),
               f.get("cx", root.get("cx")), f.get("cy", root.get("cy"))) for f in frames]
        wh = (frames[0].get("w", root.get("w")), frames[0].get("h", root.get("h")))
        ts = np.array([f["timestamp"] for f in frames]) if "timestamp" in frames[0] else None
        names = [Path(f["file_path"]).name for f in frames]
        ply = root.get("ply_file_path")
        return (c2ws, Ks, wh, ts, names,
                (path / ply if ply else None), path / "images")
    sys.exit(f"{path} 內找不到 poses.jsonl 或 transforms.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--plot", action="store_true", help="輸出軌跡俯視圖 trajectory_top.png（需 matplotlib）")
    args = ap.parse_args()

    c2ws, Ks, wh, ts, names, ply_path, images_dir = load_any(args.path)
    n = len(c2ws)
    problems = 0
    print(f"── {args.path}：{n} 幀，解析度 {wh[0]}×{wh[1]} ──\n")

    # 1. 影像檔
    missing = [nm for nm in names if not (images_dir / nm).exists()]
    if missing:
        problems += 1
        print(f"{FAIL} {len(missing)} 幀影像缺失（如 {missing[0]}）")
    else:
        print(f"{OK} 影像檔完整（{n} 張）")

    # 2. 時間戳
    if ts is not None:
        dt = np.diff(ts)
        if (dt <= 0).any():
            problems += 1
            print(f"{FAIL} 時間戳非單調：{int((dt <= 0).sum())} 處倒退")
        else:
            print(f"{OK} 時間戳單調；幀間隔 中位 {np.median(dt):.2f}s / 最大 {dt.max():.2f}s")

    # 3. 姿態連續性（跳幀 = ARKit 追蹤重定位痕跡，會毀掉訓練）
    R = c2ws[:, :3, :3]
    ortho_err = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
    print(f"{OK if ortho_err < 1e-3 else FAIL} 旋轉正交性誤差 {ortho_err:.2e}")
    if ortho_err >= 1e-3:
        problems += 1

    trans = np.linalg.norm(np.diff(c2ws[:, :3, 3], axis=0), axis=1)
    jumps = int((trans > 0.5).sum())
    if jumps:
        problems += 1
        print(f"{WARN}{jumps} 處相鄰幀平移 > 0.5m（疑似追蹤跳變，建議剔除該段或重掃）")
    else:
        print(f"{OK} 姿態連續：平移 中位 {np.median(trans)*100:.1f}cm / 最大 {trans.max()*100:.1f}cm")
    print(f"   軌跡總長 {trans.sum():.1f}m")

    # 4. 內參
    fx, fy = np.array([k[0] for k in Ks]), np.array([k[1] for k in Ks])
    cx, cy = np.array([k[2] for k in Ks]), np.array([k[3] for k in Ks])
    aspect_ok = np.abs(fx / fy - 1).max() < 0.02
    center_ok = (np.abs(cx / wh[0] - 0.5).max() < 0.05 and np.abs(cy / wh[1] - 0.5).max() < 0.05)
    drift = (fx.max() - fx.min()) / fx.mean()
    print(f"{OK if aspect_ok else WARN} fx/fy 比例（近 1）：{(fx/fy).mean():.4f}")
    print(f"{OK if center_ok else WARN} 主點居中：cx/W={cx.mean()/wh[0]:.3f}, cy/H={cy.mean()/wh[1]:.3f}")
    print(f"{OK if drift < 0.01 else WARN} 焦距漂移 {drift*100:.2f}%（>1% 表示拍攝中對焦有變動）")

    # 5. 點雲 ↔ 相機自洽：點應大多落在相機前方且在畫面內
    if ply_path is not None:
        xyz, _ = ply_io.read_ply(ply_path)
        sample = xyz[np.random.default_rng(0).choice(len(xyz), min(5000, len(xyz)), replace=False)]
        frame_idx = np.linspace(0, n - 1, min(12, n)).astype(int)
        in_front, in_view = [], []
        for i in frame_idx:
            uv, depth = G.project_gl(Ks[i], c2ws[i], sample)
            front = depth > 0
            visible = front & (uv[:, 0] >= 0) & (uv[:, 0] < wh[0]) & \
                      (uv[:, 1] >= 0) & (uv[:, 1] < wh[1])
            in_front.append(front.mean())
            in_view.append(visible.mean())
        front_ratio, view_ratio = np.mean(in_front), np.mean(in_view)
        status = OK if front_ratio > 0.5 else FAIL
        if front_ratio <= 0.5:
            problems += 1
        print(f"{status} 點雲幾何：{front_ratio*100:.0f}% 在相機前方、{view_ratio*100:.0f}% 落在畫面內"
              f"（{len(xyz):,} 點）")
        if front_ratio <= 0.5:
            print("     → 比例過低通常代表座標慣例錯了（Y/Z 翻轉遺漏）")
    else:
        print(f"{WARN}無點雲檔，略過幾何自洽檢查")

    # 6. 視角基線充足性（3DGS 需要視差）
    centroid = c2ws[:, :3, 3].mean(0)
    spread = np.linalg.norm(c2ws[:, :3, 3] - centroid, axis=1)
    print(f"{OK if spread.max() > 0.3 else WARN} 相機位置分佈半徑 {spread.max():.2f}m"
          f"{'（過小：幾乎原地旋轉，3DGS 幾何將退化）' if spread.max() <= 0.3 else ''}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            p = c2ws[:, :3, 3]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(p[:, 0], p[:, 2], "-o", ms=2, lw=0.8)
            fwd = -c2ws[:, :3, 2]
            ax.quiver(p[:, 0], p[:, 2], fwd[:, 0], fwd[:, 2], width=2e-3, scale=25, color="tab:red")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
            ax.set_title("camera trajectory (top view, ARKit world)")
            ax.set_aspect("equal")
            out_png = args.path / "trajectory_top.png"
            fig.savefig(out_png, dpi=140, bbox_inches="tight")
            print(f"\n軌跡俯視圖 → {out_png}")
        except ImportError:
            print(f"{WARN}未安裝 matplotlib，略過 --plot")

    print(f"\n{'═' * 46}")
    if problems:
        print(f"{FAIL} 發現 {problems} 個問題，建議修正後再訓練")
        sys.exit(1)
    print(f"{OK} 資料集健檢通過，可直接送入 3DGS 訓練")


if __name__ == "__main__":
    main()
