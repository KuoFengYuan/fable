#!/usr/bin/env python3
"""產生合成 ARKit 掃描資料（與 fable App 匯出格式完全一致）。

用途：
  1. 在沒有 iPhone 的機器上端到端測試 arkit2gs.py / validate_dataset.py
  2. test_math.py 的整合測試 fixture

模擬一台相機繞原點物件環拍：ARKit 世界（Y-up、公尺）、GL 相機慣例 c2w。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ply_io


def look_at_c2w_gl(pos: np.ndarray, target: np.ndarray, up=(0, 1, 0)) -> np.ndarray:
    """GL 慣例 camera-to-world：相機 -Z 指向 target。"""
    z = pos - target
    z = z / np.linalg.norm(z)                  # GL 的 +Z 指向相機後方
    x = np.cross(np.asarray(up, float), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
    return c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--radius", type=float, default=2.0)
    args = ap.parse_args()

    from PIL import Image

    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)

    W, H = args.width, args.height
    fx = fy = 0.85 * W
    cx, cy = W / 2 + 3.0, H / 2 - 2.0          # 刻意偏離中心，測試主點處理
    dw, dh = W // 4, H // 4
    target = np.array([0.0, 0.4, 0.0])         # 物件中心（略高於地面）

    t0 = 1000.0
    records = []
    for i in range(args.frames):
        theta = 2 * np.pi * i / args.frames
        pos = target + np.array([args.radius * np.cos(theta),
                                 0.35 + 0.25 * np.sin(2 * theta),
                                 args.radius * np.sin(theta)])
        c2w = look_at_c2w_gl(pos, target)

        # 影像：漸層 + 依幀變化的色塊（讓每幀可辨識）
        img = np.zeros((H, W, 3), np.uint8)
        img[..., 0] = np.linspace(30, 225, W, dtype=np.uint8)[None, :]
        img[..., 1] = np.linspace(30, 225, H, dtype=np.uint8)[:, None]
        img[..., 2] = int(255 * i / args.frames)
        img[10:60, 10:110] = (255 - int(200 * i / args.frames), 40, 200)
        name = f"frame_{i + 1:05d}"
        Image.fromarray(img).save(out / "images" / f"{name}.jpg", quality=90)

        # 深度：常數 radius 平面 + 輕微噪聲；信心全高
        depth = np.full((dh, dw), args.radius, np.float32)
        depth += np.random.default_rng(i).normal(0, 0.005, depth.shape).astype(np.float32)
        depth.tofile(out / "depth" / f"{name}_depth.bin")
        np.full((dh, dw), 2, np.uint8).tofile(out / "depth" / f"{name}_conf.bin")

        records.append({
            "id": i + 1,
            "timestamp": t0 + i * 0.45,
            "transform": [float(v) for v in c2w.reshape(-1)],   # row-major
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                           "width": W, "height": H},
            "exposureDuration": 1 / 120,
            "exposureOffsetEV": 0.0,
            "ambientLux": 900.0,
            "estimatedBlurPx": float(0.3 + 0.1 * (i % 5)),
            "imageFile": f"{name}.jpg",
            "depthFile": f"{name}_depth.bin",
            "confidenceFile": f"{name}_conf.bin",
            "depthWidth": dw,
            "depthHeight": dh,
        })

    with open(out / "poses.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    (out / "meta.json").write_text(json.dumps({
        "app": "fable-gs-capture", "version": 1,
        "device": "synthetic", "osVersion": "-",
        "startedAt": "2026-07-21T00:00:00Z", "mode": "object",
        "worldAlignment": "gravity",
        "cameraConvention": "arkit_gl_c2w_row_major",
        "imageOrientation": "sensor_landscape_right",
        "depthFormat": "float32_raw_little_endian",
        "lidarAvailable": True,
    }, indent=2))

    # 物件點雲：中心球殼，顏色 = 法向
    rng = np.random.default_rng(7)
    d = rng.normal(size=(4000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    xyz = target + d * 0.3
    rgb = ((d * 0.5 + 0.5) * 255).astype(np.uint8)
    ply_io.write_ply(out / "points.ply", xyz, rgb)

    print(f"✅ 合成掃描 → {out}（{args.frames} 幀，{len(xyz)} 點）")


if __name__ == "__main__":
    main()
