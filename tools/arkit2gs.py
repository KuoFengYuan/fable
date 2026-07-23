#!/usr/bin/env python3
"""ARKit 掃描資料 → Nerfstudio / COLMAP(Inria 3DGS) 訓練資料集。

輸入：fable App 匯出的掃描資料夾（或解壓後的 .zip）
    scan_xxx/
    ├── images/frame_00001.jpg ...     # sensor 方向 RGB
    ├── depth/frame_00001_depth.bin    # float32 raw（選用）
    │        frame_00001_conf.bin      # uint8 raw（選用）
    ├── poses.jsonl                    # 每行一幀：c2w(row-major)、內參、時間戳、曝光
    ├── points.ply                     # LiDAR 彩色點雲（ARKit 世界座標）
    └── meta.json

輸出（--format 可選其一或 both）：
    nerfstudio/  transforms.json + images/ + sparse_pc.ply     → ns-train splatfacto
    colmap/      images/ + sparse/0/{cameras,images,points3D}  → Inria 3DGS train.py

用法範例：
    python arkit2gs.py scan_20260721_103000 -o dataset --format both
    python arkit2gs.py scan_x -o out --orientation portrait --world-up z --max-blur 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import colmap_io
import geometry as G
import ply_io


# ---------------------------------------------------------------------------
# 讀取手機端資料
# ---------------------------------------------------------------------------

def load_scan(scan_dir: Path):
    # 優先讀錨點修正後姿態（App 匯出時產生，與 sparse/0 一致）
    poses_path = scan_dir / "poses_refined.jsonl"
    if poses_path.exists():
        print("ℹ️  使用修正後姿態 poses_refined.jsonl")
    else:
        poses_path = scan_dir / "poses.jsonl"
    if not poses_path.exists():
        sys.exit(f"找不到 {poses_path}（請指向 fable App 匯出的掃描資料夾）")
    records = []
    with open(poses_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️  poses.jsonl 第 {line_no} 行毀損，略過（可能是中斷時的殘行）")
    meta = {}
    meta_path = scan_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    records.sort(key=lambda r: r["id"])
    return records, meta


def filter_records(records: list[dict], scan_dir: Path, args) -> list[dict]:
    """時間戳單調性檢查 + 品質過濾 + 檔案存在性同步。"""
    out = []
    last_ts = -np.inf
    dropped_blur = dropped_ts = dropped_missing = 0
    for r in records:
        if not (scan_dir / "images" / r["imageFile"]).exists():
            dropped_missing += 1
            continue
        if r["timestamp"] <= last_ts:
            dropped_ts += 1          # 時間戳倒退＝資料異常（理論上不會發生）
            continue
        if args.max_blur is not None and r.get("estimatedBlurPx", 0) > args.max_blur:
            dropped_blur += 1
            continue
        last_ts = r["timestamp"]
        out.append(r)
    if args.max_frames and len(out) > args.max_frames:
        idx = np.linspace(0, len(out) - 1, args.max_frames).round().astype(int)
        out = [out[i] for i in idx]
    if dropped_missing:
        print(f"⚠️  {dropped_missing} 幀影像檔缺失，已略過")
    if dropped_ts:
        print(f"⚠️  {dropped_ts} 幀時間戳非單調，已略過")
    if dropped_blur:
        print(f"ℹ️  {dropped_blur} 幀模糊估計超過 {args.max_blur}px，已過濾")
    return out


# ---------------------------------------------------------------------------
# 幾何前處理：portrait 正規化 / 世界軸重定向
# ---------------------------------------------------------------------------

def preprocess(records, args):
    """回傳 (c2ws_gl (N,4,4), Ks [(fx,fy,cx,cy)], (w,h), image_transpose)。"""
    c2ws, Ks = [], []
    w = records[0]["intrinsics"]["width"]
    h = records[0]["intrinsics"]["height"]
    transpose = None
    for r in records:
        c2w = np.array(r["transform"], dtype=np.float64).reshape(4, 4)
        intr = r["intrinsics"]
        K = (intr["fx"], intr["fy"], intr["cx"], intr["cy"])
        if args.orientation == "portrait":
            c2w, K, w2, h2 = G.rotate_portrait(c2w, K, intr["width"], intr["height"])
            w, h = w2, h2
        if args.world_up == "z":
            c2w = G.yup_to_zup_c2w(c2w)
        c2ws.append(c2w)
        Ks.append(K)
    if args.orientation == "portrait":
        from PIL import Image as PILImage
        transpose = PILImage.Transpose.ROTATE_270   # 順時針 90°
    return np.stack(c2ws), Ks, (w, h), transpose


def load_points(scan_dir: Path, records, args):
    """點雲來源：--rebuild-points 從深度圖重建，否則直接用手機端 points.ply。"""
    if args.rebuild_points:
        xyz, rgb = rebuild_points_from_depth(scan_dir, records, args)
        if xyz is None:
            print("⚠️  無深度資料可重建，退回 points.ply")
        else:
            return xyz, rgb
    ply = scan_dir / "points.ply"
    if not ply.exists():
        print("⚠️  掃描內無 points.ply（無 LiDAR？）— 將以隨機點雲替代（品質較差）")
        return None, None
    xyz, rgb = ply_io.read_ply(ply)
    if rgb is None:
        rgb = np.full((len(xyz), 3), 160, np.uint8)
    return xyz, rgb


def rebuild_points_from_depth(scan_dir: Path, records, args):
    """由深度圖 + 姿態重建彩色點雲（比手機端即時累積更密、可重調 voxel）。"""
    from PIL import Image as PILImage

    voxel = args.voxel
    vox: dict[tuple, int] = {}
    all_pts, all_rgb = [], []
    n_frames = 0
    for r in records:
        depth_file = r.get("depthFile")
        if not depth_file:
            continue
        dpath = scan_dir / "depth" / depth_file
        if not dpath.exists():
            continue
        dw, dh = r["depthWidth"], r["depthHeight"]
        depth = np.fromfile(dpath, np.float32).reshape(dh, dw)
        conf_file = r.get("confidenceFile")
        conf = None
        if conf_file and (scan_dir / "depth" / conf_file).exists():
            conf = np.fromfile(scan_dir / "depth" / conf_file, np.uint8).reshape(dh, dw)

        intr = r["intrinsics"]
        sx, sy = dw / intr["width"], dh / intr["height"]
        fx, fy = intr["fx"] * sx, intr["fy"] * sy
        cx, cy = intr["cx"] * sx, intr["cy"] * sy

        stride = max(1, args.rebuild_stride)
        us, vs = np.meshgrid(np.arange(0, dw, stride), np.arange(0, dh, stride))
        z = depth[vs, us]
        mask = np.isfinite(z) & (z > 0.15) & (z < 6.0)
        if conf is not None:
            mask &= conf[vs, us] >= 2
        if not mask.any():
            continue
        z = z[mask]
        u = us[mask].astype(np.float64)
        v = vs[mask].astype(np.float64)
        # CV 反投影 → 翻 Y/Z 回 GL 相機系 → 世界（poses.jsonl 原生 ARKit 座標）
        pc = np.stack([(u - cx) / fx * z, -((v - cy) / fy * z), -z], axis=1)
        c2w = np.array(r["transform"], np.float64).reshape(4, 4)
        pw = pc @ c2w[:3, :3].T + c2w[:3, 3]

        img = np.asarray(PILImage.open(scan_dir / "images" / r["imageFile"])
                         .resize((dw, dh), PILImage.BILINEAR))
        colors = img[vs, us][mask] if img.ndim == 3 else np.full((len(z), 3), 160, np.uint8)

        # 每幀先 voxel 去重，再併入全域字典
        keys = np.floor(pw / voxel).astype(np.int64)
        _, first_idx = np.unique(keys, axis=0, return_index=True)
        for i in first_idx:
            k = (keys[i, 0], keys[i, 1], keys[i, 2])
            if k not in vox:
                vox[k] = 1
                all_pts.append(pw[i])
                all_rgb.append(colors[i])
        n_frames += 1
        if len(all_pts) >= args.max_points:
            break
    if not all_pts:
        return None, None
    print(f"ℹ️  由 {n_frames} 幀深度重建 {len(all_pts):,} 點（voxel {voxel*100:.0f}cm）")
    return np.array(all_pts), np.array(all_rgb, np.uint8)


# ---------------------------------------------------------------------------
# 影像搬移
# ---------------------------------------------------------------------------

def materialize_images(records, scan_dir: Path, out_images: Path, transpose):
    """複製（或硬連結）影像；portrait 模式下重新編碼為旋轉後的影像。"""
    out_images.mkdir(parents=True, exist_ok=True)
    if transpose is not None:
        from PIL import Image as PILImage
        for r in records:
            src = scan_dir / "images" / r["imageFile"]
            with PILImage.open(src) as im:
                im.transpose(transpose).save(out_images / r["imageFile"], quality=95)
        return
    for r in records:
        src = scan_dir / "images" / r["imageFile"]
        dst = out_images / r["imageFile"]
        if dst.exists():
            continue
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# 輸出：Nerfstudio
# ---------------------------------------------------------------------------

def write_nerfstudio(out_dir: Path, records, c2ws, Ks, wh, xyz, rgb, scan_dir, transpose):
    out_dir.mkdir(parents=True, exist_ok=True)
    materialize_images(records, scan_dir, out_dir / "images", transpose)
    w, h = wh

    fls = np.array(Ks)
    root = {
        "camera_model": "OPENCV",
        "w": w, "h": h,
        "fl_x": float(np.median(fls[:, 0])),
        "fl_y": float(np.median(fls[:, 1])),
        "cx": float(np.median(fls[:, 2])),
        "cy": float(np.median(fls[:, 3])),
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": [],
    }
    for r, c2w, K in zip(records, c2ws, Ks):
        root["frames"].append({
            "file_path": f"images/{r['imageFile']}",
            "transform_matrix": c2w.tolist(),        # c2w，GL 相機慣例
            "fl_x": float(K[0]), "fl_y": float(K[1]),
            "cx": float(K[2]), "cy": float(K[3]),
            "w": w, "h": h,
            "timestamp": r["timestamp"],
        })
    if xyz is not None:
        ply_io.write_ply(out_dir / "sparse_pc.ply", xyz, rgb)
        root["ply_file_path"] = "sparse_pc.ply"      # splatfacto 的初始化種子點
    (out_dir / "transforms.json").write_text(json.dumps(root, indent=2))
    print(f"✅ nerfstudio 資料集 → {out_dir}（{len(records)} 幀）")
    print(f"   訓練：ns-train splatfacto --data {out_dir}")


# ---------------------------------------------------------------------------
# 輸出：COLMAP / Inria 3DGS
# ---------------------------------------------------------------------------

def write_colmap(out_dir: Path, records, c2ws, Ks, wh, xyz, rgb, scan_dir, transpose,
                 also_txt: bool, flip_world_up: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    materialize_images(records, scan_dir, out_dir / "images", transpose)
    sparse = out_dir / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    w, h = wh

    # 內參逐幀差異極小（AF 鎖定後），取中位數做單一 PINHOLE 相機
    fls = np.array(Ks)
    cam = colmap_io.Camera(id=1, model="PINHOLE", width=w, height=h,
                           params=[float(np.median(fls[:, i])) for i in range(4)])

    # 世界上方向對齊 COLMAP 慣例（ARKit +Y-up 直接匯入 3DGS 會上下顛倒）；
    # 姿態與點雲成對翻轉、幾何一致
    if flip_world_up:
        c2ws = np.stack([G.flip_world_up_c2w(c2w) for c2w in c2ws])

    images = []
    for i, (r, c2w) in enumerate(zip(records, c2ws), start=1):
        qvec, tvec = G.colmap_qt_from_c2w_gl(c2w)    # → w2c（CV 慣例）
        images.append(colmap_io.Image(id=i, qvec=qvec, tvec=tvec,
                                      camera_id=1, name=r["imageFile"]))

    if xyz is None:
        # 沒有點雲時以相機包圍盒內的隨機點替代（Inria loader 需要非空 points3D）
        lo, hi = c2ws[:, :3, 3].min(0) - 1, c2ws[:, :3, 3].max(0) + 1
        xyz = np.random.default_rng(0).uniform(lo, hi, (30_000, 3))
        rgb = np.full((len(xyz), 3), 128, np.uint8)
        print("⚠️  以 30k 隨機點替代 points3D（建議改用 --rebuild-points 或補拍 LiDAR）")
    elif flip_world_up:
        xyz = G.flip_world_up_points(xyz)

    colmap_io.write_cameras_bin(sparse / "cameras.bin", [cam])
    colmap_io.write_images_bin(sparse / "images.bin", images)
    colmap_io.write_points3D_bin(sparse / "points3D.bin", xyz, rgb)
    if also_txt:
        colmap_io.write_cameras_txt(sparse / "cameras.txt", [cam])
        colmap_io.write_images_txt(sparse / "images.txt", images)
        colmap_io.write_points3D_txt(sparse / "points3D.txt", xyz, rgb)
    print(f"✅ COLMAP 資料集 → {out_dir}（{len(images)} 幀，{len(xyz):,} 點）")
    print(f"   訓練（Inria 3DGS）：python train.py -s {out_dir}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path, help="fable App 匯出的掃描資料夾")
    ap.add_argument("-o", "--out", type=Path, required=True, help="輸出資料夾")
    ap.add_argument("--format", nargs="+", default=["nerfstudio"],
                    choices=["nerfstudio", "colmap", "both"])
    ap.add_argument("--world-up", choices=["y", "z"], default="y",
                    help="世界上方向：y=ARKit 原生（預設，nerfstudio 會自動歸一），z=instant-ngp 習慣")
    ap.add_argument("--orientation", choices=["sensor", "portrait"], default="sensor",
                    help="sensor=保持原始 landscape（預設，最穩），portrait=旋正直式影像並同步修正內外參")
    ap.add_argument("--max-blur", type=float, default=None,
                    help="剔除模糊估計超過此像素數的幀（例如 3.0）")
    ap.add_argument("--max-frames", type=int, default=None, help="等距抽取至多 N 幀")
    ap.add_argument("--rebuild-points", action="store_true",
                    help="從深度圖重建點雲（更密、可重調 voxel），取代手機端 points.ply")
    ap.add_argument("--voxel", type=float, default=0.02, help="重建點雲 voxel 大小（公尺）")
    ap.add_argument("--rebuild-stride", type=int, default=4, help="重建時的深度取樣步長")
    ap.add_argument("--max-points", type=int, default=1_000_000)
    ap.add_argument("--txt", action="store_true", help="COLMAP 額外輸出 txt 版（除錯用）")
    ap.add_argument("--colmap-flip-up", action=argparse.BooleanOptionalAction, default=True,
                    help="COLMAP 輸出繞世界 X 軸翻 180° 對齊 3DGS 慣例（預設開啟，修正模型上下顛倒）；"
                         "若你的 viewer 反而變顛倒用 --no-colmap-flip-up")
    args = ap.parse_args()

    formats = set(args.format)
    if "both" in formats:
        formats = {"nerfstudio", "colmap"}

    records, meta = load_scan(args.scan_dir)
    print(f"讀入 {len(records)} 幀（裝置：{meta.get('device', '?')}，"
          f"LiDAR：{meta.get('lidarAvailable', '?')}）")
    records = filter_records(records, args.scan_dir, args)
    if not records:
        sys.exit("沒有可用的幀")

    c2ws, Ks, wh, transpose = preprocess(records, args)

    xyz, rgb = load_points(args.scan_dir, records, args)
    if xyz is not None and args.world_up == "z":
        xyz = G.yup_to_zup_points(xyz)

    dt = np.diff([r["timestamp"] for r in records])
    print(f"幀間隔：中位 {np.median(dt):.2f}s / 最大 {dt.max():.2f}s；"
          f"軌跡長度 {np.linalg.norm(np.diff(c2ws[:, :3, 3], axis=0), axis=1).sum():.1f}m")

    if "nerfstudio" in formats:
        write_nerfstudio(args.out / "nerfstudio", records, c2ws, Ks, wh,
                         xyz, rgb, args.scan_dir, transpose)
    if "colmap" in formats:
        # world-up z（instant-ngp）與 flip（COLMAP 慣例）為互斥的重定向意圖，前者優先
        flip = args.colmap_flip_up and args.world_up != "z"
        write_colmap(args.out / "colmap", records, c2ws, Ks, wh,
                     xyz, rgb, args.scan_dir, transpose, args.txt, flip_world_up=flip)


if __name__ == "__main__":
    main()
