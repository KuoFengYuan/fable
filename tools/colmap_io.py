"""COLMAP sparse model 讀寫（cameras / images / points3D，bin 與 txt 雙格式）。

二進位格式與 COLMAP scripts/python/read_write_model.py 完全一致，
Inria 3DGS 的 dataset_readers.py 直接可讀。
images 的姿態為 world-to-camera（OpenCV 相機慣例）、四元數順序 (qw, qx, qy, qz)。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

CAMERA_MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1}
CAMERA_MODEL_NAMES = {v: k for k, v in CAMERA_MODEL_IDS.items()}
CAMERA_MODEL_NUM_PARAMS = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4}


@dataclass
class Camera:
    id: int
    model: str          # "PINHOLE": params = [fx, fy, cx, cy]
    width: int
    height: int
    params: list[float]


@dataclass
class Image:
    id: int
    qvec: np.ndarray    # (qw, qx, qy, qz)
    tvec: np.ndarray    # (3,)
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------

def write_cameras_bin(path, cameras: list[Camera]):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("<iiQQ", cam.id, CAMERA_MODEL_IDS[cam.model],
                                cam.width, cam.height))
            f.write(struct.pack(f"<{len(cam.params)}d", *cam.params))


def read_cameras_bin(path) -> list[Camera]:
    cameras = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            model = CAMERA_MODEL_NAMES[model_id]
            k = CAMERA_MODEL_NUM_PARAMS[model]
            params = list(struct.unpack(f"<{k}d", f.read(8 * k)))
            cameras.append(Camera(cid, model, w, h, params))
    return cameras


def write_cameras_txt(path, cameras: list[Camera]):
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n"
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                f"# Number of cameras: {len(cameras)}\n")
        for cam in cameras:
            params = " ".join(repr(float(p)) for p in cam.params)
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params}\n")


def read_cameras_txt(path) -> list[Camera]:
    cameras = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            cameras.append(Camera(int(tokens[0]), tokens[1], int(tokens[2]),
                                  int(tokens[3]), [float(t) for t in tokens[4:]]))
    return cameras


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------

def write_images_bin(path, images: list[Image]):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<i", im.id))
            f.write(struct.pack("<4d", *im.qvec))
            f.write(struct.pack("<3d", *im.tvec))
            f.write(struct.pack("<i", im.camera_id))
            f.write(im.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(im.xys)))
            for xy, pid in zip(im.xys, im.point3D_ids):
                f.write(struct.pack("<ddq", xy[0], xy[1], pid))


def read_images_bin(path) -> list[Image]:
    images = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (iid,) = struct.unpack("<i", f.read(4))
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            (cam_id,) = struct.unpack("<i", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (n2d,) = struct.unpack("<Q", f.read(8))
            xys = np.zeros((n2d, 2))
            pids = np.zeros(n2d, np.int64)
            for i in range(n2d):
                x, y, pid = struct.unpack("<ddq", f.read(24))
                xys[i] = (x, y)
                pids[i] = pid
            images.append(Image(iid, qvec, tvec, cam_id, name.decode("utf-8"), xys, pids))
    return images


def write_images_txt(path, images: list[Image]):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                f"# Number of images: {len(images)}\n")
        for im in images:
            q = " ".join(repr(float(v)) for v in im.qvec)
            t = " ".join(repr(float(v)) for v in im.tvec)
            f.write(f"{im.id} {q} {t} {im.camera_id} {im.name}\n")
            f.write(" ".join(f"{x} {y} {pid}"
                             for (x, y), pid in zip(im.xys, im.point3D_ids)) + "\n")


def read_images_txt(path) -> list[Image]:
    images = []
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:            # 空行（不會出現在影像行位置）
            i += 1
            continue
        tokens = line.split()
        images.append(Image(int(tokens[0]),
                            np.array([float(t) for t in tokens[1:5]]),
                            np.array([float(t) for t in tokens[5:8]]),
                            int(tokens[8]), tokens[9]))
        i += 2                  # 下一行是 POINTS2D[]（可能為空），一律跳過
    return images


# ---------------------------------------------------------------------------
# points3D
# ---------------------------------------------------------------------------

def write_points3D_bin(path, xyz: np.ndarray, rgb: np.ndarray, errors=None):
    n = len(xyz)
    if errors is None:
        errors = np.full(n, 1.0)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", *xyz[i]))
            f.write(struct.pack("<3B", *rgb[i]))
            f.write(struct.pack("<d", errors[i]))
            f.write(struct.pack("<Q", 0))          # 空 track（無 SfM 觀測）


def read_points3D_bin(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        xyz = np.zeros((n, 3))
        rgb = np.zeros((n, 3), np.uint8)
        err = np.zeros(n)
        for i in range(n):
            struct.unpack("<Q", f.read(8))
            xyz[i] = struct.unpack("<3d", f.read(24))
            rgb[i] = struct.unpack("<3B", f.read(3))
            (err[i],) = struct.unpack("<d", f.read(8))
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(8 * track_len)
    return xyz, rgb, err


def write_points3D_txt(path, xyz: np.ndarray, rgb: np.ndarray, errors=None):
    n = len(xyz)
    if errors is None:
        errors = np.full(n, 1.0)
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n"
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
                f"# Number of points: {n}\n")
        for i in range(n):
            f.write(f"{i + 1} {float(xyz[i, 0])!r} {float(xyz[i, 1])!r} {float(xyz[i, 2])!r} "
                    f"{int(rgb[i, 0])} {int(rgb[i, 1])} {int(rgb[i, 2])} {float(errors[i])!r}\n")


def read_points3D_txt(path):
    xyz, rgb, err = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            xyz.append([float(t) for t in tokens[1:4]])
            rgb.append([int(t) for t in tokens[4:7]])
            err.append(float(tokens[7]))
    return np.array(xyz), np.array(rgb, np.uint8), np.array(err)
