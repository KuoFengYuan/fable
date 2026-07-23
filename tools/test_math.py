#!/usr/bin/env python3
"""座標轉換數學的自動驗證。任何一項失敗都代表輸出資料集會壞掉 —— 改動 geometry.py 後必跑。

    python test_math.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import colmap_io as C
import geometry as G
import ply_io as P

RNG = np.random.default_rng(42)


def random_rotation() -> np.ndarray:
    q = RNG.normal(size=4)
    return G.qvec2rotmat(q / np.linalg.norm(q))


def random_c2w_gl() -> np.ndarray:
    c2w = np.eye(4)
    c2w[:3, :3] = random_rotation()
    c2w[:3, 3] = RNG.uniform(-3, 3, 3)
    return c2w


def visible_world_points(c2w_gl: np.ndarray, K, w, h, n=50) -> np.ndarray:
    """在影像範圍內取像素與深度，反投影到世界座標（保證可見）。"""
    fx, fy, cx, cy = K
    u = RNG.uniform(w * 0.1, w * 0.9, n)
    v = RNG.uniform(h * 0.1, h * 0.9, n)
    d = RNG.uniform(0.5, 5.0, n)
    x_cv = (u - cx) / fx * d
    y_cv = (v - cy) / fy * d
    p_gl = np.stack([x_cv, -y_cv, -d], axis=1)     # CV → GL 相機系
    return p_gl @ c2w_gl[:3, :3].T + c2w_gl[:3, 3], np.stack([u, v], 1)


# ---------------------------------------------------------------------------

def t_quaternion_roundtrip():
    for _ in range(200):
        R = random_rotation()
        R2 = G.qvec2rotmat(G.rotmat2qvec(R))
        assert np.allclose(R, R2, atol=1e-9), f"四元數往返誤差過大\n{R}\n{R2}"


def t_gl_cv_projection_equivalence():
    """同一世界點：GL 路徑投影 == CV 路徑（c2w 右乘 FLIP_YZ 後）投影。"""
    K = (600.0, 610.0, 320.0, 235.0)
    for _ in range(20):
        c2w_gl = random_c2w_gl()
        X, uv_expected = visible_world_points(c2w_gl, K, 640, 480)
        uv_gl, depth_gl = G.project_gl(K, c2w_gl, X)
        w2c_cv = G.w2c_from_c2w(G.gl_to_cv_c2w(c2w_gl))
        uv_cv, depth_cv = G.project_cv(K, w2c_cv, X)
        assert np.allclose(uv_gl, uv_cv, atol=1e-6)
        assert np.allclose(depth_gl, depth_cv, atol=1e-9)
        assert np.allclose(uv_gl, uv_expected, atol=1e-6)   # 反投影↔投影自洽
        assert (depth_gl > 0).all()


def t_colmap_qt():
    """COLMAP (qvec,tvec) 重建的 w2c 與直接計算完全一致。"""
    K = (500.0, 505.0, 321.0, 242.0)
    for _ in range(20):
        c2w_gl = random_c2w_gl()
        X, _ = visible_world_points(c2w_gl, K, 640, 480)
        qvec, tvec = G.colmap_qt_from_c2w_gl(c2w_gl)
        w2c = np.eye(4)
        w2c[:3, :3] = G.qvec2rotmat(qvec)
        w2c[:3, 3] = tvec
        uv_a, _ = G.project_cv(K, w2c, X)
        uv_b, _ = G.project_gl(K, c2w_gl, X)
        assert np.allclose(uv_a, uv_b, atol=1e-6)


def t_world_up_rotation_invariance():
    """世界 Y-up→Z-up 同時作用於姿態與點 → 投影不變。"""
    K = (600.0, 600.0, 320.0, 240.0)
    for _ in range(10):
        c2w = random_c2w_gl()
        X, _ = visible_world_points(c2w, K, 640, 480)
        uv1, d1 = G.project_gl(K, c2w, X)
        uv2, d2 = G.project_gl(K, G.yup_to_zup_c2w(c2w), G.yup_to_zup_points(X))
        assert np.allclose(uv1, uv2, atol=1e-6)
        assert np.allclose(d1, d2, atol=1e-9)


def t_world_flip_up_invariance():
    """世界翻 180°（修正 3DGS 顛倒）同時作用於姿態與點 → 投影不變、幾何不損。
    這保證 flip 是純顯示方向調整，不會破壞資料集自洽性。"""
    K = (600.0, 610.0, 320.0, 235.0)
    for _ in range(20):
        c2w = random_c2w_gl()
        X, _ = visible_world_points(c2w, K, 640, 480)
        uv1, d1 = G.project_gl(K, c2w, X)
        uv2, d2 = G.project_gl(K, G.flip_world_up_c2w(c2w), G.flip_world_up_points(X))
        assert np.allclose(uv1, uv2, atol=1e-6)
        assert np.allclose(d1, d2, atol=1e-9)
    # 翻轉確實改變了世界上方向（+Y → -Y），而非無效操作
    p = np.array([[0.0, 1.0, 0.0]])
    assert G.flip_world_up_points(p)[0, 1] == -1.0


def t_portrait_rotation():
    """影像順時針轉 90° 後：新投影 (u',v') == (H - v, u)。"""
    w, h = 640, 480
    K = (600.0, 612.0, 322.0, 244.0)
    for _ in range(10):
        c2w = random_c2w_gl()
        X, _ = visible_world_points(c2w, K, w, h)
        uv, _ = G.project_gl(K, c2w, X)
        c2w2, K2, w2, h2 = G.rotate_portrait(c2w, K, w, h)
        assert (w2, h2) == (h, w)
        uv2, d2 = G.project_gl(K2, c2w2, X)
        expected = np.stack([h - uv[:, 1], uv[:, 0]], axis=1)
        assert np.allclose(uv2, expected, atol=1e-6), \
            f"portrait 投影不符\n{uv2[:3]}\n{expected[:3]}"
        assert (d2 > 0).all()


def t_colmap_io_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cam = C.Camera(1, "PINHOLE", 1920, 1440, [1500.1, 1501.2, 960.3, 720.4])
        images = []
        for i in range(5):
            q = G.rotmat2qvec(random_rotation())
            images.append(C.Image(i + 1, q, RNG.uniform(-2, 2, 3), 1, f"frame_{i:05d}.jpg"))
        xyz = RNG.uniform(-5, 5, (100, 3))
        rgb = RNG.integers(0, 256, (100, 3), dtype=np.uint8)

        C.write_cameras_bin(td / "cameras.bin", [cam])
        C.write_images_bin(td / "images.bin", images)
        C.write_points3D_bin(td / "points3D.bin", xyz, rgb)
        C.write_cameras_txt(td / "cameras.txt", [cam])
        C.write_images_txt(td / "images.txt", images)
        C.write_points3D_txt(td / "points3D.txt", xyz, rgb)

        for cams in (C.read_cameras_bin(td / "cameras.bin"),
                     C.read_cameras_txt(td / "cameras.txt")):
            assert cams[0].model == "PINHOLE"
            assert np.allclose(cams[0].params, cam.params)
            assert (cams[0].width, cams[0].height) == (1920, 1440)

        for ims in (C.read_images_bin(td / "images.bin"),
                    C.read_images_txt(td / "images.txt")):
            for a, b in zip(ims, images):
                assert a.name == b.name and a.camera_id == 1
                assert np.allclose(a.qvec, b.qvec, atol=1e-12)
                assert np.allclose(a.tvec, b.tvec, atol=1e-12)

        for x2, r2, _ in (C.read_points3D_bin(td / "points3D.bin"),
                          C.read_points3D_txt(td / "points3D.txt")):
            assert np.allclose(x2, xyz, atol=1e-9)
            assert (r2 == rgb).all()


def t_ply_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.ply"
        xyz = RNG.uniform(-2, 2, (500, 3))
        rgb = RNG.integers(0, 256, (500, 3), dtype=np.uint8)
        P.write_ply(path, xyz, rgb)
        x2, r2 = P.read_ply(path)
        assert np.allclose(x2, xyz, atol=1e-6)     # float32 儲存精度
        assert (r2 == rgb).all()


def main():
    tests = [
        ("四元數 ↔ 旋轉矩陣往返", t_quaternion_roundtrip),
        ("GL/ARKit ↔ OpenCV 投影等價", t_gl_cv_projection_equivalence),
        ("COLMAP qvec/tvec 正確性", t_colmap_qt),
        ("世界 Y-up→Z-up 投影不變", t_world_up_rotation_invariance),
        ("世界翻 180°（修正顛倒）投影不變", t_world_flip_up_invariance),
        ("Portrait 旋轉（影像/內參/姿態）一致", t_portrait_rotation),
        ("COLMAP bin/txt 讀寫往返", t_colmap_io_roundtrip),
        ("PLY 讀寫往返", t_ply_roundtrip),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✅ {name}")
        except AssertionError as e:
            failed += 1
            print(f"❌ {name}: {e}")
    if failed:
        raise SystemExit(f"\n{failed} 項測試失敗")
    print(f"\n全部 {len(tests)} 項通過 — 座標轉換數學驗證完成")


if __name__ == "__main__":
    main()
