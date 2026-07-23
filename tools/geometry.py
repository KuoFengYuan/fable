"""ARKit ↔ OpenCV/COLMAP ↔ Nerfstudio 座標系轉換核心。

慣例整理
========
相機「局部軸」慣例（決定影像投影方向）：

  ARKit / OpenGL / Nerfstudio :  X 右、Y 上、-Z 為視線方向（look down -Z）
  OpenCV / COLMAP / Inria 3DGS:  X 右、Y 下、+Z 為視線方向（look down +Z）

兩者同為右手系，僅相機自身的 Y/Z 軸相反 → 對 camera-to-world 矩陣
「右乘」diag(1,-1,-1,1) 即翻轉相機局部 Y/Z 軸，世界座標不動：

    c2w_cv = c2w_gl @ FLIP_YZ

儲存形式：
  - Nerfstudio transforms.json: 存 c2w（GL 慣例）→ ARKit 姿態可原樣寫入
  - COLMAP images.txt/bin:      存 w2c（CV 慣例）的四元數 (qw,qx,qy,qz) 與平移

世界座標：ARKit worldAlignment=.gravity → +Y 為反重力方向、原點為 session 起點、
單位公尺（metric）。COLMAP 世界座標本身無慣例要求；如需 Z-up（instant-ngp 習慣）
可用 yup_to_zup_* 對所有姿態與點雲同時左乘 Rx(+90°)。
"""

from __future__ import annotations

import numpy as np

# 相機局部軸翻轉（GL ↔ CV），自身即逆矩陣
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])

# 世界 Y-up → Z-up：繞 X 軸 +90°，將 +Y 映到 +Z（右手系保持）
ROT_X_90 = np.array([
    [1.0, 0.0,  0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0,  0.0, 0.0],
    [0.0, 0.0,  0.0, 1.0],
])

# 世界 Y-up(ARKit) → COLMAP/3DGS 慣例（重力 ≈ +Y、up ≈ -Y）：繞世界 X 軸 180°。
# ARKit 的 +Y-up 資料直接匯入多數 3DGS 工具會上下顛倒，套此翻轉即轉正。
FLIP_WORLD_X_180 = np.diag([1.0, -1.0, -1.0, 1.0])


def gl_to_cv_c2w(c2w_gl: np.ndarray) -> np.ndarray:
    """ARKit/GL 慣例 c2w → OpenCV 慣例 c2w（右乘只翻相機局部軸）。"""
    return c2w_gl @ FLIP_YZ


def cv_to_gl_c2w(c2w_cv: np.ndarray) -> np.ndarray:
    return c2w_cv @ FLIP_YZ


def w2c_from_c2w(c2w: np.ndarray) -> np.ndarray:
    """剛體變換的解析逆（比 np.linalg.inv 數值穩定）。"""
    R = c2w[:3, :3].T
    t = -R @ c2w[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def colmap_qt_from_c2w_gl(c2w_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ARKit c2w（GL 慣例）→ COLMAP 儲存格式：w2c 的 (qvec[wxyz], tvec)。"""
    w2c = w2c_from_c2w(gl_to_cv_c2w(c2w_gl))
    return rotmat2qvec(w2c[:3, :3]), w2c[:3, 3].copy()


def yup_to_zup_c2w(c2w: np.ndarray) -> np.ndarray:
    """世界座標 Y-up → Z-up（左乘：世界旋轉，相機局部軸不動）。"""
    return ROT_X_90 @ c2w


def yup_to_zup_points(points: np.ndarray) -> np.ndarray:
    return points @ ROT_X_90[:3, :3].T


def flip_world_up_c2w(c2w: np.ndarray) -> np.ndarray:
    """世界繞 X 軸 180°（左乘）：ARKit +Y-up → COLMAP 慣例，修正 3DGS 上下顛倒。"""
    return FLIP_WORLD_X_180 @ c2w


def flip_world_up_points(points: np.ndarray) -> np.ndarray:
    """(x, y, z) → (x, -y, -z)。須與 flip_world_up_c2w 成對套用以保持一致。"""
    return points @ FLIP_WORLD_X_180[:3, :3].T


# ---------------------------------------------------------------------------
# 四元數（COLMAP 順序：qw, qx, qy, qz）
# ---------------------------------------------------------------------------

def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """旋轉矩陣 → 四元數 (w,x,y,z)。與 COLMAP read_write_model.py 相同的特徵向量法。"""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


# ---------------------------------------------------------------------------
# Portrait 正規化（選用）
# ---------------------------------------------------------------------------
# ARKit 影像永遠是 sensor 座標（landscape-right）。直式手持拍攝時若想輸出「正立」
# 影像，需將影像順時針轉 90°，同時同步修正內參與姿態：
#
#   新相機軸（以舊 CV 相機軸表示）: x' = -y, y' = x, z' = z
#   內參: fx' = fy, fy' = fx, cx' = H - cy, cy' = cx（連續像素座標慣例）
#   像素: (u', v') = (H - v, u)
#
# 三者由 test_math.py 的投影一致性測試共同驗證。

R_PORTRAIT_CV = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])  # columns = 新相機軸 [x'|y'|z']（舊 CV 座標表示）


def rotate_portrait(c2w_gl: np.ndarray, K: tuple, w: int, h: int):
    """影像順時針轉 90° 後的 (c2w_gl', K', w', h')。

    K 為 (fx, fy, cx, cy)。影像本身需另行旋轉：
    np.rot90(img, k=-1) 或 PIL 的 Image.Transpose.ROTATE_270。
    """
    D = np.diag([1.0, -1.0, -1.0])
    R_gl = D @ R_PORTRAIT_CV @ D          # 同一旋轉在 GL 相機軸下的表示
    T = np.eye(4)
    T[:3, :3] = R_gl
    fx, fy, cx, cy = K
    return c2w_gl @ T, (fy, fx, h - cy, cx), h, w


# ---------------------------------------------------------------------------
# 投影（測試 / 驗證用）
# ---------------------------------------------------------------------------

def project_cv(K: tuple, w2c_cv: np.ndarray, points_w: np.ndarray):
    """OpenCV 慣例投影。回傳 (uv (N,2), depth (N,))，depth>0 表示在相機前方。"""
    fx, fy, cx, cy = K
    p = points_w @ w2c_cv[:3, :3].T + w2c_cv[:3, 3]
    z = p[:, 2]
    uv = np.stack([fx * p[:, 0] / z + cx, fy * p[:, 1] / z + cy], axis=1)
    return uv, z


def project_gl(K: tuple, c2w_gl: np.ndarray, points_w: np.ndarray):
    """GL 慣例投影（視線為 -Z）。回傳 (uv, depth)，depth = -z_cam。"""
    fx, fy, cx, cy = K
    w2c = w2c_from_c2w(c2w_gl)
    p = points_w @ w2c[:3, :3].T + w2c[:3, 3]
    depth = -p[:, 2]
    uv = np.stack([fx * p[:, 0] / depth + cx,
                   -fy * p[:, 1] / depth + cy], axis=1)
    return uv, depth
