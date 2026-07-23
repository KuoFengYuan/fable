# 座標系轉換完整推導：ARKit → OpenCV/COLMAP → Nerfstudio

這是整條 COLMAP-free 管線最核心（也最容易做錯）的一段。所有公式皆由
[`tools/test_math.py`](../tools/test_math.py) 的 7 項投影一致性測試自動驗證。

## 1. 各家慣例對照

所有座標系都是**右手系**，差別只在「相機局部軸」與「世界上方向」的指定：

| 系統 | 相機 X | 相機 Y | 視線方向 | 儲存的矩陣 |
|---|---|---|---|---|
| **ARKit** | 右 | 上 | **-Z** | `ARCamera.transform` = **c2w** |
| OpenGL / Blender | 右 | 上 | -Z | — |
| **Nerfstudio / instant-ngp** | 右 | 上 | -Z | `transform_matrix` = **c2w** |
| **OpenCV / COLMAP / Inria 3DGS** | 右 | **下** | **+Z** | COLMAP 存 **w2c**（qvec+tvec） |

三個關鍵事實：

1. **ARKit 與 Nerfstudio 的相機慣例完全相同** → ARKit 的 c2w 可以原樣寫進 `transforms.json`，一個矩陣都不用改。這就是 App 端能直接產出可訓練 `transforms.json` 的原因。
2. **GL ↔ CV 只差相機自身的 Y、Z 軸翻轉** → 對 c2w「**右乘**」`diag(1,-1,-1,1)`（右乘＝只轉相機局部軸，世界不動；左乘就錯了）。
3. **COLMAP 存的是 w2c 不是 c2w**，且四元數順序為 `(qw, qx, qy, qz)`（scipy 預設是 `(x,y,z,w)`，混用是最常見的翻車點）。

## 2. 核心轉換（三行數學）

設 ARKit 給出的 camera-to-world 為 `M_gl`（`ARCamera.transform`）：

```
① 相機軸 GL → CV：   M_cv = M_gl · diag(1, -1, -1, 1)
② 取逆得 w2c：       W = M_cv⁻¹ = [Rᵀ | -Rᵀt]        （剛體解析逆，勿用一般矩陣逆）
③ COLMAP 序列化：    qvec = rotmat2qvec(W[:3,:3])（wxyz），tvec = W[:3,3]
```

對應程式：[`geometry.py`](../tools/geometry.py) 的 `colmap_qt_from_c2w_gl()`。

投影自洽驗證（`test_math.py` 實測到 1e-6 像素）：

```
GL 路徑：u = fx·x/(-z) + cx，  v = cy - fy·y/(-z)     （z 為 GL 相機系座標，前方 z<0）
CV 路徑：u = fx·x'/z' + cx，   v = fy·y'/z' + cy      （x',y',z' 為 CV 相機系座標）
兩路徑對同一世界點結果完全相等。
```

## 3. 世界座標：Y-up、公制、gravity 對齊

- App 使用 `worldAlignment = .gravity` → **+Y 嚴格反重力**（ARKit 世界原生上方向）。
- **單位是公尺（公制）**：COLMAP/SfM 的任意尺度問題不存在；深度監督、物理量測直接可用。

### 3.1 為什麼 3DGS 會上下顛倒 —— 必看

ARKit 世界是 **+Y up**，但 COLMAP/OpenCV 生態隱含假設**重力 ≈ +Y（-Y 才是 up）**：
因為 OpenCV 相機 Y 軸朝下，直立相機拍攝重建出的世界，重力自然落在 +Y。
把 ARKit 的 +Y-up 資料原樣餵給 LichtFeld-Studio / Inria 3DGS / 多數 splat viewer，
就會**整個上下顛倒**。

**修正**：匯出 COLMAP 時對世界繞 X 軸轉 180°（`(x,y,z)→(x,-y,-z)`，`geometry.py: FLIP_WORLD_X_180`），
**相機姿態與點雲成對套用**。這是剛體變換，不改變重建品質、只轉正顯示方向。

- Swift 端：`CaptureConfig.flipWorldUpForExport = true`（預設），只作用於 `sparse/0`
  的 `images.bin` + `points3D.bin`；`points.ply` 與 `poses.jsonl` 保留 ARKit +Y-up
  原生幀（供 `validate_dataset.py` 與 Review 檢視器，兩者用 SceneKit Y-up 顯示正常）。
- Python 端：`arkit2gs.py` COLMAP 路徑預設 `--colmap-flip-up`；若某 viewer 反而變顛倒，
  用 `--no-colmap-flip-up`（Swift 端則設 `flipWorldUpForExport = false`）。
- ⚠️ 陷阱：只翻姿態不翻點雲（或反之）＝ 相機與點雲分離、立刻崩壞。
  `test_math.py` 的「世界翻 180° 投影不變」測試就是在守這件事。

### 3.2 其他世界朝向

- nerfstudio 路徑不翻轉：它讀入時自做 `orientation_method="up"` 歸一。
- Z-up（instant-ngp 習慣）：`--world-up z`，對所有 c2w 與點雲左乘 `Rx(+90°)`（`ROT_X_90`）。
  與 flip 互斥，前者優先。

## 4. 內參與影像方向

### 4.1 sensor 座標（預設，最穩）

`ARFrame.capturedImage` **永遠是 landscape-right sensor 方向**（1920×1440），與 `camera.intrinsics`、`camera.transform` 三者天然自洽。**影像、內參、姿態一起原樣存，訓練器根本不在乎影像的「上」是哪邊** —— 這是預設值，也是建議值。

內參讀取（`simd_float3x3` 是 column-major）：

```
fx = intrinsics[0][0]   fy = intrinsics[1][1]
cx = intrinsics[2][0]   cy = intrinsics[2][1]
畸變：ARKit 影像已經過 ISP 校正，視為零畸變針孔（k1=k2=p1=p2=0, PINHOLE/OPENCV model）
```

### 4.2 portrait 正規化（選用，`--orientation portrait`）

想要直式正立影像時，三件事必須同步改（漏一個就全錯）：

| 對象 | 變換 |
|---|---|
| 影像 | 順時針轉 90°（PIL `ROTATE_270` / `np.rot90(k=-1)`） |
| 內參 | `fx' = fy, fy' = fx, cx' = H - cy, cy' = cx` |
| 姿態（CV 相機軸） | `c2w' = c2w · R`，R 的 columns = 新軸 `[x'=-y, y'=x, z'=z]` |
| 像素對應 | `(u', v') = (H - v, u)` ← 驗證用不變量 |

GL 慣例下的同一旋轉為 `D·R·D`（`D = diag(1,-1,-1)`），見 `geometry.py: rotate_portrait()`。

### 4.3 逐幀內參

AF/OIS 會讓 fx 逐幀微動。App 端做了兩層處理：
- 開拍即鎖定對焦（`configurableCaptureDeviceForPrimaryCamera.focusMode = .locked`）
- `transforms.json` 頂層放中位數內參（instant-ngp 相容），每幀另帶各自內參（nerfstudio 支援 per-frame 覆寫）

COLMAP 輸出取中位數作單一 `PINHOLE` 相機（鎖焦後漂移 < 0.1%，`validate_dataset.py` 會回報漂移量）。

## 5. 深度圖反投影（LiDAR → 點雲）

`sceneDepth.depthMap` 為 256×192 float32，值＝**沿光軸的 Z 深度**（非 ray 距離），與 capturedImage 同為 sensor 座標：

```
內參縮放：  K_depth = K_rgb × (256/1920)          （cx,cy 同倍縮放）
CV 反投影： x = (u-cx)/fx·d,  y = (v-cy)/fy·d,  z = d
轉 GL：     p_gl = (x, -y, -z)
世界：      p_w = c2w_arkit · p_gl
```

信心過濾只取 `ARConfidenceLevel.high`（=2），2cm voxel 去重。Swift 端（`PointCloudAccumulator`）與 Python 端（`arkit2gs.py --rebuild-points`）是同一組公式的雙實作，交叉驗證過。

## 6. 時間同步

**影像、深度、內參、姿態全部取自同一個 `ARFrame`** —— ARKit 已做了感測器層級的同步，`frame.timestamp`（boottime 單調時鐘）寫入每行 `poses.jsonl`，不需要任何事後對齊。`validate_dataset.py` 仍會檢查單調性與幀間隔，作為資料毀損的保險。

## 7. 常見翻車清單

| 症狀 | 原因 |
|---|---|
| **訓練出的模型上下顛倒** | ARKit +Y-up 未對齊 COLMAP 慣例 → 開啟 `flipWorldUpForExport`（見 3.1） |
| 訓練出鏡像場景 | 把右乘 `diag(1,-1,-1,1)` 寫成左乘 |
| 相機全部背對點雲（validate 顯示 0% 在前方） | 忘了 GL→CV 翻轉，或 c2w/w2c 搞反 |
| splat 浮在半空亂轉 | 四元數順序 wxyz vs xyzw 混用 |
| 場景歪 90° | 世界軸旋轉只套了姿態、沒套點雲（或反之） |
| 翻轉後仍顛倒 | flip 只套了姿態或只套了點雲之一 —— 必須成對 |
| 重投影差半個影像 | portrait 只轉了影像、沒改內參姿態 |
| simd 讀出來的矩陣是轉置 | `simd_float4x4` 是 column-major，序列化時須明確轉 row-major |

> 為什麼幾何驗證抓不到「顛倒」：整個世界一起翻 180° 後，相機與點雲仍彼此自洽，
> 所有投影一致性測試（含 100% 點在相機前方）照樣通過。一致性測試證明「資料內部沒錯」，
> **不等於**「絕對上方向對得上 viewer」—— 後者是慣例選擇，由 `flipWorldUpForExport` 控制。
