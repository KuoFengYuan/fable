# COLMAP-free 3DGS 訓練管線設定

手機 zip 解壓後**本身就是 COLMAP 資料集**（`images/ + sparse/0/*.bin`），
LichtFeld-Studio 與 Inria 3DGS 零轉換直讀；nerfstudio 路線用 `arkit2gs.py` 轉出。

## 0. LichtFeld-Studio（高效能 C++/CUDA 實作，主要目標框架）

```bash
git clone https://github.com/MrNeRF/LichtFeld-Studio    # 依其 README 以 CMake/CUDA 建置

# 手機 zip 解壓後直接指向資料夾（COLMAP 格式自動辨識）
LichtFeld-Studio -d scan_20260721_103000 -o output/scan_20260721_103000
# 或啟動 GUI 後直接開啟該資料夾
```

對應關係：`sparse/0/cameras.bin`（單一 PINHOLE 相機）、`images.bin`（w2c 姿態，
名稱對應 `images/` 內檔案）、`points3D.bin`（擇優下採樣後的 LiDAR 點雲，作為
Gaussians 初始化）。點雲上限由 App 端 `CaptureConfig.exportMaxPoints`（預設 25 萬）
控制 —— LichtFeld 的 densification 會自行增生，初始點過多只會拖慢前期迭代。

注意：App 匯出的姿態已做過兩層優化 —— 每個關鍵幀掛 ARAnchor，ARKit 地圖優化
（迴環/重定位）的修正會回寫到錨點，停止掃描時讀回「修正後姿態」寫入 images.bin；
points3D.bin 則是以同一組修正後姿態對全部關鍵幀深度做**多視角加權重融合**的結果
（深度雜訊 ~1/√N 收斂），兩者幾何完全一致。LichtFeld 若有姿態優化選項（版本演進中，
見其 `--help`）仍建議打開吸收殘差；大場景可再做第 4 節的 point_triangulator 精修。

## 1. Nerfstudio splatfacto（想要姿態微調時的首選）

```bash
pip install nerfstudio            # 需要 CUDA；gsplat 後端

# 基本訓練 —— transforms.json 內含 ply_file_path，
# nerfstudio dataparser 會自動載入點雲作為 Gaussians 初始化種子
ns-train splatfacto --data dataset/nerfstudio

# 建議的完整參數（ARKit 資料實戰配置）
ns-train splatfacto \
  --data dataset/nerfstudio \
  --pipeline.model.camera-optimizer.mode SO3xR3 \
  --pipeline.model.rasterize-mode antialiased \
  nerfstudio-data \
  --orientation-method up \
  --center-method poses \
  --auto-scale-poses True
```

關鍵參數解讀：

| 參數 | 為什麼 |
|---|---|
| `camera-optimizer.mode SO3xR3` | **最重要**。ARKit VIO 姿態很好但非 BA 級精度（典型殘差 0.1–0.5°/數 mm 漂移），讓訓練同時微調姿態可吸收殘差，PSNR 通常 +1~2dB |
| `orientation-method up` | 用姿態平均 up 向量歸一世界（我們是 gravity 對齊，等於免費午餐） |
| `auto-scale-poses` | nerfstudio 內部歸一到單位尺度；匯出 splat 時會轉回公制 |
| 無 `ply_file_path` 時 | 加 `--pipeline.model.random-init True` 退回隨機初始化 |

檢視與匯出：

```bash
ns-viewer --load-config outputs/.../config.yml
ns-export gaussian-splat --load-config outputs/.../config.yml --output-dir exports/
```

## 2. Inria 官方 3DGS（graphdeco）

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting

python train.py -s /path/to/dataset/colmap --eval
python render.py -m output/<model_id>
```

我們產出的 `dataset/colmap/` 結構與其 loader 期望完全一致，**不需要跑任何 COLMAP 指令**：

```
colmap/
├── images/
└── sparse/0/
    ├── cameras.bin     # 1 台 PINHOLE 相機（fx fy cx cy）
    ├── images.bin      # w2c 四元數(wxyz)+平移，空 2D 觀測
    └── points3D.bin    # LiDAR 彩色點雲，空 track
```

原理：Inria 的 `dataset_readers.readColmapSceneInfo()` 只讀取這三個檔案的姿態/內參/點雲，
不依賴 SfM 的匹配資訊（track 為空完全合法；首次載入時它會自行把 points3D 轉成內部 ply）。

注意：Inria 版假設影像未畸變（我們是 ISP 校正後的 PINHOLE，符合），且**沒有**姿態微調功能 ——
若對品質敏感，優先用 splatfacto，或先做下方的「選配 BA 精修」。

## 3. gsplat / 其他框架

```bash
# gsplat 官方範例 trainer（讀 COLMAP 格式）
cd gsplat/examples
python simple_trainer.py default --data-dir dataset/colmap --data-factor 1

# OpenSplat（CPU/Metal 亦可跑，直接吃 nerfstudio 格式）
opensplat dataset/nerfstudio -n 30000

# Postshot / Brush 等 GUI 工具：匯入 COLMAP 資料夾即可
```

## 4. 選配：以 COLMAP point_triangulator 精修（仍不跑 SfM）

想要「ARKit 姿態 + SfM 級三角化點雲」的折衷方案：固定我們的姿態，只讓 COLMAP 做特徵三角化（幾秒鐘，不是幾小時的 mapper）：

```bash
# 需先把 sparse/0 轉成 txt（arkit2gs.py 加 --txt），COLMAP 以其為固定姿態先驗
colmap feature_extractor  --database_path db.db --image_path colmap/images
colmap exhaustive_matcher --database_path db.db
colmap point_triangulator --database_path db.db --image_path colmap/images \
    --input_path colmap/sparse/0 --output_path colmap/sparse/0_refined
```

得到帶 track 的高品質 points3D，姿態不變。對大場景 / 反光材質特別有感。

## 5. 深度監督（選配）

zip 內的 `depth/*.bin` 是 256×192 float32 公尺值。轉成 nerfstudio 深度格式（16-bit PNG，毫米）後可用 depth loss：

```python
import numpy as np; from PIL import Image
d = np.fromfile("frame_00001_depth.bin", np.float32).reshape(192, 256)
Image.fromarray((d * 1000).astype(np.uint16)).save("frame_00001_depth.png")
# transforms.json 每幀加 "depth_file_path"，nerfstudio 預設 depth_unit_scale_factor=1e-3
```

`ns-train depth-nerfacto` 直接支援；splatfacto 的深度正則化見 nerfstudio 文件 `--pipeline.model.use-depth-loss`（版本演進中，依安裝版本查 `ns-train splatfacto --help`）。

## 6. 品質預期與除錯順序

ARKit 姿態 vs COLMAP 姿態的訓練品質：室內小場景通常相差 < 1dB PSNR（開 camera-optimizer 後幾乎抹平）；長走廊 / 大迴圈場景 VIO 漂移會顯現 —— 這時候：

1. 先跑 `validate_dataset.py`（追蹤跳變、幀間隔異常一眼看出）
2. 掃描時避免快速甩動與長時間遮擋（跳變的來源是 ARKit relocalization）
3. 分段掃描、分段訓練，或用第 4 節的 point_triangulator + `colmap bundle_adjuster`（固定內參）做輕量 BA
