# 蘋果裝置空間採集實戰：散熱、記憶體、Rolling Shutter

## 1. 散熱（Thermal）

ARKit + LiDAR + 60fps 相機是 iPhone 最燙的工作負載之一。實測 iPhone 15 Pro 連續掃描
約 8–12 分鐘進入 `.serious`（夏天戶外更快）。過熱的後果不只是降頻：**相機會強制降 fps、
ISP 降質、VIO 精度下滑**，資料品質跟著崩。

本專案已內建的對策（`CaptureController` / `CaptureConfig`）：

| 機制 | 實作 |
|---|---|
| 熱狀態逐幀輪詢 | `ProcessInfo.processInfo.thermalState`（O(1)，不需通知監聽） |
| `.serious` → HUD 警告 | `QualityIssue.deviceHot` 橘色膠囊，提示使用者暫停 |
| `.critical` → 自動停拍 | 立即 `stopScan()` 保住已拍資料並完成匯出 |
| 降低基礎負載 | `environmentTexturing = .none`、不開 mesh 重建、SceneKit 疊加物全部 `lightingModel = .constant`（免光照計算） |
| zip 用 store 級壓縮 | JPEG 再壓縮無收益，`NSFileCoordinator` 走系統路徑省 CPU |

額外建議（依需求取捨）：

- **不要開 4K video format**（iOS 16 的 `recommendedVideoFormatFor4KResolution`）：熱預算直接減半。1920×1440 對 3DGS 已充足（訓練端通常還會 downscale）。
- 需要更高解析度時，改用 `session.captureHighResolutionFrame()`（iOS 16+）**按關鍵幀單張抓 12MP**，而不是拉高視訊流解析度 —— 平均功耗幾乎不變。
- 長場景任務：拆成多段 3–5 分鐘掃描（每段一個 zip），訓練端合併；螢幕亮度調低、拔殼、避免邊充電邊掃。

## 2. 記憶體（Memory）

崩潰路徑幾乎只有一條：**保留 ARFrame**。ARKit 的 `capturedImage` 來自固定大小的內部
buffer pool，你多握著一幀，追蹤管線就少一幀可用，先是 `didUpdate` 停止回呼，然後 OOM。

本專案的紀律（`Utils.swift` / `FrameWriter`）：

1. **絕不儲存 ARFrame 引用**。delegate 回呼內立刻把需要的東西複製出來：
   - 影像 → `CVPixelBufferPool` 自有 pool，逐 plane `memcpy`（1920×1440 NV12 ≈ 4MB，~1–2ms）
   - 深度/信心 → 去 stride 的緊湊 `Data`（196KB + 49KB）
   - 姿態/內參 → 值型別
2. **背壓限流**：`pendingWrites >= 3` 時跳過本幀。智慧快門條件仍成立，I/O 追上後自動補拍 —— 寧可少一幀，不可累積佇列。
3. **點雲密度自動管理（長掃描關鍵）**：即時融合為逐 voxel（1cm）**加權平均**（近距離 × 畫面中心 × 低模糊的觀測權重高）—— 同一區域重複掃過只會讓位置/顏色越收越準、不會變多。記憶體 60 萬 cell 觸頂時 voxel 自動 ×2 加權合併繼續累積，長掃描永不停止收點且記憶體有界；匯出時再分層擇優下採樣到 `exportMaxPoints`（預設 25 萬）。融合來源走 `smoothedSceneDepth`（時域平滑）+ 高信心 mask + 深度梯度飛點過濾 + 模糊閘門（>12px 的幀不進點雲），存檔的 depth .bin 仍為原始深度。

4. **抗漂移融合（防點雲殘影，核心）**：殘影＝同一實體表面因漂移在不同時刻被融進不同 voxel、去重救不了。解法是**融合與去重都在錨點局部座標系進行**（`PointCloudFusion.swift: TiledFusedGrid`）：空間按 1.2m 磚分塊、每磚綁一個 `ARAnchor`，cell 存錨點局部座標。世界座標會漂移，但「相機相對於鄰近錨點」的局部關係不變 → 同一表面永遠映到同一局部 voxel → 重掃時合併、不產生第二份點。換算用主執行緒每幀擷取的錨點當下變換（隨封包傳入 actor）。渲染時磚節點變換 = 錨點當下變換，ARKit 修正錨點 → 整磚跟著實體表面走。此特性以 macOS 單元測試驗證（模擬 35cm 漂移仍融合成單一 cell；對照全域格會裂成兩個）。髒磚每 tick 節流重繪 ≤2 磚。

5. **匯出/檢視點雲的清理（RefusionEngine）**：Review 與匯出的雲由 RefusionEngine 以修正後關鍵幀姿態重融合，三道品質處理讓表面貼合、去霧：
   - **反變異數深度加權** `w = 1/(0.2 + z²)`：LiDAR 雜訊 ∝ 距離²，遠點大幅降權 → 融合位置偏向近距離的準確觀測，牆面收緊（單元測試：混合觀測位置從等權的 5cm 拉回 0.3cm）。
   - **孤立點移除**（`refuseMinNeighbors`，預設 3/26）：占據 voxel 的 26 鄰域占據數不足 → 飄浮雜點剔除，清掉空間白霧（單元測試：20 顆孤立點全清、稠密面 100% 保留）。
   - **略粗融合 voxel**（`refuseVoxelSizeM`，預設 2cm）：把遠距深度雜訊造成的「厚牆」塌成薄面；即時預覽維持 1cm 清晰。要最高細節設 0.01。
   仍無法完全消除的部分：純遠距觀測的牆面（從無近距觀測）受 LiDAR 物理雜訊限制，且 ARKit VIO 殘餘漂移無 on-device 全域 BA 可修 —— 這是 on-device 的天花板。
4. JPEG 編碼用共享 `CIContext(cacheIntermediates: false)`，避免中間紋理快取膨脹。

預算參考（1920×1440、q0.9、含深度）：**每關鍵幀 ≈ 0.85MB**；10 分鐘密集掃描約
600–900 幀 ≈ 0.6–0.8GB。HUD 的容量估計即時提醒使用者。

## 3. Rolling Shutter

iPhone 相機為捲簾快門，逐行讀出時間 ≈ 8–16ms。快速**旋轉**時同一幀上下行對應不同相機姿態，
產生果凍效應 —— ARKit 姿態是「單一時刻」的，果凍幀的姿態定義本身就是壞的，
3DGS 無法建模（它假設全域快門針孔），只能**在採集端避免**：

| 對策 | 實作位置 |
|---|---|
| 模糊/果凍風險即時估計：`blur_px ≈ (ω + v/z)·fx·t_exposure` | `QualityMonitor.assess()`，陀螺儀 ω 為主導項 |
| **兩級門檻**：8px 橘色提醒（照拍）、16px 紅色遮斷（暫停抓幀） | `maxBlurPixels` / `blockBlurPixels`（紅膠囊 + 紅框 + 觸覺） |
| UX 教育「多平移、少旋轉」 | 首頁提示 + README 拍攝守則；平移的果凍效應遠小於旋轉（∝ v/z vs ∝ ω） |
| 曝光時間監控 | `camera.exposureDuration` 進模糊公式；暗場景曝光拉長，同樣角速度風險自動放大 → 警告門檻等效收緊 |
| 事後保險 | `poses.jsonl` 記錄每幀 `estimatedBlurPx`，`arkit2gs.py --max-blur 8` 離線剔除 |

門檻校準（fx≈1450px）：1/60s 曝光下，8px 警告 ≈ 角速度 0.33 rad/s（19°/s）、
16px 遮斷 ≈ 0.66 rad/s（38°/s）—— 刻意留手持掃描的自然節奏空間，只擋真正的甩動。
單級嚴格門檻（例如 2px ≈ 5°/s）在實機上會警告不斷、抓幀停滯，切勿再調回。
昏暗場景曝光拉長時警告會變頻繁，這是誠實的物理提示：補光比放慢更有效；
真的需要快速掃描的場域（工廠巡檢），外接補光把曝光壓到 1/250s 以下，等效門檻自動放寬 4 倍。

## 4. 其他細節

- **相機三鎖**（`configurableCaptureDeviceForPrimaryCamera`，iOS 16+）：掃描 HUD 有「鎖定對焦/曝光/白平衡」開關（預設開啟），按快門當下三鎖齊下 —— 對焦鎖（內參穩定、AF 不拉風箱）、曝光鎖（ISO/快門固定 → 亮度一致、動態模糊可預測）、白平衡鎖（融合不閃色、3DGS 色彩乾淨）。鎖定時機刻意選在使用者取景完成後（AE/AF 已收斂於目標）。代價：明暗差極大的場景（窗邊逆光走到暗角）會欠曝/過曝 —— 這種場景關掉開關改用自動。極近距離（<30cm）特寫建議分段掃描、每段重新鎖焦。
- **主執行緒紀律（追蹤穩定的前提）**：session delegate 回呼內**只做 buffer memcpy**，任何逐像素數學或幾何建構都會餓死 ARKit 的 VIO（尤其 Debug -Onone 組建下慢 10 倍）→ 掉幀 → 追蹤不穩與漂移。點雲反投影/融合在 actor 執行緒、磚渲染資料在 actor 端打包成 GPU-ready Data、主執行緒只做 O(1) 包裝。改動熱路徑前先想清楚它跑在哪條執行緒。
- **曝光**：ARKit 不開放手動曝光。`exposureOffset` 已記錄於 poses.jsonl，HDR 場景（窗邊逆光）建議調整走位而非硬拍。
- **無 LiDAR 機種**：`sceneDepth` 不可用時自動退回 `rawFeaturePoints` 稀疏點雲（灰色）；3DGS 仍可訓練（初始化較弱，建議 `--pipeline.model.random-init` 加大 warmup）。
- **Files App 存取**：已開 `UIFileSharingEnabled` + `LSSupportsOpeningDocumentsInPlace`，接線也能從 Finder 直接拉資料夾。
- **Niantic 概念對應**：涵蓋率圓頂 ≈ Scaniverse 的引導殼層；分段上傳 + 雲端訓練即 Niantic Spatial Platform 的 capture-to-cloud 思路 —— `ExportManager` 的 zip 之外可掛 `URLSession` background upload task（`URLSessionConfiguration.background`），App 退場也會續傳。
