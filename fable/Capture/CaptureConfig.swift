//
//  CaptureConfig.swift
//  fable — COLMAP-free 3DGS capture
//

import Foundation
import CoreGraphics

/// 所有可調參數集中於此。門檻值以 iPhone Pro（LiDAR）室內拍攝為基準。
nonisolated struct CaptureConfig: Sendable {

    // MARK: - 智慧快門（基於位移 / 轉角，而非固定時間）
    /// 相對上一關鍵幀平移超過此距離（公尺）即觸發抓幀
    var keyframeTranslationM: Float = 0.10
    /// 或視角旋轉超過此角度（度）即觸發抓幀
    var keyframeRotationDeg: Float = 6.0
    /// 兩關鍵幀最小時間間隔，避免手震造成原地連拍
    var minKeyframeInterval: TimeInterval = 0.15
    /// 關鍵幀的相機速度閘門。與即時融合同理但門檻放寬：
    /// 模糊值受曝光時間影響（亮處曝光短 → 快速移動仍判定為低模糊），無法反映「姿態延遲/誤差」，
    /// 所以只靠 blockBlurPixels 擋不住「亮處快速移動」的壞姿態。
    /// 關鍵幀的姿態同時進入 (a) 重融合點雲的反投影 (b) 3DGS 訓練的相機外參，
    /// 錯了會同時污染幾何與訓練，比即時融合的後果嚴重。
    /// 門檻刻意比預覽(0.6/0.5)寬：關鍵幀本來就是在移動中觸發的（位移 0.1m 或轉角 6°），
    /// 太嚴會抓不到幀。1.0 rad/s ≈ 57°/s、0.8 m/s ≈ 快走，正常掃描不會觸及。
    var keyframeMaxAngularSpeedRadS: Float = 1.0
    var keyframeMaxLinearSpeedMS: Float = 0.8
    /// 背景寫入佇列上限（背壓）：滿載時跳過本幀，下一幀條件仍成立會再觸發
    var maxPendingWrites = 3

    // MARK: - 品質門檻（兩級制：警告 → 提醒但照拍；遮斷 → 暫停抓幀）
    /// 幾何劣化警告門檻（像素）：
    ///   blur ≈ (角速度 + 線速度/景深) × 焦距 × (曝光時間 ＋ 捲簾讀出時間)
    ///
    /// **門檻隨捲簾快門項一併重新校準過。** 加入讀出時間後，同樣的動作在明亮環境
    /// （曝光 1/120s）算出來的值是原本的 2.2 倍 —— 舊的 8/16 是以「只算曝光」校準的，
    /// 沿用會讓正常掃描一直跳紅字（實測回報過）。以 fx≈1450、總抹動時間 18.3ms 重算：
    ///
    ///   0.3 rad/s（17°/s，很慢的平移）→  8px
    ///   0.5 rad/s（29°/s，正常掃描）  → 13px
    ///   0.9 rad/s（52°/s，明顯轉身）  → 24px
    ///   1.0 rad/s（57°/s）            → 27px ← 實機回報那張糊掉的照片就在這一級
    ///
    /// 警告設 10、遮斷設 24（≈0.9 rad/s）。
    ///
    /// 警告值由**實機資料**決定而非推導：一次回報中位數 11.4px 的掃描，
    /// 清晰度閘門丟掉了 69% 的幀，而 HUD 全程沒有警告（當時警告線是 14）。
    /// 也就是說清晰度閘門實際的作用點在 ~11px，警告必須落在它**之前**，
    /// 否則使用者會在毫無提示的情況下流失大部分資料。
    /// 遮斷維持 24：那是「連姿態都不能信」的層級，與清晰度無關。
    var maxBlurPixels: Float = 10.0
    /// 遮斷門檻：超過才紅色警告＋暫停抓幀
    var blockBlurPixels: Float = 24.0
    /// 關鍵幀的清晰度門檻：本幀清晰度 ÷「近 0.5s 內同場景的最佳清晰度」須達此比例。
    ///
    /// 為什麼需要它 —— blockBlurPixels 是**推估**（角速度 × 曝光時間），只涵蓋動態模糊，
    /// 對「失焦」完全盲目：手機靜止不動、對焦跑掉的畫面，推估值是 0，照樣被存成關鍵幀。
    /// 失焦與 AF 拉焦（重新啟用連續自動對焦後必然會發生）是「有些照片糊掉」的主因，
    /// 只有直接量影像才擋得住。
    ///
    /// 用相對值而非絕對值：清晰度與場景紋理量綁死（白牆對到極清晰也只有雜訊級的值），
    /// 絕對門檻在白牆上會全擋、在書架上會全過。
    /// 值由離線校準決定（tools/test_sharpness.py，粉紅雜訊合成場景 + 已知模糊核）：
    ///   真的該擋的：AF 拉焦（σ 8px）比值 0.03~0.27；單幀手震（σ 4px）0.05
    ///   不該擋的：平移過均勻紋理 1.00；1.5s 內從書架平移到白牆，最低 0.53
    /// 0.4 落在兩群中間 —— 對真模糊有 ~2.5× 餘裕，離自然變化的地板還有 ~1.3× 空間。
    /// （原本設 0.6，校準後發現它只對應 σ≈0.6px 的模糊，等於要求近乎完美清晰，過嚴。）
    var minSharpnessRatio: Float = 0.4
    /// CMOS 捲簾快門讀完整幀所需時間（秒）。這**不是**曝光時間，是逐列讀出的跨度：
    /// 這段時間內相機還在動 → 幀內上下兩端對應不同姿態 = 剪切變形，縮短曝光救不到。
    /// 少了這一項，明亮環境（AE 縮到 1/250s）的快速轉動會被系統性低估 3 倍以上。
    /// iPhone 主鏡 video 模式實測約 8~15ms，取 10ms。
    /// 想校準：拍直立的門框並水平快速平移，量畫面上下兩端的傾斜角 θ，
    /// 則 readout ≈ tan(θ) × 畫面寬 / (角速度 × fx)。
    var rollingShutterReadoutS: Double = 1.0 / 100
    /// 環境照度下限（lux，ARKit lightEstimate；1000 為標準室內）
    var minAmbientLux: CGFloat = 150
    /// 環境照度上限（正對強光 / 戶外直射易過曝）
    var maxAmbientLux: CGFloat = 20_000
    /// 目標距離下限（LiDAR 最近有效距離約 0.25m）
    var minTargetDistanceM: Float = 0.25
    /// 目標距離上限（LiDAR 有效範圍約 5m，遠了深度品質下降）
    var maxTargetDistanceM: Float = 4.0

    // MARK: - 影像 / 深度輸出
    /// JPEG 品質。0.90 在「有雜訊的輸入」上會讓 8×8 區塊的量化誤差變成塊狀色斑，
    /// 看起來比底下的感光元件雜訊還醜 —— 而 ARKit 影像本來就有雜訊（見下）。
    /// 感光雜訊是零均值的、多視角平均會消掉；JPEG 量化誤差是固定在該幀的系統性誤差，
    /// 對光度損失是實打實的偏差。攝影測量慣例是 ≥0.95，故調上來（120 幀約 48MB → 72MB）。
    ///
    /// 顆粒感的**主因不在這裡**：ARKit 的 capturedImage 是 video 幀，完全繞過 iOS 的
    /// 運算攝影堆疊（沒有 Deep Fusion / Smart HDR / 多幀降噪）。相機 App 的照片乾淨是因為
    /// 那是 ~9 幀融合的結果；單張 video 幀在室內 ISO 400~800 下就是這個樣子，
    /// Scaniverse / Polycam 也一樣。訓練前還會 area-average 降到 1600（雜訊再降 ~1.2×），
    /// 且 3DGS 對同一表面吃 10~30 個視角、雜訊以 √N 收斂 —— 最終成品比任一單張都乾淨得多。
    var jpegQuality: Double = 0.95
    var saveDepth = true
    /// 高 ISO 時對存檔影像做輕度降噪的門檻。低於此值不處理。
    ///
    /// **為什麼只在高 ISO 才做** —— 3DGS 對同一表面吃 10~30 個視角，感光雜訊是零均值的，
    /// 光度損失收斂到的就是多視角平均，雜訊本來就會以 √N 消掉。
    /// 也就是說：對訓練而言，逐幀降噪能拿掉的東西「平均」本來就會拿掉，
    /// 但降噪順手削掉的真實細節，平均**救不回來** —— 純以訓練論，降噪是負分。
    /// 它真正值得做的地方有兩個：(a) 匯出的照片是給人看的；
    /// (b) 雜訊會製造假梯度，讓 MRNF 的密集化把高斯浪費在雜訊上（floaters）。
    /// 所以策略是「只在雜訊真的壓過細節時才動手，而且下手要輕」。
    /// ISO 400 以下的 iPhone 主鏡雜訊遠低於 JPEG 量化誤差，動它沒有意義。
    var denoiseISOThreshold: Double = 400
    /// 降噪強度上限（CINoiseReduction 的 inputNoiseLevel；Apple 預設 0.02）。
    /// 由 ISO 在 [threshold, 4×threshold] 之間以 log 內插到此值，超過就封頂。
    /// 設 0 等於關閉降噪。
    var denoiseMaxNoiseLevel: Double = 0.022
    // 相機參數鎖定（曝光/白平衡；對焦維持連續自動）為使用者可切換選項，
    // 見 CaptureController.lockCameraParams（預設開啟）

    // MARK: - 點雲累積（3DGS 初始化 + 即時預覽共用）
    /// voxel 去重格距（公尺）。1cm 在物件距離 0.5–2m 下有 Scaniverse 級的表面密度
    var voxelSizeM: Float = 0.01
    /// 記憶體內點數上限：觸頂時 voxel 自動 ×2 粗化後繼續收（長掃描不會停止累積）
    var maxPoints = 600_000
    /// 匯出點數上限：超過時分層擇優下採樣（每格取最高品質分數的點，密度均勻）
    var exportMaxPoints = 250_000
    /// COLMAP 輸出對齊世界上方向：ARKit 為 +Y up，多數 3DGS 工具假設 -Y up，
    /// 直接匯入會上下顛倒。true = 繞世界 X 軸翻 180° 對齊 COLMAP 慣例（預設，修正顛倒）。
    /// 若你的 viewer 反而變顛倒，設為 false 即輸出 ARKit 原生 +Y up。
    var flipWorldUpForExport = true
    /// 掃描後重融合的深度取樣步長（1 = 全像素，多視角加權平均品質最佳）
    var refuseSampleStride = 1
    /// 重融合 voxel 尺寸：比即時預覽（1cm）略粗，把遠距深度雜訊造成的「厚牆」塌成薄面。
    /// 想要最高細節設 0.01；房間尺度 3DGS 初始化 2cm 已足夠且更乾淨。
    var refuseVoxelSizeM: Float = 0.02
    /// 併入 ARKit 場景重建網格（ARMeshAnchor）的頂點作為補充幾何。
    /// 價值不在「更準」，而在**覆蓋率**：ARKit 的 mesh 融合的是每一幀（60fps）的深度，
    /// 而重融合只用 ~120 個關鍵幀 → mesh 會涵蓋關鍵幀沒拍到的表面（天花板/角落黑塊的主因）。
    /// mesh 頂點以低分插入，同格若有 LiDAR 觀測會由加權平均主導，只在「沒人拍到」處補洞。
    var useSceneMesh = true
    /// mesh 頂點的可見性容差（公尺）：投影到某關鍵幀後，與該幀深度圖差距在此範圍內才採用該幀顏色。
    /// 太小 → 幾乎上不到色；太大 → 被遮擋的背面也會被錯誤上色。
    var meshColorDepthTolM: Float = 0.10
    /// 掃描後重融合的格數上限。**不可**借用 maxPoints（那是即時預覽的記憶體上限）：
    /// 重融合在停止掃描後才跑、訓練尚未配置記憶體，可用預算大得多。借用 600k 會讓大範圍掃描
    /// 一路自動粗化（2→4→8→16cm）而且完全靜默 —— 初始點距被放大數倍，
    /// 而 msplat 的初始高斯尺寸就等於 3-NN 點距，從 15cm 起跳時 24 次 refine 的 LAS 分裂
    /// （實測每顆平均只切 ~1.2 次、縮小約 2 倍）根本追不回細節。
    /// Cell 約 40B + Dictionary 開銷 ≈ 70B/格。
    ///
    /// **2M → 4M，因為整層掃描會剛好踩線。** 100m² 住家（樓高 2.6m）的表面積
    /// ≈ 384m²（地板 100 + 天花 100 + 牆 104 + 家具 80）；LiDAR 深度雜訊讓表面
    /// 不只一格厚，融合後仍約 2 層 ⇒ 2cm 格下約 1.9M 格。
    /// 也就是說舊的 2M 上限對「整層」剛好會觸頂，然後靜默粗化成 4cm ——
    /// 而初始點距直接決定 msplat 的初始高斯大小（初始 scale = 3-NN 距離），
    /// 粗一倍就是初始高斯大一倍，密集化未必追得回來。
    /// 4M 格 ≈ 280MB，只在 post-scan 尖峰存在（refuse() 回傳後就釋放，訓練尚未配置），
    /// 而觸頂粗化仍然保底：真的超過就加粗，記憶體有界。
    var refuseMaxCells = 4_000_000
    /// 孤立點移除：占據 voxel 的 26 鄰域中占據數少於此值 → 視為飄浮雜點剔除（0 = 關閉）。
    /// 專清空間中不貼表面的白霧；過大會咬掉細線/薄物，3 為保守值。
    var refuseMinNeighbors = 3
    /// 只接受此信心等級以上的深度（ARConfidenceLevel：0=low, 1=medium, 2=high）。
    ///
    /// 從 2（只收 high）放寬到 1（收 medium）。medium 大多出現在物體邊緣、
    /// 深色表面與較遠處 —— 比較吵，但**不是錯的**，而且飛點過濾
    /// （depthEdgeRejectRatio）與孤立點移除本來就會擋掉真正的壞值。
    /// 實機 log 顯示 26.4% 的格子完全沒有 LiDAR 覆蓋、只靠 mesh 撐著；
    /// 在覆蓋率這麼吃緊的情況下，把可用但較吵的觀測整片丟掉並不划算 ——
    /// 收進來給低權重，讓多視角加權平均自己決定要不要相信它。
    var minDepthConfidence: UInt8 = 1
    /// medium 信心深度的分數倍率（high = 1.0）。
    /// 0.4 使得「一次 high 觀測」勝過「兩次 medium」，high 存在時由它主導；
    /// 只有在完全沒有 high 的格子，medium 才成為唯一來源 —— 那正是要補的洞。
    var mediumConfidenceWeight: Float = 0.4
    /// 深度圖取樣步長（256×192 下 stride 2 → 每次融合約 1.2 萬個候選點）
    var depthSampleStride = 2
    /// 點雲融合的深度有效範圍（LiDAR 超過 5m 雜訊明顯）
    var pointMinDepthM: Float = 0.15
    var pointMaxDepthM: Float = 5.0
    /// 飛點過濾：與相鄰像素深度差超過 depth×此比例 → 視為物體邊緣拖影，剔除
    var depthEdgeRejectRatio: Float = 0.05
    /// 入射角上限（度）。超過就不收這個深度樣本。
    ///
    /// **這是牆面疊影的主要對策。** 掠射時一個深度像素涵蓋牆面上一大片，
    /// 深度沿光線的誤差被 1/cosθ 放大，點會落在真實表面前後好幾公分；
    /// 每一趟掃描各偏一點，就在 voxel 格上排成一片片平行的殼
    /// —— 那就是畫面上的垂直條紋與「掃多次疊在一起」。
    ///
    /// 80° 刻意留得寬：硬拒絕會在只能斜看到的牆面上開洞。
    /// 真正在做事的是 cos²θ 的權重（80° → 3%）—— 之後有正面觀測進來時，
    /// 加權平均會被拉回正確的表面，而不是留著兩層殼。
    /// 想更乾淨可以收到 70~75°，代價是斜面覆蓋率下降。
    var depthMaxIncidenceDeg: Float = 80

    // MARK: - 即時點雲預覽（AR 疊加，Scaniverse 式）
    /// 每 N 個 ARFrame 融合一次（60fps → 每 0.1s），與智慧快門解耦，點雲連續長出
    var previewFrameInterval = 6
    /// 空間磚尺寸（公尺）：點雲按磚分塊渲染，每磚掛一個 ARAnchor ——
    /// ARKit 漂移修正 / 重定位時磚跟著移動，點雲不會與實體表面錯位（防殘影核心）
    var previewTileSizeM: Float = 1.2
    /// 點雲融合的模糊閘門（比抓幀遮斷更嚴）：模糊幀的姿態-深度錯位是殘影另一來源。
    /// 與 maxBlurPixels/blockBlurPixels 同步隨捲簾項重新校準（18 ≈ 0.68 rad/s）
    /// 預覽的三個閘門**直接沿用關鍵幀的門檻**，不再各自設一套。
    ///
    /// 先前預覽比關鍵幀嚴（模糊 18 vs 24px、角速度 0.6 vs 1.0、線速度 0.5 vs 0.8），
    /// 理由是「移動快 → 姿態延遲 → 殘影」。但後果是一個幀好到足以被存成關鍵幀、
    /// 進到匯出的點雲，卻不夠格顯示在預覽上 ——
    /// 使用者看到「我明明掃過這裡但沒有點」，而掃完之後那裡是有點的。
    ///
    /// 預覽的**唯一用途就是回報覆蓋率**。顯示得比實際收到的少是直接的誤導：
    /// 熱圖會把其實已經夠的區域標成缺點，害使用者回去重掃不需要重掃的地方。
    /// 殘影本來就另有機制在防（點雲磚掛 ARAnchor，漂移修正時整磚跟著移動）。
    ///
    /// 寫成衍生值而不是複製常數 —— 兩份各自維護的門檻遲早會再漂開。
    var previewMaxBlurPixels: Float { blockBlurPixels }
    var previewMaxAngularSpeedRadS: Float { keyframeMaxAngularSpeedRadS }
    var previewMaxLinearSpeedMS: Float { keyframeMaxLinearSpeedMS }

    // MARK: - 手機端 3DGS 訓練（msplat，on-device）
    /// 訓練迭代數（手機縮規模；桌機版通常 30000）。3–7k 在物件尺度已可觀。
    /// densify 視窗＝前半段（stopSplitAt=maxSteps/2），故迭代越多、densify 也跑越久 → 點更多。
    var trainIterations = 6000
    /// SH degree（0＝僅漫反射、最省記憶體；1＝基本視角相依高光）。手機建議 0–1
    var trainSHDegree = 3
    /// gaussian 緩衝容量（固定預配置＝峰值記憶體上限，slot 數）。densify 只在「3×當前數 ≤ 此值」時進行，
    /// 故最終高斯數約為此值的 1/3～1/2；峰值記憶體恆定＝此值（不再動態倍增/一次暴衝 → 不會 OOM）。
    /// 記憶體洩漏根治後（RAII + 每步 autorelease pool），峰值有界，iPhone Pro 有大量餘裕，
    /// 故調高到 600k（活躍高斯上限 ~200k，約 2.5×）換畫質。想更細可再往上（每 +100k slot 約 +40MB）；OOM 就調低。
    var trainMaxGaussians = 300_000
    /// 訓練影像額外固定降採樣倍率（1＝不額外降；與 trainMaxImageDim 取較強的縮小）。
    /// 一般維持 1.0，用 trainMaxImageDim 控制解析度即可。
    var trainDownscale: Float = 1.0
    /// 訓練影像最長邊上限（px；0＝不限）。手機端 12MP 光柵化是 O(像素數) → 又慢又爆記憶體；
    /// 業界標準（Inria 3DGS / gsplat / Scaniverse）皆限 ~1600，對 3DGS 成品畫質實質無損。
    /// 掃描原圖與匯出的 COLMAP/PLY 全部維持原解析度，只有「訓練當下餵給光柵化的影像」受此上限。
    /// 影像解碼一次後以 uint8 常駐（1600 全 42 幀約 244MB、有界、不再每輪重解碼）。
    /// 想更快更省 → 1280/1024；想壓最高細節 → 調高（記憶體與時間相應上升）。
    var trainMaxImageDim = 1600
    /// 訓練用關鍵幀數上限（0＝不限）。訓練時每幀以 uint8 常駐（1600px 每張 ~5.8MB）→ 幀數 × 5.8MB
    /// 是記憶體大宗；長掃描動輒數百幀就會 OOM。超過上限時沿拍攝軌跡「均勻抽樣」到此數（保留視角分佈）。
    /// 120 幀 ~690MB，room 尺度 3DGS 已足夠；OOM 就調小、想更細節就調大（留意記憶體）。
    var trainMaxFrames = 120
    /// 每幾步更新一次即時預覽 render（越小越即時、越吃效能）
    var trainPreviewEvery = 50
    /// 熱狀態達 .serious 以上時暫停訓練（散熱保護，避免 thermal shutdown）
    var trainThermalThrottle = true

    // MARK: - 平面圖（RoomPlan，與 3DGS 採集共用同一個 ARSession）
    /// 掃描時同步擷取 RoomPlan 平面圖，匯出時一併輸出 usdz / json / svg。
    ///
    /// 共生的前提我們本來就滿足（gravity 對齊、sceneDepth、mesh），所以「多掃一趟」的成本是零。
    /// 但**運算成本不是零** —— RoomPlan 會持續跑牆面偵測與物件分類。
    /// 本專案已經對散熱敏感（thermalState 到 .critical 會自動停拍），
    /// 長時間整屋掃描若發現提早過熱，這是第一個該關掉的開關。
    /// 只在支援 RoomPlan 的機型生效（見 FloorPlanCapture.isSupported）。
    ///
    /// **預設關閉。** 實機在辦公室隔間／貨架／桌面的場景下，RoomPlan 反覆給出
    /// 「2 面牆、樓高 0.80m」與方向亂掉的假牆 —— 它需要場景「是個房間」
    /// （有地板、成面的牆、牆與天花板的交界），而那個前提在這裡不成立。
    /// 平面圖改走 pointCloudFloorPlan：點雲只需要表面被掃到。
    ///
    /// 關掉會一併失去的東西（都是 RoomPlan 餵的，不是 bug）：
    ///   · 掃描時的即時發光線框與 dollhouse 縮圖
    ///   · 牆高不足 / 靠太近 / 光線不足的即時引導
    ///   · floorplan.usdz（帶門窗語意的 3D 幾何）
    ///   · 門窗與家具的語意分類
    /// 場景換成一般住宅／有完整牆面的房間時，把它設回 true 會明顯更好用。
    var captureFloorPlan = false

    /// 由 LiDAR 點雲直接產生平面圖（不經過 RoomPlan）。
    ///
    /// **與 captureFloorPlan 互相獨立，兩者可以同時開。** 它們的失效模式完全不同：
    ///   · RoomPlan 需要場景「是個房間」——有地板、成面的牆、牆與天花板的交界。
    ///     辦公室隔間、貨架、桌面前它會把螢幕邊桌緣硬判成牆，
    ///     而且第一片判錯之後後續的面會跟著它對齊。回報過 2 面牆、樓高 0.80m。
    ///   · 點雲只需要表面被掃到，但分不出門窗與家具語意（那要靠 RoomPlan）。
    ///
    /// 幾乎不花錢：重融合已經算好點雲了，這一步只是再掃一遍那些點
    /// （10 萬點約數十毫秒），而且只在匯出時做，不影響掃描。
    var pointCloudFloorPlan = true

    // MARK: - 局部 BA（**目前整條關閉**）
    /// 局部 BA 的輪數。**0 = 整條路徑關閉，掃描時的特徵抽取與匹配也一起不做。**
    ///
    /// 這一個 0 同時關掉兩處成本，這是它現在為 0 的主要理由：
    ///   · 掃描中：每個關鍵幀抽 2400 個角點 + 對最近 4 幀做引導匹配
    ///   · 停止後：observations() 建表 ＋ 兩次求解（閘門解 + 全量解）各 10 輪
    /// 而使用者要的是「掃完就看到結果」。位姿改用 ARKit ＋ 錨點修正，那是免費的。
    ///
    /// **關掉不是因為它做錯了。** 保留集交叉驗證（BundleAdjuster.kHoldoutGate）在
    /// 近距離掃描量到 −21%，也就是位姿真的變好；房間尺度則持平（位姿誤差已低於
    /// 觀測雜訊，1cm 誤差的重投影量 @0.5m 是 29px、@2m 只有 7px）。
    /// 也就是說它在一半的情況下有效 —— 只是那個效益換不到後處理的等待時間。
    ///
    /// 要重新評估：設成 10，log 會印出保留集判定（那是唯一不在 BA 目標函數裡的數字），
    /// 以及新的分段耗時。程式碼與 tools/test_bundle_adjust.swift（8 項）都留著。
    var baRounds = 0

    /// 允許 BA 改動位姿。決定權不在這裡，在保留集（BundleAdjuster.kHoldoutGate）——
    /// 每次掃描各自判定，過門檻才套用。baRounds = 0 時本欄無作用。
    var baApplyPoses = true

    // MARK: - 迴環閉合（降低累積漂移，投報率最高的一項）
    /// 走多遠之後開始提示「回起點閉環」（公尺）。
    ///
    /// 為什麼需要提示：ARKit 只有在認出「我來過這裡」時才會做全域修正，
    /// 把累積誤差攤回整條軌跡。走一條開放路徑不回頭的話，誤差一路累積、
    /// 而且**不會有任何警告** —— 姿態看起來一樣正常，錯的是全域尺度與朝向。
    /// 房間尺度的 VIO 漂移約軌跡長度的 0.5~2%，走 10m 就是 5~20cm。
    var loopHintTravelM: Float = 8
    /// 回到起點多近算閉合（公尺）。ARKit 的重定位需要看到相似的視野，
    /// 1.5m 內大致就會觸發；太小會讓提示永遠不消失。
    var loopClosedRadiusM: Float = 1.5

}
