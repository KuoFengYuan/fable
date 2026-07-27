//
//  CaptureConfig.swift
//  fable — COLMAP-free 3DGS capture
//

import Foundation
import CoreGraphics

/// 掃描模式：物件環繞（顯示涵蓋率圓頂）或場景漫遊（顯示軌跡緞帶）
nonisolated enum ScanMode: String, CaseIterable, Identifiable, Sendable {
    case object
    case scene

    var id: String { rawValue }
    var label: String { self == .object ? "物件" : "場景" }
}

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
    /// 動態模糊警告門檻（像素）：blur ≈ (角速度 + 線速度/景深) × 焦距 × 曝光時間。
    /// 8px/1920 寬 ≈ 0.4%，輕微模糊、仍可訓練 —— 只給橘色提醒，不擋拍。
    var maxBlurPixels: Float = 8.0
    /// 模糊遮斷門檻：超過才紅色警告＋暫停抓幀（1/60s 曝光下約等於角速度 0.66 rad/s 的甩動）
    var blockBlurPixels: Float = 16.0
    /// 關鍵幀的清晰度門檻：本幀清晰度 ÷「近 0.5s 內同場景的最佳清晰度」須達此比例。
    ///
    /// 為什麼需要它 —— blockBlurPixels 是**推估**（角速度 × 曝光時間），只涵蓋動態模糊，
    /// 對「失焦」完全盲目：手機靜止不動、對焦跑掉的畫面，推估值是 0，照樣被存成關鍵幀。
    /// 失焦與 AF 拉焦（重新啟用連續自動對焦後必然會發生）是「有些照片糊掉」的主因，
    /// 只有直接量影像才擋得住。
    ///
    /// 用相對值而非絕對值：清晰度與場景紋理量綁死（白牆對到極清晰也只有雜訊級的值），
    /// 絕對門檻在白牆上會全擋、在書架上會全過。
    /// 0.6 ≈ 允許三分之一的細節損失。調高更嚴（可能抓不到幀）、調低更寬鬆。
    var minSharpnessRatio: Float = 0.6
    /// 環境照度下限（lux，ARKit lightEstimate；1000 為標準室內）
    var minAmbientLux: CGFloat = 150
    /// 環境照度上限（正對強光 / 戶外直射易過曝）
    var maxAmbientLux: CGFloat = 20_000
    /// 目標距離下限（LiDAR 最近有效距離約 0.25m）
    var minTargetDistanceM: Float = 0.25
    /// 目標距離上限（LiDAR 有效範圍約 5m，遠了深度品質下降）
    var maxTargetDistanceM: Float = 4.0

    // MARK: - 影像 / 深度輸出
    var jpegQuality: Double = 0.90
    var saveDepth = true
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
    /// 用單目深度（Depth Anything V2 Small, Core ML）補上 LiDAR 沒有回波的區域。
    ///
    /// **預設關閉 —— 實測會產生殘影，而且是精度層級的問題，不是調參能解決的。**
    /// 多幀 voxel 融合要不殘影，每幀深度誤差須遠小於 voxel(2cm)：3m 處的 2cm ＝ 0.67% 相對誤差。
    /// MDE 經仿射對齊後的典型相對誤差是 5~10%（室內未見場景），樂觀取 2% 也有 6cm ＝ 3 個 voxel。
    /// 差一個數量級，且誤差隨視角改變 —— 沒有機制讓不同幀對同一表面達成 2cm 內的共識，
    /// 於是同一表面被各幀放到不同深度、疊成多層殼。LiDAR 則是 σ≈1cm@2m、多幀平均後 sub-cm，
    /// 剛好在門檻內。
    ///
    /// 另一個結構性問題：能可靠填的洞沒價值，有價值的洞填不了 ——
    ///   小洞（被 LiDAR 包圍）可靠，但相鄰高斯本來就會蓋過去；
    ///   大洞（窗戶/鏡面/>5m）才是真正缺的，卻正是沒有 LiDAR 錨定、最不可靠的地方。
    ///
    /// 程式碼與守衛都保留（見 RefusionEngine.mdeHoleFill、DepthScaleFit），要實驗可打開。
    /// 若要讓它真的可用，正解是「以洞的邊界 LiDAR 做局部仿射錨定」而非全幀單一 (a,b)，
    /// 但那只改善小洞——即上表中沒價值的那一類。
    var useMDEHoleFill = false
    /// 仿射對齊（1/z ≈ a·d + b）所需的最少 LiDAR 監督樣本數；不足則該幀不用 MDE
    var mdeMinSamples = 200
    /// 對齊殘差上限（逆深度空間，1/m）。超過代表該幀 MDE 與 LiDAR 不一致（反光/透明面誤導），
    /// 寧可放棄該幀也不要注入錯誤幾何
    var mdeMaxRMSE: Float = 0.08
    /// MDE 補洞點的分數倍率（LiDAR = 1.0、mesh = 0.25）。推論值，故低於量測值
    var mdeScore: Float = 0.4
    /// 每幀補洞點數上限（ROI-aware 取樣後）。避免大片無回波區灌爆點數預算
    var mdeMaxPointsPerFrame = 8000
    /// 鄰域支撐半徑（深度圖像素）：補洞像素附近須有 LiDAR 回波才採用。
    /// 這道守衛把 MDE 限制成「內插」——只補被已知深度包圍的小洞，不碰整片無回波區。
    /// 沒有它的話，窗戶/鏡面那種大片無回波區會由每幀各自的仿射擬合給出彼此不一致的深度
    /// （沒有 LiDAR 錨定），同一表面在不同幀落到不同 voxel → 疊成多層殼＝殘影。
    /// 3 個深度像素在 256×192 下約對應 2° 視角；調大會補更多但殘影風險上升。
    var mdeSupportRadiusPx = 3
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
    /// Cell 約 40B + Dictionary 開銷 ≈ 70B/格；2M 格 ≈ 140MB，post-scan 尖峰可接受。
    var refuseMaxCells = 2_000_000
    /// 孤立點移除：占據 voxel 的 26 鄰域中占據數少於此值 → 視為飄浮雜點剔除（0 = 關閉）。
    /// 專清空間中不貼表面的白霧；過大會咬掉細線/薄物，3 為保守值。
    var refuseMinNeighbors = 3
    /// 只接受此信心等級以上的深度（2 = ARConfidenceLevel.high）
    var minDepthConfidence: UInt8 = 2
    /// 深度圖取樣步長（256×192 下 stride 2 → 每次融合約 1.2 萬個候選點）
    var depthSampleStride = 2
    /// 點雲融合的深度有效範圍（LiDAR 超過 5m 雜訊明顯）
    var pointMinDepthM: Float = 0.15
    var pointMaxDepthM: Float = 5.0
    /// 飛點過濾：與相鄰像素深度差超過 depth×此比例 → 視為物體邊緣拖影，剔除
    var depthEdgeRejectRatio: Float = 0.05

    // MARK: - 即時點雲預覽（AR 疊加，Scaniverse 式）
    /// 每 N 個 ARFrame 融合一次（60fps → 每 0.1s），與智慧快門解耦，點雲連續長出
    var previewFrameInterval = 6
    /// 空間磚尺寸（公尺）：點雲按磚分塊渲染，每磚掛一個 ARAnchor ——
    /// ARKit 漂移修正 / 重定位時磚跟著移動，點雲不會與實體表面錯位（防殘影核心）
    var previewTileSizeM: Float = 1.2
    /// 點雲融合的模糊閘門（比抓幀遮斷更嚴）：模糊幀的姿態-深度錯位是殘影另一來源
    var previewMaxBlurPixels: Float = 12
    /// 融合的「相機穩定度」閘門：只在相機夠穩時才把深度融進點雲。
    /// 模糊值受曝光時間影響（亮處曝光短→快速移動仍判定為低模糊），無法反映「姿態延遲/誤差」；
    /// 移動過快時姿態不準 → 同一表面被投影到不同世界座標 → 殘影＋點不貼合。故另設速度硬閘門。
    /// 角速度上限（rad/s）：~0.6 允許正常掃描平移、擋掉甩動。
    var previewMaxAngularSpeedRadS: Float = 0.6
    /// 線速度上限（m/s）：~0.5 允許走動掃描、擋掉「來回快步」造成的殘影。想更乾淨可調小（但覆蓋變慢）。
    var previewMaxLinearSpeedMS: Float = 0.5

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

    // MARK: - 涵蓋率圓頂（物件模式）
    var domeAzimuthBins = 24
    var domeElevationBins = 5
    var domeElevationMaxDeg: Float = 75
    /// 相機視線需與「相機→物件中心」夾角小於此值，該視角格才算已涵蓋
    var domeViewAngleDeg: Float = 35
}
