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
    /// 背景寫入佇列上限（背壓）：滿載時跳過本幀，下一幀條件仍成立會再觸發
    var maxPendingWrites = 3

    // MARK: - 品質門檻（兩級制：警告 → 提醒但照拍；遮斷 → 暫停抓幀）
    /// 動態模糊警告門檻（像素）：blur ≈ (角速度 + 線速度/景深) × 焦距 × 曝光時間。
    /// 8px/1920 寬 ≈ 0.4%，輕微模糊、仍可訓練 —— 只給橘色提醒，不擋拍。
    var maxBlurPixels: Float = 8.0
    /// 模糊遮斷門檻：超過才紅色警告＋暫停抓幀（1/60s 曝光下約等於角速度 0.66 rad/s 的甩動）
    var blockBlurPixels: Float = 16.0
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
    // 相機參數鎖定（對焦/曝光/白平衡）為使用者可切換選項，
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

    // MARK: - 手機端 3DGS 訓練（msplat，on-device）
    /// 訓練迭代數（手機縮規模；桌機版通常 30000）。3–7k 在物件尺度已可觀
    var trainIterations = 4000
    /// SH degree（0＝僅漫反射、最省記憶體；1＝基本視角相依高光）。手機建議 0–1
    var trainSHDegree = 1
    /// gaussian 數硬上限（記憶體天花板）：densify 到頂即停。注意 densify 過程 ensureCapacity 會
    /// 暫時配置到 ~3× 目前數量，故峰值記憶體 ≈ 3× 此值的緩衝。12 萬在手機上是穩定起點。
    var trainMaxGaussians = 120_000
    /// 訓練影像降採樣倍率（1＝原解析度；4＝1/4 邊長 → 記憶體/時間約 1/16）。
    /// msplat 會把所有關鍵幀以 float32 常駐（全解析度 42 幀 ≈ 1.4GB），這是手機記憶體主因。
    /// ⚠️ 全解析度很吃記憶體，若閃退請調回 2～3。
    var trainDownscale: Float = 1.0
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
