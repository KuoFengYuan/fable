//
//  PointCloudAccumulator.swift
//  fable — LiDAR 點雲：擷取（PointExtractor）＋ 品質導向累積（actor）
//
//  即時性（Scaniverse 式）：擷取與智慧快門解耦，每 previewFrameInterval 個
//  ARFrame 融合一次（~10Hz）；PointExtractor 在 delegate 回呼內同步讀取
//  depth/conf/YUV buffer（不保留 ARFrame、零 buffer 複製）。
//
//  長時間掃描的密度控制（三層）：
//  1. voxel 內擇優：每點帶品質分數（近距離、畫面中心、低模糊 → 高分），
//     同格新點分數更高就靜默替換 —— 輸出永遠是該區域「拍得最好的一次」。
//  2. 觸頂自動粗化：點數達 maxPoints 時 voxel 尺寸 ×2 重建（保留每格最高分），
//     繼續累積 —— 長掃描永遠不會「新區域收不到點」，記憶體有界。
//  3. 匯出擇優下採樣：bestPoints(target:) 分層取最高分，見下。
//

import Foundation
import ARKit
import CoreVideo
import simd

// MARK: - 擷取（回呼內只做 memcpy，數學在 actor 執行緒）
//
// 重要教訓：反投影/融合若跑在 session delegate 回呼（主執行緒），
// Debug 組建下每次 10ms+ 會餓死 ARKit 的 VIO → 掉幀 → 追蹤不穩與漂移。
// 因此拆成兩段：makePacket（主執行緒，純 buffer 複製 ~2ms）→ extract（actor，計算）。

nonisolated enum PointExtractor {

    /// 跨執行緒的一幀融合封包：深度/信心為緊湊複本、影像為自有 pool 的 YUV 副本
    struct FramePacket: @unchecked Sendable {
        let depth: Data
        let confidence: Data?
        let yuv: CVPixelBuffer
        let depthWidth: Int
        let depthHeight: Int
        let intrinsics: CameraIntrinsics      // 全解析度
        let c2w: simd_float4x4
        let blurPixels: Float
    }

    /// 主執行緒：只做 buffer 複製（memcpy 為記憶體頻寬受限，不受最佳化等級影響）
    static func makePacket(frame: ARFrame, pool: CVPixelBufferPool,
                           blurPixels: Float) -> FramePacket? {
        guard let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth else { return nil }
        let depthMap = sceneDepth.depthMap
        let dw = CVPixelBufferGetWidth(depthMap)
        let dh = CVPixelBufferGetHeight(depthMap)
        guard dw > 0, dh > 0,
              let clone = PixelBufferUtil.clone(frame.capturedImage, pool: pool) else { return nil }
        let K = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        return FramePacket(
            depth: PixelBufferUtil.tightData(depthMap, bytesPerPixel: 4),
            confidence: sceneDepth.confidenceMap.map { PixelBufferUtil.tightData($0, bytesPerPixel: 1) },
            yuv: clone,
            depthWidth: dw,
            depthHeight: dh,
            intrinsics: CameraIntrinsics(fx: Double(K[0][0]), fy: Double(K[1][1]),
                                         cx: Double(K[2][0]), cy: Double(K[2][1]),
                                         width: Int(res.width), height: Int(res.height)),
            c2w: frame.camera.transform,
            blurPixels: blurPixels)
    }

    /// actor 執行緒：反投影 + 信心/範圍/飛點過濾 + 品質評分
    static func extract(_ packet: FramePacket, config: CaptureConfig) -> [CloudPoint] {
        let dw = packet.depthWidth
        let dh = packet.depthHeight
        let K = packet.intrinsics.scaled(toWidth: dw, height: dh)
        let fx = Float(K.fx), fy = Float(K.fy), cx = Float(K.cx), cy = Float(K.cy)
        let c2w = packet.c2w
        let conf = packet.confidence.map { [UInt8]($0) }
        let sampler = YUVSampler(packet.yuv)
        defer { sampler.unlock() }

        let stride = max(1, config.depthSampleStride)
        let minD = config.pointMinDepthM
        let maxD = config.pointMaxDepthM
        let minConf = config.minDepthConfidence
        let edgeRatio = config.depthEdgeRejectRatio
        let sharpness = 1 / (1 + packet.blurPixels / 4)

        return packet.depth.withUnsafeBytes { raw -> [CloudPoint] in
            let d = raw.bindMemory(to: Float32.self)
            var out: [CloudPoint] = []
            out.reserveCapacity((dw / stride) * (dh / stride))
            var v = 0
            while v < dh {
                var u = 0
                while u < dw {
                    let i = v * dw + u
                    let z = d[i]
                    if z.isFinite, z > minD, z < maxD, (conf?[i] ?? 2) >= minConf {
                        var ok = true
                        // 飛點過濾：物體輪廓的前後景插值拖影點
                        if u + 1 < dw {
                            let dr = d[i + 1]
                            if !dr.isFinite || abs(dr - z) > z * edgeRatio { ok = false }
                        }
                        if ok, v + 1 < dh {
                            let db = d[i + dw]
                            if !db.isFinite || abs(db - z) > z * edgeRatio { ok = false }
                        }
                        if ok {
                            // CV 反投影 → 翻 Y/Z 回 GL 相機系 → 世界
                            let xc = (Float(u) - cx) / fx * z
                            let yc = (Float(v) - cy) / fy * z
                            let w4 = c2w * SIMD4<Float>(xc, -yc, -z, 1)
                            let (r, g, b) = sampler.rgb(atNormalizedU: (Float(u) + 0.5) / Float(dw),
                                                        v: (Float(v) + 0.5) / Float(dh))
                            // 品質分數：畫面中心 × 近距離 × 清晰幀
                            let ru = (Float(u) - cx) / Float(dw)
                            let rv = (Float(v) - cy) / Float(dh)
                            let central = 1 - min(1, (ru * ru + rv * rv).squareRoot() * 1.4) * 0.5
                            let near = 1 / (0.5 + z)
                            out.append(CloudPoint(x: w4.x, y: w4.y, z: w4.z, r: r, g: g, b: b,
                                                  score: central * near * sharpness))
                        }
                    }
                    u += stride
                }
                v += stride
            }
            return out
        }
    }
}

// MARK: - 累積 actor

actor PointCloudAccumulator {

    private var grid: TiledFusedGrid
    private let config: CaptureConfig

    init(config: CaptureConfig) {
        self.config = config
        grid = TiledFusedGrid(voxelSize: config.voxelSizeM,
                              tileSize: config.previewTileSizeM,
                              maxCells: config.maxPoints)
    }

    var count: Int { grid.count }

    /// 反投影 + 過濾 + 融合（全部在 actor 執行緒，不佔 delegate 回呼）。
    /// anchorTransforms：各磚錨點當下變換（主執行緒每幀擷取），用於世界↔局部換算。
    func integrate(_ packet: PointExtractor.FramePacket,
                   anchorTransforms: [Int64: simd_float4x4]) {
        grid.insert(PointExtractor.extract(packet, config: config),
                    anchorTransforms: anchorTransforms)
    }

    /// 無 LiDAR 機種退路：累積 ARKit 稀疏特徵點（無色 → 中性灰、中等分數）
    func integrateSparse(points: [SIMD3<Float>], anchorTransforms: [Int64: simd_float4x4]) {
        grid.insert(points.map { CloudPoint(x: $0.x, y: $0.y, z: $0.z,
                                            r: 160, g: 160, b: 160, score: 0.5) },
                    anchorTransforms: anchorTransforms)
    }

    /// 取走待建錨磚（主執行緒據此建立 ARAnchor）
    func takePendingAnchors() -> [(Int64, SIMD3<Float>)] {
        grid.takePendingAnchors()
    }

    /// 取出至多 limit 個待刷新磚的 GPU-ready 渲染資料
    func dirtyTileRenderData(limit: Int, mode: PointColorMode = .rgb) -> [TileRenderData] {
        grid.popDirtyTiles(limit: limit).compactMap { grid.tileRenderData($0, mode: mode) }
    }

    /// 切換上色模式後呼叫：把所有磚標記為待重畫（否則只有之後變動的磚會換色）
    func markAllDirty() { grid.markAllDirty() }

    /// 匯出用擇優下採樣（無 LiDAR 時的備援輸出；LiDAR 路徑以 RefusionEngine 重融合為準）
    func bestPoints(target: Int) -> [CloudPoint] {
        grid.exportPoints(target: target)
    }
}
