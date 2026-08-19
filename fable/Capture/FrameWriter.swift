//
//  FrameWriter.swift
//  fable — 非同步關鍵幀寫入（actor：JPEG 編碼 / 深度 raw / poses.jsonl）
//
//  actor 自帶序列化執行緒，主執行緒只做 buffer memcpy（~1-2ms @ 2Hz），
//  JPEG 編碼（~15-30ms）與磁碟 I/O 全部在此背景進行，UI 不掉幀。
//  背壓由 CaptureController.pendingWrites 控制，佇列滿時直接跳過該幀。
//

import Foundation
import CoreImage
import CoreVideo
import ImageIO

actor FrameWriter {

    private let imagesDir: URL
    private let depthDir: URL?
    private var posesHandle: FileHandle?
    private let jpegQuality: Double
    private let denoiseISOThreshold: Double
    private let denoiseMaxNoiseLevel: Double
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private let encoder: JSONEncoder
    private(set) var records: [FrameRecord] = []

    init(sessionDir: URL, saveDepth: Bool, jpegQuality: Double,
         denoiseISOThreshold: Double = .infinity, denoiseMaxNoiseLevel: Double = 0) throws {
        self.jpegQuality = jpegQuality
        self.denoiseISOThreshold = denoiseISOThreshold
        self.denoiseMaxNoiseLevel = denoiseMaxNoiseLevel

        imagesDir = sessionDir.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: imagesDir, withIntermediateDirectories: true)

        if saveDepth {
            let d = sessionDir.appendingPathComponent("depth", isDirectory: true)
            try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
            depthDir = d
        } else {
            depthDir = nil
        }

        let posesURL = sessionDir.appendingPathComponent("poses.jsonl")
        FileManager.default.createFile(atPath: posesURL.path, contents: nil)
        posesHandle = try FileHandle(forWritingTo: posesURL)

        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        encoder = enc
    }

    func write(_ kf: Keyframe) {
        var record = kf.record
        do {
            // JPEG 編碼（sensor 原始方向，與 intrinsics / transform 自洽）
            let src = CIImage(cvPixelBuffer: kf.pixelBuffer)
            var ci = src
            if let level = noiseLevel(forISO: record.iso) {
                // inputSharpness 刻意壓在 0.2（Apple 預設 0.4）：降噪後的再銳化會沿邊緣造光暈，
                // 那是憑空生出來、且各幀不一致的高頻 —— 3DGS 會試圖用高斯去解釋它。
                ci = src.applyingFilter("CINoiseReduction",
                                        parameters: ["inputNoiseLevel": level,
                                                     "inputSharpness": 0.2])
                        .cropped(to: src.extent)
            }
            let qualityKey = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
            guard let jpeg = ciContext.jpegRepresentation(of: ci, colorSpace: colorSpace,
                                                          options: [qualityKey: jpegQuality]) else {
                print("[FrameWriter] JPEG 編碼失敗 frame \(record.id)")
                return
            }
            try jpeg.write(to: imagesDir.appendingPathComponent(record.imageFile), options: [.atomic])

            // 深度 / 信心圖（raw little-endian，Python 端 np.fromfile 直接讀）
            if let depthDir, let depthData = kf.depthData, let depthName = record.depthFile {
                try depthData.write(to: depthDir.appendingPathComponent(depthName))
                if let confData = kf.confidenceData, let confName = record.confidenceFile {
                    try confData.write(to: depthDir.appendingPathComponent(confName))
                }
            } else {
                record.depthFile = nil
                record.confidenceFile = nil
            }

            // poses.jsonl：一幀一行，即寫即 flush 到 handle（中途 crash 也不丟已拍資料）
            let line = try encoder.encode(record)
            try posesHandle?.write(contentsOf: line)
            try posesHandle?.write(contentsOf: Data([0x0A]))
            records.append(record)
        } catch {
            print("[FrameWriter] 寫入失敗 frame \(record.id): \(error)")
        }
    }

    /// ISO → 降噪強度。門檻以下回 nil（完全不套濾鏡，連 GPU pass 都省）。
    /// 強度隨 ISO 以 log2 內插：門檻處 0、4× 門檻處封頂。
    /// 用 log 而非線性，是因為雜訊的標準差大致 ∝ √ISO、感知上也是對數的。
    private func noiseLevel(forISO iso: Double) -> Double? {
        guard denoiseMaxNoiseLevel > 0, iso > denoiseISOThreshold else { return nil }
        let t = min(1, log2(iso / denoiseISOThreshold) / 2)     // 2 個 stop 到頂
        let level = denoiseMaxNoiseLevel * t
        return level > 1e-4 ? level : nil
    }

    /// 讀取目前累積的紀錄（不關檔 —— review 後仍可「繼續掃描」續寫）
    func snapshotRecords() -> [FrameRecord] {
        records
    }

    /// 關檔並回傳全部成功寫入的紀錄（確認匯出時呼叫）
    func finish() -> [FrameRecord] {
        try? posesHandle?.close()
        posesHandle = nil
        return records
    }
}
