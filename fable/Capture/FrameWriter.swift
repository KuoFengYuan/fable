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
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private let encoder: JSONEncoder
    private(set) var records: [FrameRecord] = []

    init(sessionDir: URL, saveDepth: Bool, jpegQuality: Double) throws {
        self.jpegQuality = jpegQuality

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
            let ci = CIImage(cvPixelBuffer: kf.pixelBuffer)
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
