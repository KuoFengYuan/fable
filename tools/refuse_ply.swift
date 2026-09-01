//
//  refuse_ply.swift
//  fable — 在桌機上重跑掃描後的點雲融合，用來掃參數
//
//  重融合本來就是離線的：它只吃 depth/*.bin、images/*.jpg 與 poses_refined.jsonl，
//  全都留在掃描資料夾裡。所以「換一組參數會不會更好」不必重掃 —— 直接重跑，
//  再拿 tools/scan_accuracy.py 量結果。跑的是**與 App 完全同一份** RefusionEngine。
//
//  唯一跑不到的是 ARKit 場景網格（ARMeshAnchor 的頂點沒有存檔），
//  所以輸出會比 App 少了「mesh 補洞」那一份覆蓋。要跟 App 的 points.ply 比點數時
//  記得這一點；參數之間互相比則不受影響（每一組都同樣少了 mesh）。
//
//    swiftc -O -o /tmp/refuse fable/Capture/Models.swift fable/Capture/CaptureConfig.swift \
//           fable/Capture/BlurFilter.swift fable/Capture/RefusionEngine.swift \
//           tools/refuse_ply.swift
//    /tmp/refuse <scan_dir> [out.ply] [key=value ...]
//
//  可調的 key 就是 CaptureConfig 的欄位名，例如：
//    /tmp/refuse ~/scan out.ply blurWeightPower=3 depthMaxIncidenceDeg=65
//

import Foundation
import simd

@main struct RefusePLY {

    /// 只開放「影響融合結果」的欄位。刻意不做成通用反射：
    /// 打錯字要當場報錯，而不是安靜地什麼都沒改 —— 掃參數時那種靜默失敗
    /// 會讓人以為「這個參數沒有效果」，是最貴的一種錯。
    static func apply(_ kv: [String: Float], to c: inout CaptureConfig) throws {
        for (k, v) in kv {
            switch k {
            case "blurWeightPower":       c.blurWeightPower = v
            case "blurWeightHalfPx":      c.blurWeightHalfPx = v
            case "blurWeightRefPx":       c.blurWeightRefPx = v
            case "depthMaxIncidenceDeg":  c.depthMaxIncidenceDeg = v
            case "refuseVoxelSizeM":      c.refuseVoxelSizeM = v
            case "refuseSampleStride":    c.refuseSampleStride = Int(v)
            case "refuseMinNeighbors":    c.refuseMinNeighbors = Int(v)
            case "refuseMaxCells":        c.refuseMaxCells = Int(v)
            case "minDepthConfidence":    c.minDepthConfidence = UInt8(max(0, min(2, v)))
            case "mediumConfidenceWeight": c.mediumConfidenceWeight = v
            case "pointMinDepthM":        c.pointMinDepthM = v
            case "pointMaxDepthM":        c.pointMaxDepthM = v
            case "depthEdgeRejectRatio":  c.depthEdgeRejectRatio = v
            case "exportMaxPoints":       c.exportMaxPoints = Int(v)
            case "blockBlurPixels":       c.blockBlurPixels = v
            default:
                throw Fail("不認得的參數 '\(k)'。可用：blurWeightPower/blurWeightHalfPx/"
                           + "blurWeightRefPx/depthMaxIncidenceDeg/refuseVoxelSizeM/"
                           + "refuseSampleStride/refuseMinNeighbors/refuseMaxCells/"
                           + "minDepthConfidence/mediumConfidenceWeight/pointMinDepthM/"
                           + "pointMaxDepthM/depthEdgeRejectRatio/exportMaxPoints/blockBlurPixels")
            }
        }
    }

    struct Fail: Error, CustomStringConvertible {
        let description: String
        init(_ s: String) { description = s }
    }

    /// 與 ExportManager.writePLY 同一個格式（binary LE，float xyz ＋ uchar rgb）。
    /// 這裡自己寫一份而不是拉 ExportManager 進來：那支檔案依賴 UIKit / zip 打包，
    /// 為了 30 行的 header 把整條匯出路徑搬到桌機不划算。
    static func writePLY(_ points: [CloudPoint], to url: URL) throws {
        var header = "ply\nformat binary_little_endian 1.0\n"
        header += "element vertex \(points.count)\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        var data = Data(header.utf8)
        data.reserveCapacity(data.count + points.count * 15)
        for p in points {
            withUnsafeBytes(of: p.x.bitPattern.littleEndian) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: p.y.bitPattern.littleEndian) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: p.z.bitPattern.littleEndian) { data.append(contentsOf: $0) }
            data.append(contentsOf: [p.r, p.g, p.b])
        }
        try data.write(to: url, options: [.atomic])
    }

    static func main() {
        let args = Array(CommandLine.arguments.dropFirst())
        guard let dirArg = args.first else {
            print("用法: refuse_ply <scan_dir> [out.ply] [key=value ...]")
            exit(1)
        }
        let dir = URL(fileURLWithPath: (dirArg as NSString).expandingTildeInPath)
        let rest = args.dropFirst()
        let outArg = rest.first(where: { !$0.contains("=") })
        var kv: [String: Float] = [:]
        for a in rest where a.contains("=") {
            let parts = a.split(separator: "=", maxSplits: 1)
            guard parts.count == 2, let v = Float(parts[1]) else {
                print("參數格式錯誤: '\(a)'（要 key=數值）"); exit(1)
            }
            kv[String(parts[0])] = v
        }

        // poses_refined.jsonl 才是融合真正用的姿態（ARKit ＋ 錨點修正後）。
        // poses.jsonl 是掃描當下的即時值，拿它重跑會比 App 的結果更差而且不可比。
        let posesURL = dir.appendingPathComponent("poses_refined.jsonl")
        guard let text = try? String(contentsOf: posesURL, encoding: .utf8) else {
            print("讀不到 \(posesURL.path)"); exit(1)
        }
        let dec = JSONDecoder()
        var records: [FrameRecord] = []
        for line in text.split(separator: "\n") where !line.isEmpty {
            guard let r = try? dec.decode(FrameRecord.self, from: Data(line.utf8)) else { continue }
            records.append(r)
        }
        guard !records.isEmpty else { print("poses_refined.jsonl 裡沒有可用的幀"); exit(1) }

        var config = CaptureConfig()
        do { try apply(kv, to: &config) } catch { print("\(error)"); exit(1) }

        let shown = kv.isEmpty ? "（全部預設）"
                               : kv.sorted { $0.key < $1.key }
                                   .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
        print("幀 \(records.count)  參數 \(shown)")
        let blurs = records.map { Float($0.estimatedBlurPx) }.sorted()
        let w = blurs.map { RefusionEngine.blurWeight($0, config) }
        print(String(format: "模糊權重 最好 %.4f / 中位 %.4f / 最差 %.4f → 比值 %.1f×",
                     w.last ?? 0, w[w.count / 2], w.first ?? 0,
                     (w.last ?? 1) / max(1e-9, w.first ?? 1)))

        // progress 由多條 lane 呼叫，所以節流用的狀態要上鎖 —— 不鎖會是 Swift 6 的錯誤。
        final class Ticker: @unchecked Sendable {
            private let lock = NSLock()
            private var last = -1
            func step(_ p: Double) {
                let pct = Int(p * 100) / 10 * 10
                lock.lock()
                let show = pct > last
                if show { last = pct }
                lock.unlock()
                if show { FileHandle.standardError.write(Data("\(pct)% ".utf8)) }
            }
        }
        let t0 = Date()
        let ticker = Ticker()
        let points = RefusionEngine.refuse(records: records, sessionDir: dir,
                                           config: config) { ticker.step($0) }
        FileHandle.standardError.write(Data("\n".utf8))
        print(String(format: "融合 %.1fs → %d 點", Date().timeIntervalSince(t0), points.count))
        guard !points.isEmpty else { print("沒有產出任何點"); exit(1) }

        let out = URL(fileURLWithPath: outArg ?? dir.appendingPathComponent("points_refused.ply").path)
        do { try writePLY(points, to: out) } catch { print("寫檔失敗: \(error)"); exit(1) }
        print("已寫出 \(out.path)")
    }
}
