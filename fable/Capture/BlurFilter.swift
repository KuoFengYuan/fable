//
//  BlurFilter.swift
//  fable — 掃描結束後的模糊幀複核（全域、可回頭看的第二道關）
//
//  為什麼即時閘門之外還需要這一道：即時閘門必須「當下就決定」，只能拿本幀跟
//  近 0.5 秒的記憶比。掃完之後的位置好得多 —— 全部的幀都在手上，於是可以問一個
//  即時閘門問不出來的問題：
//
//      「在**看同一片表面**的那幾張裡，這張是不是最糊的？」
//
//  這才是正確的比較基準。清晰度的絕對值與場景紋理量綁死（實測跨場景差 44×，
//  見 tools/test_sharpness.py），所以全域絕對門檻一定是錯的：從書架搖到白牆，
//  白牆那幾張會被整批誤殺。但「位置相近、朝向相近」的幀看的是同一片東西，
//  它們的清晰度可以直接比。
//
//  同樣重要的是它能在丟棄前先確認**丟了不會開天窗** —— 鄰居不夠就一律保留。
//  即時閘門永遠做不到這件事，因為它不知道後面還會不會有人補拍這個角度。
//
//  ── 兩種模糊要分開處理，因為對下游的傷害方式不同 ──────────────────
//
//  1. 不適合當訓練影像（失焦、或運動模糊 >10px）：影響的是「影像鋭利度」。
//     幾何來自 LiDAR，跟 RGB 糊不糊無關；而 10px 的運動換算成世界誤差是
//     1.4cm @ 2m —— 小於一個 voxel。→ 不進訓練，深度**照樣收進點雲**。
//     丟掉它的深度只會白白開洞。
//
//  2. 幾何真的不可信（運動 >25px ≈ 3.4cm @ 2m）：影響的是「位置」。
//     深度會被反投影到**錯的世界座標**，疊出殘影／雙層殼。
//     殘影比破洞更糟 —— 破洞看得出來，殘影會被當成真的幾何。→ 連點雲一起丟。
//

import Foundation
import simd

nonisolated enum BlurVerdict: String, Codable, Sendable {
    /// 正常採用
    case keep
    /// 不當訓練影像（失焦、或運動模糊 10~25px），但**深度照常併入點雲**
    case demote
    /// 幾何也不可信（運動 >25px）：訓練與點雲都不用
    case drop
}

nonisolated enum BlurFilter {

    /// 鄰居判定：相機位置需在此距離內（公尺）。0.5m 下視差還小，畫面內容大致相同。
    static let kNeighborRadiusM: Float = 0.5
    /// 且視線夾角需在此範圍內（度）。30° 內看的大致是同一片表面。
    static let kNeighborConeDeg: Float = 30
    /// 鄰居少於此數就一律保留 —— 這是「不開天窗」的守門條件，不可為 0。
    static let kMinNeighbors = 2
    /// 本幀清晰度需達鄰域第 75 百分位的此倍率，否則判 demote。
    /// 0.5 比即時閘門的 0.4 寬鬆一點：空間鄰居的畫面內容本來就不完全相同，
    /// 變異比「同一場景的前後幀」大，門檻收太緊會誤殺。
    static let kSharpRatio: Float = 0.5
    /// **訓練影像**的幾何劣化上限（px）。超過就不當訓練影像 —— 影像不鋭利是
    /// 實打實的資訊損失，多視角平均救不回來。
    static let kTrainBlurPx: Double = 10

    /// **幾何（點雲）**的幾何劣化上限（px）。比訓練寬得多，因為換算成實際誤差後
    /// 根本不到一個 voxel：
    ///
    ///     世界橫向誤差 = blur_px × z / fx
    ///     10px @ 2m, fx=1450 → 1.4 cm   ← 小於 refuse voxel（2cm）
    ///     25px @ 2m          → 3.4 cm   ← 約 1.7 voxel，加權平均吃得下
    ///
    /// 先前兩者共用 10px，等於「用小於一個體素的誤差」丟掉整幀的深度 ——
    /// 實機 log 顯示這樣丟掉了 30% 的幀，而同一份 log 的覆蓋率是
    /// 「26.4% 的格子沒有 LiDAR、只靠 mesh 撐著」。那些深度本來就該收。
    /// 而且運動模糊的幀在 refusion 裡本來就有物理權重 1/(1+blur/4) 壓著
    /// （15px → 0.21），不需要再用硬門檻砍一次。
    static let kGeomBlurPx: Double = 25
    /// 丟棄（drop + demote）的總量上限。整段都拍糊時，全丟比留著更糟：
    /// 那代表使用者需要重拍，而不是我們該把資料集清空。
    static let kMaxRejectFraction: Double = 0.3

    /// 回傳每個 record.id 的判定。純函式、不碰檔案，方便單獨推理與測試。
    static func evaluate(_ records: [FrameRecord]) -> [Int: BlurVerdict] {
        var verdict = [Int: BlurVerdict](minimumCapacity: records.count)
        for r in records { verdict[r.id] = .keep }
        guard records.count > kMinNeighbors else { return verdict }

        let pos = records.map { position(of: $0) }
        let fwd = records.map { forward(of: $0) }
        let cosCone = cos(kNeighborConeDeg * .pi / 180)
        let r2 = kNeighborRadiusM * kNeighborRadiusM

        // 候選（連同嚴重度）先收集，最後才依上限截斷 —— 確保被丟掉的是最差的那些
        var candidates: [(id: Int, v: BlurVerdict, severity: Float)] = []

        for i in records.indices {
            var peers: [Float] = []
            for j in records.indices where j != i {
                guard simd_distance_squared(pos[i], pos[j]) <= r2,
                      simd_dot(fwd[i], fwd[j]) >= cosCone else { continue }
                // 清晰度量不到的幀（sharpness <= 0，例如舊版 App 拍的）不能當比較基準
                let sj = Float(records[j].sharpness)
                if sj > 0 { peers.append(sj) }
            }
            // 鄰居不夠 → 這是該視角唯一（或幾乎唯一）的觀測，再糊也得留
            guard peers.count >= kMinNeighbors else { continue }

            let blur = records[i].estimatedBlurPx
            // 幾何都不能用：只有真的大到超過一個多 voxel 才算
            if blur > kGeomBlurPx {
                candidates.append((records[i].id, .drop, Float(blur)))
                continue
            }
            // 影像不夠鋭利但深度仍可用 → demote（不進訓練，深度照收）
            if blur > kTrainBlurPx {
                candidates.append((records[i].id, .demote, Float(blur / kGeomBlurPx)))
                continue
            }
            let ref = percentile(peers, 0.75)
            let s = Float(records[i].sharpness)
            // s <= 0 表示本幀沒有量到清晰度 → 沒有證據，不判它有罪
            if s > 0, ref > 1e-6, s < kSharpRatio * ref {
                candidates.append((records[i].id, .demote, 1 - s / ref))   // 越大越嚴重
            }
        }

        // 上限：最嚴重的先丟。drop 排在 demote 前面（幾何錯誤比顏色糊嚴重）
        let limit = Int(Double(records.count) * kMaxRejectFraction)
        let ordered = candidates.sorted { a, b in
            if (a.v == .drop) != (b.v == .drop) { return a.v == .drop }
            return a.severity > b.severity
        }
        for c in ordered.prefix(limit) { verdict[c.id] = c.v }
        return verdict
    }

    /// 把判定寫回 records（供 poses_refined.jsonl 保留完整資訊，不是靜靜刪掉）
    static func annotate(_ records: [FrameRecord]) -> [FrameRecord] {
        let v = evaluate(records)
        return records.map { r in
            var out = r
            out.blurVerdict = v[r.id] ?? .keep
            return out
        }
    }

    // MARK: - 小工具

    private static func position(of r: FrameRecord) -> SIMD3<Float> {
        SIMD3(Float(r.transform[3]), Float(r.transform[7]), Float(r.transform[11]))
    }

    /// GL 相機慣例：視線方向為 -Z，即 row-major c2w 第 2 欄取負
    private static func forward(of r: FrameRecord) -> SIMD3<Float> {
        let v = SIMD3<Float>(-Float(r.transform[2]), -Float(r.transform[6]), -Float(r.transform[10]))
        let n = simd_length(v)
        return n > 1e-6 ? v / n : SIMD3<Float>(0, 0, -1)
    }

    private static func percentile(_ values: [Float], _ p: Float) -> Float {
        let s = values.sorted()
        let idx = min(s.count - 1, max(0, Int((Float(s.count - 1) * p).rounded())))
        return s[idx]
    }
}
