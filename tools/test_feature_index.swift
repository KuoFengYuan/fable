//
//  test_feature_index.swift
//  fable — 特徵匹配空間索引的離線驗證（不需要 iPhone）
//
//  為什麼這支測試存在：把「掃過全部特徵」換成「只掃 3×3 格」是為了讓
//  maxFeatures 從 600 提到 2000（600→2000 會讓距離檢查從 78M 變 864M，
//  而抽取＋匹配的預算是每關鍵幀 ~5ms）。
//
//  但索引化搜尋若少看了某一格，**症狀是「匹配數變少」而不是崩潰** ——
//  而少掉的那些本來就是稀少的匹配，很容易被當成場景難度而不是 bug。
//  這個檔案已經有四個「症狀是沒效果而非壞掉」的前例，所以逐點對照暴力搜尋。
//
//  編譯：
//    swiftc -O -o /tmp/fitest fable/Capture/FeatureTracker.swift \
//           fable/Capture/BundleAdjuster.swift fable/Capture/PoseRefiner.swift \
//           tools/test_feature_index.swift tools/test_stubs_core.swift && /tmp/fitest
//

import Foundation
import simd

@main struct FeatureIndexTest {

    static var seed: UInt64 = 20260820
    static func rnd() -> Float {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return Float(seed >> 33) / Float(UInt32.max >> 1)
    }

    /// 造一個特徵。patch 用隨機值 —— ZNCC 的實際數值不重要，
    /// 重要的是索引化與暴力搜尋**挑到同一個**。
    static func feature(u: Float, v: Float, taken: Bool = false) -> TrackedFeature {
        let side = FeatureParams.patchSide
        var patch = [UInt8](repeating: 0, count: side * side)
        var sum: Float = 0
        for i in 0..<patch.count { patch[i] = UInt8(rnd() * 255); sum += Float(patch[i]) }
        let mean = sum / Float(patch.count)
        var sq: Float = 0
        for p in patch { let d = Float(p) - mean; sq += d * d }
        return TrackedFeature(u: u, v: v, depth: 1 + rnd() * 3,
                              world: SIMD3<Float>(rnd(), rnd(), rnd()),
                              patch: patch, mean: mean,
                              invNorm: 1 / max(1e-6, sq.squareRoot()),
                              trackID: taken ? 7 : -1)
    }

    /// 暴力搜尋：索引化之前的原始語意
    static func bruteForce(for pf: TrackedFeature, at p: (Float, Float),
                           in feats: [TrackedFeature]) -> (idx: Int, best: Float, second: Float) {
        let r2 = FeatureParams.searchRadius * FeatureParams.searchRadius
        var bestIdx = -1
        var best: Float = -1, second: Float = -1
        for (i, f) in feats.enumerated() where f.trackID < 0 {
            let du = f.u - p.0, dv = f.v - p.1
            if du * du + dv * dv > r2 { continue }
            let s = FeatureExtractor.zncc(pf, f)
            if s > best { second = best; best = s; bestIdx = i }
            else if s > second { second = s }
        }
        return (bestIdx, best, second)
    }

    static func main() {
        var fails = 0
        func check(_ ok: Bool, _ what: String) {
            print(ok ? "✅ \(what)" : "❌ \(what)"); if !ok { fails += 1 }
        }

        let W: Float = 1920, H: Float = 1440

        // ── 1. 索引化 vs 暴力：畫面內的隨機查詢，必須逐點一致 ──
        // 密度取實機提高後的量級（每幀 ~900 個存活特徵）
        var feats = (0..<900).map { _ in feature(u: rnd() * W, v: rnd() * H) }
        // 混入已歸屬的特徵：兩邊都必須跳過（trackID >= 0）
        feats += (0..<100).map { _ in feature(u: rnd() * W, v: rnd() * H, taken: true) }
        let index = FeatureTracker.buildIndex(feats)

        var mismatch = 0, withCandidate = 0
        let probe = feature(u: 0, v: 0)
        for _ in 0..<20000 {
            let p = (rnd() * W, rnd() * H)
            let a = FeatureTracker.bestMatch(for: probe, at: p, in: feats, index: index)
            let b = bruteForce(for: probe, at: p, in: feats)
            if b.idx >= 0 { withCandidate += 1 }
            // 下標可能因掃描順序不同而不同，但「挑到的分數」必須一致
            if a.idx != b.idx || abs(a.best - b.best) > 1e-5 { mismatch += 1 }
        }
        check(mismatch == 0,
              "畫面內 20000 次查詢：索引化與暴力搜尋完全一致"
              + "（其中 \(withCandidate) 次有候選，\(mismatch) 次不符）")

        // ── 2. 畫面**外**的投影：project() 允許到 -searchRadius，
        //      而負格索引在 `y << 16 | x` 的打包裡會讓符號位吃掉另一個欄位 ──
        var edgeMismatch = 0, edgeFound = 0
        for _ in 0..<8000 {
            // 四邊之外、但仍在 searchRadius 內
            let r = FeatureParams.searchRadius
            let p: (Float, Float)
            switch Int(rnd() * 4) % 4 {
            case 0:  p = (-rnd() * r, rnd() * H)
            case 1:  p = (W + rnd() * r, rnd() * H)
            case 2:  p = (rnd() * W, -rnd() * r)
            default: p = (rnd() * W, H + rnd() * r)
            }
            let a = FeatureTracker.bestMatch(for: probe, at: p, in: feats, index: index)
            let b = bruteForce(for: probe, at: p, in: feats)
            if b.idx >= 0 { edgeFound += 1 }
            if a.idx != b.idx || abs(a.best - b.best) > 1e-5 { edgeMismatch += 1 }
        }
        check(edgeMismatch == 0,
              "畫面外投影（負座標）8000 次一致（其中 \(edgeFound) 次有候選，"
              + "\(edgeMismatch) 次不符）")

        // ── 3. 格邊長必須 ≥ searchRadius，否則 3×3 格蓋不住整個搜尋圓 ──
        let side = Float(FeatureParams.bucketSide * FeatureParams.stride)
        check(side >= FeatureParams.searchRadius,
              String(format: "格邊長 %.0f px ≥ 搜尋半徑 %.0f px（3×3 格才蓋得住搜尋圓）",
                     side, Double(FeatureParams.searchRadius)))

        // ── 4. 剛好落在半徑邊界上的候選不能被索引漏掉 ──
        //      刻意把候選放在「與投影點同一格」與「斜對角鄰格」兩種位置
        var boundaryOK = true
        for _ in 0..<2000 {
            let p = (side * 3 + rnd() * side, side * 3 + rnd() * side)   // 格內任意位置
            let ang = rnd() * 2 * .pi
            let rad = FeatureParams.searchRadius * (0.90 + rnd() * 0.09)  // 半徑內側
            let one = [feature(u: p.0 + cos(ang) * rad, v: p.1 + sin(ang) * rad)]
            let idx1 = FeatureTracker.buildIndex(one)
            let a = FeatureTracker.bestMatch(for: probe, at: p, in: one, index: idx1)
            if a.idx != 0 { boundaryOK = false; break }
        }
        check(boundaryOK, "半徑內側（0.90~0.99R）任意方向的候選都找得到，含跨格情形")

        // ── 5. maxFeatures 的密度不變式 ──
        //
        // 這一項不是在測程式碼，是在**把 maxFeatures 的推導釘住**：
        // 引導匹配的上限由特徵密度決定 —— 平均最近鄰（Poisson 下 = 0.5/√密度）
        // 大於搜尋半徑時，位姿再準也有大半的投影找不到候選。
        // 實機 259 個/幀 → 最近鄰 52px vs 半徑 28px，這就是「半徑內無候選 37%」的主因。
        // 若有人把 maxFeatures 調小，這一項會直接指出論證破在哪，
        // 而不是讓匹配率悄悄掉回去、又被當成場景難度。
        let area = 1920.0 * 1440
        let nn = { (n: Double) in 0.5 / (n / area).squareRoot() }
        let R = Double(FeatureParams.searchRadius)
        let needed = (0.5 / R) * (0.5 / R) * area          // 最近鄰 = R 時的存活特徵數
        let expected = Double(FeatureParams.maxFeatures) * FeatureParams.depthSurvivalRate
        check(nn(259) > R && expected >= needed,
              String(format: "密度不變式：實機 259 個/幀 → 最近鄰 %.0fpx（>%.0f，太稀）；"
                     + "maxFeatures %d × 存活率 %.0f%% = %.0f 個 ≥ 需要的 %.0f 個"
                     + "（→ 最近鄰 %.0fpx）",
                     nn(259), R, FeatureParams.maxFeatures,
                     FeatureParams.depthSurvivalRate * 100, expected, needed, nn(expected)))

        print()
        print(fails == 0 ? "全部通過 — 匹配空間索引驗證完成" : "\(fails) 項失敗")
        exit(fails == 0 ? 0 : 1)
    }
}
