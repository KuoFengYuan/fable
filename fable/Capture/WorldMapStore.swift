//
//  WorldMapStore.swift
//  fable — ARWorldMap 持久化：讓「關掉 app 之後再掃的下一個房間」落在同一個座標系
//
//  沒有這個東西的話，每次 session.run(.resetTracking) 都是全新的世界原點，
//  水平朝向（yaw）也完全任意 —— 於是同一間房子的兩次掃描之間沒有任何共同參考，
//  拼不起來。同一次 session 內（含 review → 續掃）ARKit 會自動重定位，不需要它；
//  它解決的是**跨 session** 的問題，也就是整屋掃描真正的限制。
//
//  運作方式：停止掃描時把 ARKit 當下的地圖（特徵點 ＋ 錨點）存檔，
//  下一次以 initialWorldMap 帶入 —— ARKit 進入 relocalizing，
//  使用者把鏡頭對回上次掃描過的區域就會接上，新的姿態與舊的在同一個座標系。
//
//  兩份副本各有用途：
//    latest.arworldmap        App 層級，「延續上次」讀的就是它
//    <session>/worldmap.bin   隨掃描資料一起打包，之後要重現/除錯有據可查
//

import Foundation
import ARKit

nonisolated enum WorldMapStore {

    /// 地圖檔上限。ARWorldMap 會隨掃描範圍成長，整層樓可能到數十 MB；
    /// 過大的地圖 relocalization 會變慢且記憶體吃緊，超過就不留（寧可下次重新開始）。
    static let maxBytes = 64 * 1024 * 1024

    static var latestURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("latest.arworldmap")
    }

    static var hasLatest: Bool {
        FileManager.default.fileExists(atPath: latestURL.path)
    }

    /// 上次地圖的大小與時間，給 UI 顯示「延續上次（3.2 MB・12 分鐘前）」
    static func latestInfo() -> (bytes: Int, modified: Date)? {
        guard let a = try? FileManager.default.attributesOfItem(atPath: latestURL.path),
              let size = a[.size] as? Int, let date = a[.modificationDate] as? Date
        else { return nil }
        return (size, date)
    }

    /// 存檔。同時寫進 session 資料夾（隨匯出打包）與 App 層級的 latest。
    /// 回傳位元組數；失敗回 nil。
    @discardableResult
    static func save(_ map: ARWorldMap, sessionDir: URL?) -> Int? {
        do {
            let data = try NSKeyedArchiver.archivedData(withRootObject: map,
                                                        requiringSecureCoding: true)
            guard data.count <= maxBytes else {
                print("[WorldMap] 地圖 \(data.count / 1_048_576)MB 超過上限，不保存")
                try? FileManager.default.removeItem(at: latestURL)   // 舊的也作廢，避免接到過時地圖
                return nil
            }
            try data.write(to: latestURL, options: [.atomic])
            if let sessionDir {
                try? data.write(to: sessionDir.appendingPathComponent("worldmap.bin"),
                                options: [.atomic])
            }
            return data.count
        } catch {
            print("[WorldMap] 存檔失敗: \(error)")
            return nil
        }
    }

    /// 讀回上次的地圖。壞檔直接刪掉並回 nil —— 帶著壞地圖跑會讓 session 永遠卡在
    /// relocalizing，比重新開始還糟。
    static func loadLatest() -> ARWorldMap? {
        guard let data = try? Data(contentsOf: latestURL) else { return nil }
        do {
            return try NSKeyedUnarchiver.unarchivedObject(ofClass: ARWorldMap.self, from: data)
        } catch {
            print("[WorldMap] 讀取失敗，刪除毀損的地圖: \(error)")
            try? FileManager.default.removeItem(at: latestURL)
            return nil
        }
    }

    static func clearLatest() {
        try? FileManager.default.removeItem(at: latestURL)
    }
}
