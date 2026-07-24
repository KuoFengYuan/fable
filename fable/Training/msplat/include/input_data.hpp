#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
#include <cstdint>
#include <unordered_map>
#include "metal_tensor.hpp"

// Simple float32 RGB image — replaces cv::Mat
struct Image {
    std::vector<float> data;  // width * height * 3 floats, RGB, [0,1]
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct Camera {
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
    float camToWorld[16] = {};  // 4x4 row-major, camera-to-world (OpenGL: Y-up, Z-back)
    std::string filePath;

    // 解碼一次後常駐的像素（RGB, uint8, 已套用 maxImageDim 上限與內參校正）。
    // uint8 而非 float32：省 4×（1600px 全 42 幀約 244MB、有界），訓練用的 GPU tensor
    // 由此常駐副本即時重建（不需重新解碼 JPEG）→ 串流不再每輪重解碼（解決變慢）。
    std::vector<uint8_t> pixels;
    std::unordered_map<int, MTensor> mtensorImageCache;
    MTensor cachedViewMat, cachedProjViewMat;
    float cachedCamPos[3] = {};
    float cachedFovX = 0, cachedFovY = 0;

    // 相機姿態優化（SE3 左擾動於 world→cam，Adam）。viewCorr = 修正後的 world→cam（4x4 row-major）；
    // 首次由 prepareCam 以 base viewmat 初始化，之後每步左乘 exp(Δδ̂) 精修。見 Model::cameraPoseStep。
    bool  poseInit = false;
    float viewCorr[16] = {};
    float poseAdamM[6] = {};   // (ω, τ) Adam 一階動量
    float poseAdamV[6] = {};   // Adam 二階動量
    int   poseAdamStep = 0;

    // 外觀校正（per-image 學習式仿射，bilateral guided 核心）：M(3x3 row-major,9)+t(3)=12。
    // init 為 identity（起始無作用），Adam 精修以吸收本幀曝光/白平衡差異；僅作用於訓練損失。
    bool  appInit = false;
    float appAffine[12] = {};
    float appAdamM[12] = {};
    float appAdamV[12] = {};
    int   appAdamStep = 0;

    // 訓練解析度上限（最長邊 px；0=不限）。手機端 12MP 光柵化是 O(像素數) → 又慢又吃記憶體；
    // 業界標準（Inria/gsplat/Scaniverse）皆限 ~1600。掃描原圖與匯出檔案不受此影響。
    int maxImageDim = 0;
    float datasetDownscale = 1.0f;   // 額外固定降採樣倍率（與 maxImageDim 取較強的縮小）
    bool calibrated = false;         // 內參校正只做一次

    // LiDAR 深度監督用（度量真值）：depth/<stem>_depth.bin（ARKit sceneDepth 256×192 float32）+ _conf.bin。
    // 由 filePath 推導路徑、lazy 載入；PINHOLE 才用（畸變 undistort 裁切會破壞 FOV 對應）。
    std::vector<float> lidarDepth;
    std::vector<uint8_t> lidarConf;
    int lidarW = 0, lidarH = 0;
    bool lidarTried = false;
    void loadLidarDepth();

    // 誤差圖引導（LichtFeld MRNF use_error_map 的等價實作，但不需要新的光柵器 pass）。
    // 1/8 解析度的「高頻能量圖」：密集化只需要「這裡紋理多不多」的空間先驗，
    // 不需要 Canny 的精確邊緣定位。1/8 → 200×150 uint8 ≈ 30KB/幀（120 幀約 3.6MB），
    // 由常駐 pixels 一次算好後快取。見 Model::mrnfAccumEdge。
    std::vector<uint8_t> edgeMap;
    int edgeW = 0, edgeH = 0;
    bool edgeTried = false;
    void buildEdgeMap();

    void loadImage(float downscaleFactor);
    Image getImage(int downscaleFactor);
    MTensor& getGPUImage(int downscaleFactor);
    void releaseImage() {
        // 只釋放 GPU tensor（LRU 驅逐）；常駐 uint8 pixels 保留 → 下次用到直接重建、免重解碼。
        // MTensor 無解構子（靠手動 reset() CFRelease，ARC=NO 下 __bridge_retained 為 no-op）：
        // 直接 clear() 只移除 map、不釋放 MTLBuffer → 每次驅逐洩漏 GPU 記憶體。必須先 reset()。
        for (auto& kv : mtensorImageCache) kv.second.reset();
        mtensorImageCache.clear();
    }
    bool hasDistortion() const { return k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0; }
};

struct Points {
    std::vector<float> xyz;     // N*3 flattened
    std::vector<uint8_t> rgb;   // N*3 flattened
    int64_t count = 0;
};

struct InputData {
    std::vector<Camera> cameras;
    float scale = 1.0f;
    float translation[3] = {};
    Points points;

    std::tuple<std::vector<Camera>, Camera*> getCameras(bool validate, const std::string &valImage = "random");
    std::tuple<std::vector<Camera>, std::vector<Camera>> splitTrainTest(int testEvery);
    void saveCameras(const std::string &filename, bool keepCrs) const;
};

// Auto-detect format and load dataset
InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath = "");

#endif
