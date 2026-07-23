#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
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

    Image image;
    std::unordered_map<int, Image> imagePyramids;
    std::unordered_map<int, MTensor> mtensorImageCache;
    MTensor cachedViewMat, cachedProjViewMat;
    float cachedCamPos[3] = {};
    float cachedFovX = 0, cachedFovY = 0;

    // 影像串流：不預載全部關鍵幀（全解析度下全常駐是手機 OOM 主因），
    // 逐幀 lazy load、用完由 LRU 釋放。datasetDownscale 供 lazy 重載使用；
    // calibrated 確保 loadImage 的內參調整只做一次（重載時只讀像素、不再改內參）。
    float datasetDownscale = 1.0f;
    bool calibrated = false;

    void loadImage(float downscaleFactor);
    Image getImage(int downscaleFactor);
    MTensor& getGPUImage(int downscaleFactor);
    void releaseImage() { image = Image(); imagePyramids.clear(); mtensorImageCache.clear(); }
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
