#include "input_data.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Image loading ───────────────────────────────────────────────────────────

void Camera::loadImage(float downscaleFactor) {
    Image raw = imreadRGB(filePath);
    if (raw.empty()) return;

    if (!calibrated) {
        // 首次載入：對齊宣告的 (width,height) → 套 downscale + maxImageDim 上限 → 校正內參（只做一次）
        if (width > 0 && height > 0 && (raw.width != width || raw.height != height)) {
            float sx = (float)raw.width / (float)width;
            float sy = (float)raw.height / (float)height;
            fx *= sx; fy *= sy; cx *= sx; cy *= sy;
            width = raw.width; height = raw.height;
        } else if (width == 0 || height == 0) {
            width = raw.width; height = raw.height;
        }

        // 目標尺寸：先套固定 downscale，再套最長邊上限（maxImageDim）——取較強的縮小。
        // 12MP → 最長邊 1600 約降 6.4× 像素：光柵化/記憶體同步降一個量級（畫質對 3DGS 無損）。
        int tw = width, th = height;
        float ds = std::max(1.0f, downscaleFactor);
        if (ds > 1.0f) { tw = (int)(width / ds); th = (int)(height / ds); }
        if (maxImageDim > 0 && std::max(tw, th) > maxImageDim) {
            float f = (float)std::max(tw, th) / (float)maxImageDim;
            tw = (int)(tw / f); th = (int)(th / f);
        }
        tw = std::max(1, tw); th = std::max(1, th);
        if (tw != width || th != height) {
            raw = resizeArea(raw, tw, th);
            float sx = (float)tw / (float)width;
            float sy = (float)th / (float)height;
            fx *= sx; fy *= sy; cx *= sx; cy *= sy;
            width = tw; height = th;
        }

        if (hasDistortion()) {
            auto result = undistortImage(raw, fx, fy, cx, cy, k1, k2, p1, p2, k3);
            raw = std::move(result.image);
            fx = result.fx; fy = result.fy;
            cx = result.cx; cy = result.cy;
            width = result.width; height = result.height;
            k1 = k2 = k3 = p1 = p2 = 0;
        }
        calibrated = true;
    } else {
        // 已校正（理論上 pixels 常駐不會再走此路）；保險：縮到已定案的 (width,height)
        if (raw.width != width || raw.height != height) raw = resizeArea(raw, width, height);
    }

    // 存成 uint8 常駐（解碼一次）：float [0,1] → uint8 [0,255]。之後 GPU tensor 由此重建、免重解碼。
    size_t n = (size_t)width * height * 3;
    pixels.resize(n);
    const float* s = raw.ptr();
    for (size_t i = 0; i < n; i++) {
        float v = s[i] * 255.0f + 0.5f;
        pixels[i] = (uint8_t)std::clamp(v, 0.0f, 255.0f);
    }
}

Image Camera::getImage(int downscaleFactor) {
    if (pixels.empty()) loadImage(datasetDownscale);   // 串流：首用時解碼，之後常駐

    // uint8 常駐副本 → float 全（已上限）解析度
    Image full;
    full.width = width; full.height = height;
    size_t n = (size_t)width * height * 3;
    full.data.resize(n);
    for (size_t i = 0; i < n; i++) full.data[i] = pixels[i] * (1.0f / 255.0f);

    if (downscaleFactor <= 1) return full;
    int newW = width / downscaleFactor;
    int newH = height / downscaleFactor;
    return resizeArea(full, newW, newH);   // coarse-to-fine 降採樣（由常駐副本，非重解碼）
}

MTensor& Camera::getGPUImage(int downscaleFactor) {
    auto it = mtensorImageCache.find(downscaleFactor);
    if (it != mtensorImageCache.end()) return it->second;
    Image img = getImage(downscaleFactor);
    MTensor mt = gpu_empty({img.height, img.width, 3}, DType::Float32);
    memcpy(mt.data_ptr(), img.ptr(), img.width * img.height * 3 * sizeof(float));
    mtensorImageCache[downscaleFactor] = mt;
    return mtensorImageCache[downscaleFactor];
}

// ── Scale & center ──────────────────────────────────────────────────────────

void autoScaleAndCenter(InputData &data) {
    if (data.cameras.empty()) return;

    // Compute mean camera position
    float mean[3] = {};
    for (auto &cam : data.cameras) {
        mean[0] += cam.camToWorld[3];   // column 3 of row 0
        mean[1] += cam.camToWorld[7];   // column 3 of row 1
        mean[2] += cam.camToWorld[11];  // column 3 of row 2
    }
    int n = (int)data.cameras.size();
    mean[0] /= n; mean[1] /= n; mean[2] /= n;

    data.translation[0] = mean[0];
    data.translation[1] = mean[1];
    data.translation[2] = mean[2];

    // Center camera poses
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  -= mean[0];
        cam.camToWorld[7]  -= mean[1];
        cam.camToWorld[11] -= mean[2];
    }

    // Compute scale from max absolute camera position
    float maxAbs = 0;
    for (auto &cam : data.cameras) {
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[3]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[7]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[11]));
    }
    data.scale = (maxAbs > 0) ? (1.0f / maxAbs) : 1.0f;

    // Apply scale to camera positions
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  *= data.scale;
        cam.camToWorld[7]  *= data.scale;
        cam.camToWorld[11] *= data.scale;
    }

    // Apply to point cloud
    for (int64_t i = 0; i < data.points.count; i++) {
        data.points.xyz[i*3+0] = (data.points.xyz[i*3+0] - mean[0]) * data.scale;
        data.points.xyz[i*3+1] = (data.points.xyz[i*3+1] - mean[1]) * data.scale;
        data.points.xyz[i*3+2] = (data.points.xyz[i*3+2] - mean[2]) * data.scale;
    }
}

// ── Train/test split ────────────────────────────────────────────────────────

std::tuple<std::vector<Camera>, Camera*> InputData::getCameras(bool validate, const std::string &valImage) {
    if (!validate) return {cameras, nullptr};

    // Find validation camera
    int valIdx = -1;
    if (valImage == "random") {
        std::mt19937 rng(42);
        valIdx = rng() % cameras.size();
    } else {
        for (int i = 0; i < (int)cameras.size(); i++) {
            if (cameras[i].filePath.find(valImage) != std::string::npos) { valIdx = i; break; }
        }
    }
    if (valIdx < 0) valIdx = 0;

    Camera *valCam = &cameras[valIdx];
    std::vector<Camera> train;
    for (int i = 0; i < (int)cameras.size(); i++)
        if (i != valIdx) train.push_back(cameras[i]);

    return {train, valCam};
}

std::tuple<std::vector<Camera>, std::vector<Camera>> InputData::splitTrainTest(int testEvery) {
    std::vector<Camera> train, test;
    for (int i = 0; i < (int)cameras.size(); i++) {
        if (i % testEvery == 0)
            test.push_back(cameras[i]);
        else
            train.push_back(cameras[i]);
    }
    return {train, test};
}

// ── Save cameras ────────────────────────────────────────────────────────────

void InputData::saveCameras(const std::string &filename, bool keepCrs) const {
    json arr = json::array();
    for (auto &cam : cameras) {
        json c;
        c["file_path"] = fs::path(cam.filePath).filename().string();
        c["width"] = cam.width;
        c["height"] = cam.height;
        c["fx"] = cam.fx; c["fy"] = cam.fy;
        c["cx"] = cam.cx; c["cy"] = cam.cy;

        // Extract rotation and translation from camToWorld
        float R[9], T[3];
        // Undo OpenGL flip (negate columns 1,2 back to OpenCV convention)
        R[0] =  cam.camToWorld[0]; R[1] = -cam.camToWorld[1]; R[2] = -cam.camToWorld[2];
        R[3] =  cam.camToWorld[4]; R[4] = -cam.camToWorld[5]; R[5] = -cam.camToWorld[6];
        R[6] =  cam.camToWorld[8]; R[7] = -cam.camToWorld[9]; R[8] = -cam.camToWorld[10];
        T[0] =  cam.camToWorld[3]; T[1] =  cam.camToWorld[7]; T[2] =  cam.camToWorld[11];

        if (keepCrs) {
            T[0] = T[0] / scale + translation[0];
            T[1] = T[1] / scale + translation[1];
            T[2] = T[2] / scale + translation[2];
        }

        c["rotation"] = {{R[0],R[1],R[2]},{R[3],R[4],R[5]},{R[6],R[7],R[8]}};
        c["translation"] = {T[0], T[1], T[2]};
        arr.push_back(c);
    }

    std::ofstream f(filename);
    f << arr.dump(2);
}

// ── Format dispatcher ───────────────────────────────────────────────────────

InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath) {
    fs::path root(path);

    // Nerfstudio: transforms.json
    if (fs::exists(root / "transforms.json"))
        return loaders::loadNerfstudio(path);

    // COLMAP: cameras.bin (direct or in sparse/0/)
    if (fs::exists(root / "cameras.bin") || fs::exists(root / "sparse" / "0" / "cameras.bin"))
        return loaders::loadColmap(path, colmapImagePath);

    // Polycam: keyframes/ directory or cameras.json
    if (fs::exists(root / "keyframes" / "corrected_cameras") || fs::exists(root / "cameras.json"))
        return loaders::loadPolycam(path);

    throw std::runtime_error("Unrecognized dataset format in: " + path +
        "\nSupported: COLMAP (cameras.bin), Nerfstudio (transforms.json), Polycam (keyframes/)");
}
