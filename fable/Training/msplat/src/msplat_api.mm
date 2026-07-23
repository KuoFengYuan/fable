// ObjC++ implementation of the Swift-facing C++ API.
// This is the ONLY file that touches internal C++ types (Model, Camera, MTensor).

#include "msplat_api.hpp"

#include "model.hpp"
#include "input_data.hpp"
#include "msplat.hpp"
#include "ssim.hpp"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <deque>

namespace msplat {

// ── Dataset::Impl ───────────────────────────────────────────────────────────

struct Dataset::Impl {
    InputData data;
    std::vector<Camera> trainCams;
    std::vector<Camera> testCams;
};

Dataset::Dataset(const std::string& path, float downscaleFactor,
                 bool evalMode, int testEvery)
    : impl(std::make_unique<Impl>())
{
    impl->data = inputDataFromX(path);

    // 影像串流：不預載全部關鍵幀（全解析度下全常駐是手機 OOM 主因）。
    // 只記錄 downscale，訓練時逐幀 lazy load、由 LRU 釋放（見 Trainer::step）。
    for (auto& cam : impl->data.cameras)
        cam.datasetDownscale = downscaleFactor;

    if (evalMode) {
        auto split = impl->data.splitTrainTest(testEvery);
        impl->trainCams = std::get<0>(split);
        impl->testCams = std::get<1>(split);
    } else {
        auto t = impl->data.getCameras(false);
        impl->trainCams = std::get<0>(t);
    }
}

Dataset::~Dataset() = default;
Dataset::Dataset(Dataset&&) noexcept = default;
Dataset& Dataset::operator=(Dataset&&) noexcept = default;

int Dataset::numTrain() const { return (int)impl->trainCams.size(); }
int Dataset::numTest() const { return (int)impl->testCams.size(); }
void Dataset::cameraPose(int index, float camToWorld[16]) const {
    if (index >= 0 && index < (int)impl->trainCams.size())
        memcpy(camToWorld, impl->trainCams[index].camToWorld, 16 * sizeof(float));
}
void Dataset::pointsBounds(float* center3, float* radius) const {
    const auto& p = impl->data.points;
    if (p.count == 0) { center3[0] = center3[1] = center3[2] = 0; *radius = 1.f; return; }
    double cx = 0, cy = 0, cz = 0;
    for (int64_t i = 0; i < p.count; i++) { cx += p.xyz[i*3]; cy += p.xyz[i*3+1]; cz += p.xyz[i*3+2]; }
    cx /= p.count; cy /= p.count; cz /= p.count;
    center3[0] = (float)cx; center3[1] = (float)cy; center3[2] = (float)cz;
    double s = 0;
    for (int64_t i = 0; i < p.count; i++) {
        double dx = p.xyz[i*3]-cx, dy = p.xyz[i*3+1]-cy, dz = p.xyz[i*3+2]-cz;
        s += dx*dx + dy*dy + dz*dz;
    }
    *radius = (float)std::sqrt(s / (double)p.count);   // RMS 半徑（對離群點較穩）
}

void* Dataset::_handle() const { return impl.get(); }

// ── Trainer::Impl ───────────────────────────────────────────────────────────

struct Trainer::Impl {
    std::unique_ptr<Model> model;
    Config config;
    Dataset::Impl* ds = nullptr;
    int currentStep = 0;

    // Camera iteration
    std::vector<size_t> camIndices;
    size_t camIterPos = 0;
    std::mt19937 rng{42};
    std::deque<size_t> lru;   // 影像串流：最近使用的相機（超出視窗即釋放影像）

    void shuffleCameras() {
        std::shuffle(camIndices.begin(), camIndices.end(), rng);
        camIterPos = 0;
    }

    size_t nextCamera() {
        if (camIterPos >= camIndices.size()) shuffleCameras();
        return camIndices[camIterPos++];
    }
};

Trainer::Trainer(Dataset& dataset, const Config& config)
    : impl(std::make_unique<Impl>())
{
    impl->config = config;
    impl->ds = static_cast<Dataset::Impl*>(dataset._handle());

    impl->model = std::make_unique<Model>(
        impl->ds->data,
        (int)impl->ds->trainCams.size(),
        config.numDownscales, config.resolutionSchedule,
        config.shDegree, config.shDegreeInterval,
        config.refineEvery, config.warmupLength, config.resetAlphaEvery,
        config.densifyGradThresh, config.densifySizeThresh,
        config.stopScreenSizeAt, config.splitScreenSize,
        config.iterations, config.keepCrs,
        config.maxGaussians,
        config.bgColor
    );

    impl->camIndices.resize(impl->ds->trainCams.size());
    std::iota(impl->camIndices.begin(), impl->camIndices.end(), 0);
    impl->shuffleCameras();
}

Trainer::~Trainer() = default;

Stats Trainer::step() {
    impl->currentStep++;
    size_t camIdx = impl->nextCamera();

    // 影像串流 LRU：只保留最近 kImageWindow 個相機的影像常駐，其餘釋放
    // （全解析度下全部常駐 ~數 GB 是 OOM 主因）。驅逐的是數步前用過的 → GPU 早已用完，安全。
    // 畸變相機不驅逐（其 undistort 裁切重載無法重現；fable 為 PINHOLE 無此問題）。
    constexpr size_t kImageWindow = 6;
    impl->lru.push_back(camIdx);
    if (impl->lru.size() > kImageWindow) {
        size_t old = impl->lru.front();
        impl->lru.pop_front();
        if (old != camIdx && !impl->ds->trainCams[old].hasDistortion())
            impl->ds->trainCams[old].releaseImage();
    }

    Camera& cam = impl->ds->trainCams[camIdx];

    int ds = impl->model->getDownscaleFactor(impl->currentStep);
    MTensor& gt = cam.getGPUImage(ds);

    auto t0 = std::chrono::high_resolution_clock::now();

    impl->model->fullIteration(cam, impl->currentStep, gt, impl->config.ssimWeight);
    impl->model->schedulersStep(impl->currentStep);
    impl->model->afterTrain(impl->currentStep);
    msplat_commit();

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;

    Stats s;
    s.iteration = impl->currentStep;
    s.splatCount = (int)impl->model->means.size(0);
    s.msPerStep = ms;
    return s;
}

void Trainer::train(int callbackEvery) {
    while (impl->currentStep < impl->config.iterations) {
        step();
        // Note: callbacks handled at the Swift level via polling iteration()
        // to keep the C++ API free of function pointer complexity
    }
}

EvalMetrics Trainer::evaluate() {
    auto& testCams = impl->ds->testCams;
    if (testCams.empty())
        return {};

    double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
    int n = (int)testCams.size();

    for (int i = 0; i < n; i++) {
        Camera& cam = testCams[i];
        MTensor rgb = impl->model->render(cam, impl->config.iterations);
        msplat_gpu_sync();
        MTensor rgbCpu = rgb.cpu();
        int dsf = impl->model->getDownscaleFactor(impl->config.iterations);
        MTensor gtCpu = cam.getGPUImage(dsf).cpu();

        sumPsnr += psnr(rgbCpu, gtCpu);
        sumSsim += ssim_eval(rgbCpu, gtCpu);
        sumL1 += l1_loss(rgbCpu, gtCpu);
    }

    EvalMetrics m;
    m.psnr = (float)(sumPsnr / n);
    m.ssim = (float)(sumSsim / n);
    m.l1 = (float)(sumL1 / n);
    m.numTest = n;
    m.numGaussians = (int)impl->model->means.size(0);
    return m;
}

PixelBuffer Trainer::render(int cameraIndex, bool useTest) {
    auto& cams = useTest ? impl->ds->testCams : impl->ds->trainCams;
    if (cameraIndex < 0 || cameraIndex >= (int)cams.size())
        return {};

    Camera& cam = cams[cameraIndex];
    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    // Use malloc so callers can free() — PixelBuffer destructor handles both
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));

    return PixelBuffer(buf, w, h);
}

PixelBuffer Trainer::renderFromPose(const float camToWorld[16], int refCameraIndex) {
    auto& cams = impl->ds->trainCams;
    if (refCameraIndex < 0 || refCameraIndex >= (int)cams.size())
        return {};

    Camera cam = cams[refCameraIndex];  // copy intrinsics
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));
    // Invalidate cached matrices so prepareCam recomputes from the new pose
    cam.cachedViewMat = MTensor();
    cam.cachedProjViewMat = MTensor();

    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));
    return PixelBuffer(buf, w, h);
}

void Trainer::renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                                  uint8_t* outRGBA, int* outWidth, int* outHeight) {
    auto& cams = impl->ds->trainCams;
    if (refCameraIndex < 0 || refCameraIndex >= (int)cams.size()) {
        *outWidth = 0; *outHeight = 0; return;
    }

    Camera cam = cams[refCameraIndex];
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));
    cam.cachedViewMat = MTensor();
    cam.cachedProjViewMat = MTensor();

    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();

    int h = (int)rgb.size(0), w = (int)rgb.size(1);
    *outWidth = w;
    *outHeight = h;
    if (!outRGBA) return;

    // Read directly from GPU tensor (unified memory on Apple Silicon)
    const float* src = (const float*)rgb.data_ptr();
    int n = w * h;
    for (int i = 0; i < n; i++) {
        outRGBA[i * 4]     = (uint8_t)(fminf(fmaxf(src[i*3],   0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 1] = (uint8_t)(fminf(fmaxf(src[i*3+1], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 2] = (uint8_t)(fminf(fmaxf(src[i*3+2], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 3] = 255;
    }
}

void Trainer::renderFromPoseIntrinsics(const float camToWorld[16], float fx, float fy, float cx, float cy,
                                  int width, int height, uint8_t* outRGBA, int* outWidth, int* outHeight) {
    Camera cam;
    cam.width = width; cam.height = height;
    cam.fx = fx; cam.fy = fy; cam.cx = cx; cam.cy = cy;
    cam.k1 = cam.k2 = cam.p1 = cam.p2 = 0;
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));
    cam.cachedViewMat = MTensor();
    cam.cachedProjViewMat = MTensor();

    // 用大 step 強制全解析度（避開訓練 coarse-to-fine 降採樣）：輸出尺寸＝指定 width×height，
    // 也用完整 SH。否則輸出會是 width/sf×height/sf，與呼叫端 buffer 尺寸不符。
    MTensor rgb = impl->model->render(cam, 1 << 20);
    msplat_gpu_sync();

    int h = (int)rgb.size(0), w = (int)rgb.size(1);
    *outWidth = w; *outHeight = h;
    if (!outRGBA) return;
    const float* src = (const float*)rgb.data_ptr();
    int n = w * h;
    for (int i = 0; i < n; i++) {
        outRGBA[i * 4]     = (uint8_t)(fminf(fmaxf(src[i*3],   0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 1] = (uint8_t)(fminf(fmaxf(src[i*3+1], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 2] = (uint8_t)(fminf(fmaxf(src[i*3+2], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 3] = 255;
    }
}

void Trainer::exportPly(const std::string& path) {
    impl->model->savePly(path, impl->currentStep);
}

void Trainer::exportSplat(const std::string& path) {
    impl->model->saveSplat(path);
}

void Trainer::saveCheckpoint(const std::string& path) {
    impl->model->saveCheckpoint(path, impl->currentStep);
}

int Trainer::loadCheckpoint(const std::string& path) {
    impl->currentStep = impl->model->loadCheckpoint(path);
    // Re-shuffle cameras for resumed training
    impl->shuffleCameras();
    return impl->currentStep;
}

int Trainer::splatCount() const {
    return (int)impl->model->means.size(0);
}

int Trainer::iteration() const {
    return impl->currentStep;
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

void sync() { msplat_gpu_sync(); }
void cleanup() { cleanup_msplat_metal(); }

} // namespace msplat

// ── C API (for Swift interop) ───────────────────────────────────────────────

#include "msplat_c_api.h"

static msplat::Config configFromC(MsplatConfig c) {
    msplat::Config cfg;
    cfg.iterations = c.iterations;
    cfg.shDegree = c.shDegree;
    cfg.shDegreeInterval = c.shDegreeInterval;
    cfg.ssimWeight = c.ssimWeight;
    cfg.numDownscales = c.numDownscales;
    cfg.resolutionSchedule = c.resolutionSchedule;
    cfg.refineEvery = c.refineEvery;
    cfg.warmupLength = c.warmupLength;
    cfg.resetAlphaEvery = c.resetAlphaEvery;
    cfg.densifyGradThresh = c.densifyGradThresh;
    cfg.densifySizeThresh = c.densifySizeThresh;
    cfg.stopScreenSizeAt = c.stopScreenSizeAt;
    cfg.splitScreenSize = c.splitScreenSize;
    cfg.maxGaussians = c.maxGaussians;
    cfg.keepCrs = c.keepCrs;
    cfg.downscaleFactor = c.downscaleFactor;
    memcpy(cfg.bgColor, c.bgColor, sizeof(cfg.bgColor));
    return cfg;
}

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor,
                                     bool evalMode, int testEvery) {
    auto* ds = new msplat::Dataset(std::string(path), downscaleFactor, evalMode, testEvery);
    return static_cast<MsplatDataset>(ds);
}

void msplat_dataset_destroy(MsplatDataset ds) {
    delete static_cast<msplat::Dataset*>(ds);
}

int msplat_dataset_num_train(MsplatDataset ds) {
    return static_cast<msplat::Dataset*>(ds)->numTrain();
}

int msplat_dataset_num_test(MsplatDataset ds) {
    return static_cast<msplat::Dataset*>(ds)->numTest();
}

void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]) {
    static_cast<msplat::Dataset*>(ds)->cameraPose(cameraIndex, camToWorld);
}

void msplat_dataset_points_bounds(MsplatDataset ds, float* outCenter3, float* outRadius) {
    static_cast<msplat::Dataset*>(ds)->pointsBounds(outCenter3, outRadius);
}

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config) {
    auto* dataset = static_cast<msplat::Dataset*>(ds);
    auto cfg = configFromC(config);
    auto* trainer = new msplat::Trainer(*dataset, cfg);
    return static_cast<MsplatTrainer>(trainer);
}

void msplat_trainer_destroy(MsplatTrainer t) {
    delete static_cast<msplat::Trainer*>(t);
}

MsplatStats msplat_trainer_step(MsplatTrainer t) {
    auto stats = static_cast<msplat::Trainer*>(t)->step();
    return MsplatStats{stats.iteration, stats.splatCount, stats.msPerStep};
}

void msplat_trainer_train(MsplatTrainer t) {
    static_cast<msplat::Trainer*>(t)->train(0);
}

MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t) {
    auto m = static_cast<msplat::Trainer*>(t)->evaluate();
    return MsplatEvalMetrics{m.psnr, m.ssim, m.l1, m.numTest, m.numGaussians};
}

MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest) {
    auto buf = static_cast<msplat::Trainer*>(t)->render(cameraIndex, useTest);
    MsplatPixelBuffer result{buf.data, buf.width, buf.height};
    buf.data = nullptr; // Transfer ownership to caller
    return result;
}

MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex) {
    auto buf = static_cast<msplat::Trainer*>(t)->renderFromPose(camToWorld, refCameraIndex);
    MsplatPixelBuffer result{buf.data, buf.width, buf.height};
    buf.data = nullptr;
    return result;
}

void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight) {
    static_cast<msplat::Trainer*>(t)->renderFromPoseToBuffer(
        camToWorld, refCameraIndex, outRGBA, outWidth, outHeight);
}

void msplat_trainer_render_pose_intrinsics(MsplatTrainer t, const float camToWorld[16],
                                      float fx, float fy, float cx, float cy,
                                      int width, int height, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight) {
    static_cast<msplat::Trainer*>(t)->renderFromPoseIntrinsics(
        camToWorld, fx, fy, cx, cy, width, height, outRGBA, outWidth, outHeight);
}

void msplat_trainer_export_ply(MsplatTrainer t, const char* path) {
    static_cast<msplat::Trainer*>(t)->exportPly(std::string(path));
}

void msplat_trainer_export_splat(MsplatTrainer t, const char* path) {
    static_cast<msplat::Trainer*>(t)->exportSplat(std::string(path));
}

void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path) {
    static_cast<msplat::Trainer*>(t)->saveCheckpoint(std::string(path));
}

int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path) {
    return static_cast<msplat::Trainer*>(t)->loadCheckpoint(std::string(path));
}

int msplat_trainer_splat_count(MsplatTrainer t) {
    return static_cast<msplat::Trainer*>(t)->splatCount();
}

int msplat_trainer_iteration(MsplatTrainer t) {
    return static_cast<msplat::Trainer*>(t)->iteration();
}

void msplat_sync(void) { msplat::sync(); }
void msplat_cleanup(void) { msplat::cleanup(); }
