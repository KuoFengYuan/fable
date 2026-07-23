// C API for Swift interop. Thin wrapper around msplat C++ types.
// Opaque handles + free functions — works with any Swift version.

#ifndef MSPLAT_C_API_H
#define MSPLAT_C_API_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Config ──────────────────────────────────────────────────────────────────

typedef struct {
    int iterations;
    int shDegree;
    int shDegreeInterval;
    float ssimWeight;
    int numDownscales;
    int resolutionSchedule;
    int refineEvery;
    int warmupLength;
    int resetAlphaEvery;
    float densifyGradThresh;
    float densifySizeThresh;
    int stopScreenSizeAt;
    float splitScreenSize;
    int maxGaussians;   // 0 = unlimited (upstream default); >0 caps densification for on-device memory safety
    bool keepCrs;
    float downscaleFactor;
    float bgColor[3];
} MsplatConfig;

static inline MsplatConfig msplat_default_config(void) {
    MsplatConfig c;
    c.iterations = 30000;
    c.shDegree = 3;
    c.shDegreeInterval = 1000;
    c.ssimWeight = 0.2f;
    // coarse-to-fine：ds = 1<<max(numDownscales - step/resolutionSchedule, 0)。
    // 舊值 (2,3000) 在 4000~6000 步內最細只到 ds=2（800px）、永遠到不了 1600 全解析度 → 糊。
    // 改 (1,2000)：step<2000 用 ds=2(800px) 粗訓，step>=2000 用 ds=1(1600px 全上限)，
    // 且落在 densify 視窗(500~maxSteps/2) 內 → 能在全解析度上長出細節點。
    c.numDownscales = 1;
    c.resolutionSchedule = 2000;
    c.refineEvery = 100;
    c.warmupLength = 500;
    c.resetAlphaEvery = 30;
    c.densifyGradThresh = 0.0002f;
    c.densifySizeThresh = 0.01f;
    c.stopScreenSizeAt = 4000;
    c.splitScreenSize = 0.05f;
    c.maxGaussians = 0;
    c.keepCrs = false;
    c.downscaleFactor = 1.0f;
    c.bgColor[0] = 0.6130f; c.bgColor[1] = 0.0101f; c.bgColor[2] = 0.3984f;
    return c;
}

// ── Stats ───────────────────────────────────────────────────────────────────

typedef struct {
    int iteration;
    int splatCount;
    float msPerStep;
} MsplatStats;

typedef struct {
    float psnr;
    float ssim;
    float l1;
    int numTest;
    int numGaussians;
} MsplatEvalMetrics;

// ── Pixel buffer ────────────────────────────────────────────────────────────

typedef struct {
    float* data;   // RGB float32, HWC layout. Caller must free() this.
    int width;
    int height;
} MsplatPixelBuffer;

// ── Dataset ─────────────────────────────────────────────────────────────────

typedef void* MsplatDataset;

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor, int maxImageDim,
                                     bool evalMode, int testEvery);
void msplat_dataset_destroy(MsplatDataset ds);
int msplat_dataset_num_train(MsplatDataset ds);
int msplat_dataset_num_test(MsplatDataset ds);
/// 點雲（初始化點）的質心與 RMS 半徑（msplat 內部置中/縮放後的 frame）。
/// 供環繞檢視以「物件本身」為樞軸（而非相機群中心）。
void msplat_dataset_points_bounds(MsplatDataset ds, float* outCenter3, float* outRadius);

// ── Trainer ─────────────────────────────────────────────────────────────────

typedef void* MsplatTrainer;

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config);
void msplat_trainer_destroy(MsplatTrainer t);

MsplatStats msplat_trainer_step(MsplatTrainer t);
void msplat_trainer_train(MsplatTrainer t);
MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t);
MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest);
MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex);

/// Render into a caller-provided RGBA uint8 buffer (no allocation, no float copy).
/// outRGBA must be at least width*height*4 bytes. Returns dimensions via outWidth/outHeight.
/// Call once with outRGBA=NULL to get dimensions, then allocate and call again.
void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight);
/// 用「自訂內參 + 輸出尺寸」render（直式檢視用：可指定 portrait 寬高與焦距）。
/// outRGBA 需 width*height*4 bytes；傳 NULL 只查詢尺寸。
void msplat_trainer_render_pose_intrinsics(MsplatTrainer t, const float camToWorld[16],
                                      float fx, float fy, float cx, float cy,
                                      int width, int height, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight);
void msplat_trainer_export_ply(MsplatTrainer t, const char* path);
void msplat_trainer_export_splat(MsplatTrainer t, const char* path);
void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path);
int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path);
int msplat_trainer_splat_count(MsplatTrainer t);
int msplat_trainer_iteration(MsplatTrainer t);
void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]);

// ── Lifecycle ───────────────────────────────────────────────────────────────

void msplat_set_metallib_path(const char* path);
void msplat_sync(void);
void msplat_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // MSPLAT_C_API_H
