#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <random>
#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"

int numShBases(int degree);
float psnr(const MTensor& rendered, const MTensor& gt);
float l1_loss(const MTensor& rendered, const MTensor& gt);

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs, int maxGaussians,
        const float* bgColor = nullptr);

  ~Model(){ releaseOptimizers(); }

  void setupOptimizers();
  void releaseOptimizers();

  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  void saveSplat(const std::string &filename);
  int loadPly(const std::string &filename);
  void saveCheckpoint(const std::string &filename, int step);
  int loadCheckpoint(const std::string &filename);
  struct CamSetup {
    float fx, fy, cx, cy;
    int height, width, degree, degreesToUse;
    std::tuple<int,int,int> tileBounds;
    float cam_pos[3];
  };
  CamSetup prepareCam(Camera& cam, int step);
  void fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight);
  MTensor render(Camera& cam, int step);

  MTensor means;
  MTensor scales;
  MTensor quats;
  MTensor featuresDc;
  MTensor featuresRest;
  MTensor opacities;

  static constexpr int N_ADAM_GROUPS = 6;
  MTensor adam_exp_avg[N_ADAM_GROUPS];
  MTensor adam_exp_avg_sq[N_ADAM_GROUPS];
  int adam_step_count = 0;
  float adam_lr[N_ADAM_GROUPS] = {};
  float adam_beta1 = 0.9f, adam_beta2 = 0.999f, adam_eps = 1e-8f;
  float means_lr_init = 0, means_lr_final = 0;

  MTensor means_buf, scales_buf, quats_buf, featuresDc_buf, featuresRest_buf, opacities_buf;
  MTensor adam_exp_avg_buf[N_ADAM_GROUPS], adam_exp_avg_sq_buf[N_ADAM_GROUPS];
  int num_active = 0, buf_capacity = 0;
  void refreshViews();
  void ensureCapacity(int needed);

  MTensor densify_split_flag, densify_dup_flag;
  MTensor densify_split_prefix, densify_dup_prefix;
  MTensor densify_keep_flag, densify_keep_prefix;
  MTensor densify_block_totals;
  MTensor densify_compact_scratch;
  MTensor densify_random_samples;

  MTensor radii;
  int lastHeight;
  int lastWidth;

  MTensor xysGradNorm;
  MTensor visCounts;
  MTensor max2DSize;

  MTensor backgroundColor;
  MTensor window2d;  // SSIM window (11,11) f32

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  bool keepCrs;
  int maxGaussians;   // 0 = unlimited; >0 halts densification once reached

  // ── MCMC 密集化（3DGS-MCMC, arXiv:2404.09591）──
  // 取代梯度啟發式 clone/split/prune：固定預算、把死高斯(低透明度)relocate 到高透明度處、
  // 每步注入 SGLD 位置噪聲探索。手機端優勢＝預算固定（記憶體有界）＋同預算下畫質更好。
  bool useMcmc = true;
  std::mt19937 mcmcRng{1234567u};
  void mcmcRefine(int step);          // MRNF 密集化：梯度引導候選 + Gumbel top-k + Long Axis Split
  void mrnfLongAxisSplit(int parent, int child);  // 沿最長軸切兩半並 ±偏移（取代原地複製）

  // ── 誤差圖引導（MRNF use_error_map 等價；per-gaussian 累加，無新 Metal kernel）──
  // 梯度引導回答「哪裡還沒重建好」，細節圖回答「哪裡本來就有紋理」——兩者互補。
  bool useEdgeGuidance = true;
  std::vector<float> mrnfEdgeScore;   // [buf_capacity] refine 視窗內累加的細節分
  std::vector<float> mrnfEdgeCount;   // [buf_capacity] 對應被觀測次數
  void mrnfAccumEdge(Camera& cam, int step);   // 投影取樣細節圖 → 累加
  void mcmcInjectNoise(float lr);     // SGLD 位置噪聲
  // 正則化參考量（初始點雲統計，只算一次）
  float mcmcScaleRef = 0.0f;          // 初始線性 scale 均值 ≈ 局部點距（尺度正則的無量綱化基準）
  float mcmcMaxScale = 0.0f;          // 世界系 scale 硬上限 = 0.1 × 場景 RMS 半徑
  void mcmcComputeScaleRefs();
  void mcmcRegularize(int step);      // opacity/scale 退火衰減 + 巨大高斯夾限（對齊 LichtFeld MRNF）
  void mcmcCopyGaussian(int dst, int src);  // 複製全部參數 src→dst
  void mcmcResetAdam(int idx);              // 清零該 slot 的 Adam 動量（6 群）

  // ── 相機姿態優化（SO3×R3 / SE3 左擾動，per-camera Adam）──
  // ARKit 姿態非完美；訓練中聯合精修每台相機的 6-DOF 修正 → 銳化重建。預設開。
  // 梯度由 backward 的世界系 v_mean3d 在 CPU 端組裝（g_cam=W·v_mean3d；grad_τ=Σg_cam、grad_ω=Σp_cam×g_cam）。
  bool useCameraOpt = true;
  void cameraPoseStep(Camera& cam, int step);

  // ── 外觀校正（per-image 學習式仿射；bilateral guided）──
  // 吸收逐幀曝光/白平衡差異，避免多視角顏色不一致污染幾何（霧感/飄浮物）。作用於損失層。
  bool useAppearance = true;
  void appearanceEnsureInit(Camera& cam);   // 首次把 affine 設為 identity
  void appearanceStep(Camera& cam, int step);  // 讀 GPU 累加的 affine 梯度 → per-camera Adam + L2 拉回 identity

  // ── LiDAR 深度監督（per-gaussian，CPU 端）──
  // 用 LiDAR 度量深度真值把「近表面」高斯沿光學軸拉到正確深度 → 表面更薄更準、幾何更硬。
  // band 閘門只碰接近 LiDAR 表面的高斯（不亂拉遮擋/浮點，安全）。
  bool useDepthSupervision = true;
  void depthRefineStep(Camera& cam, int step);

  float scale;
  float translation[3] = {};
};

#endif
