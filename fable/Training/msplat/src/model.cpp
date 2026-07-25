#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include "model.hpp"
#include "kdtree_tensor.hpp"
#include "msplat.hpp"
#include "loaders.hpp"

namespace fs = std::filesystem;

static const double C0 = 0.28209479177387814;

int numShBases(int degree){
    switch(degree){
        case 0: return 1;
        case 1: return 4;
        case 2: return 9;
        case 3: return 16;
        default: return 25;
    }
}

// Metrics on CPU MTensor data
float psnr(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double mse = 0;
    for (int64_t i = 0; i < n; i++) { double d = r[i] - g[i]; mse += d * d; }
    mse /= n;
    return 10.0f * std::log10(1.0 / mse);
}

float l1_loss(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double sum = 0;
    for (int64_t i = 0; i < n; i++) sum += std::abs(r[i] - g[i]);
    return (float)(sum / n);
}

// Model constructor
Model::Model(const InputData &inputData, int numCameras,
    int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
    int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
    int maxSteps, bool keepCrs, int maxGaussians,
    const float* bgColor)
    : numCameras(numCameras), numDownscales(numDownscales), resolutionSchedule(resolutionSchedule),
      shDegree(shDegree), shDegreeInterval(shDegreeInterval),
      refineEvery(refineEvery), warmupLength(warmupLength), resetAlphaEvery(resetAlphaEvery),
      stopSplitAt(maxSteps / 2), densifyGradThresh(densifyGradThresh), densifySizeThresh(densifySizeThresh),
      stopScreenSizeAt(stopScreenSizeAt), splitScreenSize(splitScreenSize),
      maxSteps(maxSteps), keepCrs(keepCrs), maxGaussians(maxGaussians) {

    int64_t numPoints = inputData.points.count;
    scale = inputData.scale;
    memcpy(translation, inputData.translation, sizeof(translation));

    // Means: copy xyz directly to GPU
    means = gpu_empty({numPoints, 3}, DType::Float32);
    memcpy(means.data_ptr(), inputData.points.xyz.data(), numPoints * 3 * sizeof(float));

    // Scales: KD-tree nearest neighbor distances, log'd, repeated 3x
    {
        PointsTensor pt(inputData.points.xyz.data(), numPoints);
        auto sc = pt.scales();  // vector<float> of length numPoints
        scales = gpu_empty({numPoints, 3}, DType::Float32);
        float *sp = scales.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            float v = std::log(sc[i]);
            sp[i*3] = sp[i*3+1] = sp[i*3+2] = v;
        }
    }

    // Random quaternions
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        quats = gpu_empty({numPoints, 4}, DType::Float32);
        float *qp = quats.data<float>();
        for (int64_t i = 0; i < numPoints; i++) {
            float u = dist(rng), v = dist(rng), w = dist(rng);
            qp[i*4+0] = std::sqrt(1-u) * std::sin(2*M_PI*v);
            qp[i*4+1] = std::sqrt(1-u) * std::cos(2*M_PI*v);
            qp[i*4+2] = std::sqrt(u) * std::sin(2*M_PI*w);
            qp[i*4+3] = std::sqrt(u) * std::cos(2*M_PI*w);
        }
    }

    // SH features: f_dc = rgb2sh(rgb), f_rest = zeros
    int dimSh = numShBases(shDegree);
    {
        featuresDc = gpu_empty({numPoints, 3}, DType::Float32);
        float *dp = featuresDc.data<float>();
        const uint8_t *rgb = inputData.points.rgb.data();
        for (int64_t i = 0; i < numPoints; i++) {
            for (int c = 0; c < 3; c++)
                dp[i*3+c] = (float)((rgb[i*3+c] / 255.0 - 0.5) / C0);
        }
        featuresRest = gpu_zeros({numPoints, (int64_t)(dimSh - 1), 3}, DType::Float32);
    }

    // 初始 opacity。原版 3DGS（梯度啟發式）用 0.1；MCMC 系列一律用 0.5
    // （3DGS-MCMC 論文、gsplat MCMC strategy、LichtFeld-Studio init_opacity=0.5）。
    // 理由：MCMC 靠 relocate 重分配透明度，new_op = 1−(1−o)^(1/n)。從 0.1 起跳時
    // n=2 只得到 0.051（幾乎貼著 0.005 死亡門檻），relocation 的 scale 修正也跟著失真；
    // 從 0.5 起跳得到 0.293，才是這套公式設計的工作點。稀疏點雲(19k)下尤其明顯。
    {
        float logitInit = std::log(0.5f / 0.5f);   // = 0
        opacities = gpu_empty({numPoints, 1}, DType::Float32);
        float *op = opacities.data<float>();
        for (int64_t i = 0; i < numPoints; i++) op[i] = logitInit;
    }

    // Background color — default is magenta (high-contrast against typical scenes,
    // makes under-reconstructed regions obvious during training)
    backgroundColor = gpu_empty({3}, DType::Float32);
    static const float defaultBg[3] = {0.6130f, 0.0101f, 0.3984f};
    memcpy(backgroundColor.data_ptr(), bgColor ? bgColor : defaultBg, 3 * sizeof(float));
    setupOptimizers();
}

// ─────────────────────────────────────────────────────────────────────────────
// MCMC / MRNF 密集化。基底是 3DGS-MCMC（arXiv:2404.09591）的 SGLD 噪聲探索，
// 密集化本體改用 LichtFeld-Studio 的 MRNF strategy（見下方 kMrnf* 常數）。
// msplat 的參數緩衝為 MTLStorageModeShared → 直接在 CPU 端操作（免新寫 Metal kernel，低風險）。
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float kMcmcNoiseLr = 5e5f;    // SGLD 噪聲尺度（× 當前 means lr）
static constexpr float kMcmcNoiseT  = 0.005f;  // 噪聲 opacity 閘門轉折點
static constexpr float kMcmcNoiseK  = 100.0f;  // 噪聲 opacity 閘門銳度
static constexpr float kMcmcGrowRate= 1.12f;   // 成長倍率下界/回退值
static constexpr float kMcmcGrowMax = 1.35f;   // 單次 refine 成長上限（避免一次暴衝）
// 正則化：形式與強度對齊 LichtFeld-Studio 的 MRNF decay
//   （src/training/kernels/mrnf_kernels.cu，MrNeRF/LichtFeld-Studio）：
//     opac  = sigmoid(u) − opacity_decay·(1−t)      → 再 logit 回去（作用在「σ 空間」，非 logit）
//     scale = exp(w) · (1 − scale_decay·(1−t))      → 乘性衰減（＝log 空間平移）
//   (1−t) 隨訓練退火 ⇒ 後期收斂不再被壓縮，穩定。
//
// 關鍵：不能照抄它的 per-application 常數。它每次 refine 才 apply_decay（30000 步 / 245 次），
// 我們是每 kMcmcCpuEvery 步施加（6000 步 / ~1375 次）→ 必須換算「整場總量」才等價。
// LichtFeld 總量（refine 落在 600..24900，245 次，avg(1−t)=0.575）：
//   scale  : ln∏(1−0.002·(1−t)) = −0.002×245×0.575 = −0.282  ⇒ 平均尺度縮至 75%
//   opacity: Σ 0.004·(1−t)      =  0.004×245×0.575 =  0.564  ⇒ σ 總共減 0.564（壓力極大，
//            刻意讓光度損失不護的高斯沉到死亡門檻 → 被 relocate 到需要的地方，這是 MCMC 的引擎）
// 以「總量」定義常數、per-application 係數在執行期換算 → 改 maxSteps/攤提間隔都不會失準。
static constexpr float kMcmcScaleShrinkTotal  = 0.282f;  // 整場 log 空間總收縮量
static constexpr float kMcmcOpacityDecayTotal = 0.564f;  // 整場 σ 空間總衰減量
// LichtFeld 用 prune_scale3d=0.1（×場景範圍）真的「剪掉」超大高斯；我們無剪枝路徑，改用夾限
// （較溫和：高斯留著但不得再長大），這是 MCMC 路徑唯一的尺寸上界。
static constexpr float kMcmcMaxScaleFrac = 0.05f;

// ── MRNF 密集化（LichtFeld-Studio 的 MRNF strategy）──
// 取代原本「依 opacity 的 CDF 取樣 + 原地複製」。三個關鍵差異：
//   ① 候選由螢幕空間梯度引導：只在殘差大、且本輪真的被看到的地方長點
//      （原本對全體依 opacity 取樣 → 點長在「已經很亮」的地方，跟「哪裡還沒重建好」無關）
//   ② Gumbel top-k 不重複取樣（原本 CDF 可重複 → 同一顆被抽中 N 次 → 一堆完全重合的副本）
//   ③ Long Axis Split：沿最長軸切成兩半並 ±偏移
//      ← 這是畫面糊掉的根因。原地複製產生「位置相同、尺度相同」的子高斯，不帶任何新幾何
//        資訊，只能靠 SGLD 噪聲推開，但噪聲閘門 sigmoid(-100(σ-0.005)) 只對接近死亡的高斯
//        開啟 → 19k 長到 212k 實際上是每顆疊了 ~11 份 9cm 的重合副本。
static constexpr float kMrnfPruneOpacityRaw = -5.54126358f; // logit(1/255)：剪枝門檻
static constexpr float kMrnfLogMinScale     = -23.0258509f; // log(1e-10)：退化軸
static constexpr float kMrnfLasLong    = 0.5f;   // LAS：長軸 ×0.5
static constexpr float kMrnfLasOther   = 0.85f;  // LAS：其餘兩軸 ×0.85
static constexpr float kMrnfLasOpacity = 0.6f;   // LAS：父子各 0.6×σ
// 候選池比例。必須 >> 每次 refine 要挑的數量，否則 Gumbel top-k 沒有選擇餘地（實測 7% 時
// split==cand，引導形同關閉）。取梯度最高的 30%：以 n_add≈6% 計，池子約為需求的 5 倍。
static constexpr float kMrnfCandFraction = 0.30f;

// Gumbel top-k：依權重 w 做「不重複」加權取樣（Gumbel-max trick, Vieira 2014）。
// key_i = log(w_i) + G_i，G_i = −log(−log U_i) ⇒ 取 key 最大的 k 個
//   ⇔ 依 w 不重複加權取樣。用 nth_element → O(M)，20 萬顆可接受。
static void mrnfGumbelTopK(const std::vector<int>& cand, const std::vector<float>& w,
                           int k, std::mt19937& rng, std::vector<int>& out){
    out.clear();
    const int M = (int)cand.size();
    if(M <= 0 || k <= 0) return;
    if(k >= M){ out = cand; return; }
    std::vector<std::pair<float,int>> keys((size_t)M);
    std::uniform_real_distribution<float> U(1e-12f, 1.0f);
    for(int j=0;j<M;j++){
        const float wj = w[(size_t)j];
        const float key = (wj > 0.0f)
            ? (std::log(wj) - std::log(-std::log(U(rng))))
            : -std::numeric_limits<float>::infinity();
        keys[(size_t)j] = {key, cand[(size_t)j]};
    }
    std::nth_element(keys.begin(), keys.begin()+k, keys.end(),
                     [](const std::pair<float,int>& a, const std::pair<float,int>& b){
                         return a.first > b.first;
                     });
    out.reserve((size_t)k);
    for(int j=0;j<k;j++) if(std::isfinite(keys[(size_t)j].first)) out.push_back(keys[(size_t)j].second);
}
// CPU 端 O(N) 維護的攤提間隔。SGLD 是隨機漫步 → 位移變異數線性累加，
// 「每 k 步、尺度 ×√k」與「每步、尺度 ×1」擴散係數相同（kσ²），CPU 成本卻只有 1/k。
static constexpr int   kMcmcCpuEvery  = 4;
static constexpr float kMcmcNoiseSqrtK = 2.0f;    // = √kMcmcCpuEvery

static inline float mcmcSigmoid(float x){ return 1.0f/(1.0f+std::exp(-x)); }
static inline float mcmcLogit(float p){ p=std::min(std::max(p,1e-7f),1.0f-1e-7f); return std::log(p/(1.0f-p)); }


void Model::mcmcCopyGaussian(int dst,int src){
    if(dst==src) return;
    int fr=(int)featuresRest_buf.stride0();
    auto cp=[&](MTensor& buf,int dim){ float* p=buf.data<float>(); std::memcpy(p+(size_t)dst*dim, p+(size_t)src*dim, (size_t)dim*sizeof(float)); };
    cp(means_buf,3); cp(scales_buf,3); cp(quats_buf,4); cp(featuresDc_buf,3); cp(featuresRest_buf,fr); cp(opacities_buf,1);
}

// Long Axis Split（LichtFeld densification_kernels.cu: long_axis_split_gaussians_inplace_kernel）
// 沿最長主軸把一顆高斯切成兩顆：
//   長軸 ×0.5、其餘兩軸 ×0.85、父子各 0.6×σ，位置沿該主軸的世界方向 ±exp(s_long)/2。
// 父高斯就地變成其中一半，child slot 收另一半 → 一次呼叫真正細化了幾何。
void Model::mrnfLongAxisSplit(int parent, int child){
    float* mean = means_buf.data<float>();
    float* scb  = scales_buf.data<float>();   // log
    float* qtb  = quats_buf.data<float>();    // (w,x,y,z)
    float* opb  = opacities_buf.data<float>();// logit

    // 最長軸（log 單調 ⇒ 在 log 空間比較等價於線性空間比較）
    int li = 0;
    if(scb[parent*3+1] > scb[parent*3+li]) li = 1;
    if(scb[parent*3+2] > scb[parent*3+li]) li = 2;
    const float offMag = std::exp(scb[parent*3+li]) * 0.5f;

    float qw=qtb[parent*4+0], qx=qtb[parent*4+1], qy=qtb[parent*4+2], qz=qtb[parent*4+3];
    const float qn=std::sqrt(qw*qw+qx*qx+qy*qy+qz*qz);
    if(qn>1e-9f){ const float iv=1.0f/qn; qw*=iv; qx*=iv; qy*=iv; qz*=iv; }
    // row-major 3×3；取第 li 直行（R[li], R[li+3], R[li+6]）＝該主軸的世界方向
    const float R[9]={
        1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy),
        2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx),
        2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)
    };
    const float o0=R[li]*offMag, o1=R[li+3]*offMag, o2=R[li+6]*offMag;

    // 新尺度（log 空間加常數 ＝ 線性空間乘常數）
    const float lnLong=std::log(kMrnfLasLong), lnOther=std::log(kMrnfLasOther);
    float ns[3];
    for(int j=0;j<3;j++) ns[j] = scb[parent*3+j] + (j==li ? lnLong : lnOther);

    const float newOp = mcmcLogit(mcmcSigmoid(opb[parent]) * kMrnfLasOpacity);

    // 先存下原始位置，再複製（複製會把父的當前值搬過去）
    const float p0=mean[parent*3+0], p1=mean[parent*3+1], p2=mean[parent*3+2];
    mcmcCopyGaussian(child, parent);          // quat / SH / … 全部承襲

    mean[parent*3+0]=p0+o0; mean[parent*3+1]=p1+o1; mean[parent*3+2]=p2+o2;
    mean[child *3+0]=p0-o0; mean[child *3+1]=p1-o1; mean[child *3+2]=p2-o2;
    for(int j=0;j<3;j++){ scb[parent*3+j]=ns[j]; scb[child*3+j]=ns[j]; }
    opb[parent]=newOp; opb[child]=newOp;
    // 父子的 Adam 動量都必須歸零（幾何被硬改，舊的一二階估計已失效）——與 LichtFeld 一致
    mcmcResetAdam(parent); mcmcResetAdam(child);
}

void Model::mcmcResetAdam(int idx){
    int fr=(int)featuresRest_buf.stride0();
    int dims[N_ADAM_GROUPS]={3,3,4,3,fr,1};
    for(int g=0;g<N_ADAM_GROUPS;g++){
        std::memset(adam_exp_avg_buf[g].data<float>()+(size_t)idx*dims[g], 0, (size_t)dims[g]*sizeof(float));
        std::memset(adam_exp_avg_sq_buf[g].data<float>()+(size_t)idx*dims[g], 0, (size_t)dims[g]*sizeof(float));
    }
}

void Model::mcmcRefine(int step){
    const int N=num_active;
    if(N<=1) return;
    float* opb=opacities_buf.data<float>();  // logit
    float* scb=scales_buf.data<float>();      // log

    // ---- 1) 剪枝（MRNF refine 的條件）→ 產生可回收的 slot ----
    // 我們的 num_active 是連續前綴、不能有洞，所以「回收」＝把 child 寫進死掉的 slot
    // （功能等價於 LichtFeld 的 _free_mask soft-prune + fill_free_slots）。
    std::vector<uint8_t> isDead((size_t)N, 0);
    std::vector<int> dead; dead.reserve((size_t)N/8);
    for(int i=0;i<N;i++){
        bool bad = (opb[i] < kMrnfPruneOpacityRaw);
        if(!bad){
            const float smin = std::min(scb[i*3+0], std::min(scb[i*3+1], scb[i*3+2]));
            bad = (smin < kMrnfLogMinScale);
        }
        if(bad){ isDead[(size_t)i] = 1; dead.push_back(i); }
    }

    // ---- 2) 候選集：MRNF 的 refine_candidates ----
    //   (平均螢幕梯度 > densifyGradThresh) && (本輪可見)
    //   avgGrad 與 msplat densify kernel 同單位：(Σ‖∇xy‖ / vis) × half_max_dim
    //   → 直接沿用已校準的 densifyGradThresh，不用去換算 LichtFeld 的 0.003。
    std::vector<int> cand; std::vector<float> cw;
    const bool haveStats = xysGradNorm.defined() && visCounts.defined()
                        && xysGradNorm.numel() >= N && visCounts.numel() >= N;
    const bool have2D = max2DSize.defined() && max2DSize.numel() >= N;
    if(haveStats){
        const float* gn = xysGradNorm.data<float>();
        const float* vc = visCounts.data<float>();
        const float* m2d = have2D ? max2DSize.data<float>() : nullptr;
        const float halfMaxDim = 0.5f * (float)std::max(lastWidth, lastHeight);
        // 門檻改為「百分位」而非絕對值。實測用絕對值 densifyGradThresh 時候選只佔 ~7%，
        // 導致 nSplit == cand.size()（池子被抽乾）→ Gumbel top-k 退化成「全拿」→
        // 權重完全不起作用 → 梯度/細節引導實質失效。池子必須顯著大於要挑的數量才有選擇壓力。
        // 百分位還順帶修掉解析度相依性：avgGrad ∝ halfMaxDim，coarse-to-fine 在 ds 切換那一步
        // 會讓固定門檻的選擇性突然跳 2 倍（實測 step 1000 候選數暴增 2.4 倍）。
        {
            std::vector<float> g; g.reserve((size_t)N);
            for(int i=0;i<N;i++){
                if(isDead[(size_t)i] || vc[i] <= 0.0f) continue;
                g.push_back((gn[i]/vc[i]) * halfMaxDim);
            }
            if(!g.empty()){
                const size_t keep = std::max<size_t>(1, (size_t)(g.size() * kMrnfCandFraction));
                std::nth_element(g.begin(), g.begin()+(g.size()-keep), g.end());
                mrnfGradCut = g[g.size()-keep];      // 取梯度最高的 kMrnfCandFraction 比例
            } else {
                mrnfGradCut = densifyGradThresh;
            }
        }
        cand.reserve((size_t)N/3); cw.reserve((size_t)N/3);
        for(int i=0;i<N;i++){
            if(isDead[(size_t)i] || vc[i] <= 0.0f) continue;
            const float avg = (gn[i]/vc[i]) * halfMaxDim;
            // 螢幕空間過大者「優先分裂」而非剪枝：剪掉會留洞，LAS 切一半直接把螢幕面積砍 4×，
            // 正是 per-tile overflow(>2048/tile) 的解。max2DSize 是視窗內最大正規化螢幕半徑
            // （accumulate_grad_stats_kernel: radii × 1/max(H,W)），splitScreenSize=0.05 是
            // msplat 既有已校準的門檻（原本只有梯度啟發式路徑在用，MRNF 路徑白白忽略了它）。
            const float over = (m2d && splitScreenSize > 0.0f) ? (m2d[i] / splitScreenSize) : 0.0f;
            float w = avg;
            if(over > 1.0f) w = std::max(avg, mrnfGradCut) * over;  // 超出越多、權重越高
            else if(!(avg >= mrnfGradCut)) continue;                // 梯度低且螢幕不大 → 跳過
            cand.push_back(i); cw.push_back(w);
        }
        // 細節圖引導：×(1 + 0.25·score/median)，與 LichtFeld edge_guidance_factor 同形式
        // （MRNF_EDGE_SCORE_WEIGHT=0.25，並以「正值中位數」正規化）。
        if(useEdgeGuidance && (int)mrnfEdgeScore.size() >= N && !cand.empty()){
            std::vector<float> es; es.reserve(cand.size());
            for(int idx : cand){
                const float c = mrnfEdgeCount[(size_t)idx];
                es.push_back(c > 0.0f ? mrnfEdgeScore[(size_t)idx]/c : 0.0f);
            }
            std::vector<float> pos; pos.reserve(es.size());
            for(float v : es) if(v > 0.0f) pos.push_back(v);
            if(!pos.empty()){
                std::nth_element(pos.begin(), pos.begin()+pos.size()/2, pos.end());
                const float med = pos[pos.size()/2];
                if(med > 1e-6f)
                    for(size_t j=0;j<cw.size();j++) cw[j] *= (1.0f + 0.25f * (es[j]/med));
            }
        }
    }
    // 首次 refine（統計剛重配、全為 0）時退回全體、權重用 opacity → 不會卡住成長。
    if(cand.empty()){
        cand.reserve((size_t)N); cw.reserve((size_t)N);
        for(int i=0;i<N;i++){
            if(isDead[(size_t)i]) continue;
            cand.push_back(i); cw.push_back(mcmcSigmoid(opb[i]));
        }
    }

    // ---- 3) 成長量：保留「在短排程內填滿預算」的自適應率 ----
    // MRNF 原式是 候選數×grow_fraction(0.07)，那是為 145 次 refine 設計的；我們只有 24 次，
    // 照抄會長不到預算（記憶體照付、畫質沒拿到）。改為解 rate^n = budget/N ⇒ rate=(budget/N)^(1/n)，
    // 每次 refine 重算 → 自我修正，剛好在 stopSplitAt 前用滿。
    int n_left = (stopSplitAt - 1 - step) / refineEvery + 1;   // 含本次，尚餘幾次 refine
    if(n_left < 1) n_left = 1;
    float rate = kMcmcGrowRate;
    if(buf_capacity > N){
        rate = (float)std::pow((double)buf_capacity/(double)N, 1.0/(double)n_left);
        rate = std::min(std::max(rate, 1.02f), kMcmcGrowMax);
    }
    int n_add = std::min((int)std::ceil((double)N*rate) - N, buf_capacity - N);
    if(n_add < 0) n_add = 0;

    // ---- 4) Gumbel top-k 選 parent（不重複），每個 parent 做一次 Long Axis Split ----
    int nSplit = std::min((int)dead.size() + n_add, (int)cand.size());
    std::vector<int> parents;
    mrnfGumbelTopK(cand, cw, nSplit, mcmcRng, parents);

    int usedDead = 0, appended = 0;
    for(int p : parents){
        int childSlot;
        if(usedDead < (int)dead.size())            childSlot = dead[(size_t)usedDead++];
        else if(N + appended < buf_capacity)       childSlot = N + appended++;
        else                                      break;
        mrnfLongAxisSplit(p, childSlot);
    }
    num_active = N + appended;
    if(!parents.empty()){
        xysGradNorm.reset(); visCounts.reset(); max2DSize.reset();  // 下步重配並歸零累加窗
    }
    // 細節分也要歸零：它是「本 refine 視窗」的統計，跨窗累積會讓引導失去時效
    if(!mrnfEdgeScore.empty()){
        std::fill(mrnfEdgeScore.begin(), mrnfEdgeScore.end(), 0.0f);
        std::fill(mrnfEdgeCount.begin(), mrnfEdgeCount.end(), 0.0f);
    }
    std::cout << "MRNF step " << step << ": active=" << num_active
              << " (cand " << cand.size() << ", split " << parents.size()
              << ", reused " << usedDead << ", appended " << appended << ")" << std::endl;
}

// 初始點雲統計 → 正則化的兩個參考量（只算一次；用初始值才不會隨訓練漂移）。
void Model::mcmcComputeScaleRefs(){
    const int N = num_active;
    if(N <= 0) return;
    const float* mn = means_buf.data<float>();
    const float* sc = scales_buf.data<float>();
    double cx=0, cy=0, cz=0, ssum=0;
    for(int i=0;i<N;i++){ cx+=mn[i*3+0]; cy+=mn[i*3+1]; cz+=mn[i*3+2]; }
    cx/=N; cy/=N; cz/=N;
    double r2=0;
    for(int i=0;i<N;i++){
        double dx=mn[i*3+0]-cx, dy=mn[i*3+1]-cy, dz=mn[i*3+2]-cz;
        r2 += dx*dx+dy*dy+dz*dz;
        ssum += std::exp(sc[i*3+0]) + std::exp(sc[i*3+1]) + std::exp(sc[i*3+2]);
    }
    float rms = (float)std::sqrt(r2/(double)N);          // 場景 RMS 半徑
    mcmcScaleRef = (float)(ssum/(3.0*(double)N));
    if(!(mcmcScaleRef > 1e-9f)) mcmcScaleRef = 1e-3f;
    // 下界綁在 3×scaleRef：小/薄場景不會被夾到無法表達平面（牆面等大平板仍可用）。
    // 舊值 8× 讓下界永遠勝出（實測 8×0.0924=0.739 > 0.1×5.333=0.533）→ 夾限形同虛設。
    mcmcMaxScale = std::max(kMcmcMaxScaleFrac * rms, mcmcScaleRef * 3.0f);
    fprintf(stderr, "MCMC reg refs: N=%d sceneRMS=%.4f scaleRef=%.5f maxScale=%.5f (ratio %.1fx)\n",
            N, rms, mcmcScaleRef, mcmcMaxScale, mcmcMaxScale / mcmcScaleRef);
}

// opacity / scale 退火衰減 + 巨大高斯硬上限（形式對齊 LichtFeld MRNF，見上方常數推導）。
// 為什麼這同時提升品質與速度：
//   · opacity 衰減把沒貢獻的高斯壓到死亡門檻 → 下次 refine 被 relocate 到真正需要的地方
//     （＝固定預算用在刀口上，這是 MCMC 能贏梯度啟發式的核心機制）
//   · scale 乘性衰減讓表面更薄 → 每顆觸及的 tile 更少（rasterize 成本是 O(觸及面積)）
//   · 硬上限補上 MCMC 路徑完全缺失的尺寸控制（useMcmc 會 early-return 掉 checkHuge 剪枝）
//     → 抑制覆蓋整幀的巨大高斯造成的 per-tile overflow(>2048/tile) 掉點與方塊感
void Model::mcmcRegularize(int step){
    const int N = num_active;
    if(N <= 0 || mcmcMaxScale <= 0.0f) return;
    // 把「整場總量」換算成本次施加的係數：總量 = c · Σ(1−t)，Σ(1−t) ≈ nApply · avg(1−t)。
    const int nApply = std::max(1, (maxSteps - warmupLength) / kMcmcCpuEvery);
    const float t0 = (float)warmupLength / (float)std::max(maxSteps, 1);
    const float wSum = (float)nApply * (1.0f - 0.5f * (t0 + 1.0f));   // avg(1−t) = 1 − (t0+1)/2
    if(wSum <= 1e-6f) return;
    const float t = std::min(std::max((float)step / (float)std::max(maxSteps, 1), 0.0f), 1.0f);
    const float tShrink = 1.0f - t;                                   // 退火：後期壓力趨零
    const float od   =  (kMcmcOpacityDecayTotal / wSum) * tShrink;    // σ 空間減量
    const float lnSf = -(kMcmcScaleShrinkTotal  / wSum) * tShrink;    // log 空間收縮量（乘性衰減）
    float* opb = opacities_buf.data<float>();
    float* scb = scales_buf.data<float>();
    const float wMax = std::log(mcmcMaxScale);
    for(int i=0;i<N;i++){
        opb[i] = mcmcLogit(mcmcSigmoid(opb[i]) - od);   // mcmcLogit 已夾 [1e-7, 1−1e-7]
        for(int j=0;j<3;j++){
            float& w = scb[i*3+j];
            w += lnSf;
            if(w > wMax) w = wMax;
        }
    }
}

void Model::mcmcInjectNoise(float lr){
    const int N=num_active;
    const float noise_scale=lr*kMcmcNoiseLr;
    if(noise_scale<=0.0f) return;
    float* mean=means_buf.data<float>();
    const float* scb=scales_buf.data<float>();
    const float* qtb=quats_buf.data<float>();
    const float* opb=opacities_buf.data<float>();
    std::normal_distribution<float> nd(0.0f,1.0f);
    for(int i=0;i<N;i++){
        float o=mcmcSigmoid(opb[i]);
        float gate=1.0f/(1.0f+std::exp(kMcmcNoiseK*(o-kMcmcNoiseT)));  // = sigmoid(-k(o-t))
        if(gate<1e-3f) continue;   // 高透明度幾乎不動 → 早退省 CPU
        float s0=std::exp(scb[i*3+0]),s1=std::exp(scb[i*3+1]),s2=std::exp(scb[i*3+2]);
        float qw=qtb[i*4+0],qx=qtb[i*4+1],qy=qtb[i*4+2],qz=qtb[i*4+3];
        float qn=std::sqrt(qw*qw+qx*qx+qy*qy+qz*qz); if(qn>1e-9f){float iv=1.0f/qn;qw*=iv;qx*=iv;qy*=iv;qz*=iv;}
        float R[3][3]={
            {1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)},
            {2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)},
            {2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx*qx+qy*qy)}
        };
        float d0=s0*s0,d1=s1*s1,d2=s2*s2, cov[3][3];
        for(int a=0;a<3;a++)for(int b=0;b<3;b++)
            cov[a][b]=R[a][0]*d0*R[b][0]+R[a][1]*d1*R[b][1]+R[a][2]*d2*R[b][2];
        float e0=nd(mcmcRng)*gate*noise_scale, e1=nd(mcmcRng)*gate*noise_scale, e2=nd(mcmcRng)*gate*noise_scale;
        mean[i*3+0]+=cov[0][0]*e0+cov[0][1]*e1+cov[0][2]*e2;
        mean[i*3+1]+=cov[1][0]*e0+cov[1][1]*e1+cov[1][2]*e2;
        mean[i*3+2]+=cov[2][0]*e0+cov[2][1]*e1+cov[2][2]*e2;
    }
}

void Model::setupOptimizers(){
    releaseOptimizers();


    num_active = means.size(0);
    // 固定預配置到 maxGaussians（若設定）：訓練全程 buf_capacity 不變 → 峰值記憶體確定，
    // 消除動態倍增（44k→88k→…）與 grow 時 old+new 併存的暫時尖峰（on-device OOM 主因）。
    buf_capacity = (maxGaussians > 0) ? std::max((int)num_active, maxGaussians) : (int)(num_active * 4);
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);
    if (mcmcScaleRef <= 0.0f) mcmcComputeScaleRefs();   // 用初始點雲統計；載入 checkpoint 後不重算

    static constexpr float lr_init[] = {0.00016f, 0.005f, 0.001f, 0.0025f, 0.000125f, 0.05f};
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        auto shape = params[g]->shape();
        shape[0] = buf_capacity;
        adam_exp_avg_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_exp_avg_sq_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_lr[g] = lr_init[g];
    }
    adam_step_count = 0;
    means_lr_init = 0.00016f;
    means_lr_final = 0.0000016f;

    densify_split_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_split_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    int max_blocks = (buf_capacity + 1023) / 1024;
    densify_block_totals = gpu_zeros({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest.numel() / featuresRest.size(0);
    densify_compact_scratch = gpu_zeros({(int64_t)buf_capacity * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({buf_capacity, 3}, DType::Float32);

    refreshViews();
}

void Model::releaseOptimizers(){
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g].reset(); adam_exp_avg_sq[g].reset();
        adam_exp_avg_buf[g].reset(); adam_exp_avg_sq_buf[g].reset();
    }
    means_buf.reset(); scales_buf.reset(); quats_buf.reset();
    featuresDc_buf.reset(); featuresRest_buf.reset(); opacities_buf.reset();
    densify_split_flag.reset(); densify_dup_flag.reset();
    densify_split_prefix.reset(); densify_dup_prefix.reset();
    densify_keep_flag.reset(); densify_keep_prefix.reset();
    densify_block_totals.reset(); densify_compact_scratch.reset(); densify_random_samples.reset();
}

void Model::schedulersStep(int step){
    float t = std::clamp((float)step / (float)maxSteps, 0.f, 1.f);
    adam_lr[0] = std::exp(std::log(means_lr_init) * (1.f - t) + std::log(means_lr_final) * t);
}

void Model::refreshViews(){
    means = means_buf.view(num_active);
    scales = scales_buf.view(num_active);
    quats = quats_buf.view(num_active);
    featuresDc = featuresDc_buf.view(num_active);
    featuresRest = featuresRest_buf.view(num_active);
    opacities = opacities_buf.view(num_active);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g] = adam_exp_avg_buf[g].view(num_active);
        adam_exp_avg_sq[g] = adam_exp_avg_sq_buf[g].view(num_active);
    }
}

void Model::ensureCapacity(int needed){
    if (needed <= buf_capacity) return;
    int new_cap = std::max(needed, buf_capacity * 2);

    auto grow = [&](MTensor &buf) {
        auto shape = buf.shape();
        shape[0] = new_cap;
        MTensor new_buf = gpu_zeros(shape, DType::Float32);
        size_t copy_bytes = num_active * buf.stride0() * sizeof(float);
        memcpy(new_buf.data_ptr(), buf.data_ptr(), copy_bytes);
        buf = new_buf;
    };
    grow(means_buf); grow(scales_buf); grow(quats_buf);
    grow(featuresDc_buf); grow(featuresRest_buf); grow(opacities_buf);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        grow(adam_exp_avg_buf[g]);
        grow(adam_exp_avg_sq_buf[g]);
    }
    densify_split_flag = gpu_zeros({new_cap}, DType::Int32);
    densify_dup_flag = gpu_zeros({new_cap}, DType::Int32);
    densify_split_prefix = gpu_zeros({new_cap}, DType::Int32);
    densify_dup_prefix = gpu_zeros({new_cap}, DType::Int32);
    densify_keep_flag = gpu_zeros({new_cap}, DType::Int32);
    densify_keep_prefix = gpu_zeros({new_cap}, DType::Int32);
    int max_blocks = (new_cap + 1023) / 1024;
    densify_block_totals = gpu_zeros({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest_buf.stride0();
    densify_compact_scratch = gpu_zeros({(int64_t)new_cap * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({new_cap, 3}, DType::Float32);

    buf_capacity = new_cap;
    refreshViews();
}

int Model::getDownscaleFactor(int step) {
    int remaining = numDownscales - step / resolutionSchedule;
    return 1 << std::max(remaining, 0);
}

void Model::afterTrain(int step){
    if (!radii.defined()) return;

    if (useMcmc) {
        // MCMC 路徑：取代梯度啟發式 clone/split/prune 與週期性 opacity reset。
        // 先 sync 確保本步 GPU Adam 已寫回 shared buffer，CPU 才能安全讀寫參數。
        msplat_gpu_sync();
        // CPU 端 O(N) 維護攤提到每 kMcmcCpuEvery 步（速度）：
        //   · SGLD 噪聲 ×√k → 擴散係數 kσ² 不變（隨機漫步變異數線性累加），統計行為等價
        //   · 正則化 ×k    → 累積衰減量不變（單步衰減量極小，一階等價）
        // 點數長到 300k 後這一趟正是成長最快的 O(N) 項，攤提剛好抵掉它。
        if (step > warmupLength && step % kMcmcCpuEvery == 0) {
            mcmcInjectNoise(adam_lr[0] * kMcmcNoiseSqrtK);
            mcmcRegularize(step);
        }
        if (step % refineEvery == 0 && step > warmupLength && step < stopSplitAt) {
            mcmcRefine(step);      // relocate 死高斯 + 成長到固定預算
            refreshViews();
        }
        return;
    }

    if (step % refineEvery == 0 && step > warmupLength){
        int resetInterval = resetAlphaEvery * refineEvery;
        bool doDensification = step < stopSplitAt && step % resetInterval > numCameras + refineEvery;
        // 記憶體硬界（數學根治）：densify 內部最壞會產生 3×num_active 顆並需 3N 的緩衝
        // （見 msplat_densify 的 worst_case=3N 與 assert(3N<=buf_capacity)）。
        // 只在「3N 不超過固定 buf_capacity」時才 densify → 這一趟不會要求成長、也不會一次暴衝，
        // 峰值記憶體恆等於 buf_capacity（確定、不隨場景爆掉）。0 = 不限。
        if (maxGaussians > 0 && 3 * num_active > buf_capacity) doDensification = false;

        if (doDensification){
            int numPointsBefore = num_active;
            ensureCapacity(3 * num_active);  // worst case: every gaussian splits

            // Fill random samples for splits (CPU randn, shared memory)
            {
                std::mt19937 rng(step);
                std::normal_distribution<float> dist(0.0f, 1.0f);
                float *p = densify_random_samples.data<float>();
                for (int64_t i = 0; i < 2 * num_active * 3; i++) p[i] = dist(rng);
            }

            float half_max_dim = 0.5f * static_cast<float>((std::max)(lastWidth, lastHeight));
            int check_screen = (step < stopScreenSizeAt) ? 1 : 0;
            // 巨大高斯剪枝啟用時機。原始 msplat 用 refineEvery*resetAlphaEvery（=3000，為 30000 步排程設計），
            // 但本專案 maxSteps=4000 → stopSplitAt=2000，densify 只在 step<2000 跑 → 3000 這門檻永不觸發 →
            // 覆蓋整幀的巨大高斯（初始化稀疏遠點造成）從不被剪 → 渲染時 tile overflow(>2048/tile) → 方塊。
            // 改為 warmup 後即啟用（落在 densify 視窗內），順帶降低高斯數/記憶體。
            bool checkHuge = step > warmupLength;
            int fr_stride = (int)featuresRest_buf.stride0();

            int new_count = msplat_densify(
                num_active, buf_capacity,
                densifyGradThresh, densifySizeThresh, splitScreenSize, check_screen,
                0.1f, 0.5f, 0.15f, checkHuge ? 1 : 0,
                xysGradNorm, visCounts, max2DSize, half_max_dim,
                means_buf, scales_buf, quats_buf,
                featuresDc_buf, featuresRest_buf, opacities_buf, fr_stride,
                adam_exp_avg_buf, adam_exp_avg_sq_buf,
                densify_split_flag, densify_dup_flag,
                densify_split_prefix, densify_dup_prefix,
                densify_keep_flag, densify_keep_prefix,
                densify_block_totals, densify_compact_scratch,
                densify_random_samples
            );

            num_active = new_count;
            refreshViews();
            std::cout << "Densified: " << numPointsBefore << " -> " << num_active << " gaussians" << std::endl;
        }

        if (step < stopSplitAt && step % resetInterval == refineEvery){
            msplat_gpu_sync();
            constexpr float resetLogit = -1.3862943611198906f;
            float *op = opacities.data<float>();
            for (int64_t i = 0; i < opacities.numel(); i++)
                if (op[i] > resetLogit) op[i] = resetLogit;

            adam_exp_avg[5].zero();
            adam_exp_avg_sq[5].zero();
            fprintf(stderr, "Opacity reset at step %d\n", step);
        }

        xysGradNorm.reset();
        visCounts.reset();
        max2DSize.reset();
    }
}

void Model::save(const std::string &filename, int step) {
    std::string ext = fs::path(filename).extension().string();
    if (ext == ".splat")
        saveSplat(filename);
    else
        savePly(filename, step);
    fprintf(stderr, "Saved %s\n", filename.c_str());
}

void Model::savePly(const std::string &filename, int step){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianPly(filename, p, step);
}

void Model::saveSplat(const std::string &filename){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs};
    saveGaussianSplat(filename, p);
}

int Model::loadPly(const std::string &filename){
    auto g = loadGaussianPly(filename, scale, translation, keepCrs);
    means = g.means;
    scales = g.scales;
    quats = g.quats;
    featuresDc = g.featuresDc;
    featuresRest = g.featuresRest;
    opacities = g.opacities;
    setupOptimizers();
    return g.step;
}

// ── Checkpoint save/load ────────────────────────────────────────────────────

static constexpr uint32_t CKPT_MAGIC = 0x4C50534D; // "MSPL"
static constexpr uint32_t CKPT_VERSION = 1;

static void writeTensor(std::ofstream &f, MTensor &t) {
    uint32_t ndim = t.ndim();
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int i = 0; i < (int)ndim; i++) {
        int64_t s = t.size(i);
        f.write(reinterpret_cast<const char*>(&s), sizeof(s));
    }
    uint64_t bytes = t.nbytes();
    f.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    f.write(reinterpret_cast<const char*>(t.data_ptr()), bytes);
}

static MTensor readTensor(std::ifstream &f) {
    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    std::vector<int64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++)
        f.read(reinterpret_cast<char*>(&shape[i]), sizeof(int64_t));
    uint64_t bytes;
    f.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
    MTensor t = gpu_empty(shape, DType::Float32);
    f.read(reinterpret_cast<char*>(t.data_ptr()), bytes);
    return t;
}

void Model::saveCheckpoint(const std::string &filename, int step) {
    msplat_gpu_sync();

    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file for writing: " + filename);

    // Header
    f.write(reinterpret_cast<const char*>(&CKPT_MAGIC), sizeof(CKPT_MAGIC));
    f.write(reinterpret_cast<const char*>(&CKPT_VERSION), sizeof(CKPT_VERSION));

    // Scalar state
    uint32_t u;
    u = (uint32_t)step;            f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)num_active;      f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)shDegree;        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)adam_step_count;  f.write(reinterpret_cast<const char*>(&u), sizeof(u));

    // Adam learning rates
    f.write(reinterpret_cast<const char*>(adam_lr), sizeof(adam_lr));
    f.write(reinterpret_cast<const char*>(&means_lr_init), sizeof(means_lr_init));
    f.write(reinterpret_cast<const char*>(&means_lr_final), sizeof(means_lr_final));

    // Gaussian parameters (views — only num_active elements)
    writeTensor(f, means);
    writeTensor(f, scales);
    writeTensor(f, quats);
    writeTensor(f, featuresDc);
    writeTensor(f, featuresRest);
    writeTensor(f, opacities);

    // Optimizer state
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg[g]);
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg_sq[g]);

    f.close();
    std::cout << "Checkpoint saved: " << filename << " (step " << step
              << ", " << num_active << " gaussians, "
              << fs::file_size(filename) / (1024*1024) << " MB)" << std::endl;
}

int Model::loadCheckpoint(const std::string &filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file: " + filename);

    // Header
    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != CKPT_MAGIC) throw std::runtime_error("Not a valid msplat checkpoint file");
    if (version != CKPT_VERSION) throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(version));

    // Scalar state
    uint32_t step, numPts, shDeg, adamSteps;
    f.read(reinterpret_cast<char*>(&step), sizeof(step));
    f.read(reinterpret_cast<char*>(&numPts), sizeof(numPts));
    f.read(reinterpret_cast<char*>(&shDeg), sizeof(shDeg));
    f.read(reinterpret_cast<char*>(&adamSteps), sizeof(adamSteps));

    f.read(reinterpret_cast<char*>(adam_lr), sizeof(adam_lr));
    f.read(reinterpret_cast<char*>(&means_lr_init), sizeof(means_lr_init));
    f.read(reinterpret_cast<char*>(&means_lr_final), sizeof(means_lr_final));
    adam_step_count = (int)adamSteps;

    // Gaussian parameters — read into fresh tensors
    means = readTensor(f);
    scales = readTensor(f);
    quats = readTensor(f);
    featuresDc = readTensor(f);
    featuresRest = readTensor(f);
    opacities = readTensor(f);

    // Optimizer state
    for (int g = 0; g < N_ADAM_GROUPS; g++) adam_exp_avg[g] = readTensor(f);
    for (int g = 0; g < N_ADAM_GROUPS; g++) adam_exp_avg_sq[g] = readTensor(f);

    f.close();

    // Rebuild backing buffers with loaded data (don't call setupOptimizers —
    // it would zero the optimizer state we just loaded)
    num_active = (int)numPts;
    buf_capacity = num_active * 4;

    // Copy gaussian params into oversized backing buffers
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);

    // Copy optimizer state into oversized backing buffers
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        auto shape = adam_exp_avg[g].shape();
        shape[0] = buf_capacity;
        MTensor avg_buf = gpu_zeros(shape, DType::Float32);
        MTensor sq_buf = gpu_zeros(shape, DType::Float32);
        memcpy(avg_buf.data_ptr(), adam_exp_avg[g].data_ptr(), adam_exp_avg[g].nbytes());
        memcpy(sq_buf.data_ptr(), adam_exp_avg_sq[g].data_ptr(), adam_exp_avg_sq[g].nbytes());
        adam_exp_avg_buf[g] = avg_buf;
        adam_exp_avg_sq_buf[g] = sq_buf;
    }

    // Allocate densification scratch buffers
    densify_split_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_split_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    int max_blocks = (buf_capacity + 1023) / 1024;
    densify_block_totals = gpu_zeros({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest.numel() / featuresRest.size(0);
    densify_compact_scratch = gpu_zeros({(int64_t)buf_capacity * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({buf_capacity, 3}, DType::Float32);

    refreshViews();

    std::cout << "Checkpoint loaded: " << filename << " (step " << step
              << ", " << num_active << " gaussians)" << std::endl;

    return (int)step;
}

Model::CamSetup Model::prepareCam(Camera& cam, int step) {
    const float sf = getDownscaleFactor(step);
    CamSetup s;
    s.fx = cam.fx / sf; s.fy = cam.fy / sf;
    s.cx = cam.cx / sf; s.cy = cam.cy / sf;
    s.height = static_cast<int>(cam.height / sf);
    s.width = static_cast<int>(cam.width / sf);

    float fovX = 2.0f * std::atan(s.width / (2.0f * s.fx));
    float fovY = 2.0f * std::atan(s.height / (2.0f * s.fy));

    // world→cam（CV）。useCameraOpt 時用逐步精修的 cam.viewCorr（首次以 base 初始化、之後每步重算）；
    // 否則用 base viewmat，依 fov 快取。
    bool recompute = useCameraOpt || !cam.cachedViewMat.defined()
                     || cam.cachedFovX != fovX || cam.cachedFovY != fovY;
    if (recompute) {
        float vm[16];
        if (useCameraOpt) {
            if (!cam.poseInit) {
                const float *d = cam.camToWorld;
                float R[3][3], Rinv[3][3], T[3], Tinv[3];
                for (int i=0;i<3;i++){ R[i][0]=d[i*4+0]; R[i][1]=-d[i*4+1]; R[i][2]=-d[i*4+2]; T[i]=d[i*4+3]; }
                for (int i=0;i<3;i++) for (int j=0;j<3;j++) Rinv[i][j]=R[j][i];
                for (int i=0;i<3;i++) Tinv[i]=-(Rinv[i][0]*T[0]+Rinv[i][1]*T[1]+Rinv[i][2]*T[2]);
                float b[16]={Rinv[0][0],Rinv[0][1],Rinv[0][2],Tinv[0], Rinv[1][0],Rinv[1][1],Rinv[1][2],Tinv[1], Rinv[2][0],Rinv[2][1],Rinv[2][2],Tinv[2], 0,0,0,1};
                std::memcpy(cam.viewCorr, b, sizeof(b));
                cam.poseInit = true;
            }
            std::memcpy(vm, cam.viewCorr, sizeof(vm));
        } else {
            const float *d = cam.camToWorld;
            float R[3][3], Rinv[3][3], T[3], Tinv[3];
            for (int i=0;i<3;i++){ R[i][0]=d[i*4+0]; R[i][1]=-d[i*4+1]; R[i][2]=-d[i*4+2]; T[i]=d[i*4+3]; }
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) Rinv[i][j]=R[j][i];
            for (int i=0;i<3;i++) Tinv[i]=-(Rinv[i][0]*T[0]+Rinv[i][1]*T[1]+Rinv[i][2]*T[2]);
            float b[16]={Rinv[0][0],Rinv[0][1],Rinv[0][2],Tinv[0], Rinv[1][0],Rinv[1][1],Rinv[1][2],Tinv[1], Rinv[2][0],Rinv[2][1],Rinv[2][2],Tinv[2], 0,0,0,1};
            std::memcpy(vm, b, sizeof(b));
        }
        // camPos(world) = -Wᵀ t，W=vm 旋轉、t=vm 平移
        float camPos[3] = {
            -(vm[0]*vm[3] + vm[4]*vm[7] + vm[8]*vm[11]),
            -(vm[1]*vm[3] + vm[5]*vm[7] + vm[9]*vm[11]),
            -(vm[2]*vm[3] + vm[6]*vm[7] + vm[10]*vm[11])
        };
        float t_p = 0.001f * std::tan(0.5f * fovY), r_p = 0.001f * std::tan(0.5f * fovX);
        float pm[16] = { 0.001f/r_p,0,0,0, 0,0.001f/t_p,0,0, 0,0,(1000.0f+0.001f)/(1000.0f-0.001f),-1000.0f*0.001f/(1000.0f-0.001f), 0,0,1,0 };
        float pvm[16] = {};
        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<4;k++) pvm[i*4+j] += pm[i*4+k] * vm[k*4+j];

        if (!cam.cachedViewMat.defined())     cam.cachedViewMat     = gpu_empty({4, 4}, DType::Float32);
        if (!cam.cachedProjViewMat.defined()) cam.cachedProjViewMat = gpu_empty({4, 4}, DType::Float32);
        std::memcpy(cam.cachedViewMat.data_ptr(), vm, sizeof(vm));
        std::memcpy(cam.cachedProjViewMat.data_ptr(), pvm, sizeof(pvm));
        cam.cachedCamPos[0]=camPos[0]; cam.cachedCamPos[1]=camPos[1]; cam.cachedCamPos[2]=camPos[2];
        cam.cachedFovX = fovX; cam.cachedFovY = fovY;
    }

    s.degreesToUse = (std::min<int>)(step / shDegreeInterval, shDegree);
    int b = featuresRest.size(-2) + 1;
    s.degree = (b <= 1) ? 0 : (b <= 4) ? 1 : (b <= 9) ? 2 : (b <= 16) ? 3 : 4;
    s.tileBounds = std::make_tuple(
        (s.width + BLOCK_X - 1) / BLOCK_X,
        (s.height + BLOCK_Y - 1) / BLOCK_Y, 1);
    s.cam_pos[0] = cam.cachedCamPos[0];
    s.cam_pos[1] = cam.cachedCamPos[1];
    s.cam_pos[2] = cam.cachedCamPos[2];

    return s;
}

// 相機姿態優化：由 backward 世界系 v_mean3d 在 CPU 端組裝 SE3 位姿梯度，per-camera Adam 精修。
// 左擾動於 world→cam：corrected = exp(δ̂)·viewCorr。安全網：延遲啟動 + 保守 lr + 單步幅度上限
//（無法在此跑有限差分驗證，故以安全網防梯度號誤造成的發散；如品質退步可用 useCameraOpt 關閉對照）。
void Model::cameraPoseStep(Camera& cam, int step){
    if (!useCameraOpt) return;
    static constexpr int   kPoseStartIter = 800;    // 等幾何/密集化稍穩再開
    static constexpr float kPoseLrRot     = 1e-4f;  // ω 學習率（保守）
    static constexpr float kPoseLrTrans   = 1e-3f;  // τ 學習率
    static constexpr float kPoseMaxStep   = 3e-3f;  // 單步修正幅度上限（防發散）
    static constexpr float kB1=0.9f, kB2=0.999f, kEps=1e-8f;
    if (step < kPoseStartIter) return;

    const float* vmean = msplat_v_mean3d_data();     // 世界系 ∂loss/∂mean（須已 sync）
    if (!vmean || !cam.cachedViewMat.defined()) return;
    const int N = num_active;
    const float* mw = means_buf.data<float>();
    const float* vm = (const float*)cam.cachedViewMat.data_ptr();  // corrected world→cam (row-major)
    const float W[3][3]={{vm[0],vm[1],vm[2]},{vm[4],vm[5],vm[6]},{vm[8],vm[9],vm[10]}};
    const float t[3]={vm[3],vm[7],vm[11]};

    double gw[3]={0,0,0}, gt[3]={0,0,0};             // grad_ω, grad_τ
    for (int i=0;i<N;i++){
        const float* vi=&vmean[i*3];
        if (vi[0]==0.f && vi[1]==0.f && vi[2]==0.f) continue;   // 不可見/未貢獻
        float gc0=W[0][0]*vi[0]+W[0][1]*vi[1]+W[0][2]*vi[2];    // g_cam = W·v_mean3d
        float gc1=W[1][0]*vi[0]+W[1][1]*vi[1]+W[1][2]*vi[2];
        float gc2=W[2][0]*vi[0]+W[2][1]*vi[1]+W[2][2]*vi[2];
        const float* p=&mw[i*3];
        float pc0=W[0][0]*p[0]+W[0][1]*p[1]+W[0][2]*p[2]+t[0];  // p_cam = W·p_world + t
        float pc1=W[1][0]*p[0]+W[1][1]*p[1]+W[1][2]*p[2]+t[1];
        float pc2=W[2][0]*p[0]+W[2][1]*p[1]+W[2][2]*p[2]+t[2];
        gt[0]+=gc0; gt[1]+=gc1; gt[2]+=gc2;                     // grad_τ += g_cam
        gw[0]+=pc1*gc2-pc2*gc1;                                 // grad_ω += p_cam × g_cam
        gw[1]+=pc2*gc0-pc0*gc2;
        gw[2]+=pc0*gc1-pc1*gc0;
    }

    // Adam（6-vec：ω,τ）。adam 對梯度尺度不變 → 步長由 lr 控制，與高斯數無關。
    float g[6]={(float)gw[0],(float)gw[1],(float)gw[2],(float)gt[0],(float)gt[1],(float)gt[2]};
    cam.poseAdamStep++;
    float bc1=1.f-std::pow(kB1,(float)cam.poseAdamStep), bc2=1.f-std::pow(kB2,(float)cam.poseAdamStep);
    float delta[6];
    for(int j=0;j<6;j++){
        cam.poseAdamM[j]=kB1*cam.poseAdamM[j]+(1-kB1)*g[j];
        cam.poseAdamV[j]=kB2*cam.poseAdamV[j]+(1-kB2)*g[j]*g[j];
        float mhat=cam.poseAdamM[j]/bc1, vhat=cam.poseAdamV[j]/bc2;
        float lr=(j<3)?kPoseLrRot:kPoseLrTrans;
        float d=-lr*mhat/(std::sqrt(vhat)+kEps);
        delta[j]=std::min(std::max(d,-kPoseMaxStep),kPoseMaxStep);   // 幅度上限（安全網）
    }

    // 左乘 exp(δ̂)：R'=dR·W, t'=dR·t+dτ。SO3 exp 用 Rodrigues。
    float wx=delta[0],wy=delta[1],wz=delta[2];
    float th=std::sqrt(wx*wx+wy*wy+wz*wz);
    float dR[3][3];
    if (th<1e-8f){
        dR[0][0]=1;dR[0][1]=-wz;dR[0][2]=wy; dR[1][0]=wz;dR[1][1]=1;dR[1][2]=-wx; dR[2][0]=-wy;dR[2][1]=wx;dR[2][2]=1;
    } else {
        float a=std::sin(th)/th, bb=(1-std::cos(th))/(th*th);
        float K[3][3]={{0,-wz,wy},{wz,0,-wx},{-wy,wx,0}}, K2[3][3];
        for(int r=0;r<3;r++)for(int c=0;c<3;c++) K2[r][c]=K[r][0]*K[0][c]+K[r][1]*K[1][c]+K[r][2]*K[2][c];
        for(int r=0;r<3;r++)for(int c=0;c<3;c++) dR[r][c]=(r==c?1.f:0.f)+a*K[r][c]+bb*K2[r][c];
    }
    float nW[3][3], nt[3];
    for(int r=0;r<3;r++){
        for(int c=0;c<3;c++) nW[r][c]=dR[r][0]*W[0][c]+dR[r][1]*W[1][c]+dR[r][2]*W[2][c];
        nt[r]=dR[r][0]*t[0]+dR[r][1]*t[1]+dR[r][2]*t[2]+delta[3+r];
    }
    // 對 nW 做 Gram-Schmidt 再正交化，避免長時間累積漂移
    auto n3=[](float* v){ float n=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); if(n>1e-12f){v[0]/=n;v[1]/=n;v[2]/=n;} };
    float r0[3]={nW[0][0],nW[0][1],nW[0][2]}; n3(r0);
    float r1[3]={nW[1][0],nW[1][1],nW[1][2]};
    float d01=r0[0]*r1[0]+r0[1]*r1[1]+r0[2]*r1[2];
    r1[0]-=d01*r0[0]; r1[1]-=d01*r0[1]; r1[2]-=d01*r0[2]; n3(r1);
    float r2[3]={r0[1]*r1[2]-r0[2]*r1[1], r0[2]*r1[0]-r0[0]*r1[2], r0[0]*r1[1]-r0[1]*r1[0]};

    float* vc=cam.viewCorr;
    vc[0]=r0[0];vc[1]=r0[1];vc[2]=r0[2];vc[3]=nt[0];
    vc[4]=r1[0];vc[5]=r1[1];vc[6]=r1[2];vc[7]=nt[1];
    vc[8]=r2[0];vc[9]=r2[1];vc[10]=r2[2];vc[11]=nt[2];
    vc[12]=0;vc[13]=0;vc[14]=0;vc[15]=1;
}

// 外觀校正：affine init 為 identity（M=I, t=0）→ 起始不改變影像，之後 Adam 精修。
void Model::appearanceEnsureInit(Camera& cam){
    if (cam.appInit) return;
    for (int i = 0; i < 12; i++) { cam.appAffine[i] = 0.f; cam.appAdamM[i] = 0.f; cam.appAdamV[i] = 0.f; }
    cam.appAffine[0] = cam.appAffine[4] = cam.appAffine[8] = 1.f;  // M = I
    cam.appAdamStep = 0;
    cam.appInit = true;
}

// 讀 GPU 累加的 affine 梯度（12，須 caller 已 sync）→ per-camera Adam；加 L2 拉回 identity 防退化。
void Model::appearanceStep(Camera& cam, int step){
    if (!useAppearance) return;
    const float* grad = msplat_appearance_grad_data();
    if (!grad) return;
    static constexpr float kLr = 1e-3f, kB1 = 0.9f, kB2 = 0.999f, kEps = 1e-8f, kL2 = 1e-2f;
    static const float identity[12] = {1,0,0, 0,1,0, 0,0,1, 0,0,0};
    cam.appAdamStep++;
    float bc1 = 1.f - std::pow(kB1, (float)cam.appAdamStep);
    float bc2 = 1.f - std::pow(kB2, (float)cam.appAdamStep);
    for (int j = 0; j < 12; j++) {
        float g = grad[j] + kL2 * (cam.appAffine[j] - identity[j]);   // L2 → identity（避免外觀吃掉幾何）
        cam.appAdamM[j] = kB1 * cam.appAdamM[j] + (1 - kB1) * g;
        cam.appAdamV[j] = kB2 * cam.appAdamV[j] + (1 - kB2) * g * g;
        float mh = cam.appAdamM[j] / bc1, vh = cam.appAdamV[j] / bc2;
        cam.appAffine[j] -= kLr * mh / (std::sqrt(vh) + kEps);
    }
}

// LiDAR 深度監督（per-gaussian）：把近 LiDAR 表面的高斯沿光學軸拉到度量深度真值。
// 使用修正後 viewmat（cam.cachedViewMat）；band 閘門確保只精修近表面高斯（安全）。
// 誤差圖引導的累加（MRNF use_error_map 的等價實作）。
// LichtFeld 的做法是 Canny → edge_rasterize（alpha 加權 splat 回高斯），需要一支新的光柵器 pass。
// 這裡改成「投影取樣」：把高斯中心投到影像、直接讀 1/8 細節圖。差別是少了 alpha 加權，
// 但這個量只當作 1.0~1.25 的權重乘數（不是損失項），點取樣的誤差不影響選點決策。
// 好處：零新 Metal kernel、天然用到訓練當下的相機（等於 LichtFeld 的隨機取視角）。
void Model::mrnfAccumEdge(Camera& cam, int step){
    if (!useEdgeGuidance) return;
    // 攤提：一個 refine 視窗(100 步)累積 ~12 個視角就足夠代表性，不必每步都掃 O(N)。
    if (step % (kMcmcCpuEvery * 2) != 0) return;
    if (!cam.edgeTried) cam.buildEdgeMap();        // lazy 建圖（每台相機只算一次）
    if (cam.edgeW == 0 || !cam.cachedViewMat.defined()) return;

    const int N = num_active;
    if ((int)mrnfEdgeScore.size() < N){
        mrnfEdgeScore.assign((size_t)buf_capacity, 0.0f);
        mrnfEdgeCount.assign((size_t)buf_capacity, 0.0f);
    }
    const float* vm = (const float*)cam.cachedViewMat.data_ptr();  // world→cam (row-major)
    const float W0=vm[0],W1=vm[1],W2=vm[2], W4=vm[4],W5=vm[5],W6=vm[6], W8=vm[8],W9=vm[9],W10=vm[10];
    const float t0=vm[3],t1=vm[7],t2=vm[11];
    const float fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy;
    const int W=cam.width, H=cam.height;
    if (W<=0 || H<=0) return;
    const float* mean = means_buf.data<float>();
    for (int i=0;i<N;i++){
        const float x=mean[i*3], y=mean[i*3+1], z=mean[i*3+2];
        const float vz = W8*x + W9*y + W10*z + t2;
        if (vz <= 0.05f) continue;
        const float px = fx*((W0*x + W1*y + W2*z + t0)/vz) + cx;
        const float py = fy*((W4*x + W5*y + W6*z + t1)/vz) + cy;
        if (px<0 || py<0 || px>=W || py>=H) continue;
        // cam.fx/cx 是對「常駐(已裁上限)解析度 cam.width/height」校正過的 → 直接 /8 對到細節圖，
        // 與訓練當下的 coarse-to-fine ds 無關。
        const int eu = std::min((int)(px) / 8, cam.edgeW - 1);
        const int ev = std::min((int)(py) / 8, cam.edgeH - 1);
        mrnfEdgeScore[(size_t)i] += (float)cam.edgeMap[(size_t)ev * cam.edgeW + eu];
        mrnfEdgeCount[(size_t)i] += 1.0f;
    }
}

void Model::depthRefineStep(Camera& cam, int step){
    if (!useDepthSupervision) return;
    // 攤提（速度）：這是往 LiDAR 表面的收縮映射（每次施加把誤差砍 kLr），不是梯度累積 —— 少施加幾次
    // 只影響收斂步數，不改收斂點。6000 步 ÷ 4 仍有 ~1400 次施加，遠超收斂所需。
    if (step % kMcmcCpuEvery != 0) return;
    if (!cam.lidarTried) cam.loadLidarDepth();     // lazy 載入
    if (cam.lidarW == 0 || !cam.cachedViewMat.defined()) return;   // 無深度 / 無 viewmat
    static constexpr float kBand = 0.10f;          // 只精修 |view_z - d_lidar| < 10cm 的高斯
    static constexpr float kLr   = 0.5f;           // 每次校正誤差的比例
    static constexpr float kMaxD = 8.0f;           // LiDAR 有效上限

    const float* vm = (const float*)cam.cachedViewMat.data_ptr();  // world→cam (row-major)
    const float W0=vm[0],W1=vm[1],W2=vm[2], W4=vm[4],W5=vm[5],W6=vm[6], W8=vm[8],W9=vm[9],W10=vm[10];
    const float t0=vm[3],t1=vm[7],t2=vm[11];
    const float fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy;
    const int   W=cam.width, H=cam.height;
    if (W<=0 || H<=0) return;
    float* mean = means_buf.data<float>();
    const int N = num_active;
    for (int i=0;i<N;i++){
        float x=mean[i*3], y=mean[i*3+1], z=mean[i*3+2];
        float vz = W8*x + W9*y + W10*z + t2;                 // view-space depth
        if (vz <= 0.05f) continue;
        float vx = W0*x + W1*y + W2*z + t0;
        float vy = W4*x + W5*y + W6*z + t1;
        float px = fx*(vx/vz) + cx, py = fy*(vy/vz) + cy;    // 訓練影像像素座標
        if (px<0 || py<0 || px>=W || py>=H) continue;
        int lu = (int)(px / (float)W * cam.lidarW);          // → LiDAR 像素（FOV 不變的正規化對應）
        int lv = (int)(py / (float)H * cam.lidarH);
        if (lu<0 || lv<0 || lu>=cam.lidarW || lv>=cam.lidarH) continue;
        int li = lv*cam.lidarW + lu;
        if (!cam.lidarConf.empty() && cam.lidarConf[li] < 2) continue;   // 只用高信心
        float d = cam.lidarDepth[li];
        if (d <= 0.05f || d > kMaxD) continue;
        float err = d - vz;
        if (fabsf(err) > kBand) continue;                    // 只精修近表面高斯（安全閘門）
        float dz = kLr * err;                                 // 沿光學軸移動：Δp = dz·(W row2)
        mean[i*3+0] += dz*W8; mean[i*3+1] += dz*W9; mean[i*3+2] += dz*W10;
    }
}

MTensor Model::render(Camera& cam, int step){
    auto s = prepareCam(cam, step);
    return msplat_render(
        means.size(0), means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor);
}

void Model::fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight){
    auto s = prepareCam(cam, step);
    lastHeight = s.height; lastWidth = s.width;
    int numPoints = means.size(0);

    // Initialize SSIM window (once)
    if (!window2d.defined()) {
        auto w = createSSIMWindow(11, 1.5f);
        window2d = gpu_empty({11, 11}, DType::Float32);
        memcpy(window2d.data_ptr(), w.data(), w.size() * sizeof(float));
    }

    adam_step_count++;
    float bc1 = 1.0f - std::pow(adam_beta1, adam_step_count);
    float bc2 = 1.0f - std::pow(adam_beta2, adam_step_count);
    MTensor adam_p[N_ADAM_GROUPS];
    MTensor adam_ea[N_ADAM_GROUPS], adam_eas[N_ADAM_GROUPS];
    float adam_ss[N_ADAM_GROUPS], adam_bc2s[N_ADAM_GROUPS];
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int i = 0; i < N_ADAM_GROUPS; ++i) {
        adam_p[i] = *params[i];
        adam_ea[i] = adam_exp_avg[i];
        adam_eas[i] = adam_exp_avg_sq[i];
        adam_ss[i] = adam_lr[i] / bc1;
        adam_bc2s[i] = std::sqrt(bc2);
    }

    if (!xysGradNorm.defined()) {
    
        xysGradNorm = gpu_zeros({numPoints}, DType::Float32);
        visCounts = gpu_zeros({numPoints}, DType::Float32);
        max2DSize = gpu_zeros({numPoints}, DType::Float32);
    }

    float invMaxDim = 1.0f / static_cast<float>((std::max)(lastHeight, lastWidth));
    float lossInvN = 1.0f / (float)(s.height * s.width * 3);

    auto [r, loss] = msplat_train_step(
        numPoints, means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, backgroundColor, gt, window2d, ssimWeight,
        lossInvN, (int)featuresRest.size(-2),
        N_ADAM_GROUPS,
        adam_p, adam_ea, adam_eas,
        adam_ss, adam_bc2s,
        adam_beta1, adam_beta2, adam_eps,
        visCounts, xysGradNorm, max2DSize, invMaxDim);

    radii = r;
}
