#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include <cstdint>

using namespace AscendC;

namespace {
// 32B 对齐（AscendC 向量/搬运常用对齐）
__aicore__ inline uint32_t AlignUpBytes32(uint32_t bytes)
{
    constexpr uint32_t kAlign = 32U;
    if (bytes == 0U) {
        return kAlign;
    }
    return ((bytes + kAlign - 1U) / kAlign) * kAlign;
}

__aicore__ inline uint32_t AlignUpU32(uint32_t v, uint32_t align)
{
    if (align == 0U) {
        return v;
    }
    return ((v + align - 1U) / align) * align;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (b == 0U) ? 0U : ((a + b - 1U) / b);
}

__aicore__ inline uint32_t GcdU32(uint32_t a, uint32_t b)
{
    while (b != 0U) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

template <typename T>
__aicore__ inline uint32_t VecWidth()
{
    constexpr uint32_t bytes = static_cast<uint32_t>(sizeof(T));
    return bytes == 0U ? 1U : (32U / bytes);
}

// 2D 搬运封装：用 DataCopyPad(DataCopyExtParams) 一次搬 N 行，显著降低 per-row 标量开销。
// 说明：
// - GM->UB：srcStride 用字节；dstStride 用 32B block 数（UB 访问粒度）。
// - UB->GM：srcStride 用 32B block 数；dstStride 用字节。
// - 为保证 32B block 对齐，dstDim0/srcDim0 必须是 8 的倍数（float32 下 8*4B=32B）。
template <typename T>
__aicore__ inline void DataCopyPad2DGm2UbFixedStride(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                                    uint32_t dim1, uint32_t dim0, uint32_t srcDim0,
                                                    uint32_t dstDim0)
{
    if (dim1 == 0U || dim0 == 0U) {
        return;
    }
    constexpr uint32_t kBlockBytes = 32U;
    constexpr uint64_t kAddrAlignBytes = 64ULL;
    const bool full_tile = (dim0 == dstDim0);
    const uint32_t bytes = dim0 * static_cast<uint32_t>(sizeof(T));
    const uint32_t elem_per_block = kBlockBytes / static_cast<uint32_t>(sizeof(T));
    const bool aligned =
        (bytes % kBlockBytes == 0U) && ((srcDim0 % elem_per_block) == 0U) && ((dstDim0 % elem_per_block) == 0U);
    if (full_tile && aligned) {
        const uint64_t src_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src.GetPhyAddr()));
        const uint64_t dst_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.GetPhyAddr()));
        if (((src_addr & (kAddrAlignBytes - 1ULL)) == 0ULL) && ((dst_addr & (kAddrAlignBytes - 1ULL)) == 0ULL)) {
            const uint32_t block_len = bytes / kBlockBytes;
            const uint32_t src_stride = ((srcDim0 - dim0) * static_cast<uint32_t>(sizeof(T))) / kBlockBytes;
            const uint32_t dst_stride = ((dstDim0 - dim0) * static_cast<uint32_t>(sizeof(T))) / kBlockBytes;
            DataCopyParams params(static_cast<uint16_t>(dim1), static_cast<uint16_t>(block_len),
                                  static_cast<uint16_t>(src_stride), static_cast<uint16_t>(dst_stride));
            DataCopy(dst, src, params);
            return;
        }
    }
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(dim1);
    params.blockLen = bytes;
    // GM 侧 stride：字节
    params.srcStride = (srcDim0 - dim0) * static_cast<uint32_t>(sizeof(T));

    // UB 侧 stride：32B block 数。每行在 UB 内固定占 dstDim0 个元素。
    const uint32_t aligned_bytes = AlignUpBytes32(bytes);
    const uint32_t row_blocks = aligned_bytes / 32U;
    const uint32_t dst_row_bytes = dstDim0 * static_cast<uint32_t>(sizeof(T));
    const uint32_t dst_blocks = dst_row_bytes / 32U;
    params.dstStride = (dst_blocks > row_blocks) ? (dst_blocks - row_blocks) : 0U;

    const uint8_t rpad = static_cast<uint8_t>((aligned_bytes - bytes) / static_cast<uint32_t>(sizeof(T)));
    DataCopyPadExtParams<T> padParams{true, 0, rpad, static_cast<T>(0)};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2DUb2GmFixedStride(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                                    uint32_t dim1, uint32_t dim0, uint32_t srcDim0,
                                                    uint32_t dstDim0)
{
    if (dim1 == 0U || dim0 == 0U) {
        return;
    }
    constexpr uint32_t kBlockBytes = 32U;
    constexpr uint64_t kAddrAlignBytes = 64ULL;
    const bool full_tile = (dim0 == srcDim0);
    const uint32_t bytes = dim0 * static_cast<uint32_t>(sizeof(T));
    const uint32_t elem_per_block = kBlockBytes / static_cast<uint32_t>(sizeof(T));
    const bool aligned =
        (bytes % kBlockBytes == 0U) && ((srcDim0 % elem_per_block) == 0U) && ((dstDim0 % elem_per_block) == 0U);
    if (full_tile && aligned) {
        const uint64_t src_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src.GetPhyAddr()));
        const uint64_t dst_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst.GetPhyAddr()));
        if (((src_addr & (kAddrAlignBytes - 1ULL)) == 0ULL) && ((dst_addr & (kAddrAlignBytes - 1ULL)) == 0ULL)) {
            const uint32_t block_len = bytes / kBlockBytes;
            const uint32_t src_stride = ((srcDim0 - dim0) * static_cast<uint32_t>(sizeof(T))) / kBlockBytes;
            const uint32_t dst_stride = ((dstDim0 - dim0) * static_cast<uint32_t>(sizeof(T))) / kBlockBytes;
            DataCopyParams params(static_cast<uint16_t>(dim1), static_cast<uint16_t>(block_len),
                                  static_cast<uint16_t>(src_stride), static_cast<uint16_t>(dst_stride));
            DataCopy(dst, src, params);
            return;
        }
    }
    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(dim1);
    params.blockLen = bytes;

    // UB 侧 stride：32B block 数。注意 UB 搬运以 32B 为粒度，实际读取按 blockLen 向上取整。
    const uint32_t aligned_bytes = AlignUpBytes32(bytes);
    const uint32_t src_row_bytes = srcDim0 * static_cast<uint32_t>(sizeof(T));
    const uint32_t src_stride_blocks = (src_row_bytes > aligned_bytes) ? ((src_row_bytes - aligned_bytes) / 32U) : 0U;
    params.srcStride = src_stride_blocks;

    // GM 侧 stride：字节
    params.dstStride = (dstDim0 - dim0) * static_cast<uint32_t>(sizeof(T));
    DataCopyPad(dst, src, params);
}

// float32: 32B / 4B = 8
constexpr uint32_t kVecWidthF32 = 8U;
// CompareScalar 对 float32 的 calCount 有 256B 对齐约束：64 * 4B = 256B
constexpr uint32_t kCompareAlignF32 = 64U;
constexpr uint32_t kTileColsDefault = 64U;
// 小 N 场景允许放大 tile_cols 到 256（仍满足 64 对齐）
constexpr uint32_t kTileColsMax = 256U;
// p=3 小 N：批量处理的行数（用于降低 Sub/Abs/Mul 指令数）
constexpr uint32_t kP3VecBlockRows = 32U;
// row_block 的理论上限：row_bytes=m*4，gcd(row_bytes,32) 最小为 1 => row_block=32
constexpr uint32_t kMaxRowBlock = 32U;
// Sign 共享临时缓冲（bytes）：避免每次 Sign 调用 PopStackBuffer 带来的额外标量开销
constexpr uint32_t kSignTmpBytes = 2048U;
// PairReduce(workspace) 计算强度阈值（pair_total * tile_cols）
constexpr uint64_t kLargePairsWork = 2ULL * 1024ULL * 1024ULL;
// PairReduce(workspace) pair 总量阈值：避免中大 N 场景过度收缩并行度
constexpr uint32_t kLargePairsPairMin = 24000U;
// p=2 小 M 原子路径的最低计算强度阈值（pair_total * m）
constexpr uint64_t kP2AtomicMinWork = 4ULL * 1024ULL * 1024ULL;
// p=1/p=inf RowParallel 触发阈值（pair_total * m）与 M 上限：小工作量优先避开原子
constexpr uint64_t kP1PinfRowParallelWork = 4ULL * 1024ULL * 1024ULL;
constexpr uint32_t kP1PinfRowParallelMaxM = 256U;
// p=1/p=inf：在大计算量场景尝试启用全 AIV 并行（使用 2x AIV task）
constexpr uint64_t kP1PinfFullAivMinPairsPerCore = 512ULL;
constexpr uint32_t kP1PinfFullAivMinM = 128U;
// p=1/p=inf：非对齐尾块时，允许原子前缀的最小阈值（避免小形状被原子开销拖慢）
constexpr uint32_t kP1PinfAtomicTailMinPairs = 1024U;
constexpr uint32_t kP1PinfAtomicTailMinCols = 64U;
// PairReduce(workspace) 小 pair 总量下的有效并行上限（降低 partial 数量）
// kernel 类型：MIX_AIC_1_2（每个 block group: 1个AIC + 2个AIV）
constexpr uint32_t kAivPerBlock = 2U;
// user workspace layout（matmul 分支）：
// - [0, 512B)：header（预留，保证后续 Laplace 起点满足 512B 对齐）
// - [512B, 512B + N*N*4B)：Laplace(N,N)（FP32，仅 1 份，全 AIC 复用）
constexpr uint32_t kUserWsHeaderBytes = 512U;
// AIV->AIC 组内同步 flag（参考官方 quant_group_matmul / baremix 示例）
constexpr uint32_t kLaplaceReadyFlagId = 3U;
// Laplace 构造：按行分块在 UB 暂存后 2D 写回 GM，降低同步/搬运开销
// 说明：
// - Matmul 快路径下 AIV 数量通常大于 N/16，因此把行 tile 调小到 4 行，让更多 AIV 参与构造，降低 wall time；
// - N 不强制 16 对齐：Matmul 支持 tail，行写回使用 DataCopyPad 保证正确性。
constexpr uint32_t kLaplaceRowTile = 4U;
// Laplace 构造：grad/pdist 先批量搬到 UB 再计算 w=grad/dist
// - 需满足 CompareScalar(float32) 的 calCount 对齐约束（256B=64*4B），因此取 2048（=32*64）
// - 兼顾减少 DataCopy 次数与 UB 占用
constexpr uint32_t kLaplaceWChunk = 2048U;
// RowParallel/基线 pair 循环：对 j>i 段做 GM->UB 预取的最小长度，避免小段频繁 MTE2
constexpr uint32_t kRowPrefetchMin = 16U;

// (i, j) -> k 的前缀起点：k = i*(2n-i-1)/2 + (j-i-1)
__aicore__ inline uint32_t PairStart(uint32_t i, uint32_t n)
{
    // n<=256，返回值 <= 32640，u32 安全
    return (i * (2U * n - i - 1U)) / 2U;
}

// GM->UB：优先走 DataCopy / DataCopyPad；尾部跨行风险时退化为标量读。
// gm_total 为“允许访问的绝对末尾 offset(元素数)”，用于限制不要跨过每行末尾。
template <typename T>
__aicore__ inline void LoadGmToUb(GlobalTensor<T> &gm, uint64_t gm_total, uint64_t gm_offset, LocalTensor<T> &ub,
                                 uint32_t len)
{
    if (len == 0U || gm_total == 0ULL) {
        return;
    }
    if (gm_offset >= gm_total) {
        for (uint32_t i = 0; i < len; ++i) {
            ub.SetValue(i, static_cast<T>(0));
        }
        return;
    }
    const uint64_t remain = gm_total - gm_offset;
    const uint32_t safe_len = (remain < static_cast<uint64_t>(len)) ? static_cast<uint32_t>(remain) : len;
    if (safe_len == 0U) {
        for (uint32_t i = 0; i < len; ++i) {
            ub.SetValue(i, static_cast<T>(0));
        }
        return;
    }

    // 注意：仅检查 gm_offset 是否对齐是不够的（base 指针可能本身非 32B 对齐，例如 NPU view/切片导致 storage offset）。
    // DataCopy 对 GM 起始地址和长度都有严格对齐要求，不满足时可能出现“静默写错/读错”。这里改为检查“真实地址”对齐。
    const uint64_t gm_base_u64 = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(gm.GetPhyAddr()));
    const uint64_t gm_addr_u64 = gm_base_u64 + gm_offset * static_cast<uint64_t>(sizeof(T));
    const uint64_t ub_addr_u64 = static_cast<uint64_t>(ub.GetPhyAddr());
    // 说明：DataCopy 的对齐约束比“32B字节对齐”更严格：实测在部分地址模式下会出现“静默读错”（首个 32B block 丢失/为0）。
    // 将对齐约束提升到 64B（地址与长度同时满足），不满足则退化为标量读，优先保证正确性。
    constexpr uint64_t kDataCopyAlignBytes = 64ULL;
    const uint64_t copy_bytes = static_cast<uint64_t>(safe_len) * static_cast<uint64_t>(sizeof(T));
    const bool ub_aligned = (ub_addr_u64 & (kDataCopyAlignBytes - 1ULL)) == 0ULL;
    const bool gm_aligned = (gm_addr_u64 & (kDataCopyAlignBytes - 1ULL)) == 0ULL;
    const bool len_aligned = (copy_bytes % kDataCopyAlignBytes) == 0ULL;

    if (gm_aligned && ub_aligned && len_aligned) {
        DataCopy(ub, gm[gm_offset], safe_len);
        for (uint32_t i = safe_len; i < len; ++i) {
            ub.SetValue(i, static_cast<T>(0));
        }
        return;
    }
    // 非对齐场景：优先使用 DataCopyPad(GM->UB)，避免逐元素 GetValue 的标量访存开销。
    // 注意：DataCopyPad 可能会对齐下探/上探触及相邻地址；这里依赖调用者传入 gm_total 做边界裁剪，
    // 并在 safe_len < len 时显式将 UB 尾部补 0，避免 padding 元素参与后续计算。
    const uint32_t bytes = safe_len * static_cast<uint32_t>(sizeof(T));
    const uint32_t aligned_bytes = AlignUpBytes32(bytes);
    const uint8_t rpad =
        static_cast<uint8_t>((aligned_bytes - bytes) / static_cast<uint32_t>(sizeof(T))); // pad元素个数
    DataCopyExtParams copyParams = {1, bytes, 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {true, 0, rpad, 0};
    DataCopyPad(ub, gm[gm_offset], copyParams, padParams);
    for (uint32_t i = safe_len; i < len; ++i) {
        ub.SetValue(i, static_cast<T>(0));
    }
}

} // namespace

class PdistGradKernel {
public:
    __aicore__ inline PdistGradKernel() = default;

		    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR input, GM_ADDR pdist, GM_ADDR out, GM_ADDR workspace,
		                                const PdistGradTilingData &tiling_data)
		    {
        n_ = tiling_data.n;
        m_ = tiling_data.m;
        p_type_ = tiling_data.p_type;
        block_dim_ = tiling_data.block_dim;
        sys_ws_size_ = tiling_data.sys_ws_size;
        p_ = tiling_data.p_value;
        workspace_ = workspace;
        // 仅 p in (0,1) 且非 0.5 需要高精度 log2/exp2 近似
        need_log_work_ = ((p_type_ == 5U) && (p_ < 1.0f)) ? 1U : 0U;

        // 全局状态兜底复位：避免上一 kernel 遗留的 mask 状态干扰当前计算
        SetMaskNorm();
        ResetMask();

        // tile_cols：host 侧下发，kernel 侧二次校验（必须 8 对齐且 <=256）
        tile_cols_ = tiling_data.tile_cols;
        if (tile_cols_ == 0U || tile_cols_ > kTileColsMax || ((tile_cols_ % kVecWidthF32) != 0U)) {
            tile_cols_ = kTileColsDefault;
        }
        if ((p_type_ == 3U) && ((tile_cols_ % 64U) != 0U)) {
            tile_cols_ = kTileColsDefault;
        }
        if ((p_type_ != 1U) && (p_type_ != 2U) && (p_type_ != 3U)) {
            tile_cols_ = kTileColsDefault;
        }
        total_tiles_ = (m_ == 0U) ? 0U : ((m_ + tile_cols_ - 1U) / tile_cols_);

	        // matmul 快路径二次保护：
	        // - 避免 tiling 数据异常/未初始化导致小 case 误入 matmul 分支（会访问 workspace，引发 507057 等严重错误）
        use_matmul_ = (tiling_data.use_matmul != 0U) && (p_type_ == 2U) && (n_ >= 64U) && (m_ >= 64U) &&
                      (workspace_ != 0) && (sys_ws_size_ != 0U);
        // PairReduce/RowParallel 快路径二次保护：
        // - 仅 baseline(AIV) 分支可用；
        // - p=1/p=inf 允许对齐 workspace（避免 RowParallel 双倍计算）。
        use_pair_reduce_ =
            (tiling_data.use_pair_reduce != 0U) && (use_matmul_ == 0U) && (total_tiles_ != 0U) && (n_ >= 2U);
        p1pinf_ws_bytes_ = tiling_data.p1pinf_ws_bytes;
        m_padded_ = m_;
        use_p1pinf_ws_ = 0U;
        if ((use_pair_reduce_ != 0U) && ((p_type_ == 1U) || (p_type_ == 3U)) &&
            ((m_ % kVecWidthF32) != 0U) && (workspace_ != 0) && (sys_ws_size_ != 0U) &&
            (p1pinf_ws_bytes_ != 0U)) {
            const uint32_t align = (p_type_ == 3U) ? kCompareAlignF32 : kVecWidthF32;
            m_padded_ = AlignUpU32(m_, align);
            const uint64_t need_bytes =
                static_cast<uint64_t>(kUserWsHeaderBytes) +
                static_cast<uint64_t>(n_) * static_cast<uint64_t>(m_padded_) * sizeof(float);
            if (need_bytes <= static_cast<uint64_t>(p1pinf_ws_bytes_)) {
                use_p1pinf_ws_ = 1U;
            } else {
                m_padded_ = m_;
            }
        }

        // p=3 小 N：启用批量矢量处理（仅 baseline 路径，避免影响 RowParallel/PairReduce）
        const bool enable_p3_vecblk =
            (use_pair_reduce_ == 0U) && (p_type_ == 4U) && (p_ == 3.0f) && (n_ <= 128U) && (tile_cols_ == 64U);
        use_p3_vecblk_ = enable_p3_vecblk ? 1U : 0U;

        input_total_ = static_cast<uint64_t>(n_) * static_cast<uint64_t>(m_);
        pair_total_ = static_cast<uint64_t>(n_) * static_cast<uint64_t>(n_ > 0U ? (n_ - 1U) : 0U) / 2ULL;

        grad_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(grad), pair_total_);
        pdist_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pdist), pair_total_);
        input_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input), input_total_);
        out_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out), input_total_);
        // 大量顺序读：关闭 L2 缓存，避免无效缓存污染（对小 shape 保持默认）
        if (pair_total_ >= 8192ULL) {
            grad_gm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            pdist_gm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

	        if (use_matmul_ != 0U) {
	            // Cube Matmul 快路径：p==2 的大 shape 走 Laplace 矩阵 + Matmul，显著降低 case5。
	            // 重要：Laplace 构造放在 AIV（矢量侧）执行，再用 CrossCore flag 通知 AIC 开始 Matmul。
	            // 原因：AIC 侧不保证支持 GlobalTensor::GetValue 等矢量/标量访存接口，容易出现“读到0 -> 输出全0”。
		            if ASCEND_IS_AIC {
		                cube_tiling_ = tiling_data.cubeTilingData;
		            } else {
		                // AIV 侧任务索引（matmul 分支也需要，用于并行构造 Laplace 的行 tile）
		                aiv_block_idx_ = static_cast<uint32_t>(GetBlockIdx());
		                aiv_block_dim_ = block_dim_ * kAivPerBlock;
		                if (aiv_block_dim_ == 0U) {
		                    aiv_block_dim_ = 1U;
		                }
		                // AIV 侧负责：构造 Laplace（FP32），并写入 user workspace。
		                // matmul 分支：为 Laplace 分配 UB tile buffer（按行分块构造 + 2D 写回），避免每行强同步带来的性能劣化。
		                const uint32_t tile_rows = MinU32(kLaplaceRowTile, n_);
		                pipe_.InitBuffer(buf_a_tile_,
		                                 AlignUpBytes32(tile_rows * n_ * static_cast<uint32_t>(sizeof(float))));
		                a_tile_f32_ = buf_a_tile_.Get<float>();
		                // matmul 分支：每个 AIV core 都需要构造自己负责的 Laplace 行 tile，因此每个 core 都申请 w/dist/mask 的 UB buffer。
		                const uint32_t pair_total_u32 = static_cast<uint32_t>(pair_total_);
		                const uint32_t w_elems = AlignUpU32(pair_total_u32, kLaplaceWChunk);
		                pipe_.InitBuffer(buf_lap_w_, AlignUpBytes32(w_elems * static_cast<uint32_t>(sizeof(float))));
		                pipe_.InitBuffer(buf_lap_d_, AlignUpBytes32(kLaplaceWChunk * static_cast<uint32_t>(sizeof(float))));
		                pipe_.InitBuffer(buf_lap_mask_,
		                                 AlignUpBytes32(kLaplaceWChunk * static_cast<uint32_t>(sizeof(uint8_t))));
		                lap_w_ = buf_lap_w_.Get<float>();
		                lap_d_ = buf_lap_d_.Get<float>();
		                lap_mask_ = buf_lap_mask_.Get<uint8_t>();
		            }
		            // matmul 分支不再使用按列 tile 的输入/输出大 buffer，避免占用 UB 影响 Matmul 性能/稳定性
		            start_tile_ = total_tiles_;
		            tile_count_ = 0U;
		            return;
	        }

        // baseline 分支仅在 AIV 执行（AIC 直接返回）
        if ASCEND_IS_AIC {
            start_tile_ = total_tiles_;
            tile_count_ = 0U;
            return;
        }

        // AIV 侧任务索引（MIX_AIC_1_2）：
        // - 仍以 GetBlockIdx 作为 task id（AIV 上通常为 0..mix_block_dim-1）；
        // - 物理并行度按 block_dim * kAivPerBlock 计算。
        aiv_block_idx_ = static_cast<uint32_t>(GetBlockIdx());
        aiv_block_dim_ = block_dim_ * kAivPerBlock;
        if (aiv_block_dim_ == 0U) {
            aiv_block_dim_ = 1U;
        }
        // 默认：实际参与计算的 AIV 并行度等于物理 AIV 并行度
        aiv_effective_dim_ = aiv_block_dim_;

        if (total_tiles_ == 0U || aiv_block_idx_ >= aiv_block_dim_) {
            start_tile_ = total_tiles_;
            tile_count_ = 0U;
            return;
        }
        const uint32_t base = total_tiles_ / aiv_block_dim_;
        const uint32_t rem = total_tiles_ - base * aiv_block_dim_;
        tile_count_ = base + (aiv_block_idx_ < rem ? 1U : 0U);
        start_tile_ = base * aiv_block_idx_ + (aiv_block_idx_ < rem ? aiv_block_idx_ : rem);

        // need_sync 场景下（M 非 8 对齐 && 多核并行）：DataCopyPad 的 32B RMW 可能在“跨行边界”(last_tile <-> tile0)发生竞争。
        // 原 baseline 采用 2-phase + SyncAll 规避，但在 total_tiles 很小（例如 case4: total_tiles=2）时，
        // SyncAll 的开销会远大于计算本身，导致极慢。
        //
        // 这里对“极小 tile”场景启用单核串行：仅让 aiv_core0 处理所有 tile，其它 core 不参与，避免 SyncAll。
        const bool need_sync = (aiv_block_dim_ > 1U) && (total_tiles_ > 1U) && ((m_ % kVecWidthF32) != 0U);
        const bool force_single_core = need_sync && (total_tiles_ <= 4U);
        if (force_single_core) {
            aiv_effective_dim_ = 1U;
            if (aiv_block_idx_ != 0U) {
                start_tile_ = total_tiles_;
                tile_count_ = 0U;
            } else {
                start_tile_ = 0U;
                tile_count_ = total_tiles_;
            }
	        } else {
	            // 当 total_tiles 为奇数且 base==0（即：每个 tile 默认分配给不同 core）时，将最后一个 tile 合并到倒数第二个 core。
	            // 这样配合后续的 2-phase（偶/奇，且 last_tile 强制放到 phase=1）即可只用 1 次 SyncAll，避免多次 SyncAll 导致的 hang/timeout。
	            if (need_sync && ((total_tiles_ & 1U) != 0U) && (base == 0U) && (total_tiles_ > 1U)) {
	                if (aiv_block_idx_ + 1U == total_tiles_) { // 原本负责 last_tile 的 core：改为不处理 tile
                    start_tile_ = total_tiles_;
                    tile_count_ = 0U;
                } else if (aiv_block_idx_ + 2U == total_tiles_) { // 倒数第二个 core：额外处理 last_tile
                    tile_count_ = 2U;
                }
	            }
	        }

	        // row_block：保证 DataCopyPad(UB->GM) 的 32B RMW 不跨 block 竞争。
	        const uint32_t row_bytes = m_ * static_cast<uint32_t>(sizeof(float));
	        const uint32_t g = GcdU32(row_bytes == 0U ? 32U : row_bytes, 32U);
        row_block_ = (g == 0U) ? 1U : (32U / g);
        if (row_block_ == 0U) {
            row_block_ = 1U;
        }
        if (row_block_ > kMaxRowBlock) {
            row_block_ = kMaxRowBlock;
        }
        // 记录 base row_block（仅由对齐决定），后续 UB 优化可能增大 row_block_
        if (((p_type_ == 1U) || (p_type_ == 3U)) && ((m_ % kVecWidthF32) != 0U) && (total_tiles_ <= 2U)) {
            // 非对齐 + 小 tile：尽量放大 row_block，减少 RowParallel 的 block 数与同步压力
            const uint32_t base_block = row_block_;
            uint32_t candidate = base_block;
            while ((candidate + base_block) <= kMaxRowBlock && (candidate + base_block) <= n_) {
                candidate += base_block;
            }
            if (candidate > row_block_) {
                row_block_ = candidate;
            }
        }

	        // UB：input_tile + out_tile + 若干计算 buffer
	        const uint32_t in_tile_elems = n_ * tile_cols_;
	        // pair reduce 路径需要 [N, tile_cols] 的 out_tile（用于局部累加与写回）
	        const uint32_t out_tile_elems = in_tile_elems;
        pipe_.InitBuffer(buf_in_, AlignUpBytes32(in_tile_elems * static_cast<uint32_t>(sizeof(float))));
        pipe_.InitBuffer(buf_out_, AlignUpBytes32(out_tile_elems * static_cast<uint32_t>(sizeof(float))));
        pipe_.InitBuffer(buf_diff_, AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(float))));
        pipe_.InitBuffer(buf_abs_, AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(float))));
        pipe_.InitBuffer(buf_tmp_, AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(float))));
        if (use_p3_vecblk_ != 0U) {
            const uint32_t blk_elems = tile_cols_ * kP3VecBlockRows;
            pipe_.InitBuffer(buf_diff_blk_, AlignUpBytes32(blk_elems * static_cast<uint32_t>(sizeof(float))));
            pipe_.InitBuffer(buf_abs_blk_, AlignUpBytes32(blk_elems * static_cast<uint32_t>(sizeof(float))));
            pipe_.InitBuffer(buf_tmp_blk_, AlignUpBytes32(blk_elems * static_cast<uint32_t>(sizeof(float))));
        }
        if (need_log_work_ != 0U) {
            // 额外 scratch：用于 p<1 fast log2/exp2 计算（保存常量 t 等）
            pipe_.InitBuffer(buf_log_, AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(float))));
            pipe_.InitBuffer(buf_work_,
                             AlignUpBytes32((tile_cols_ + kVecWidthF32) * static_cast<uint32_t>(sizeof(float))));
        }
        pipe_.InitBuffer(buf_mask_, AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(uint8_t))));
        pipe_.InitBuffer(buf_sign_tmp_, AlignUpBytes32(kSignTmpBytes));
        // pair reduce/row 路径：
        // - grad/pdist 按行段预取（每行一次），减少 GM 标量 GetValue
        grad_seg_elems_ = n_;
        const bool need_pdist_seg = (p_type_ != 1U) && (p_type_ != 0U);
        pdist_seg_elems_ = need_pdist_seg ? n_ : 0U;
        pipe_.InitBuffer(buf_grad_seg_, AlignUpBytes32(grad_seg_elems_ * static_cast<uint32_t>(sizeof(float))));
        if (need_pdist_seg && pdist_seg_elems_ != 0U) {
            pipe_.InitBuffer(buf_pdist_seg_, AlignUpBytes32(pdist_seg_elems_ * static_cast<uint32_t>(sizeof(float))));
        }
        // 把 grad/pdist 预取到 UB，避免在 pair 双循环内做 GM 标量 GetValue（对 wall time 影响极大）
        //
        // 说明：
        // - 默认覆盖到 N<=128（pair_total<=8192）；
        // - p=1 场景尝试放宽到 N=256 的完整 grad cache，但需满足 UB 预算约束（防止 UB 超限）。
        constexpr uint32_t kPairCacheMaxDefault = 8192U;  // 8192 * 4B = 32KB
        constexpr uint32_t kPairCacheMaxP1N256 = 32768U; // 覆盖 pair_total=32640
        uint32_t pair_cache_max = kPairCacheMaxDefault;
        const uint32_t pair_total_u32 = static_cast<uint32_t>(pair_total_);
        // p=inf + 非 8 对齐 + 大 tile：默认收紧 cache，但小 pair 仍允许整段预取以保证稳定性
        const bool pinf_disable_cache =
            (p_type_ == 3U) && ((m_ % kVecWidthF32) != 0U) && (tile_cols_ >= 128U);
        if (pinf_disable_cache && (pair_total_u32 > kPairCacheMaxDefault)) {
            pair_cache_max = 0U;
        } else if (p_type_ == 1U) {
            // 大 N + 小 tile：优先释放 UB，降低标量与搬运开销
            if ((n_ >= 128U) && (total_tiles_ <= 2U)) {
                pair_cache_max = kPairCacheMaxDefault;
            } else {
                pair_cache_max = kPairCacheMaxP1N256;
            }
        }
        if (pair_total_u32 != 0U && pair_total_u32 <= pair_cache_max) {
            uint64_t ub_budget = static_cast<uint64_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
            if (ub_budget > 1024ULL) {
                ub_budget -= 1024ULL;
            }
            uint64_t ub_used = 0ULL;
            ub_used += AlignUpBytes32(in_tile_elems * static_cast<uint32_t>(sizeof(float)));
            ub_used += AlignUpBytes32(out_tile_elems * static_cast<uint32_t>(sizeof(float)));
            ub_used += AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(float))); // diff
            ub_used += AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(float))); // abs
            ub_used += AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(float))); // tmp
            if (use_p3_vecblk_ != 0U) {
                const uint32_t blk_elems = tile_cols_ * kP3VecBlockRows;
                ub_used += AlignUpBytes32(blk_elems * static_cast<uint32_t>(sizeof(float))); // diff_blk
                ub_used += AlignUpBytes32(blk_elems * static_cast<uint32_t>(sizeof(float))); // abs_blk
                ub_used += AlignUpBytes32(blk_elems * static_cast<uint32_t>(sizeof(float))); // tmp_blk
            }
            if (need_log_work_ != 0U) {
                ub_used += AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(float))); // log
                ub_used +=
                    AlignUpBytes32((tile_cols_ + kVecWidthF32) * static_cast<uint32_t>(sizeof(float))); // work
            }
            ub_used += AlignUpBytes32(tile_cols_ * static_cast<uint32_t>(sizeof(uint8_t))); // mask
            ub_used += AlignUpBytes32(kSignTmpBytes); // sign_tmp
            ub_used += AlignUpBytes32(grad_seg_elems_ * static_cast<uint32_t>(sizeof(float)));
            ub_used += AlignUpBytes32(pdist_seg_elems_ * static_cast<uint32_t>(sizeof(float)));

            uint64_t cache_bytes = static_cast<uint64_t>(pair_total_u32) * sizeof(float);
            const bool need_pdist_cache = (p_type_ != 1U);
            if (need_pdist_cache) {
                cache_bytes *= 2ULL;
            }
            if (ub_used + cache_bytes <= ub_budget) {
                use_grad_cache_ = 1U;
                use_pdist_cache_ = need_pdist_cache ? 1U : 0U;
                pipe_.InitBuffer(buf_grad_cache_,
                                 AlignUpBytes32(pair_total_u32 * static_cast<uint32_t>(sizeof(float))));
                grad_cache_ = buf_grad_cache_.Get<float>();
                if (use_pdist_cache_ != 0U) {
                    pipe_.InitBuffer(buf_pdist_cache_,
                                     AlignUpBytes32(pair_total_u32 * static_cast<uint32_t>(sizeof(float))));
                    pdist_cache_ = buf_pdist_cache_.Get<float>();
                }
            }
        }


        in_tile_ = buf_in_.Get<float>();
        out_tile_ = buf_out_.Get<float>();
        diff_ = buf_diff_.Get<float>();
        abs_ = buf_abs_.Get<float>();
        tmp_ = buf_tmp_.Get<float>();
        if (use_p3_vecblk_ != 0U) {
            diff_blk_ = buf_diff_blk_.Get<float>();
            abs_blk_ = buf_abs_blk_.Get<float>();
            tmp_blk_ = buf_tmp_blk_.Get<float>();
        }
        if (need_log_work_ != 0U) {
            log_ = buf_log_.Get<float>();
            work_ = buf_work_.Get<float>();
        }
        mask_ = buf_mask_.Get<uint8_t>();
        sign_tmp_ = buf_sign_tmp_.Get<uint8_t>();
        grad_seg_ = buf_grad_seg_.Get<float>();
        if (pdist_seg_elems_ != 0U) {
            pdist_seg_ = buf_pdist_seg_.Get<float>();
        } else {
            // p=1/0 场景不使用 pdist_seg_，复用 grad_seg_ 避免误用未初始化指针
            pdist_seg_ = grad_seg_;
        }

    }

	    __aicore__ inline void Process()
	    {
        if (use_matmul_ != 0U) {
            // Matmul 快路径：AIV 构造 Laplace，AIC 等待后执行 Matmul
            if ASCEND_IS_AIV {
                ProcessP2BuildLaplace();
                return;
            }
            if ASCEND_IS_AIC {
                ProcessP2Matmul();
                return;
            }
            return;
        }
	        // baseline 分支仅在 AIV 执行（AIC 直接返回）
        if ASCEND_IS_AIC {
            return;
        }
        // RowParallel 快路径：按 row 切分并行（无 workspace 归约）
        if (use_pair_reduce_ != 0U) {
            ProcessPairReduce();
            return;
        }
        if (n_ == 0U || m_ == 0U || total_tiles_ == 0U) {
            return;
        }
	        // 防止上一算子残留的 mask 模式/寄存器状态影响本算子矢量指令（会导致结果随机缺失）。
	        SetMaskNorm();
	        ResetMask();
        // case1 专用极简路径（严格命中 n=3,m=32,p=2）
        if (IsP2Case1Fast()) {
            ProcessP2Case1Fast();
            return;
        }
        // 小 N 场景预取 grad/pdist 到 UB，避免在 pair 双循环内做 GM 标量 GetValue（AIV scalar_time 会非常大）。
        EnsurePairCacheLoaded();
        // p=3 小 N：预计算 scale=g/(d*d) 到 cache，避免在每个 pair 上做标量除法
        if ((p_type_ == 4U) && (p_ == 3.0f)) {
            PrepareP3ScaleCache();
        }

        const bool need_sync = NeedSyncFix();
        if (!need_sync) {
            if (tile_count_ == 0U) {
                return;
            }
            for (uint32_t t = 0U; t < tile_count_; ++t) {
                ComputeAndStoreOneTile(start_tile_ + t);
            }
            return;
        }

        // 当 M 非 8 对齐且多核并行时：DataCopyPad(UB->GM) 可能触发 32B block 的 RMW，
        // 并在相邻 tile 之间（含跨行边界：last_tile <-> tile0）产生写回竞争。
        // 采用“2-phase（偶/奇）写回 + 1 次核间同步”：
        // - 对于 total_tiles 为奇数的情况：last_tile 强制放到 phase=1，并在 Init 中将 last_tile 合并到倒数第二个 core（base==0 时），
        //   使 last_tile 与其相邻的 (total_tiles-2) 在同一 core 内串行执行，避免额外 SyncAll（多次 SyncAll 会导致 hang/timeout）。
        constexpr uint32_t phase_cnt = 2U;
        // 2-phase（偶 -> 奇）写回：
        // - 奇数 tile（含 total_tiles 为奇时强制到 phase=1 的 last_tile）会通过 DataCopyPad 触发 32B block RMW，
        //   若与偶数 tile（例如 tile0）并行写回同一 block 会发生覆盖/随机错；
        // - 这里用 phase0(even) -> SyncAll -> phase1(odd) 串行化，避免跨核竞争。
        for (uint32_t phase = 0U; phase < phase_cnt; ++phase) {
            if (tile_count_ != 0U) {
                for (uint32_t t = 0U; t < tile_count_; ++t) {
                    const uint32_t tile_idx = start_tile_ + t;
                    if (TilePhaseId(tile_idx) != phase) {
                        continue;
                    }
                    ComputeAndStoreOneTile(tile_idx);
                }
            }
            if (phase + 1U < phase_cnt) {
                // 需要确保本 phase 的所有写回（含标量 SetValue）完成后再做核间同步，否则可能出现 RMW 覆盖/随机丢写。
                PipeBarrier<PIPE_ALL>();
                SyncAll();
            }
        }
    }

private:
    __aicore__ inline void CalcMatmulGMOffset(int32_t block_idx, const TCubeTiling &tiling, int32_t &offsetA,
                                              int32_t &offsetB, int32_t &offsetC, int32_t &tailM, int32_t &tailN) const
    {
        const uint32_t m = static_cast<uint32_t>(tiling.M);
        const uint32_t n = static_cast<uint32_t>(tiling.N);
        const uint32_t ka = static_cast<uint32_t>(tiling.Ka);
        const uint32_t single_m = static_cast<uint32_t>(tiling.singleCoreM);
        const uint32_t single_n = static_cast<uint32_t>(tiling.singleCoreN);

        const uint32_t m_single_blocks = CeilDivU32(m, single_m);
        const uint32_t m_core_idx = (m_single_blocks == 0U) ? 0U : (static_cast<uint32_t>(block_idx) % m_single_blocks);
        const uint32_t n_core_idx =
            (m_single_blocks == 0U) ? 0U : (static_cast<uint32_t>(block_idx) / m_single_blocks);

        offsetA = static_cast<int32_t>(m_core_idx * ka * single_m);
        offsetB = static_cast<int32_t>(n_core_idx * single_n);
        offsetC = static_cast<int32_t>(m_core_idx * n * single_m + n_core_idx * single_n);

        uint32_t tm = m - m_core_idx * single_m;
        tm = (tm < single_m) ? tm : single_m;
        uint32_t tn = n - n_core_idx * single_n;
        tn = (tn < single_n) ? tn : single_n;
        tailM = static_cast<int32_t>(tm);
        tailN = static_cast<int32_t>(tn);
    }

		    __aicore__ inline void BuildLaplaceF32(GlobalTensor<float> &lap_gm)
		    {
		        // Laplace 矩阵（N,N）：
		        // - L[i,i] = sum_{j!=i} w(i,j)
	        // - L[i,j] = -w(i,j), i!=j
	        // 其中 w(i,j) = grad(k)/dist(k), dist==0 时按 0 处理（对齐 PyTorch 行为）
		        //
		        // 性能关键点：
		        // - 禁止在双重(i,j)循环内使用 grad_gm_/pdist_gm_ 的 GetValue 标量访存（会把大 shape 拉到 ms 级）
		        // - 改为：先用 MTE(DataCopy/DataCopyPad) 把 grad/pdist 批量搬到 UB，算出 w=grad/dist 存在 UB，再用 UB 标量读构造 Laplace。
		        if (n_ < 2U) {
		            return;
		        }
		        if (pair_total_ == 0ULL) {
		            return;
		        }

		        // 0) 预计算 w(k)=grad(k)/dist(k) 到 UB（lap_w_）
		        // 说明：lap_w_/lap_d_/lap_mask_ 仅在 builder(core0) 初始化；本函数也只在 builder 调用。
		        const uint32_t pair_total_u32 = static_cast<uint32_t>(pair_total_);
		        const uint32_t w_elems = AlignUpU32(pair_total_u32, kLaplaceWChunk);
		        for (uint32_t off = 0U; off < w_elems; off += kLaplaceWChunk) {
		            auto w_chunk = lap_w_[off];
		            LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(off), w_chunk, kLaplaceWChunk);
		            LoadGmToUb(pdist_gm_, pair_total_, static_cast<uint64_t>(off), lap_d_, kLaplaceWChunk);
		            PipeBarrier<PIPE_ALL>();
		            // mask = (dist != 0)
		            CompareScalar(lap_mask_, lap_d_, 0.0f, CMPMODE::NE, kLaplaceWChunk);
		            // w = grad / dist（dist==0 的位置随后用 Select 置 0）
		            Div(w_chunk, w_chunk, lap_d_, static_cast<int32_t>(kLaplaceWChunk));
		            Select(w_chunk, lap_mask_, w_chunk, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, kLaplaceWChunk);
		            PipeBarrier<PIPE_V>();
		        }
		        PipeBarrier<PIPE_ALL>();

		        // 1) 并行构造 Laplace：按“行 tile”在 AIV 之间切分，避免单核构造导致 AIC 长时间空转等待。
		        const uint32_t tile_rows = MinU32(kLaplaceRowTile, n_);
		        const uint32_t total_row_tiles = (n_ + tile_rows - 1U) / tile_rows;
		        const uint32_t tid = aiv_block_idx_;
		        const uint32_t tdim = aiv_block_dim_ == 0U ? 1U : aiv_block_dim_;
		        for (uint32_t tile_id = tid; tile_id < total_row_tiles; tile_id += tdim) {
		            const uint32_t row_base = tile_id * tile_rows;
		            const uint32_t rows = MinU32(tile_rows, n_ - row_base);
		            for (uint32_t rr = 0U; rr < rows; ++rr) {
		                const uint32_t i = row_base + rr;
		                float diag = 0.0f;
		                const uint32_t row_off = rr * n_;

		                // j < i：k = PairStart(j) + (i-j-1)
		                for (uint32_t j = 0U; j < i; ++j) {
		                    const uint32_t k = PairStart(j, n_) + (i - j - 1U);
		                    const float w = lap_w_.GetValue(static_cast<uint32_t>(k));
		                    diag += w;
		                    a_tile_f32_.SetValue(row_off + j, -w);
		                }
		                // j > i：k = PairStart(i) + (j-i-1)
		                const uint32_t base = PairStart(i, n_);
		                for (uint32_t j = i + 1U; j < n_; ++j) {
		                    const uint32_t k = base + (j - i - 1U);
		                    const float w = lap_w_.GetValue(static_cast<uint32_t>(k));
		                    diag += w;
		                    a_tile_f32_.SetValue(row_off + j, -w);
		                }
		                a_tile_f32_.SetValue(row_off + i, diag);
		            }

		            // 2) 按 burst 写回 GM（UB->GM），写回前后用强栅栏，避免 UB 复用导致的极端错
		            PipeBarrier<PIPE_ALL>();
		            const uint64_t gm_off = static_cast<uint64_t>(row_base) * static_cast<uint64_t>(n_);
		            const uint32_t bytes = n_ * static_cast<uint32_t>(sizeof(float));
		            DataCopyExtParams copyParams = {static_cast<uint16_t>(rows), bytes, 0, 0, 0};
		            DataCopyPad(lap_gm[gm_off], a_tile_f32_, copyParams);
		            PipeBarrier<PIPE_ALL>();
		        }
		    }

			    __aicore__ inline void ProcessP2Matmul()
			    {
		        // 仅用于 host 侧标记的 p==2 大 shape 场景
		        if (p_type_ != 2U || n_ < 2U || m_ == 0U) {
		            return;
		        }
		        if (workspace_ == 0) {
		            return;
		        }
		        // AIC 上若存在 subblock，多个 subblock 并发进入 Matmul 可能共享内部 workspace/pipe 导致随机错，
		        // 这里统一只让 subblock0 工作（参考 SegmentReduceGrad 的 subblock 处理方式）。
		        if (GetSubBlockIdx() != 0U) {
		            return;
		        }
	        // 重要：
	        // - Matmul(cube) 的 tiling/内部workspace切分依赖 “唯一且连续的 block_idx”；
	        // - 在 MIX_AIC_1_2 场景下，AIV 侧 GetBlockIdx() 可能包含 task_ration（例如 idx=block*2+sublk），
	        //   但 AIC 侧 GetBlockIdx() 一般就是连续的物理 block idx（0..blockDim-1），不应再做 >>1 映射；
	        // - 若把 AIC block_idx 右移，会导致两个 AIC core 映射到同一个 block_idx，进而 Matmul 内部 workspace 冲突，
	        //   表现为结果随机错/不稳定。
	        const int32_t block_idx = static_cast<int32_t>(GetBlockIdx());
	        // 空闲 core 必须提前 return，避免无意义等待 flag 导致 hang
	        if (block_idx < 0 || block_idx >= static_cast<int32_t>(cube_tiling_.usedCoreNum)) {
	            return;
	        }

        // 等待本 block group 的 AIV builder 构造 Laplace 完成（group 内同步）
        CrossCoreWaitFlag(kLaplaceReadyFlagId);

        // workspace layout（host 侧按 sys_ws + user_ws 申请）：
        // - system workspace：供高阶 API（Matmul/SyncAll/DataCopyPad 等）内部使用；
        // - user workspace：我们自定义的 Laplace GM 缓冲。
        //
        // 注意：不同框架/版本下，workspace 入参可能是 sys workspace 基址，也可能已经是 user workspace 起点。
        // 这里参考 SegmentReduceGrad 的做法做一次判别，避免“重复偏移”导致 507057（MTE out-of-range）。
        __gm__ uint8_t *ws_u8 = reinterpret_cast<__gm__ uint8_t *>(workspace_);
        GM_ADDR sys_ws_base = reinterpret_cast<GM_ADDR>(GetSysWorkSpacePtr());
        const bool ws_is_sys = (sys_ws_base != nullptr) && (workspace_ == sys_ws_base);
        __gm__ uint8_t *user_ws_u8 = nullptr;
        if (ws_is_sys) {
            user_ws_u8 = ws_u8 + static_cast<uint64_t>(sys_ws_size_);
        } else {
            user_ws_u8 = ws_u8;
        }

	        // user workspace：仅构造 1 份 Laplace（FP32），所有 AIC core 复用
	        const uint64_t lap_off = static_cast<uint64_t>(kUserWsHeaderBytes);
	        auto lap_ptr = reinterpret_cast<__gm__ float *>(user_ws_u8 + lap_off);
        GlobalTensor<float> lap_gm;
        lap_gm.SetGlobalBuffer(lap_ptr, static_cast<uint64_t>(n_) * static_cast<uint64_t>(n_));

        int32_t offsetA = 0;
        int32_t offsetB = 0;
        int32_t offsetC = 0;
        int32_t tailM = 0;
        int32_t tailN = 0;
        CalcMatmulGMOffset(block_idx, cube_tiling_, offsetA, offsetB, offsetC, tailM, tailN);

        auto gmA = lap_gm[static_cast<uint64_t>(offsetA)];
        auto gmB = input_gm_[static_cast<uint64_t>(offsetB)];
        auto gmC = out_gm_[static_cast<uint64_t>(offsetC)];

        using MmA = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
	        using MmB = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
	        using MmC = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

	        // host 侧 MultiCoreMatmulTiling 默认使用 MDL 配置生成 tiling（mmConfigType=1），
	        // kernel 侧必须与之保持一致，否则会出现结果随机错/不稳定（尤其是大 shape + MIX 场景）。
	        Matmul<MmA, MmB, MmC, MmC, CFG_MDL> mm;
	        REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mm, &cube_tiling_);
	        mm.SetOrgShape(cube_tiling_.M, cube_tiling_.N, cube_tiling_.Ka, cube_tiling_.Kb);
	        mm.SetTensorA(gmA, false);
	        mm.SetTensorB(gmB, false);
	        mm.SetTail(tailM, tailN, -1);
        mm.IterateAll(gmC, 0);
        mm.End();
		    }

			    __aicore__ inline void ProcessP2BuildLaplace()
			    {
		        // 仅构造 1 份 Laplace（FP32）供所有 AIC 复用，并用各 block group 内 flag 唤醒对应的 AIC matmul。
		        const bool valid = (p_type_ == 2U) && (n_ >= 2U) && (m_ != 0U) && (workspace_ != 0) && (sys_ws_size_ != 0U);
		        if (!valid) {
		            CrossCoreSetFlag<2, PIPE_MTE2>(kLaplaceReadyFlagId);
		            return;
		        }
	        // 注意：workspace 入参可能已经是 user workspace 起点（不同调用链存在差异）。
	        // 这里与 AIC 侧保持一致：若 workspace==sys_ws_base 则按 sys_ws_size_ 偏移，否则直接当作 user workspace。
		        // workspace 指针语义在不同封装链路下可能存在差异（workspace 入参可能是 sys 或 user 起点）。
		        // 这里优先使用 GetSysWorkSpacePtr() 作为 sys workspace 起点，再按 host 下发的 sys_ws_size_ 做确定性切分。
		        // 若获取失败（极少见），退化为使用 workspace_ 入参作为 user workspace 起点。
		        __gm__ uint8_t *sys_ws_u8 = reinterpret_cast<__gm__ uint8_t *>(GetSysWorkSpacePtr());
		        __gm__ uint8_t *user_ws_u8 =
		            (sys_ws_u8 != nullptr) ? (sys_ws_u8 + static_cast<uint64_t>(sys_ws_size_))
		                                   : reinterpret_cast<__gm__ uint8_t *>(workspace_);

	        const uint64_t lap_off = static_cast<uint64_t>(kUserWsHeaderBytes);
	        auto lap_ptr = reinterpret_cast<__gm__ float *>(user_ws_u8 + lap_off);
	        GlobalTensor<float> lap_gm;
	        lap_gm.SetGlobalBuffer(lap_ptr, static_cast<uint64_t>(n_) * static_cast<uint64_t>(n_) * 1ULL);

		        SetMaskNorm();
		        ResetMask();
		        BuildLaplaceF32(lap_gm);
		        PipeBarrier<PIPE_ALL>();
		        // AIV 全局同步：确保 builder 写回完成后再通知 AIC（避免 AIC 读到未就绪 Laplace 导致随机错/超时）
		        // 说明：显式使用 SyncAll<true>()（仅同步 AIV），避免默认 SyncAll 的更大同步域引入额外开销/风险。
		        SyncAll<true>();
		        CrossCoreSetFlag<2, PIPE_MTE2>(kLaplaceReadyFlagId);
			    }

    __aicore__ inline bool NeedSyncFix() const
    {
        return (aiv_effective_dim_ > 1U) && (total_tiles_ > 1U) && ((m_ % kVecWidthF32) != 0U);
    }

    __aicore__ inline __gm__ uint8_t *GetUserWorkspaceBase() const
    {
        __gm__ uint8_t *ws_u8 = reinterpret_cast<__gm__ uint8_t *>(workspace_);
        if (ws_u8 == nullptr) {
            return nullptr;
        }
        __gm__ uint8_t *sys_ws_base = reinterpret_cast<__gm__ uint8_t *>(GetSysWorkSpacePtr());
        if (sys_ws_base != nullptr && ws_u8 == sys_ws_base) {
            return ws_u8 + static_cast<uint64_t>(sys_ws_size_);
        }
        return ws_u8;
    }

    __aicore__ inline bool UseRowParallelP1Pinf() const
    {
        // p=1/p=inf：小工作量优先用 RowParallel，避免原子/同步开销压过计算
        if (!((p_type_ == 1U) || (p_type_ == 3U))) {
            return false;
        }
        if (use_p1pinf_ws_ != 0U) {
            return false;
        }
        if (n_ < 2U || m_ == 0U || total_tiles_ == 0U) {
            return false;
        }
        if (total_tiles_ > 2U) {
            return false;
        }
        const uint64_t work = pair_total_ * static_cast<uint64_t>(m_);
        if ((m_ <= kP1PinfRowParallelMaxM) && (work <= kP1PinfRowParallelWork)) {
            return true;
        }
        // 兜底：小 N 仍走 RowParallel
        if (n_ <= 96U) {
            return true;
        }
        return false;
    }

    __aicore__ inline bool UseFullAivPairReduceP1Pinf() const
    {
        // p=1/p=inf：大计算量场景使用全 AIV task 并行，降低每核 pair 循环开销
        if (!((p_type_ == 1U) || (p_type_ == 3U))) {
            return false;
        }
        if (aiv_block_dim_ <= block_dim_) {
            return false;
        }
        if (m_ < kP1PinfFullAivMinM || pair_total_ == 0ULL) {
            return false;
        }
        const uint64_t per_core =
            (pair_total_ + static_cast<uint64_t>(aiv_block_dim_) - 1ULL) / static_cast<uint64_t>(aiv_block_dim_);
        if (per_core < kP1PinfFullAivMinPairsPerCore) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool UsePairReduceP2Atomic() const
    {
        // p=2 小 M：尝试原子 pair-reduce，避免 RowParallel 双倍计算
        if (p_type_ != 2U) {
            return false;
        }
        if (n_ < 2U || m_ < 16U || total_tiles_ == 0U) {
            return false;
        }
        const uint64_t work = pair_total_ * static_cast<uint64_t>(m_);
        if (work < kP2AtomicMinWork) {
            return false;
        }
        // 只覆盖小 M，避免影响 matmul 已经占优的大 M 形态
        if (m_ >= 64U) {
            return false;
        }
        return true;
    }


    __aicore__ inline uint32_t TilePhaseId(uint32_t tile_idx) const
    {
        // total_tiles 为奇数时，last_tile 与 tile0 会在“跨行边界”发生潜在 32B RMW 冲突，
        // 这里强制把 last_tile 放到 phase=1（odd phase），配合 Init 中的 tile 合并避免额外 SyncAll。
        if (((total_tiles_ & 1U) != 0U) && (tile_idx + 1U == total_tiles_)) {
            return 1U;
        }
        return tile_idx & 1U;
    }

    __aicore__ inline void ComputeAndStoreOneTile(uint32_t tile_idx)
    {
        const uint32_t col_start = tile_idx * tile_cols_;
        if (col_start >= m_) {
            return;
        }
        const uint32_t cols = MinU32(tile_cols_, m_ - col_start);

        LoadInputTile(col_start);
        Duplicate(out_tile_, 0.0f, static_cast<int32_t>(n_ * tile_cols_));
        PipeBarrier<PIPE_ALL>();
        ComputeTile(cols);
        StoreOutputTile(col_start, cols);
    }

    __aicore__ inline void ComputeTile(uint32_t cols)
    {
        if (p_type_ == 0U || n_ < 2U) {
            return;
        }
        if ((p_type_ == 4U) && (p_ == 3.0f)) {
            ComputeP3(cols);
            return;
        }
        switch (p_type_) {
            case 2U:
                ComputeP2();
                break;
            case 1U:
                ComputeP1Strided(0U, 1U, tile_cols_);
                break;
            case 3U:
                ComputePInfStrided(0U, 1U, tile_cols_);
                break;
            case 5U:
            case 6U:
            case 7U:
                ComputePLessThan2();
                break;
            default:
                ComputePGeneral();
                break;
        }
    }

    __aicore__ inline bool IsP2Case1Fast() const
    {
        // case1: n=3, m=32, p=2（严格命中，避免影响其它远端 case）
        return (p_type_ == 2U) && (n_ == 3U) && (m_ == 32U) && (total_tiles_ == 1U) && (tile_cols_ >= 32U);
    }

    __aicore__ inline void ProcessP2Case1Fast()
    {
        constexpr uint32_t cols = 32U;
        const uint32_t cols_aligned = AlignUpU32(cols, kVecWidthF32);
        LoadInputTile(0U);
        Duplicate(out_tile_, 0.0f, static_cast<int32_t>(n_ * tile_cols_));
        PipeBarrier<PIPE_ALL>();

        // 读取 3 个 pair 的 grad/pdist（k=0,1,2）
        const float g0 = grad_gm_.GetValue(0ULL);
        const float g1 = grad_gm_.GetValue(1ULL);
        const float g2 = grad_gm_.GetValue(2ULL);
        const float d0 = pdist_gm_.GetValue(0ULL);
        const float d1 = pdist_gm_.GetValue(1ULL);
        const float d2 = pdist_gm_.GetValue(2ULL);
        const float s0 = (d0 != 0.0f) ? (g0 / d0) : 0.0f;
        const float s1 = (d1 != 0.0f) ? (g1 / d1) : 0.0f;
        const float s2 = (d2 != 0.0f) ? (g2 / d2) : 0.0f;

        auto x0 = in_tile_[0U];
        auto x1 = in_tile_[tile_cols_];
        auto x2 = in_tile_[2U * tile_cols_];
        auto out0 = out_tile_[0U];
        auto out1 = out_tile_[tile_cols_];
        auto out2 = out_tile_[2U * tile_cols_];

        // (0,1)
        if (s0 != 0.0f) {
            Sub(diff_, x0, x1, static_cast<int32_t>(cols_aligned));
            Muls(tmp_, diff_, s0, static_cast<int32_t>(cols_aligned));
            Add(out0, out0, tmp_, static_cast<int32_t>(cols_aligned));
            Sub(out1, out1, tmp_, static_cast<int32_t>(cols_aligned));
        }
        // (0,2)
        if (s1 != 0.0f) {
            Sub(diff_, x0, x2, static_cast<int32_t>(cols_aligned));
            Muls(tmp_, diff_, s1, static_cast<int32_t>(cols_aligned));
            Add(out0, out0, tmp_, static_cast<int32_t>(cols_aligned));
            Sub(out2, out2, tmp_, static_cast<int32_t>(cols_aligned));
        }
        // (1,2)
        if (s2 != 0.0f) {
            Sub(diff_, x1, x2, static_cast<int32_t>(cols_aligned));
            Muls(tmp_, diff_, s2, static_cast<int32_t>(cols_aligned));
            Add(out1, out1, tmp_, static_cast<int32_t>(cols_aligned));
            Sub(out2, out2, tmp_, static_cast<int32_t>(cols_aligned));
        }

        PipeBarrier<PIPE_ALL>();
        StoreOutputTile(0U, cols);
    }

    __aicore__ inline void ComputeP3(uint32_t cols)
    {
        const uint32_t cols_aligned = AlignUpU32(cols, kVecWidthF32);
        const bool use_scale_cache = (p3_scale_ready_ != 0U);
        const bool use_vecblk = (use_p3_vecblk_ != 0U) && use_scale_cache;
        BinaryRepeatParams sub_params;
        UnaryRepeatParams abs_params;
        BinaryRepeatParams mul_params;
        BinaryRepeatParams red2;
        BinaryRepeatParams red4;
        BinaryRepeatParams red8;
        BinaryRepeatParams red16;
        uint8_t vec_rep_stride = 0U;
        if (use_vecblk) {
            // repeat stride 以 32B block 为单位：float32 下 1 block = 8 elems
            vec_rep_stride = static_cast<uint8_t>(tile_cols_ / kVecWidthF32);
            sub_params = BinaryRepeatParams(1, 1, 1, vec_rep_stride, 0, vec_rep_stride);
            abs_params = UnaryRepeatParams(1, 1, vec_rep_stride, vec_rep_stride);
            mul_params = BinaryRepeatParams(1, 1, 1, vec_rep_stride, vec_rep_stride, vec_rep_stride);
            const uint8_t stride2 = static_cast<uint8_t>(vec_rep_stride * 2U);
            const uint8_t stride4 = static_cast<uint8_t>(vec_rep_stride * 4U);
            const uint8_t stride8 = static_cast<uint8_t>(vec_rep_stride * 8U);
            const uint8_t stride16 = static_cast<uint8_t>(vec_rep_stride * 16U);
            red2 = BinaryRepeatParams(1, 1, 1, stride2, stride2, stride2);
            red4 = BinaryRepeatParams(1, 1, 1, stride4, stride4, stride4);
            red8 = BinaryRepeatParams(1, 1, 1, stride8, stride8, stride8);
            red16 = BinaryRepeatParams(1, 1, 1, stride16, stride16, stride16);
        }
        if (use_scale_cache) {
            uint32_t base = 0U;
            for (uint32_t i = 0U; i + 1U < n_; ++i) {
                auto xi = in_tile_[i * tile_cols_];
                auto out_i = out_tile_[i * tile_cols_];
                const uint32_t seg_len = n_ - i - 1U;
                uint32_t k = base;
                uint32_t off = (i + 1U) * tile_cols_;
                uint32_t j = i + 1U;
                // scale 已预计算为 g / (d*d)，直接使用，减少标量除法开销
                if (use_vecblk) {
                    // 批量处理 8 行：Sub/Abs/Mul 使用高维 repeat，降低矢量指令数
                    for (; (j + kP3VecBlockRows - 1U) < n_; j += kP3VecBlockRows, k += kP3VecBlockRows,
                           off += (tile_cols_ * kP3VecBlockRows)) {
                        Sub(diff_blk_, xi, in_tile_[off], cols_aligned, static_cast<uint8_t>(kP3VecBlockRows),
                            sub_params);
                        Abs(abs_blk_, diff_blk_, cols_aligned, static_cast<uint8_t>(kP3VecBlockRows), abs_params);
                        Mul(tmp_blk_, diff_blk_, abs_blk_, cols_aligned, static_cast<uint8_t>(kP3VecBlockRows),
                            mul_params);
                        // 先逐行缩放，再一次性更新 out_j（减少标量三目 Axpy 次数）
                        for (uint32_t r = 0U; r < kP3VecBlockRows; ++r) {
                            const float scale = grad_cache_.GetValue(k + r);
                            auto tmp_r = tmp_blk_[r * tile_cols_];
                            Muls(tmp_r, tmp_r, scale, static_cast<int32_t>(cols_aligned));
                        }
                        Sub(out_tile_[off], out_tile_[off], tmp_blk_, cols_aligned,
                            static_cast<uint8_t>(kP3VecBlockRows), mul_params);
                        // 将 32 行累加到第 0 行（使用 repeat 降低指令开销），最后一次性更新 out_i
                        Add(tmp_blk_, tmp_blk_, tmp_blk_[tile_cols_], cols_aligned, 16U, red2);
                        Add(tmp_blk_, tmp_blk_, tmp_blk_[2U * tile_cols_], cols_aligned, 8U, red4);
                        Add(tmp_blk_, tmp_blk_, tmp_blk_[4U * tile_cols_], cols_aligned, 4U, red8);
                        Add(tmp_blk_, tmp_blk_, tmp_blk_[8U * tile_cols_], cols_aligned, 2U, red16);
                        Add(tmp_blk_, tmp_blk_, tmp_blk_[16U * tile_cols_], static_cast<int32_t>(cols_aligned));
                        Add(out_i, out_i, tmp_blk_, static_cast<int32_t>(cols_aligned));
                    }
                } else {
                    for (; (j + 7U) < n_; j += 8U, k += 8U, off += (tile_cols_ * 8U)) {
                        const float scale0 = grad_cache_.GetValue(k);
                        auto xj0 = in_tile_[off];
                        auto out_j0 = out_tile_[off];
                        Sub(diff_, xi, xj0, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale0, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j0, tmp_, -scale0, static_cast<int32_t>(cols_aligned));

                        const uint32_t off1 = off + tile_cols_;
                        const float scale1 = grad_cache_.GetValue(k + 1U);
                        auto xj1 = in_tile_[off1];
                        auto out_j1 = out_tile_[off1];
                        Sub(diff_, xi, xj1, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale1, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j1, tmp_, -scale1, static_cast<int32_t>(cols_aligned));

                        const uint32_t off2 = off1 + tile_cols_;
                        const float scale2 = grad_cache_.GetValue(k + 2U);
                        auto xj2 = in_tile_[off2];
                        auto out_j2 = out_tile_[off2];
                        Sub(diff_, xi, xj2, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale2, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j2, tmp_, -scale2, static_cast<int32_t>(cols_aligned));

                        const uint32_t off3 = off2 + tile_cols_;
                        const float scale3 = grad_cache_.GetValue(k + 3U);
                        auto xj3 = in_tile_[off3];
                        auto out_j3 = out_tile_[off3];
                        Sub(diff_, xi, xj3, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale3, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j3, tmp_, -scale3, static_cast<int32_t>(cols_aligned));

                        const uint32_t off4 = off3 + tile_cols_;
                        const float scale4 = grad_cache_.GetValue(k + 4U);
                        auto xj4 = in_tile_[off4];
                        auto out_j4 = out_tile_[off4];
                        Sub(diff_, xi, xj4, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale4, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j4, tmp_, -scale4, static_cast<int32_t>(cols_aligned));

                        const uint32_t off5 = off4 + tile_cols_;
                        const float scale5 = grad_cache_.GetValue(k + 5U);
                        auto xj5 = in_tile_[off5];
                        auto out_j5 = out_tile_[off5];
                        Sub(diff_, xi, xj5, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale5, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j5, tmp_, -scale5, static_cast<int32_t>(cols_aligned));

                        const uint32_t off6 = off5 + tile_cols_;
                        const float scale6 = grad_cache_.GetValue(k + 6U);
                        auto xj6 = in_tile_[off6];
                        auto out_j6 = out_tile_[off6];
                        Sub(diff_, xi, xj6, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale6, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j6, tmp_, -scale6, static_cast<int32_t>(cols_aligned));

                        const uint32_t off7 = off6 + tile_cols_;
                        const float scale7 = grad_cache_.GetValue(k + 7U);
                        auto xj7 = in_tile_[off7];
                        auto out_j7 = out_tile_[off7];
                        Sub(diff_, xi, xj7, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale7, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j7, tmp_, -scale7, static_cast<int32_t>(cols_aligned));
                    }
                }
                if (use_vecblk) {
                    // 尾部行数用一次 repeat 处理（<=31），减少多段 16/8/4/2/1 的开销
                    const uint32_t rem = n_ - j;
                    if (rem != 0U) {
                        const uint8_t rep = static_cast<uint8_t>(rem);
                        Sub(diff_blk_, xi, in_tile_[off], cols_aligned, rep, sub_params);
                        Abs(abs_blk_, diff_blk_, cols_aligned, rep, abs_params);
                        Mul(tmp_blk_, diff_blk_, abs_blk_, cols_aligned, rep, mul_params);
                        for (uint32_t r = 0U; r < rem; ++r) {
                            const float scale = grad_cache_.GetValue(k + r);
                            auto tmp_r = tmp_blk_[r * tile_cols_];
                            Muls(tmp_r, tmp_r, scale, static_cast<int32_t>(cols_aligned));
                        }
                        Sub(out_tile_[off], out_tile_[off], tmp_blk_, cols_aligned, rep, mul_params);
                        // 将 rem 行按 16/8/4/2/1 分块归约，减少 out_i 的逐行 Add 次数
                        uint32_t rem_left = rem;
                        uint32_t base_row = 0U;
                        while (rem_left != 0U) {
                            uint32_t chunk = 1U;
                            if (rem_left >= 16U) {
                                chunk = 16U;
                            } else if (rem_left >= 8U) {
                                chunk = 8U;
                            } else if (rem_left >= 4U) {
                                chunk = 4U;
                            } else if (rem_left >= 2U) {
                                chunk = 2U;
                            }
                            auto tmp_base = tmp_blk_[base_row * tile_cols_];
                            if (chunk == 16U) {
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 1U) * tile_cols_], cols_aligned, 8U, red2);
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 2U) * tile_cols_], cols_aligned, 4U, red4);
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 4U) * tile_cols_], cols_aligned, 2U, red8);
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 8U) * tile_cols_], cols_aligned, 1U, red16);
                            } else if (chunk == 8U) {
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 1U) * tile_cols_], cols_aligned, 4U, red2);
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 2U) * tile_cols_], cols_aligned, 2U, red4);
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 4U) * tile_cols_], cols_aligned, 1U, red8);
                            } else if (chunk == 4U) {
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 1U) * tile_cols_], cols_aligned, 2U, red2);
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 2U) * tile_cols_], cols_aligned, 1U, red4);
                            } else if (chunk == 2U) {
                                Add(tmp_base, tmp_base, tmp_blk_[(base_row + 1U) * tile_cols_], cols_aligned, 1U, red2);
                            }
                            Add(out_i, out_i, tmp_base, static_cast<int32_t>(cols_aligned));
                            base_row += chunk;
                            rem_left -= chunk;
                        }
                        j += rem;
                        k += rem;
                        off += (tile_cols_ * rem);
                    }
                } else {
                    for (; (j + 3U) < n_; j += 4U, k += 4U, off += (tile_cols_ * 4U)) {
                        const float scale0 = grad_cache_.GetValue(k);
                        auto xj0 = in_tile_[off];
                        auto out_j0 = out_tile_[off];
                        Sub(diff_, xi, xj0, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale0, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j0, tmp_, -scale0, static_cast<int32_t>(cols_aligned));

                        const uint32_t off1 = off + tile_cols_;
                        const float scale1 = grad_cache_.GetValue(k + 1U);
                        auto xj1 = in_tile_[off1];
                        auto out_j1 = out_tile_[off1];
                        Sub(diff_, xi, xj1, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale1, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j1, tmp_, -scale1, static_cast<int32_t>(cols_aligned));

                        const uint32_t off2 = off1 + tile_cols_;
                        const float scale2 = grad_cache_.GetValue(k + 2U);
                        auto xj2 = in_tile_[off2];
                        auto out_j2 = out_tile_[off2];
                        Sub(diff_, xi, xj2, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale2, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j2, tmp_, -scale2, static_cast<int32_t>(cols_aligned));

                        const uint32_t off3 = off2 + tile_cols_;
                        const float scale3 = grad_cache_.GetValue(k + 3U);
                        auto xj3 = in_tile_[off3];
                        auto out_j3 = out_tile_[off3];
                        Sub(diff_, xi, xj3, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale3, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j3, tmp_, -scale3, static_cast<int32_t>(cols_aligned));
                    }
                    for (; (j + 1U) < n_; j += 2U, k += 2U, off += (tile_cols_ * 2U)) {
                        const float scale0 = grad_cache_.GetValue(k);
                        auto xj0 = in_tile_[off];
                        auto out_j0 = out_tile_[off];
                        Sub(diff_, xi, xj0, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale0, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j0, tmp_, -scale0, static_cast<int32_t>(cols_aligned));

                        const uint32_t off1 = off + tile_cols_;
                        const float scale1 = grad_cache_.GetValue(k + 1U);
                        auto xj1 = in_tile_[off1];
                        auto out_j1 = out_tile_[off1];
                        Sub(diff_, xi, xj1, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale1, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j1, tmp_, -scale1, static_cast<int32_t>(cols_aligned));
                    }
                    for (; j < n_; ++j, ++k, off += tile_cols_) {
                        const float scale = grad_cache_.GetValue(k);
                        auto xj = in_tile_[off];
                        auto out_j = out_tile_[off];
                        Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j, tmp_, -scale, static_cast<int32_t>(cols_aligned));
                    }
                }
                // PairStart(i+1) = PairStart(i) + (n - i - 1)
                base += seg_len;
            }
        } else {
            uint32_t base = 0U;
            for (uint32_t i = 0U; i + 1U < n_; ++i) {
                auto xi = in_tile_[i * tile_cols_];
                auto out_i = out_tile_[i * tile_cols_];
                const uint32_t seg_len = n_ - i - 1U;
                bool use_grad_seg = false;
                bool use_pdist_seg = false;
                bool need_barrier = false;
                if ((use_grad_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                    LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(base), grad_seg_, seg_len);
                    use_grad_seg = true;
                    need_barrier = true;
                }
                if ((use_pdist_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                    LoadGmToUb(pdist_gm_, pair_total_, static_cast<uint64_t>(base), pdist_seg_, seg_len);
                    use_pdist_seg = true;
                    need_barrier = true;
                }
                if (need_barrier) {
                    PipeBarrier<PIPE_MTE2>();
                }
                uint32_t k = base;
                uint32_t seg_idx = 0U;
                uint32_t off = (i + 1U) * tile_cols_;
                uint32_t j = i + 1U;
                // 未预计算 scale：按原始 g/d 计算，保持兼容性
                for (; (j + 3U) < n_; j += 4U, k += 4U, seg_idx += 4U, off += (tile_cols_ * 4U)) {
                    const float g0 = (use_grad_cache_ != 0U)
                                         ? grad_cache_.GetValue(k)
                                         : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                         : grad_gm_.GetValue(static_cast<uint64_t>(k)));
                    const float d0 = (use_pdist_cache_ != 0U)
                                         ? pdist_cache_.GetValue(k)
                                         : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx)
                                                          : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
                    if (d0 != 0.0f) {
                        const float scale0 = g0 / (d0 * d0);
                        auto xj0 = in_tile_[off];
                        auto out_j0 = out_tile_[off];
                        Sub(diff_, xi, xj0, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale0, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j0, tmp_, -scale0, static_cast<int32_t>(cols_aligned));
                    }

                    const uint32_t off1 = off + tile_cols_;
                    const float g1 = (use_grad_cache_ != 0U)
                                         ? grad_cache_.GetValue(k + 1U)
                                         : (use_grad_seg ? grad_seg_.GetValue(seg_idx + 1U)
                                                         : grad_gm_.GetValue(static_cast<uint64_t>(k + 1U)));
                    const float d1 = (use_pdist_cache_ != 0U)
                                         ? pdist_cache_.GetValue(k + 1U)
                                         : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx + 1U)
                                                          : pdist_gm_.GetValue(static_cast<uint64_t>(k + 1U)));
                    if (d1 != 0.0f) {
                        const float scale1 = g1 / (d1 * d1);
                        auto xj1 = in_tile_[off1];
                        auto out_j1 = out_tile_[off1];
                        Sub(diff_, xi, xj1, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale1, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j1, tmp_, -scale1, static_cast<int32_t>(cols_aligned));
                    }

                    const uint32_t off2 = off1 + tile_cols_;
                    const float g2 = (use_grad_cache_ != 0U)
                                         ? grad_cache_.GetValue(k + 2U)
                                         : (use_grad_seg ? grad_seg_.GetValue(seg_idx + 2U)
                                                         : grad_gm_.GetValue(static_cast<uint64_t>(k + 2U)));
                    const float d2 = (use_pdist_cache_ != 0U)
                                         ? pdist_cache_.GetValue(k + 2U)
                                         : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx + 2U)
                                                          : pdist_gm_.GetValue(static_cast<uint64_t>(k + 2U)));
                    if (d2 != 0.0f) {
                        const float scale2 = g2 / (d2 * d2);
                        auto xj2 = in_tile_[off2];
                        auto out_j2 = out_tile_[off2];
                        Sub(diff_, xi, xj2, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale2, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j2, tmp_, -scale2, static_cast<int32_t>(cols_aligned));
                    }

                    const uint32_t off3 = off2 + tile_cols_;
                    const float g3 = (use_grad_cache_ != 0U)
                                         ? grad_cache_.GetValue(k + 3U)
                                         : (use_grad_seg ? grad_seg_.GetValue(seg_idx + 3U)
                                                         : grad_gm_.GetValue(static_cast<uint64_t>(k + 3U)));
                    const float d3 = (use_pdist_cache_ != 0U)
                                         ? pdist_cache_.GetValue(k + 3U)
                                         : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx + 3U)
                                                          : pdist_gm_.GetValue(static_cast<uint64_t>(k + 3U)));
                    if (d3 != 0.0f) {
                        const float scale3 = g3 / (d3 * d3);
                        auto xj3 = in_tile_[off3];
                        auto out_j3 = out_tile_[off3];
                        Sub(diff_, xi, xj3, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale3, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j3, tmp_, -scale3, static_cast<int32_t>(cols_aligned));
                    }
                }
                for (; (j + 1U) < n_; j += 2U, k += 2U, seg_idx += 2U, off += (tile_cols_ * 2U)) {
                    const float g0 = (use_grad_cache_ != 0U)
                                         ? grad_cache_.GetValue(k)
                                         : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                         : grad_gm_.GetValue(static_cast<uint64_t>(k)));
                    const float d0 = (use_pdist_cache_ != 0U)
                                         ? pdist_cache_.GetValue(k)
                                         : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx)
                                                          : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
                    if (d0 != 0.0f) {
                        const float scale0 = g0 / (d0 * d0);
                        auto xj0 = in_tile_[off];
                        auto out_j0 = out_tile_[off];
                        Sub(diff_, xi, xj0, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale0, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j0, tmp_, -scale0, static_cast<int32_t>(cols_aligned));
                    }

                    const uint32_t off1 = off + tile_cols_;
                    const float g1 = (use_grad_cache_ != 0U)
                                         ? grad_cache_.GetValue(k + 1U)
                                         : (use_grad_seg ? grad_seg_.GetValue(seg_idx + 1U)
                                                         : grad_gm_.GetValue(static_cast<uint64_t>(k + 1U)));
                    const float d1 = (use_pdist_cache_ != 0U)
                                         ? pdist_cache_.GetValue(k + 1U)
                                         : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx + 1U)
                                                          : pdist_gm_.GetValue(static_cast<uint64_t>(k + 1U)));
                    if (d1 != 0.0f) {
                        const float scale1 = g1 / (d1 * d1);
                        auto xj1 = in_tile_[off1];
                        auto out_j1 = out_tile_[off1];
                        Sub(diff_, xi, xj1, static_cast<int32_t>(cols_aligned));
                        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                        Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                        Axpy(out_i, tmp_, scale1, static_cast<int32_t>(cols_aligned));
                        Axpy(out_j1, tmp_, -scale1, static_cast<int32_t>(cols_aligned));
                    }
                }
                for (; j < n_; ++j, ++k, ++seg_idx, off += tile_cols_) {
                    const float g = (use_grad_cache_ != 0U)
                                        ? grad_cache_.GetValue(k)
                                        : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                        : grad_gm_.GetValue(static_cast<uint64_t>(k)));
                    const float d = (use_pdist_cache_ != 0U)
                                        ? pdist_cache_.GetValue(k)
                                        : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx)
                                                         : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
                    if (d == 0.0f) {
                        continue;
                    }
                    const float scale = g / (d * d);
                    auto xj = in_tile_[off];
                    auto out_j = out_tile_[off];
                    Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
                    Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                    Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                    Axpy(out_i, tmp_, scale, static_cast<int32_t>(cols_aligned));
                    Axpy(out_j, tmp_, -scale, static_cast<int32_t>(cols_aligned));
                }
                // PairStart(i+1) = PairStart(i) + (n - i - 1)
                base += seg_len;
            }
        }
    }
    __aicore__ inline void LoadInputTile(uint32_t col_start)
    {
        if (col_start >= m_ || n_ == 0U || m_ == 0U) {
            return;
        }
        const uint32_t cols = MinU32(tile_cols_, m_ - col_start);

        // 2D DMA：一次搬 N 行，显著降低 AIV scalar_time / MTE2 调用次数。
        // 说明：当 cols < tile_cols_ 时，DataCopyPad 只会写入前 AlignUpBytes32(cols*4) 字节，
        // 若对齐后仍不足 tile_cols_，需要清零剩余尾部（否则计算会读到脏数据）。
        if (cols < tile_cols_) {
            // tail 起点不对齐时，改为整块清零，避免 UB 非对齐访问异常
            if ((cols % kVecWidthF32) != 0U) {
                Duplicate(in_tile_, 0.0f, static_cast<int32_t>(n_ * tile_cols_));
            } else {
                const uint32_t tail = tile_cols_ - cols;
                for (uint32_t r = 0U; r < n_; ++r) {
                    Duplicate(in_tile_[r * tile_cols_ + cols], 0.0f, static_cast<int32_t>(tail));
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyPad2DGm2UbFixedStride<float>(in_tile_, input_gm_[static_cast<uint64_t>(col_start)], n_, cols, m_,
                                            tile_cols_);
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void StoreOutputTile(uint32_t col_start, uint32_t cols)
    {
        // 确保所有矢量/标量流水线的计算完成后再写回，避免出现“部分写回/随机缺失”。
        PipeBarrier<PIPE_ALL>();
        if (n_ == 0U || cols == 0U || m_ == 0U) {
            return;
        }
        // 2D 写回：对齐列用 2D DMA，非对齐尾列用逐行写回避免跨行 RMW。
        if ((cols % kVecWidthF32) != 0U) {
            for (uint32_t r = 0U; r < n_; ++r) {
                const uint64_t gm_off =
                    static_cast<uint64_t>(r) * static_cast<uint64_t>(m_) + static_cast<uint64_t>(col_start);
                DataCopyPad2DUb2GmFixedStride<float>(out_gm_[gm_off], out_tile_[r * tile_cols_], 1U, cols, tile_cols_,
                                                    m_);
                // 非对齐行步长下 DataCopyPad 可能触发跨行 32B RMW，逐行串行化避免随机覆盖
                PipeBarrier<PIPE_MTE2>();
            }
        } else {
            DataCopyPad2DUb2GmFixedStride<float>(out_gm_[static_cast<uint64_t>(col_start)], out_tile_, n_, cols,
                                                tile_cols_, m_);
        }
        // 同时覆盖 DataCopy/DataCopyPad 以及标量写回的完成，避免跨 tile/phase 出现写回竞争。
        PipeBarrier<PIPE_ALL>();
    }



    __aicore__ inline void ProcessPairReduce()
    {
        // p=1/p=inf：默认走原子写回的 pair reduce（每个 pair 只计算一次）
        if ((p_type_ == 1U) || (p_type_ == 3U)) {
            if (use_p1pinf_ws_ != 0U) {
                ProcessPairReduceP1PinfWorkspace();
            } else if (UseRowParallelP1Pinf()) {
                ProcessRowParallel();
            } else {
                ProcessPairReduceAtomic();
            }
            return;
        }
        // p=2：小 M 尝试原子 pair-reduce，其余走 RowParallel
        if (UsePairReduceP2Atomic()) {
            ProcessPairReduceAtomic();
        } else {
            ProcessRowParallel();
        }
    }


    __aicore__ inline void ClearOutputGm(GlobalTensor<float> &out_gm, uint32_t out_m, uint32_t clear_m)
    {
        if (aiv_block_idx_ != 0U) {
            return;
        }
        if (n_ == 0U || clear_m == 0U || total_tiles_ == 0U) {
            return;
        }
        const bool use_2d = ((out_m % kVecWidthF32) == 0U) && ((clear_m % kVecWidthF32) == 0U);
        Duplicate(out_tile_, 0.0f, static_cast<int32_t>(n_ * tile_cols_));
        PipeBarrier<PIPE_ALL>();
        for (uint32_t tile_idx = 0U; tile_idx < total_tiles_; ++tile_idx) {
            const uint32_t col_start = tile_idx * tile_cols_;
            if (col_start >= clear_m) {
                continue;
            }
            const uint32_t cols = MinU32(tile_cols_, clear_m - col_start);
            if (cols == 0U) {
                continue;
            }
            if (use_2d && ((cols % kVecWidthF32) == 0U)) {
                DataCopyPad2DUb2GmFixedStride<float>(out_gm[static_cast<uint64_t>(col_start)], out_tile_, n_, cols,
                                                    tile_cols_, out_m);
            } else {
                for (uint32_t r = 0U; r < n_; ++r) {
                    const uint64_t gm_off =
                        static_cast<uint64_t>(r) * static_cast<uint64_t>(out_m) + static_cast<uint64_t>(col_start);
                    DataCopy(out_gm[gm_off], out_tile_[r * tile_cols_], cols);
                }
            }
            PipeBarrier<PIPE_MTE2>();
        }
    }

    __aicore__ inline void CopyWorkspaceToOut(const GlobalTensor<float> &ws_gm)
    {
        if (aiv_block_idx_ != 0U) {
            return;
        }
        if (n_ == 0U || m_ == 0U || total_tiles_ == 0U) {
            return;
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t tile_idx = 0U; tile_idx < total_tiles_; ++tile_idx) {
            const uint32_t col_start = tile_idx * tile_cols_;
            if (col_start >= m_) {
                continue;
            }
            const uint32_t cols = MinU32(tile_cols_, m_ - col_start);
            DataCopyPad2DGm2UbFixedStride<float>(out_tile_, ws_gm[static_cast<uint64_t>(col_start)], n_, cols,
                                                m_padded_, tile_cols_);
            PipeBarrier<PIPE_MTE2>();
            DataCopyPad2DUb2GmFixedStride<float>(out_gm_[static_cast<uint64_t>(col_start)], out_tile_, n_, cols,
                                                tile_cols_, m_);
            PipeBarrier<PIPE_MTE2>();
        }
    }

    __aicore__ inline void ProcessPairReduceP1PinfWorkspace()
    {
        if (n_ < 2U || m_ == 0U || total_tiles_ == 0U) {
            return;
        }
        if (use_p1pinf_ws_ == 0U) {
            return;
        }
        __gm__ uint8_t *user_ws_u8 = GetUserWorkspaceBase();
        if (user_ws_u8 == nullptr) {
            return;
        }
        auto ws_ptr = reinterpret_cast<__gm__ float *>(user_ws_u8 + static_cast<uint64_t>(kUserWsHeaderBytes));
        GlobalTensor<float> ws_gm;
        ws_gm.SetGlobalBuffer(ws_ptr, static_cast<uint64_t>(n_) * static_cast<uint64_t>(m_padded_));
        ProcessPairReduceAtomicOut(ws_gm, m_padded_, m_padded_, true);
        if (aiv_block_dim_ > 1U) {
            SyncAll<true>();
        }
        CopyWorkspaceToOut(ws_gm);
    }

    __aicore__ inline void ProcessPairReduceAtomicOut(GlobalTensor<float> &out_gm, uint32_t out_m, uint32_t compute_m,
                                                      bool use_ws)
    {
        // 原子写回：每个 pair 只计算一次，写回时按对齐段/尾段分段原子累加。
        if (n_ < 2U || m_ == 0U || total_tiles_ == 0U) {
            return;
        }
        if (!((p_type_ == 1U) || (p_type_ == 3U) || (p_type_ == 2U))) {
            return;
        }
        if ((tile_cols_ % kVecWidthF32) != 0U) {
            return;
        }

        SetMaskNorm();
        ResetMask();

        // 原子写回前需要保证 out_gm 初值为 0，否则会把历史值叠加进去。
        // 仅由 core0 清零一次，其他 core 等待同步。
        ClearOutputGm(out_gm, out_m, compute_m);
        if (aiv_block_dim_ > 1U) {
            SyncAll<true>();
        }

        // MIX_AIC_1_2 默认 2 个 AIV task/组；p=1/p=inf 大计算量允许启用全 AIV task。
        uint32_t task_ration = kAivPerBlock;
        uint32_t group_dim = block_dim_;
        if (UseFullAivPairReduceP1Pinf()) {
            task_ration = 1U;
            group_dim = aiv_block_dim_;
        }
        if (group_dim == 0U) {
            group_dim = 1U;
        }
        const uint32_t group_id = (task_ration == 0U) ? aiv_block_idx_ : (aiv_block_idx_ / task_ration);
        const uint32_t sub_idx = (task_ration == 0U) ? 0U : (aiv_block_idx_ % task_ration);

        uint32_t core_dim = group_dim;
        if (core_dim == 0U) {
            core_dim = 1U;
        }
        uint32_t core_id = group_id % core_dim;

        uint32_t eff_core_dim = core_dim;
        if (pair_total_ != 0ULL) {
            // 动态提高并行度：中等/大 N 场景优先用更多核减少向量/标量循环时间
            uint32_t pairs_per_core = 1024U;
            if (pair_total_ >= 16384ULL) {
                pairs_per_core = 256U;
            } else if (pair_total_ >= 4096ULL) {
                pairs_per_core = 512U;
            } else if (((p_type_ == 1U) || (p_type_ == 3U)) && (m_ >= kP1PinfAtomicTailMinCols) &&
                       (pair_total_ >= static_cast<uint64_t>(kP1PinfAtomicTailMinPairs))) {
                // p=1/p=inf：中小 N 且 M 不小的场景，适度加并行，降低每核 pair 循环时间
                pairs_per_core = 512U;
            }
            uint32_t target_core = CeilDivU32(static_cast<uint32_t>(pair_total_), pairs_per_core);
            if (target_core == 0U) {
                target_core = 1U;
            }
            if (eff_core_dim > target_core) {
                eff_core_dim = target_core;
            }
        }
        if (eff_core_dim == 0U) {
            eff_core_dim = 1U;
        }
        const bool is_worker = (sub_idx == 0U) && (core_id < eff_core_dim);
        const bool has_unaligned_tail = (!use_ws) && ((m_ % kVecWidthF32) != 0U);
        if (!has_unaligned_tail && !is_worker) {
            return;
        }

        if (is_worker) {
            EnsurePairCacheLoaded();
        }

        uint64_t k_start = 0ULL;
        uint64_t k_end = 0ULL;
        uint32_t row_start = 0U;
        uint32_t row_end = 0U;
        if (is_worker) {
            GetPairRange(core_id, eff_core_dim, k_start, k_end);
            ComputeRowRangeFromKRange(k_start, k_end, row_start, row_end);
        }

        RowParallelInfo rp_info{};
        bool rp_ready = false;
        const bool is_p2 = (p_type_ == 2U);
        const bool is_p1 = (p_type_ == 1U);
        const bool is_pinf = (p_type_ == 3U);

        for (uint32_t tile_idx = 0U; tile_idx < total_tiles_; ++tile_idx) {
            const uint32_t col_start = tile_idx * tile_cols_;
            if (col_start >= m_) {
                continue;
            }
            const uint32_t cols = MinU32(tile_cols_, m_ - col_start);
            const uint32_t cols_compute = use_ws ? MinU32(tile_cols_, compute_m - col_start) : cols;
            const bool tail_non_aligned = (!use_ws) && ((cols % kVecWidthF32) != 0U);
            if (tail_non_aligned) {
                if (is_p2) {
                    // p=2 非对齐：前缀对齐段走原子，尾段走 RowParallel
                    const uint32_t aligned_cols = (cols / 16U) * 16U; // 64B 对齐
                    const uint32_t tail_cols = cols - aligned_cols;
                    if (aligned_cols >= 16U) {
                        if (is_worker) {
                            LoadInputTile(col_start);
                            PipeBarrier<PIPE_ALL>();
                            if (row_start < row_end) {
                                const uint32_t row_elems = (row_end - row_start) * tile_cols_;
                                Duplicate(out_tile_[row_start * tile_cols_], 0.0f,
                                          static_cast<int32_t>(row_elems));
                            }
                            ComputeP2StridedSlice(core_id, eff_core_dim, 0U, aligned_cols);
                            PipeBarrier<PIPE_ALL>();
                            SetAtomicType<float>();
                            SetAtomicAdd<float>();
                            for (uint32_t r = row_start; r < row_end; ++r) {
                                const uint64_t gm_off =
                                    static_cast<uint64_t>(r) * static_cast<uint64_t>(out_m) +
                                    static_cast<uint64_t>(col_start);
                                DataCopy(out_gm[gm_off], out_tile_[r * tile_cols_], aligned_cols);
                            }
                            SetAtomicNone();
                            PipeBarrier<PIPE_ALL>();
                        }
                    }
                    if (tail_cols != 0U) {
                        if (!rp_ready) {
                            if (!InitRowParallelInfo(rp_info)) {
                                return;
                            }
                            rp_ready = true;
                        }
                        if (rp_info.is_worker) {
                            EnsurePairCacheLoaded();
                        }
                        ProcessRowParallelTile(rp_info, col_start + aligned_cols, tail_cols);
                    }
                    continue;
                }
                if (is_p1 || is_pinf) {
                    const bool allow_atomic_prefix =
                        (pair_total_ >= static_cast<uint64_t>(kP1PinfAtomicTailMinPairs)) &&
                        (cols >= kP1PinfAtomicTailMinCols);
                    if (!allow_atomic_prefix) {
                        // 小 pair_total 或小 cols：保持 RowParallel，全量计算避免原子前缀额外调度开销
                        if (!rp_ready) {
                            if (!InitRowParallelInfo(rp_info)) {
                                return;
                            }
                            rp_ready = true;
                        }
                        if (rp_info.is_worker) {
                            EnsurePairCacheLoaded();
                        }
                        ProcessRowParallelTile(rp_info, col_start, cols);
                        continue;
                    }
                    // p=1/p=inf 非对齐：前缀对齐段走原子，尾段走 RowParallel
                    const uint32_t align = is_pinf ? kCompareAlignF32 : kVecWidthF32;
                    const uint32_t aligned_cols = (cols / align) * align;
                    const uint32_t tail_cols = cols - aligned_cols;
                    if (aligned_cols != 0U) {
                        if (is_worker) {
                            LoadInputTile(col_start);
                            PipeBarrier<PIPE_ALL>();
                            if (row_start < row_end) {
                                const uint32_t row_elems = (row_end - row_start) * tile_cols_;
                                Duplicate(out_tile_[row_start * tile_cols_], 0.0f,
                                          static_cast<int32_t>(row_elems));
                            }
                            if (is_p1) {
                                ComputeP1Strided(core_id, eff_core_dim, aligned_cols);
                            } else {
                                ComputePInfStrided(core_id, eff_core_dim, aligned_cols);
                            }
                            PipeBarrier<PIPE_ALL>();
                            SetAtomicType<float>();
                            SetAtomicAdd<float>();
                            for (uint32_t r = row_start; r < row_end; ++r) {
                                const uint64_t gm_off =
                                    static_cast<uint64_t>(r) * static_cast<uint64_t>(out_m) +
                                    static_cast<uint64_t>(col_start);
                                DataCopy(out_gm[gm_off], out_tile_[r * tile_cols_], aligned_cols);
                            }
                            SetAtomicNone();
                            PipeBarrier<PIPE_ALL>();
                        }
                    }
                    if (tail_cols != 0U) {
                        if (!rp_ready) {
                            if (!InitRowParallelInfo(rp_info)) {
                                return;
                            }
                            rp_ready = true;
                        }
                        if (rp_info.is_worker) {
                            EnsurePairCacheLoaded();
                        }
                        ProcessRowParallelTile(rp_info, col_start + aligned_cols, tail_cols);
                    }
                    continue;
                }
                if (!rp_ready) {
                    if (!InitRowParallelInfo(rp_info)) {
                        return;
                    }
                    rp_ready = true;
                }
                if (rp_info.is_worker) {
                    EnsurePairCacheLoaded();
                }
                ProcessRowParallelTile(rp_info, col_start, cols);
                continue;
            }
            if (!is_worker) {
                continue;
            }
            LoadInputTile(col_start);
            // 额外保障：LoadInputTile 的 MTE2 完成后再进入矢量计算，避免随机 WA
            PipeBarrier<PIPE_ALL>();
            if (row_start < row_end) {
                const uint32_t row_elems = (row_end - row_start) * tile_cols_;
                Duplicate(out_tile_[row_start * tile_cols_], 0.0f, static_cast<int32_t>(row_elems));
            }
            if (p_type_ == 1U) {
                ComputeP1Strided(core_id, eff_core_dim, cols_compute);
            } else if (p_type_ == 3U) {
                ComputePInfStrided(core_id, eff_core_dim, cols_compute);
            } else {
                ComputeP2StridedSlice(core_id, eff_core_dim, 0U, cols_compute);
            }

            PipeBarrier<PIPE_ALL>();
            SetAtomicType<float>();
            SetAtomicAdd<float>();
            if (!AtomicStoreRowsAlignedOut(out_gm, out_m, col_start, cols_compute, row_start, row_end)) {
                for (uint32_t r = row_start; r < row_end; ++r) {
                    const uint64_t gm_off =
                        static_cast<uint64_t>(r) * static_cast<uint64_t>(out_m) + static_cast<uint64_t>(col_start);
                    DataCopy(out_gm[gm_off], out_tile_[r * tile_cols_], cols_compute);
                }
            }
            SetAtomicNone();
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessPairReduceAtomic()
    {
        ProcessPairReduceAtomicOut(out_gm_, m_, m_, false);
    }

    struct RowParallelInfo {
        uint32_t group_dim;
        uint32_t group_id;
        uint32_t sub_idx;
        uint32_t row_block;
        uint32_t total_blocks;
        uint32_t eff_group_dim;
        bool is_worker;
        bool need_block_sync_base;
    };

    __aicore__ inline bool InitRowParallelInfo(RowParallelInfo &info) const
    {
        uint32_t group_dim = block_dim_;
        if (group_dim == 0U) {
            group_dim = 1U;
        }
        // MIX_AIC_1_2 固定 2 个 AIV task/组，避免 GetTaskRation 异常导致分组遗漏
        const uint32_t task_ration = kAivPerBlock;
        const uint32_t task_id = aiv_block_idx_;
        const uint32_t sub_idx = (task_ration == 0U) ? 0U : (task_id % task_ration);
        uint32_t group_id = (task_ration == 0U) ? task_id : (task_id / task_ration);
        if (group_dim != 0U) {
            group_id = group_id % group_dim;
        }

        // 计算 row_block（32B 对齐 block），并以 block 为单位并行。
        const uint32_t row_bytes = m_ * static_cast<uint32_t>(sizeof(float));
        const uint32_t g = GcdU32(row_bytes == 0U ? 32U : row_bytes, 32U);
        const uint32_t row_block = (g == 0U) ? 1U : (32U / g); // >=1
        const uint32_t total_blocks = (row_block == 0U) ? 0U : CeilDivU32(n_, row_block);
        if (total_blocks == 0U) {
            info = RowParallelInfo{};
            return false;
        }
        const uint32_t eff_group_dim = (group_dim > total_blocks) ? total_blocks : group_dim;
        const bool is_worker = (sub_idx == 0U) && (group_id < total_blocks);
        // M 非 8 对齐时：若 tile 的 cols 非 32B 对齐，DataCopyPad 的 32B RMW 可能跨 row_block，
        // 需要 2-phase 串行化写回；对齐 tile 可跳过同步，减少 barrier 开销。
        const bool need_block_sync_base =
            ((m_ % kVecWidthF32) != 0U) && (total_blocks > 1U) && (eff_group_dim > 1U);

        info.group_dim = group_dim;
        info.group_id = group_id;
        info.sub_idx = sub_idx;
        info.row_block = row_block;
        info.total_blocks = total_blocks;
        info.eff_group_dim = eff_group_dim;
        info.is_worker = is_worker;
        info.need_block_sync_base = need_block_sync_base;
        return true;
    }

    __aicore__ inline void ProcessRowParallelTile(const RowParallelInfo &info, uint32_t col_start, uint32_t cols)
    {
        if (col_start >= m_ || cols == 0U) {
            return;
        }
        const bool tile_need_sync = info.need_block_sync_base && ((cols % kVecWidthF32) != 0U);
        const uint32_t cols_aligned =
            (p_type_ == 3U) ? AlignUpU32(cols, kCompareAlignF32) : AlignUpU32(cols, kVecWidthF32);
        const bool full_cols = (cols_aligned == tile_cols_);
        if (!tile_need_sync && !info.is_worker) {
            return;
        }
        if (info.is_worker) {
            LoadInputTile(col_start);
            // 额外保障：LoadInputTile 的 MTE2 完成后再进入矢量计算，避免随机 WA
            PipeBarrier<PIPE_ALL>();
        }

        const uint32_t phase_cnt = tile_need_sync ? 2U : 1U;
        for (uint32_t phase = 0U; phase < phase_cnt; ++phase) {
            if (info.is_worker) {
                for (uint32_t blk = info.group_id; blk < info.total_blocks; blk += info.eff_group_dim) {
                    if (tile_need_sync && ((blk & 1U) != phase)) {
                        continue;
                    }
                    const uint32_t row_begin = blk * info.row_block;
                    uint32_t row_end = row_begin + info.row_block;
                    if (row_end > n_) {
                        row_end = n_;
                    }
                    const uint32_t block_rows = (row_end > row_begin) ? (row_end - row_begin) : 0U;
                    if (block_rows == 0U) {
                        continue;
                    }
                    if (full_cols) {
                        const uint32_t block_elems = block_rows * tile_cols_;
                        // 1) 清零本 block 的 out 行（一次性清零，减少矢量指令与标量循环开销）
                        Duplicate(out_tile_[row_begin * tile_cols_], 0.0f, static_cast<int32_t>(block_elems));
                        // 2) 计算本 block 的 out 行（按行调用对应 p 的 row 计算）
                        for (uint32_t i = row_begin; i < row_end; ++i) {
                            if (p_type_ == 2U) {
                                ComputeRowP2(i, tile_cols_);
                            } else if (p_type_ == 1U) {
                                ComputeRowP1(i, tile_cols_);
                            } else if ((p_type_ == 4U) && (p_ == 3.0f)) {
                                ComputeRowP3(i, tile_cols_);
                            } else {
                                ComputeRowPInf(i, tile_cols_);
                            }
                        }
                    } else {
                        // tail tile：只计算 cols 范围，避免对齐不足导致的大量冗余矢量计算
                        for (uint32_t i = row_begin; i < row_end; ++i) {
                            Duplicate(out_tile_[i * tile_cols_], 0.0f, static_cast<int32_t>(cols_aligned));
                            if (p_type_ == 2U) {
                                ComputeRowP2(i, cols);
                            } else if (p_type_ == 1U) {
                                ComputeRowP1(i, cols);
                            } else if ((p_type_ == 4U) && (p_ == 3.0f)) {
                                ComputeRowP3(i, cols);
                            } else {
                                ComputeRowPInf(i, cols);
                            }
                        }
                    }
                    PipeBarrier<PIPE_ALL>();
                    // 3) 写回 GM（每个 block 的 GM 区间 32B 对齐且不重叠，无需 SyncAll）
                    const uint64_t gm_off = static_cast<uint64_t>(row_begin) * static_cast<uint64_t>(m_) +
                                            static_cast<uint64_t>(col_start);
                    DataCopyPad2DUb2GmFixedStride<float>(out_gm_[gm_off], out_tile_[row_begin * tile_cols_],
                                                        block_rows, cols, tile_cols_, m_);
                    PipeBarrier<PIPE_ALL>();
                }
            }
            if (tile_need_sync && (phase + 1U < phase_cnt)) {
                PipeBarrier<PIPE_ALL>();
                SyncAll<true>();
            }
        }
    }

    __aicore__ inline bool AtomicStoreRowsAlignedOut(const GlobalTensor<float> &out_gm, uint32_t out_m,
                                                     uint32_t col_start, uint32_t cols, uint32_t row_start,
                                                     uint32_t row_end)
    {
        if (row_start >= row_end) {
            return true;
        }
        if (cols != tile_cols_ || (cols % kVecWidthF32) != 0U) {
            return false;
        }
        if ((out_m % kVecWidthF32) != 0U) {
            return false;
        }
        const uint32_t rows = row_end - row_start;
        const uint32_t bytes = cols * static_cast<uint32_t>(sizeof(float));
        if ((bytes % 64U) != 0U) {
            return false;
        }
        if (rows > 0xFFFFU) {
            return false;
        }
        const uint32_t src_stride = ((tile_cols_ - cols) * static_cast<uint32_t>(sizeof(float))) / 32U;
        const uint32_t dst_stride = ((out_m - cols) * static_cast<uint32_t>(sizeof(float))) / 32U;
        if (src_stride > 0xFFFFU || dst_stride > 0xFFFFU) {
            return false;
        }
        const uint64_t gm_off =
            static_cast<uint64_t>(row_start) * static_cast<uint64_t>(out_m) + static_cast<uint64_t>(col_start);
        auto gm_row = out_gm[gm_off];
        auto ub_row = out_tile_[row_start * tile_cols_];
        const uint64_t gm_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(gm_row.GetPhyAddr()));
        const uint64_t ub_addr = static_cast<uint64_t>(ub_row.GetPhyAddr());
        if (((gm_addr | ub_addr) & 63ULL) != 0ULL) {
            return false;
        }
        const uint32_t block_len = bytes / 32U;
        DataCopyParams params(static_cast<uint16_t>(rows), static_cast<uint16_t>(block_len),
                              static_cast<uint16_t>(src_stride), static_cast<uint16_t>(dst_stride));
        DataCopy(gm_row, ub_row, params);
        return true;
    }

		    __aicore__ inline void ProcessRowParallel()
		    {
	        // RowParallel 快路径：
	        // - 按 row_block 切分，保证 GM 写回 32B 对齐，避免跨行 RMW。
	        if (n_ < 2U || m_ == 0U || total_tiles_ == 0U) {
	            return;
	        }
        // p=2 始终可用；p=1/p=inf 仅在小形状下启用（避免双倍计算）
        const bool allow_p3 = (p_type_ == 4U) && (p_ == 3.0f);
        if ((p_type_ != 2U) && !UseRowParallelP1Pinf() && !allow_p3) {
            return;
        }

		        // 防止上一算子残留的 mask 模式/寄存器状态影响本算子矢量指令（会导致结果随机缺失）。
		        SetMaskNorm();
		        ResetMask();

        RowParallelInfo info{};
        if (!InitRowParallelInfo(info)) {
            return;
        }
        // 无需同步时：非 worker 直接退出，减少调度开销
        if (!info.need_block_sync_base && !info.is_worker) {
            return;
        }

        // 复用 baseline 的 pair cache（小 N 预取 grad/pdist），减少 GM 标量访存
        // 放在 subblock/group 过滤之后，避免大量空转 task 也做 MTE2 搬运。
        if (info.is_worker) {
            EnsurePairCacheLoaded();
            if (allow_p3) {
                PrepareP3ScaleCache();
            }
        }

        for (uint32_t tile_idx = 0U; tile_idx < total_tiles_; ++tile_idx) {
            const uint32_t col_start = tile_idx * tile_cols_;
            if (col_start >= m_) {
                continue;
            }
            const uint32_t cols = MinU32(tile_cols_, m_ - col_start);
            ProcessRowParallelTile(info, col_start, cols);
        }
    }

    __aicore__ inline void ComputeRowP1(uint32_t i, uint32_t cols)
    {
        // p=1：仅更新当前 row 的梯度（RowParallel 路径）
        const uint32_t cols_aligned = AlignUpU32(cols, kVecWidthF32);
        auto xi = in_tile_[i * tile_cols_];
        auto out_i = out_tile_[i * tile_cols_];

        // j < i：pair index 不连续，使用 cache/GM 标量读取
        for (uint32_t j = 0U; j < i; ++j) {
            const uint32_t k = PairStart(j, n_) + (i - j - 1U);
            const float g = GetGrad(k);
            auto xj = in_tile_[j * tile_cols_];
            Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
            ComputeP1Sign(cols_aligned);
            Axpy(out_i, tmp_, g, static_cast<int32_t>(cols_aligned));
        }

        // j > i：pair index 连续，允许段预取
        const uint32_t base = PairStart(i, n_);
        const uint32_t seg_len = n_ - i - 1U;
        bool use_grad_seg = false;
        if ((use_grad_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
            LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(base), grad_seg_, seg_len);
            use_grad_seg = true;
            PipeBarrier<PIPE_MTE2>();
        }
        uint32_t k = base;
        uint32_t seg_idx = 0U;
        uint32_t off = (i + 1U) * tile_cols_;
        for (uint32_t j = i + 1U; j < n_; ++j, ++k, ++seg_idx, off += tile_cols_) {
            const float g = (use_grad_cache_ != 0U)
                                ? grad_cache_.GetValue(k)
                                : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                : grad_gm_.GetValue(static_cast<uint64_t>(k)));
            auto xj = in_tile_[off];
            Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
            ComputeP1Sign(cols_aligned);
            Axpy(out_i, tmp_, g, static_cast<int32_t>(cols_aligned));
        }
    }

    __aicore__ inline void ComputeRowPInf(uint32_t i, uint32_t cols)
    {
        // p=inf：仅更新当前 row 的梯度（RowParallel 路径）
        const uint32_t cols_aligned = AlignUpU32(cols, kCompareAlignF32);
        auto xi = in_tile_[i * tile_cols_];
        auto out_i = out_tile_[i * tile_cols_];

        // j < i：pair index 不连续，使用 cache/GM 标量读取
        for (uint32_t j = 0U; j < i; ++j) {
            const uint32_t k = PairStart(j, n_) + (i - j - 1U);
            const float g = GetGrad(k);
            const float d = GetPdist(k);
            auto xj = in_tile_[j * tile_cols_];
            AccumPInfRow(out_i, xi, xj, g, d, cols_aligned);
        }

        // j > i：pair index 连续，允许段预取
        const uint32_t base = PairStart(i, n_);
        const uint32_t seg_len = n_ - i - 1U;
        bool use_grad_seg = false;
        bool use_pdist_seg = false;
        bool need_barrier = false;
        if ((use_grad_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
            LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(base), grad_seg_, seg_len);
            use_grad_seg = true;
            need_barrier = true;
        }
        if ((use_pdist_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
            LoadGmToUb(pdist_gm_, pair_total_, static_cast<uint64_t>(base), pdist_seg_, seg_len);
            use_pdist_seg = true;
            need_barrier = true;
        }
        if (need_barrier) {
            PipeBarrier<PIPE_MTE2>();
        }
        uint32_t k = base;
        uint32_t seg_idx = 0U;
        uint32_t off = (i + 1U) * tile_cols_;
        for (uint32_t j = i + 1U; j < n_; ++j, ++k, ++seg_idx, off += tile_cols_) {
            const float g = (use_grad_cache_ != 0U)
                                ? grad_cache_.GetValue(k)
                                : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                : grad_gm_.GetValue(static_cast<uint64_t>(k)));
            const float d = (use_pdist_cache_ != 0U)
                                ? pdist_cache_.GetValue(k)
                                : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx)
                                                 : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
            auto xj = in_tile_[off];
            AccumPInfRow(out_i, xi, xj, g, d, cols_aligned);
        }
    }

    __aicore__ inline void ComputeRowP2(uint32_t i, uint32_t cols)
    {
        // p=2：out[i] = sum_{j!=i} (grad(k)/dist(k)) * (x_i - x_j)
        const uint32_t cols_aligned = AlignUpU32(cols, kVecWidthF32);
        auto xi = in_tile_[i * tile_cols_];
        auto out_i = out_tile_[i * tile_cols_];
        const uint32_t base = PairStart(i, n_);
        const uint32_t seg_len = n_ - i - 1U;
        bool use_grad_seg = false;
        bool use_pdist_seg = false;
        bool need_barrier = false;
        float sum_scale = 0.0f;
        if ((use_grad_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
            LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(base), grad_seg_, seg_len);
            use_grad_seg = true;
            need_barrier = true;
        }
        if ((use_pdist_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
            LoadGmToUb(pdist_gm_, pair_total_, static_cast<uint64_t>(base), pdist_seg_, seg_len);
            use_pdist_seg = true;
            need_barrier = true;
        }

        // j < i：k = PairStart(j) + (i-j-1)
        uint32_t off = 0U;
        uint32_t k = (i == 0U) ? 0U : (i - 1U);
        for (uint32_t j = 0U; j < i; ++j) {
            const float g = GetGrad(k);
            const float d = GetPdist(k);
            if (d != 0.0f) {
                const float scale = g / d;
                auto xj = in_tile_[off];
                sum_scale += scale;
                Axpy(out_i, xj, scale, static_cast<int32_t>(cols_aligned));
            }
            off += tile_cols_;
            k += (n_ - j - 2U);
        }

        // j > i：k = PairStart(i) + (j-i-1)
        if (need_barrier) {
            PipeBarrier<PIPE_MTE2>();
        }
        k = base;
        uint32_t seg_idx = 0U;
        off = (i + 1U) * tile_cols_;
        const uint32_t seg_base = 0U;
        for (uint32_t j = i + 1U; j < n_; ++j, ++k) {
            const float g = (use_grad_cache_ != 0U)
                                ? grad_cache_.GetValue(k)
                                : (use_grad_seg ? grad_seg_.GetValue(seg_base + seg_idx)
                                                : grad_gm_.GetValue(static_cast<uint64_t>(k)));
            const float d = (use_pdist_cache_ != 0U)
                                ? pdist_cache_.GetValue(k)
                                : (use_pdist_seg ? pdist_seg_.GetValue(seg_base + seg_idx)
                                                 : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
            if (d != 0.0f) {
                const float scale = g / d;
                auto xj = in_tile_[off];
                sum_scale += scale;
                Axpy(out_i, xj, scale, static_cast<int32_t>(cols_aligned));
            }
            off += tile_cols_;
            ++seg_idx;
        }
        // out_i = xi * sum_scale - sum_scaled_xj
        Muls(tmp_, xi, sum_scale, static_cast<int32_t>(cols_aligned));
        Sub(out_i, tmp_, out_i, static_cast<int32_t>(cols_aligned));
    }

    __aicore__ inline void ComputeRowP3(uint32_t i, uint32_t cols)
    {
        // p=3：out[i] = sum_{j!=i} (grad(k)/dist^2) * (x_i - x_j) * |x_i - x_j|
        const uint32_t cols_aligned = AlignUpU32(cols, kVecWidthF32);
        auto xi = in_tile_[i * tile_cols_];
        auto out_i = out_tile_[i * tile_cols_];
        const bool use_scale_cache = (p3_scale_ready_ != 0U);

        // j < i：pair index 不连续，使用 cache/GM 标量读取
        for (uint32_t j = 0U; j < i; ++j) {
            const uint32_t k = PairStart(j, n_) + (i - j - 1U);
            float scale = 0.0f;
            if (use_scale_cache) {
                scale = grad_cache_.GetValue(k);
            } else {
                const float g = GetGrad(k);
                const float d = GetPdist(k);
                if (d == 0.0f) {
                    continue;
                }
                scale = g / (d * d);
            }
            auto xj = in_tile_[j * tile_cols_];
            Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
            Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
            Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
            Axpy(out_i, tmp_, scale, static_cast<int32_t>(cols_aligned));
        }

        // j > i：pair index 连续，允许段预取
        const uint32_t base = PairStart(i, n_);
        const uint32_t seg_len = n_ - i - 1U;
        bool use_grad_seg = false;
        bool use_pdist_seg = false;
        bool need_barrier = false;
        if (!use_scale_cache) {
            if ((use_grad_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(base), grad_seg_, seg_len);
                use_grad_seg = true;
                need_barrier = true;
            }
            if ((use_pdist_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                LoadGmToUb(pdist_gm_, pair_total_, static_cast<uint64_t>(base), pdist_seg_, seg_len);
                use_pdist_seg = true;
                need_barrier = true;
            }
            if (need_barrier) {
                PipeBarrier<PIPE_MTE2>();
            }
        }
        uint32_t k = base;
        uint32_t seg_idx = 0U;
        uint32_t off = (i + 1U) * tile_cols_;
        if (use_scale_cache) {
            for (uint32_t j = i + 1U; j < n_; ++j, ++k, off += tile_cols_) {
                const float scale = grad_cache_.GetValue(k);
                auto xj = in_tile_[off];
                Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
                Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                Axpy(out_i, tmp_, scale, static_cast<int32_t>(cols_aligned));
            }
        } else {
            for (uint32_t j = i + 1U; j < n_; ++j, ++k, ++seg_idx, off += tile_cols_) {
                const float g = (use_grad_cache_ != 0U)
                                    ? grad_cache_.GetValue(k)
                                    : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                    : grad_gm_.GetValue(static_cast<uint64_t>(k)));
                const float d = (use_pdist_cache_ != 0U)
                                    ? pdist_cache_.GetValue(k)
                                    : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx)
                                                     : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
                if (d == 0.0f) {
                    continue;
                }
                const float scale = g / (d * d);
                auto xj = in_tile_[off];
                Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
                Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
                Mul(tmp_, diff_, abs_, static_cast<int32_t>(cols_aligned));
                Axpy(out_i, tmp_, scale, static_cast<int32_t>(cols_aligned));
            }
        }
    }

    __aicore__ inline void ComputeP1Sign(uint32_t cols)
    {
        const uint32_t cols_aligned = AlignUpU32(cols, kVecWidthF32);
        Sign<float, false>(tmp_, diff_, sign_tmp_, cols_aligned);
    }



    __aicore__ inline void AccumPInfPair(const LocalTensor<float> &out_i, const LocalTensor<float> &out_j,
                                         const LocalTensor<float> &xi, const LocalTensor<float> &xj, float g, float d,
                                         uint32_t cols)
    {
        // p=inf：仅在达到 max(|diff|) 的位置传梯度
        if (d == 0.0f) {
            return;
        }
        const uint32_t cols_aligned = AlignUpU32(cols, kCompareAlignF32);
        Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
        CompareScalar(mask_, abs_, d, CMPMODE::EQ, cols_aligned);
        const float scale = g / d;
        Muls(tmp_, diff_, scale, static_cast<int32_t>(cols_aligned));
        Select(tmp_, mask_, tmp_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, cols_aligned);
        Add(out_i, out_i, tmp_, cols_aligned);
        Sub(out_j, out_j, tmp_, cols_aligned);
    }

    __aicore__ inline void AccumPInfRow(const LocalTensor<float> &out_i, const LocalTensor<float> &xi,
                                        const LocalTensor<float> &xj, float g, float d, uint32_t cols)
    {
        // p=inf：仅更新当前 row 的梯度（RowParallel 路径）
        if (d == 0.0f) {
            return;
        }
        const uint32_t cols_aligned = AlignUpU32(cols, kCompareAlignF32);
        Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
        Abs(abs_, diff_, static_cast<int32_t>(cols_aligned));
        CompareScalar(mask_, abs_, d, CMPMODE::EQ, cols_aligned);
        const float scale = g / d;
        Muls(tmp_, diff_, scale, static_cast<int32_t>(cols_aligned));
        Select(tmp_, mask_, tmp_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, cols_aligned);
        Add(out_i, out_i, tmp_, cols_aligned);
    }


    __aicore__ inline uint32_t PairRowFromK(uint64_t k) const
    {
        uint32_t i = 0U;
        uint32_t row_len = (n_ > 0U) ? (n_ - 1U) : 0U;
        while ((row_len != 0U) && (k >= static_cast<uint64_t>(row_len))) {
            k -= static_cast<uint64_t>(row_len);
            ++i;
            if ((i + 1U) >= n_) {
                return n_;
            }
            row_len = n_ - i - 1U;
        }
        return i;
    }

    __aicore__ inline void PairIndexToIJ(uint64_t k, uint32_t &i, uint32_t &j, uint32_t &row_rem) const
    {
        i = 0U;
        j = 0U;
        row_rem = 0U;
        uint32_t row_len = (n_ > 0U) ? (n_ - 1U) : 0U;
        while ((row_len != 0U) && (k >= static_cast<uint64_t>(row_len))) {
            k -= static_cast<uint64_t>(row_len);
            ++i;
            if ((i + 1U) >= n_) {
                row_len = 0U;
                break;
            }
            row_len = n_ - i - 1U;
        }
        if ((row_len == 0U) || ((i + 1U) >= n_)) {
            i = n_;
            j = n_;
            row_rem = 0U;
            return;
        }
        j = i + 1U + static_cast<uint32_t>(k);
        row_rem = row_len - static_cast<uint32_t>(k);
    }

    __aicore__ inline void ComputeRowRangeFromKRange(uint64_t k_start, uint64_t k_end, uint32_t &row_start,
                                                     uint32_t &row_end) const
    {
        row_start = (k_start < pair_total_) ? PairRowFromK(k_start) : n_;
        row_end = row_start;
        if (k_start < k_end) {
            // row_end 取“本段 k 覆盖的最大 j”（只看起始行即可）
            uint32_t i0 = 0U;
            uint32_t j0 = 0U;
            uint32_t row_rem = 0U;
            PairIndexToIJ(k_start, i0, j0, row_rem);
            const uint64_t span = k_end - k_start;
            if (span <= static_cast<uint64_t>(row_rem)) {
                const uint32_t j_max = j0 + static_cast<uint32_t>(span - 1ULL);
                row_end = j_max + 1U;
            } else {
                row_end = n_;
            }
            if (row_end > n_) {
                row_end = n_;
            }
            if (row_end < row_start) {
                row_end = row_start;
            }
        }
    }

    __aicore__ inline void GetPairRange(uint32_t core_id, uint32_t core_dim, uint64_t &k_start,
                                        uint64_t &k_end) const
    {
        if (core_dim == 0U) {
            core_dim = 1U;
        }
        const uint64_t total = pair_total_;
        if (total == 0ULL) {
            k_start = 0ULL;
            k_end = 0ULL;
            return;
        }
        const uint64_t k_per_core =
            (total + static_cast<uint64_t>(core_dim) - 1ULL) / static_cast<uint64_t>(core_dim);
        k_start = k_per_core * static_cast<uint64_t>(core_id);
        if (k_start >= total) {
            k_end = k_start;
            return;
        }
        k_end = k_start + k_per_core;
        if (k_end > total) {
            k_end = total;
        }
    }

    __aicore__ inline void ComputeP1StridedImpl(uint32_t core_id, uint32_t core_dim, uint32_t cols)
    {
        // 按 k 连续分摊 pair（k 为 pair index）：减少 per-i 小 DMA 次数
        uint64_t k_start = 0ULL;
        uint64_t k_end = 0ULL;
        GetPairRange(core_id, core_dim, k_start, k_end);
        if (k_start >= k_end || n_ < 2U) {
            return;
        }
        const uint32_t cols_aligned = AlignUpU32(cols, kVecWidthF32);
        uint32_t i = 0U;
        uint32_t j = 0U;
        uint32_t row_rem = 0U;
        PairIndexToIJ(k_start, i, j, row_rem);
        if (row_rem == 0U) {
            return;
        }
        uint64_t k = k_start;
        while (k < k_end) {
            uint32_t run = row_rem;
            const uint64_t remain = k_end - k;
            if (run > remain) {
                run = static_cast<uint32_t>(remain);
            }
            uint32_t processed = 0U;
            auto xi = in_tile_[i * tile_cols_];
            auto out_i = out_tile_[i * tile_cols_];
            uint32_t off = j * tile_cols_;
            while (processed < run) {
                uint32_t seg = run - processed;
                if (seg > grad_seg_elems_) {
                    seg = grad_seg_elems_;
                }
                if (use_grad_cache_ == 0U && seg != 0U) {
                    LoadGmToUb(grad_gm_, pair_total_, k, grad_seg_, seg);
                    PipeBarrier<PIPE_MTE2>();
                }
                // 2-way unroll：降低标量循环与分支开销
                uint32_t t = 0U;
                for (; (t + 1U) < seg; t += 2U) {
                    const float g0 =
                        (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k) : grad_seg_.GetValue(t);
                    const float g1 =
                        (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k + 1ULL) : grad_seg_.GetValue(t + 1U);
                    auto xj0 = in_tile_[off];
                    auto out_j0 = out_tile_[off];
                    Sub(diff_, xi, xj0, static_cast<int32_t>(cols_aligned));
                    ComputeP1Sign(cols_aligned);
                    Axpy(out_i, tmp_, g0, static_cast<int32_t>(cols_aligned));
                    Axpy(out_j0, tmp_, -g0, static_cast<int32_t>(cols_aligned));

                    const uint32_t off1 = off + tile_cols_;
                    auto xj1 = in_tile_[off1];
                    auto out_j1 = out_tile_[off1];
                    Sub(diff_, xi, xj1, static_cast<int32_t>(cols_aligned));
                    ComputeP1Sign(cols_aligned);
                    Axpy(out_i, tmp_, g1, static_cast<int32_t>(cols_aligned));
                    Axpy(out_j1, tmp_, -g1, static_cast<int32_t>(cols_aligned));

                    k += 2ULL;
                    processed += 2U;
                    j += 2U;
                    off += (tile_cols_ * 2U);
                }
                for (; t < seg; ++t, ++k, ++processed, ++j, off += tile_cols_) {
                    const float g =
                        (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k) : grad_seg_.GetValue(t);
                    auto xj = in_tile_[off];
                    auto out_j = out_tile_[off];
                    Sub(diff_, xi, xj, static_cast<int32_t>(cols_aligned));
                    ComputeP1Sign(cols_aligned);
                    Axpy(out_i, tmp_, g, static_cast<int32_t>(cols_aligned));
                    Axpy(out_j, tmp_, -g, static_cast<int32_t>(cols_aligned));
                }
            }
            if (run == row_rem) {
                ++i;
                if ((i + 1U) >= n_) {
                    return;
                }
                j = i + 1U;
                row_rem = n_ - i - 1U;
            } else {
                row_rem -= run;
            }
        }
    }

    __aicore__ inline void ComputeP1Strided(uint32_t core_id, uint32_t core_dim, uint32_t cols)
    {
        ComputeP1StridedImpl(core_id, core_dim, cols);
    }

    __aicore__ inline void ComputePInfStrided(uint32_t core_id, uint32_t core_dim, uint32_t cols)
    {
        // inf-norm：按 k 连续分摊 pair，减少 per-i 小 DMA 次数
        uint64_t k_start = 0ULL;
        uint64_t k_end = 0ULL;
        GetPairRange(core_id, core_dim, k_start, k_end);
        if (k_start >= k_end || n_ < 2U) {
            return;
        }
        const uint32_t cols_aligned = AlignUpU32(cols, kCompareAlignF32);
        uint32_t i = 0U;
        uint32_t j = 0U;
        uint32_t row_rem = 0U;
        PairIndexToIJ(k_start, i, j, row_rem);
        if (row_rem == 0U) {
            return;
        }
        uint64_t k = k_start;
        while (k < k_end) {
            uint32_t run = row_rem;
            const uint64_t remain = k_end - k;
            if (run > remain) {
                run = static_cast<uint32_t>(remain);
            }
            uint32_t processed = 0U;
            auto xi = in_tile_[i * tile_cols_];
            auto out_i = out_tile_[i * tile_cols_];
            uint32_t off = j * tile_cols_;
            while (processed < run) {
                uint32_t seg = run - processed;
                if (seg > grad_seg_elems_) {
                    seg = grad_seg_elems_;
                }
                if (use_grad_cache_ == 0U && seg != 0U) {
                    LoadGmToUb(grad_gm_, pair_total_, k, grad_seg_, seg);
                }
                if (use_pdist_cache_ == 0U && seg != 0U) {
                    LoadGmToUb(pdist_gm_, pair_total_, k, pdist_seg_, seg);
                }
                if (((use_grad_cache_ == 0U) || (use_pdist_cache_ == 0U)) && seg != 0U) {
                    PipeBarrier<PIPE_MTE2>();
                }
                // 2-way unroll：降低标量循环与分支开销
                uint32_t t = 0U;
                for (; (t + 1U) < seg; t += 2U) {
                    const float g0 =
                        (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k) : grad_seg_.GetValue(t);
                    const float d0 =
                        (use_pdist_cache_ != 0U) ? pdist_cache_.GetValue(k) : pdist_seg_.GetValue(t);
                    auto xj0 = in_tile_[off];
                    auto out_j0 = out_tile_[off];
                    AccumPInfPair(out_i, out_j0, xi, xj0, g0, d0, cols_aligned);

                    const uint32_t off1 = off + tile_cols_;
                    const float g1 =
                        (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k + 1ULL) : grad_seg_.GetValue(t + 1U);
                    const float d1 =
                        (use_pdist_cache_ != 0U) ? pdist_cache_.GetValue(k + 1ULL) : pdist_seg_.GetValue(t + 1U);
                    auto xj1 = in_tile_[off1];
                    auto out_j1 = out_tile_[off1];
                    AccumPInfPair(out_i, out_j1, xi, xj1, g1, d1, cols_aligned);

                    k += 2ULL;
                    processed += 2U;
                    j += 2U;
                    off += (tile_cols_ * 2U);
                }
                for (; t < seg; ++t, ++k, ++processed, ++j, off += tile_cols_) {
                    const float g =
                        (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k) : grad_seg_.GetValue(t);
                    const float d =
                        (use_pdist_cache_ != 0U) ? pdist_cache_.GetValue(k) : pdist_seg_.GetValue(t);
                    auto xj = in_tile_[off];
                    auto out_j = out_tile_[off];
                    AccumPInfPair(out_i, out_j, xi, xj, g, d, cols_aligned);
                }
            }
            if (run == row_rem) {
                ++i;
                if ((i + 1U) >= n_) {
                    return;
                }
                j = i + 1U;
                row_rem = n_ - i - 1U;
            } else {
                row_rem -= run;
            }
        }
    }

    __aicore__ inline void ComputeP2StridedSlice(uint32_t core_id, uint32_t core_dim, uint32_t col_off,
                                                 uint32_t cols)
    {
        if (cols == 0U) {
            return;
        }
        uint64_t k_start = 0ULL;
        uint64_t k_end = 0ULL;
        GetPairRange(core_id, core_dim, k_start, k_end);
        if (k_start >= k_end || n_ < 2U) {
            return;
        }
        uint32_t i = 0U;
        uint32_t j = 0U;
        uint32_t row_rem = 0U;
        PairIndexToIJ(k_start, i, j, row_rem);
        if (row_rem == 0U) {
            return;
        }
        uint64_t k = k_start;
        while (k < k_end) {
            uint32_t run = row_rem;
            const uint64_t remain = k_end - k;
            if (run > remain) {
                run = static_cast<uint32_t>(remain);
            }
            uint32_t processed = 0U;
            auto xi = in_tile_[i * tile_cols_ + col_off];
            auto out_i = out_tile_[i * tile_cols_ + col_off];
            uint32_t off = j * tile_cols_ + col_off;
            while (processed < run) {
                uint32_t seg = run - processed;
                if (seg > grad_seg_elems_) {
                    seg = grad_seg_elems_;
                }
                if (use_grad_cache_ == 0U && seg != 0U) {
                    LoadGmToUb(grad_gm_, pair_total_, k, grad_seg_, seg);
                }
                if (use_pdist_cache_ == 0U && seg != 0U) {
                    LoadGmToUb(pdist_gm_, pair_total_, k, pdist_seg_, seg);
                }
                if (((use_grad_cache_ == 0U) || (use_pdist_cache_ == 0U)) && seg != 0U) {
                    PipeBarrier<PIPE_MTE2>();
                }
                uint32_t t = 0U;
                for (; (t + 1U) < seg; t += 2U) {
                    const float g0 =
                        (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k) : grad_seg_.GetValue(t);
                    const float d0 =
                        (use_pdist_cache_ != 0U) ? pdist_cache_.GetValue(k) : pdist_seg_.GetValue(t);
                    if (d0 != 0.0f) {
                        const float scale0 = g0 / d0;
                        auto xj0 = in_tile_[off];
                        auto out_j0 = out_tile_[off];
                        Sub(diff_, xi, xj0, static_cast<int32_t>(cols));
                        Axpy(out_i, diff_, scale0, static_cast<int32_t>(cols));
                        Axpy(out_j0, diff_, -scale0, static_cast<int32_t>(cols));
                    }

                    const uint32_t off1 = off + tile_cols_;
                    const float g1 =
                        (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k + 1ULL) : grad_seg_.GetValue(t + 1U);
                    const float d1 =
                        (use_pdist_cache_ != 0U) ? pdist_cache_.GetValue(k + 1ULL) : pdist_seg_.GetValue(t + 1U);
                    if (d1 != 0.0f) {
                        const float scale1 = g1 / d1;
                        auto xj1 = in_tile_[off1];
                        auto out_j1 = out_tile_[off1];
                        Sub(diff_, xi, xj1, static_cast<int32_t>(cols));
                        Axpy(out_i, diff_, scale1, static_cast<int32_t>(cols));
                        Axpy(out_j1, diff_, -scale1, static_cast<int32_t>(cols));
                    }

                    k += 2ULL;
                    processed += 2U;
                    j += 2U;
                    off += (tile_cols_ * 2U);
                }
                for (; t < seg; ++t, ++k, ++processed, ++j, off += tile_cols_) {
                    const float g =
                        (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k) : grad_seg_.GetValue(t);
                    const float d =
                        (use_pdist_cache_ != 0U) ? pdist_cache_.GetValue(k) : pdist_seg_.GetValue(t);
                    if (d != 0.0f) {
                        const float scale = g / d;
                        auto xj = in_tile_[off];
                        auto out_j = out_tile_[off];
                        Sub(diff_, xi, xj, static_cast<int32_t>(cols));
                        Axpy(out_i, diff_, scale, static_cast<int32_t>(cols));
                        Axpy(out_j, diff_, -scale, static_cast<int32_t>(cols));
                    }
                }
            }
            if (run == row_rem) {
                ++i;
                if ((i + 1U) >= n_) {
                    return;
                }
                j = i + 1U;
                row_rem = n_ - i - 1U;
            } else {
                row_rem -= run;
            }
        }
    }


    __aicore__ inline void ComputeP2()
	    {
        for (uint32_t i = 0U; i + 1U < n_; ++i) {
            const uint32_t base = PairStart(i, n_);
            auto xi = in_tile_[i * tile_cols_];
            auto out_i = out_tile_[i * tile_cols_];
            const uint32_t seg_len = n_ - i - 1U;
            const bool use_grad_seg = (use_grad_cache_ == 0U) && (seg_len >= kRowPrefetchMin);
            const bool use_pdist_seg = (use_pdist_cache_ == 0U) && (seg_len >= kRowPrefetchMin);
            if (use_grad_seg) {
                LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(base), grad_seg_, seg_len);
            }
            if (use_pdist_seg) {
                LoadGmToUb(pdist_gm_, pair_total_, static_cast<uint64_t>(base), pdist_seg_, seg_len);
            }
            if (use_grad_seg || use_pdist_seg) {
                PipeBarrier<PIPE_MTE2>();
            }
            uint32_t k = base;
            uint32_t seg_idx = 0U;
            uint32_t off = (i + 1U) * tile_cols_;
            for (uint32_t j = i + 1U; j < n_; ++j, ++k) {
                const float g = (use_grad_cache_ != 0U)
                                    ? grad_cache_.GetValue(k)
                                    : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                    : grad_gm_.GetValue(static_cast<uint64_t>(k)));
                const float d = (use_pdist_cache_ != 0U)
                                    ? pdist_cache_.GetValue(k)
                                    : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx)
                                                     : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
                if (d != 0.0f) {
                    const float scale = g / d;
                    auto xj = in_tile_[off];
                    auto out_j = out_tile_[off];
                    Sub(diff_, xi, xj, static_cast<int32_t>(tile_cols_));
                    Axpy(out_i, diff_, scale, static_cast<int32_t>(tile_cols_));
                    Axpy(out_j, diff_, -scale, static_cast<int32_t>(tile_cols_));
                }
                off += tile_cols_;
                ++seg_idx;
            }
        }
    }


    __aicore__ inline void ComputePLessThan2()
    {
        // 0<p<2 且 p!=1：
        // PyTorch(lttdist_calc) 等价形式：
        //   sign(diff) * |diff|^(p-1) * grad / dist^(p-1)
        //
        // 说明：
        // - 直接按 sign(diff)*|diff|^(p-1) 计算（而不是 diff*|diff|^(p-2)），可避免 |diff| 极小时先算 |diff|^(p-2) 溢出为 inf；
        // - 将 |diff| 中的 0 替换为 1，避免 0^负数 -> inf，进而 0*inf 产生 NaN（对齐 PyTorch 行为：该位置梯度为 0）；
        // - 合并指数项，计算 sign(diff) * grad * exp((p-1) * (ln(|diff|) - ln(dist)))。
        const float exp_abs = p_ - 1.0f; // (p-1)

        for (uint32_t i = 0U; i + 1U < n_; ++i) {
            const uint32_t base = PairStart(i, n_);
            auto xi = in_tile_[i * tile_cols_];
            auto out_i = out_tile_[i * tile_cols_];
            const uint32_t seg_len = n_ - i - 1U;
            bool use_grad_seg = false;
            bool use_pdist_seg = false;
            bool need_barrier = false;
            if ((use_grad_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(base), grad_seg_, seg_len);
                use_grad_seg = true;
                need_barrier = true;
            }
            if ((use_pdist_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                LoadGmToUb(pdist_gm_, pair_total_, static_cast<uint64_t>(base), pdist_seg_, seg_len);
                use_pdist_seg = true;
                need_barrier = true;
            }
            if (need_barrier) {
                PipeBarrier<PIPE_MTE2>();
            }
            uint32_t k = base;
            uint32_t seg_idx = 0U;
            uint32_t off = (i + 1U) * tile_cols_;
            for (uint32_t j = i + 1U; j < n_; ++j, ++k, ++seg_idx, off += tile_cols_) {
                const float g = (use_grad_cache_ != 0U)
                                    ? grad_cache_.GetValue(k)
                                    : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                    : grad_gm_.GetValue(static_cast<uint64_t>(k)));
                const float d = (use_pdist_cache_ != 0U)
                                    ? pdist_cache_.GetValue(k)
                                    : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx)
                                                     : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
                if (d == 0.0f) {
                    continue;
                }
                auto xj = in_tile_[off];
                auto out_j = out_tile_[off];

                Sub(diff_, xi, xj, static_cast<int32_t>(tile_cols_));

                // tmp_ = sign(diff)：-1/0/1
                Sign<float, false>(tmp_, diff_, sign_tmp_, tile_cols_);

                // abs_ = (|diff| / dist)^(p-1)
                // 说明：
                // - p<1 时 (p-1)<0，且 |diff|/dist 往往非常小，会被放大为非常大的数；再叠加多对(pair)累加时的“消除”，对精度极其敏感；
                // - 这里对 p<1 使用更高精度的 log2/exp2 近似（更高阶多项式），尽量对齐 PyTorch CPU 参考实现；p>=1 继续使用 Ln/Exp（更快）。
                Abs(abs_, diff_, static_cast<int32_t>(tile_cols_));
                CompareScalar(mask_, abs_, 0.0f, CMPMODE::NE, tile_cols_);
                Select(abs_, mask_, abs_, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, tile_cols_);
                // 先用标量除法求 inv_d，再做 1 次牛顿迭代修正（提升 p<1 场景 ratio 的精度）
                float inv_d = 1.0f / d;
                inv_d = inv_d * (2.0f - d * inv_d);
                Muls(abs_, abs_, inv_d, static_cast<int32_t>(tile_cols_)); // ratio in (0, 1]

                if (p_ < 1.0f) {
                    constexpr float kLn2 = 0.6931471805599453f;
                    constexpr float kInvLn2 = 1.4426950408889634f;          // 1/ln(2)
                    constexpr float kMantissaScale = 1.1920928955078125e-7f; // 2^-23

                    // ---------- log2(ratio) ----------
                    // ratio = m * 2^e, m in [1,2)
                    auto ratio_bits = abs_.ReinterpretCast<int32_t>();
                    auto exp_bits = diff_.ReinterpretCast<int32_t>();
                    auto mant_bits = log_.ReinterpretCast<int32_t>();
                    ShiftRight(exp_bits, ratio_bits, static_cast<int32_t>(23), static_cast<int32_t>(tile_cols_));
                    ShiftLeft(mant_bits, exp_bits, static_cast<int32_t>(23), static_cast<int32_t>(tile_cols_));
                    Sub(mant_bits, ratio_bits, mant_bits, static_cast<int32_t>(tile_cols_));
                    Adds(exp_bits, exp_bits, static_cast<int32_t>(-127), static_cast<int32_t>(tile_cols_)); // e

                    // abs_ = m
                    Cast(abs_, mant_bits, RoundMode::CAST_NONE, static_cast<int32_t>(tile_cols_));
                    Muls(abs_, abs_, kMantissaScale, static_cast<int32_t>(tile_cols_));
                    Adds(abs_, abs_, 1.0f, static_cast<int32_t>(tile_cols_));

                    // t = (m-1)/(m+1)
                    Adds(abs_, abs_, -1.0f, static_cast<int32_t>(tile_cols_)); // abs_=n=m-1
                    Adds(log_, abs_, 2.0f, static_cast<int32_t>(tile_cols_));  // log_=denom=n+2
                    Reciprocal(log_, log_, static_cast<int32_t>(tile_cols_));  // inv0
                    // inv1 = inv0*(2 - denom*inv0)
                    Adds(work_, abs_, 2.0f, static_cast<int32_t>(tile_cols_));
                    Mul(work_, work_, log_, static_cast<int32_t>(tile_cols_));
                    Muls(work_, work_, -1.0f, static_cast<int32_t>(tile_cols_));
                    Adds(work_, work_, 2.0f, static_cast<int32_t>(tile_cols_));
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_)); // inv1
                    // inv2 = inv1*(2 - denom*inv1)（再做 1 次牛顿迭代，进一步提升 t 的精度）
                    Adds(work_, abs_, 2.0f, static_cast<int32_t>(tile_cols_));
                    Mul(work_, work_, log_, static_cast<int32_t>(tile_cols_));
                    Muls(work_, work_, -1.0f, static_cast<int32_t>(tile_cols_));
                    Adds(work_, work_, 2.0f, static_cast<int32_t>(tile_cols_));
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_)); // inv2

                    Mul(abs_, abs_, log_, static_cast<int32_t>(tile_cols_));   // abs_=t
                    Muls(work_, abs_, 1.0f, static_cast<int32_t>(tile_cols_)); // work_=t(const)

                    // ln(m) = 2 * (t + t^3/3 + ... + t^13/13)
                    Muls(log_, abs_, 1.0f, static_cast<int32_t>(tile_cols_)); // log_=term=t
                    // t^3
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Axpy(abs_, log_, 0.3333333333333333f, static_cast<int32_t>(tile_cols_));
                    // t^5
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Axpy(abs_, log_, 0.2f, static_cast<int32_t>(tile_cols_));
                    // t^7
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Axpy(abs_, log_, 0.14285714285714285f, static_cast<int32_t>(tile_cols_));
                    // t^9
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Axpy(abs_, log_, 0.1111111111111111f, static_cast<int32_t>(tile_cols_));
                    // t^11
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Axpy(abs_, log_, 0.09090909090909091f, static_cast<int32_t>(tile_cols_));
                    // t^13
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Mul(log_, log_, work_, static_cast<int32_t>(tile_cols_));
                    Axpy(abs_, log_, 0.07692307692307693f, static_cast<int32_t>(tile_cols_));

                    // abs_ = log2(m) = ln(m)/ln2
                    Muls(abs_, abs_, 2.0f, static_cast<int32_t>(tile_cols_));
                    Muls(abs_, abs_, kInvLn2, static_cast<int32_t>(tile_cols_));

                    // abs_ = log2(ratio) = e + log2(m)
                    Cast(log_, exp_bits, RoundMode::CAST_NONE, static_cast<int32_t>(tile_cols_));
                    Add(abs_, abs_, log_, static_cast<int32_t>(tile_cols_));

                    // ---------- exp2(y) ----------
                    // y = (p-1) * log2(ratio)
                    Muls(abs_, abs_, exp_abs, static_cast<int32_t>(tile_cols_));

                    // i = floor(y), f = y - i
                    Cast(exp_bits, abs_, RoundMode::CAST_FLOOR, static_cast<int32_t>(tile_cols_));
                    Cast(log_, exp_bits, RoundMode::CAST_NONE, static_cast<int32_t>(tile_cols_));
                    Sub(abs_, abs_, log_, static_cast<int32_t>(tile_cols_)); // abs_=f

                    // exp2(f) = exp(f*ln2)，10 阶 Taylor（z in [0, ln2]）
                    Muls(abs_, abs_, kLn2, static_cast<int32_t>(tile_cols_)); // abs_=z
                    Duplicate(log_, 1.0f / 3628800.0f, static_cast<int32_t>(tile_cols_)); // 1/10!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 1.0f / 362880.0f, static_cast<int32_t>(tile_cols_)); // 1/9!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 1.0f / 40320.0f, static_cast<int32_t>(tile_cols_)); // 1/8!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 1.0f / 5040.0f, static_cast<int32_t>(tile_cols_)); // 1/7!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 1.0f / 720.0f, static_cast<int32_t>(tile_cols_)); // 1/6!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 1.0f / 120.0f, static_cast<int32_t>(tile_cols_)); // 1/5!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 1.0f / 24.0f, static_cast<int32_t>(tile_cols_)); // 1/4!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 1.0f / 6.0f, static_cast<int32_t>(tile_cols_)); // 1/3!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 0.5f, static_cast<int32_t>(tile_cols_)); // 1/2!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 1.0f, static_cast<int32_t>(tile_cols_)); // 1/1!
                    Mul(log_, log_, abs_, static_cast<int32_t>(tile_cols_));
                    Adds(log_, log_, 1.0f, static_cast<int32_t>(tile_cols_)); // 1/0!

                    // 2^i：构造 float 指数位（i>=0）
                    Adds(exp_bits, exp_bits, static_cast<int32_t>(127), static_cast<int32_t>(tile_cols_));
                    ShiftLeft(exp_bits, exp_bits, static_cast<int32_t>(23), static_cast<int32_t>(tile_cols_));

                    // abs_ = 2^y = 2^i * 2^f
                    Mul(abs_, diff_, log_, static_cast<int32_t>(tile_cols_));
                    PipeBarrier<PIPE_V>();
                } else {
                    // p in (1,2)：直接用 Ln/Exp（更快）
                    Ln(abs_, abs_, static_cast<int32_t>(tile_cols_));
                    Muls(abs_, abs_, exp_abs, static_cast<int32_t>(tile_cols_));
                    Exp(abs_, abs_, static_cast<int32_t>(tile_cols_));
                    PipeBarrier<PIPE_V>();
                }

                // tmp_ *= (|diff|/dist)^(p-1)，标量 g 直接在 Axpy 中融合
                Mul(tmp_, tmp_, abs_, static_cast<int32_t>(tile_cols_));
                PipeBarrier<PIPE_V>();
                Axpy(out_i, tmp_, g, static_cast<int32_t>(tile_cols_));
                Axpy(out_j, tmp_, -g, static_cast<int32_t>(tile_cols_));
            }
        }
    }


    __aicore__ inline void ComputePGeneral()
    {
        // 通用 p：diff * |diff|^(p-2) * grad / dist^(p-1)
        if (p_ == 3.0f) {
            // p=3：diff * |diff| * grad / dist^2，避免 Ln/Exp
            for (uint32_t i = 0U; i + 1U < n_; ++i) {
                const uint32_t base = PairStart(i, n_);
                auto xi = in_tile_[i * tile_cols_];
                auto out_i = out_tile_[i * tile_cols_];
                const uint32_t seg_len = n_ - i - 1U;
                bool use_grad_seg = false;
                bool use_pdist_seg = false;
                bool need_barrier = false;
                if ((use_grad_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                    LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(base), grad_seg_, seg_len);
                    use_grad_seg = true;
                    need_barrier = true;
                }
                if ((use_pdist_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                    LoadGmToUb(pdist_gm_, pair_total_, static_cast<uint64_t>(base), pdist_seg_, seg_len);
                    use_pdist_seg = true;
                    need_barrier = true;
                }
                if (need_barrier) {
                    PipeBarrier<PIPE_MTE2>();
                }
                uint32_t k = base;
                uint32_t seg_idx = 0U;
                uint32_t off = (i + 1U) * tile_cols_;
                for (uint32_t j = i + 1U; j < n_; ++j, ++k, ++seg_idx, off += tile_cols_) {
                    const float g = (use_grad_cache_ != 0U)
                                        ? grad_cache_.GetValue(k)
                                        : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                        : grad_gm_.GetValue(static_cast<uint64_t>(k)));
                    const float d = (use_pdist_cache_ != 0U)
                                        ? pdist_cache_.GetValue(k)
                                        : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx)
                                                         : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
                    if (d == 0.0f) {
                        continue;
                    }
                    const float scale = g / (d * d);
                    auto xj = in_tile_[off];
                    auto out_j = out_tile_[off];
                    Sub(diff_, xi, xj, static_cast<int32_t>(tile_cols_));
                    Abs(abs_, diff_, static_cast<int32_t>(tile_cols_));
                    Mul(tmp_, diff_, abs_, static_cast<int32_t>(tile_cols_));
                    PipeBarrier<PIPE_V>();
                    Axpy(out_i, tmp_, scale, static_cast<int32_t>(tile_cols_));
                    Axpy(out_j, tmp_, -scale, static_cast<int32_t>(tile_cols_));
                }
            }
            return;
        }
        const float exp_diff = p_ - 2.0f; // |diff|^(p-2)
        const float exp_dist = 1.0f - p_; // dist^(1-p) = 1/dist^(p-1)

        for (uint32_t i = 0U; i + 1U < n_; ++i) {
            const uint32_t base = PairStart(i, n_);
            auto xi = in_tile_[i * tile_cols_];
            auto out_i = out_tile_[i * tile_cols_];
            const uint32_t seg_len = n_ - i - 1U;
            bool use_grad_seg = false;
            bool use_pdist_seg = false;
            bool need_barrier = false;
            if ((use_grad_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                LoadGmToUb(grad_gm_, pair_total_, static_cast<uint64_t>(base), grad_seg_, seg_len);
                use_grad_seg = true;
                need_barrier = true;
            }
            if ((use_pdist_cache_ == 0U) && (seg_len >= kRowPrefetchMin)) {
                LoadGmToUb(pdist_gm_, pair_total_, static_cast<uint64_t>(base), pdist_seg_, seg_len);
                use_pdist_seg = true;
                need_barrier = true;
            }
            if (need_barrier) {
                PipeBarrier<PIPE_MTE2>();
            }
            uint32_t k = base;
            uint32_t seg_idx = 0U;
            uint32_t off = (i + 1U) * tile_cols_;
            for (uint32_t j = i + 1U; j < n_; ++j, ++k, ++seg_idx, off += tile_cols_) {
                const float g = (use_grad_cache_ != 0U)
                                    ? grad_cache_.GetValue(k)
                                    : (use_grad_seg ? grad_seg_.GetValue(seg_idx)
                                                    : grad_gm_.GetValue(static_cast<uint64_t>(k)));
                const float d = (use_pdist_cache_ != 0U)
                                    ? pdist_cache_.GetValue(k)
                                    : (use_pdist_seg ? pdist_seg_.GetValue(seg_idx)
                                                     : pdist_gm_.GetValue(static_cast<uint64_t>(k)));
                if (d == 0.0f) {
                    continue;
                }
                // dist^(1-p)：用短向量计算标量，避免对整 tile 做 Ln/Exp
                Duplicate(tmp_, d, static_cast<int32_t>(kVecWidthF32));
                Ln(tmp_, tmp_, static_cast<int32_t>(kVecWidthF32));
                Muls(tmp_, tmp_, exp_dist, static_cast<int32_t>(kVecWidthF32));
                Exp(tmp_, tmp_, static_cast<int32_t>(kVecWidthF32));
                PipeBarrier<PIPE_V>();
                const float scale = g * tmp_.GetValue(0);
                auto xj = in_tile_[off];
                auto out_j = out_tile_[off];

                Sub(diff_, xi, xj, static_cast<int32_t>(tile_cols_));
                Abs(abs_, diff_, static_cast<int32_t>(tile_cols_));
                // abs_ 中的 0 替换为 1，避免 Ln(0) 以及 0*inf 产生 NaN
                CompareScalar(mask_, abs_, 0.0f, CMPMODE::NE, tile_cols_);
                Select(abs_, mask_, abs_, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, tile_cols_);
                Ln(abs_, abs_, static_cast<int32_t>(tile_cols_));
                Muls(abs_, abs_, exp_diff, static_cast<int32_t>(tile_cols_));
                Exp(abs_, abs_, static_cast<int32_t>(tile_cols_));
                PipeBarrier<PIPE_V>();
                Mul(tmp_, diff_, abs_, static_cast<int32_t>(tile_cols_));

                // 标量 scale 直接在 Axpy 中融合，减少一次向量 Muls
                PipeBarrier<PIPE_V>();
                Axpy(out_i, tmp_, scale, static_cast<int32_t>(tile_cols_));
                Axpy(out_j, tmp_, -scale, static_cast<int32_t>(tile_cols_));
            }
        }
    }

private:
	    __aicore__ inline void EnsurePairCacheLoaded()
	    {
	        if (pair_cache_loaded_ != 0U || use_grad_cache_ == 0U || pair_total_ == 0ULL) {
	            return;
	        }
	        const uint32_t k_total = static_cast<uint32_t>(pair_total_);
	        LoadGmToUb(grad_gm_, pair_total_, 0ULL, grad_cache_, k_total);
	        if (use_pdist_cache_ != 0U) {
	            LoadGmToUb(pdist_gm_, pair_total_, 0ULL, pdist_cache_, k_total);
	        }
	        PipeBarrier<PIPE_ALL>();
	        pair_cache_loaded_ = 1U;
	    }

        __aicore__ inline void PrepareP3ScaleCache()
        {
            if (p3_scale_ready_ != 0U) {
                return;
            }
            if (!((p_type_ == 4U) && (p_ == 3.0f))) {
                return;
            }
            if ((use_grad_cache_ == 0U) || (use_pdist_cache_ == 0U) || (pair_total_ == 0ULL)) {
                return;
            }
            if (pair_cache_loaded_ == 0U) {
                EnsurePairCacheLoaded();
            }
            const uint32_t total = static_cast<uint32_t>(pair_total_);
            // CompareScalar 对 float32 的 calCount 需 64 对齐，这里用 64 为 chunk
            constexpr uint32_t kP3ScaleChunk = kCompareAlignF32;
            uint32_t off = 0U;
            for (; (off + kP3ScaleChunk) <= total; off += kP3ScaleChunk) {
                auto g_chunk = grad_cache_[off];
                auto d_chunk = pdist_cache_[off];
                // d = d * d
                Mul(d_chunk, d_chunk, d_chunk, static_cast<int32_t>(kP3ScaleChunk));
                // g = g / d
                Div(g_chunk, g_chunk, d_chunk, static_cast<int32_t>(kP3ScaleChunk));
                // dist==0 -> scale=0
                CompareScalar(mask_, d_chunk, 0.0f, CMPMODE::NE, kP3ScaleChunk);
                Select(g_chunk, mask_, g_chunk, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, kP3ScaleChunk);
                PipeBarrier<PIPE_V>();
            }
            // tail：用标量处理，避免 CompareScalar 对齐约束
            for (; off < total; ++off) {
                const float d = pdist_cache_.GetValue(off);
                if (d != 0.0f) {
                    const float g = grad_cache_.GetValue(off);
                    grad_cache_.SetValue(off, g / (d * d));
                    pdist_cache_.SetValue(off, d * d);
                } else {
                    grad_cache_.SetValue(off, 0.0f);
                }
            }
            PipeBarrier<PIPE_ALL>();
            p3_scale_ready_ = 1U;
        }

        template <bool kUseCache>
        __aicore__ inline float GetGradRange(uint32_t k, uint32_t idx) const
        {
            if constexpr (kUseCache) {
                return grad_cache_.GetValue(k);
            } else {
                return grad_seg_.GetValue(idx);
            }
        }

        template <bool kUseCache>
        __aicore__ inline float GetPdistRange(uint32_t k, uint32_t idx) const
        {
            if constexpr (kUseCache) {
                return pdist_cache_.GetValue(k);
            } else {
                return pdist_seg_.GetValue(idx);
            }
        }

	    __aicore__ inline float GetGrad(uint32_t k) const
	    {
	        return (use_grad_cache_ != 0U) ? grad_cache_.GetValue(k) : grad_gm_.GetValue(static_cast<uint64_t>(k));
	    }

	    __aicore__ inline float GetPdist(uint32_t k) const
	    {
	        return (use_pdist_cache_ != 0U) ? pdist_cache_.GetValue(k) : pdist_gm_.GetValue(static_cast<uint64_t>(k));
	    }

    TPipe pipe_;
    TBuf<QuePosition::VECIN> buf_in_;
    TBuf<QuePosition::VECOUT> buf_out_;
    TBuf<QuePosition::VECCALC> buf_diff_;
    TBuf<QuePosition::VECCALC> buf_abs_;
    TBuf<QuePosition::VECCALC> buf_tmp_;
    TBuf<QuePosition::VECCALC> buf_diff_blk_;
    TBuf<QuePosition::VECCALC> buf_abs_blk_;
    TBuf<QuePosition::VECCALC> buf_tmp_blk_;
    TBuf<QuePosition::VECCALC> buf_log_;
    TBuf<QuePosition::VECCALC> buf_work_;
    TBuf<QuePosition::VECCALC> buf_mask_;
    TBuf<QuePosition::VECCALC> buf_sign_tmp_;
	    // baseline：小 N 场景把 grad/pdist 预取到 UB，降低 GM 标量访存开销
	    TBuf<QuePosition::VECCALC> buf_grad_cache_;
	    TBuf<QuePosition::VECCALC> buf_pdist_cache_;
	    // matmul 分支：Laplace UB tile buffer（rows * n_）
	    TBuf<QuePosition::VECCALC> buf_a_tile_;
	    // matmul 分支（仅 builder）：Laplace 权重 w=grad/dist 的 UB 缓冲 + 临时 dist/mask
	    TBuf<QuePosition::VECCALC> buf_lap_w_;
	    TBuf<QuePosition::VECCALC> buf_lap_d_;
	    TBuf<QuePosition::VECCALC> buf_lap_mask_;
	    // pair reduce/row 路径：grad/pdist 分段预取
		    TBuf<QuePosition::VECCALC> buf_grad_seg_;
		    TBuf<QuePosition::VECCALC> buf_pdist_seg_;

    GlobalTensor<float> grad_gm_;
    GlobalTensor<float> input_gm_;
    GlobalTensor<float> pdist_gm_;
    GlobalTensor<float> out_gm_;

    LocalTensor<float> in_tile_;
    LocalTensor<float> out_tile_;
    LocalTensor<float> diff_;
    LocalTensor<float> abs_;
    LocalTensor<float> tmp_;
    LocalTensor<float> diff_blk_;
    LocalTensor<float> abs_blk_;
    LocalTensor<float> tmp_blk_;
    LocalTensor<float> log_;
    LocalTensor<float> work_;
            LocalTensor<uint8_t> mask_;
            LocalTensor<uint8_t> sign_tmp_;
			    LocalTensor<float> grad_cache_;
			    LocalTensor<float> pdist_cache_;
	    LocalTensor<float> a_tile_f32_;
	    LocalTensor<float> lap_w_;
	    LocalTensor<float> lap_d_;
	    LocalTensor<uint8_t> lap_mask_;
		    LocalTensor<float> grad_seg_;
		    LocalTensor<float> pdist_seg_;

	    uint32_t n_ = 0U;
	    uint32_t m_ = 0U;
    // 32B 对齐的 row-block（由 m 决定）：用于 DataCopyPad 写回避免跨行 RMW 竞争
    uint32_t row_block_ = 1U;
    uint32_t tile_cols_ = kTileColsDefault;
	    uint32_t total_tiles_ = 0U;
    uint32_t p_type_ = 0U;
    uint32_t block_dim_ = 1U;
    uint32_t sys_ws_size_ = 0U;
    uint32_t p1pinf_ws_bytes_ = 0U;
    uint32_t m_padded_ = 0U;
    uint32_t use_p1pinf_ws_ = 0U;
    // baseline(AIV) 逻辑 block 信息（MIX_AIC_1_2：block_dim = block group 数，AIV 并行度=block_dim*2）
    uint32_t aiv_block_idx_ = 0U;
    uint32_t aiv_block_dim_ = 1U;
    uint32_t aiv_effective_dim_ = 1U;
    float p_ = 2.0f;
    uint32_t need_log_work_ = 0U;
    uint32_t use_matmul_ = 0U;
    uint32_t use_pair_reduce_ = 0U;
    GM_ADDR workspace_ = 0;
    TCubeTiling cube_tiling_ = {};

	    uint32_t start_tile_ = 0U;
	    uint32_t tile_count_ = 0U;
	    uint64_t input_total_ = 0ULL;
	    uint64_t pair_total_ = 0ULL;
	    // PairReduce(workspace) 预取 buffer 的实际元素数（Init 时按 n_ 与 ceil(pair_total/block_dim) 取最大，并做上限裁剪）
	    uint32_t grad_seg_elems_ = 0U;
	    uint32_t pdist_seg_elems_ = 0U;
    uint32_t use_grad_cache_ = 0U;
    uint32_t use_pdist_cache_ = 0U;
    uint32_t pair_cache_loaded_ = 0U;
    uint32_t p3_scale_ready_ = 0U;
    uint32_t use_p3_vecblk_ = 0U;
};

extern "C" __global__ __aicore__ void pdist_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR pdist, GM_ADDR out,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    // 启用 AIC/AIV 混合执行：baseline 在 AIV，上大 shape(p=2) matmul 快路径在 AIC
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);
    // workspace 指针语义说明（非常重要）：
    // - AscendC 内置高阶 API（如 SyncAll/DataCopyPad/Matmul 注册）依赖系统预留 workspace（GetSysWorkSpacePtr）；
    // - 部分调用链会在 kernel launch 前设置好 sys workspace 基址；workspace 入参可能并非 sys workspace 基址（可能是 user workspace 起点）；
    // - 若这里强制 SetSysWorkspaceForce(workspace)，可能覆盖运行时已设置的 sys workspace，导致内部 API 使用错误 workspace，
    //   表现为结果随机错/不稳定，甚至触发 507057（MTE out-of-range）。
    // 因此：仅当运行时尚未设置 sys workspace（GetSysWorkSpacePtr()==nullptr）时，才用入参 workspace 兜底设置。
    if (workspace != nullptr && GetSysWorkSpacePtr() == nullptr) {
        SetSysWorkspaceForce(workspace);
    }
    PdistGradKernel op;
    op.Init(grad, input, pdist, out, workspace, tiling_data);
    op.Process();
}
