#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PdistGradTilingData)
  // 输入 input 的二维 shape: [N, M]
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  // 列 tile 大小（必须是 8 的倍数，且 <=256）：
  // - p=1/p=2/p=inf 在小 N 场景允许放大到 128/256（仍满足 CompareScalar 的 64 对齐约束）
  // - 其它 p（0<p<2）仍固定为 64；p>=3 允许放大（便于降低 total_tiles）
  TILING_DATA_FIELD_DEF(uint32_t, tile_cols);
  // p 类型编码：
  // 0(p=0), 1(p=1), 2(p=2), 3(p=inf),
  // 4(有限 p>2), 5(有限 0<p<2 且 p!=1)
  TILING_DATA_FIELD_DEF(uint32_t, p_type);
  // 实际使用的 blockDim（<= AI Core 数 && <= total_tiles）
  TILING_DATA_FIELD_DEF(uint32_t, block_dim);
  // system workspace 大小（bytes，256B 对齐），用于 kernel 侧确定性切分 system/user workspace
  TILING_DATA_FIELD_DEF(uint32_t, sys_ws_size);
  // p=1/p=inf 对齐 workspace 大小（bytes，256B 对齐）；为 0 表示禁用
  TILING_DATA_FIELD_DEF(uint32_t, p1pinf_ws_bytes);
  // p 数值（p_type=4/5 需要；其它类型可忽略）
  TILING_DATA_FIELD_DEF(float, p_value);

  // Cube Matmul 快路径开关：
  // - 仅在 p==2 且 shape 足够大时启用（host 侧判定），用于显著优化大 case 的性能。
  TILING_DATA_FIELD_DEF(uint32_t, use_matmul);

  // pair reduce / RowParallel 开关：
  // - p=1/p=inf：原子写回的 pair reduce；
  // - p=2：total_tiles 很小（例如 m≈1~2 个 tile）时启用 RowParallel。
  // - 仅在非 matmul 分支下启用（use_matmul=0）。
  TILING_DATA_FIELD_DEF(uint32_t, use_pair_reduce);

  // Matmul tiling（仅 use_matmul=1 时有效）
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PdistGrad, PdistGradTilingData)
}
