# MSD 模板深度解析（Multi-Scale Dequant）

> 论文：《Multi-Scale Dequant: Eliminating Dequantization Bottleneck via Activation Decomposition for Efficient LLM Inference》arXiv:2605.13915
> 实现：CANN ops-nn 仓库 `matmul/weight_quant_batch_matmul_v2` 的 MSD 模板族
> 记号统一：权重 W n×k（INT8/INT4），激活 x m×k（FP16/BF16），输出 m×n

## 1. MSD 是什么、解决什么问题

**问题**：weight-quant matmul 的普通流程是"Vector 先把 INT8/INT4 权重反量化成 BF16 再送 Cube"，反量化的 Vector 计算 + 中间 BF16 权重在 HBM 的往返成为 decode 阶段的瓶颈。

**机制**：
1. FP16 激活按行归一化（÷ rowmax）
2. 二阶展开为两个 INT8：`a1 = round(x_norm × 127.499)`，`a2 = round((x_norm − a1/127.499) × 127.499² × 2)`
3. **权重保持 INT8/INT4 不动**（这是关键——不再反量化权重）
4. 两次 INT8×INT8 MMAD（INT32 精确累加）
5. Vector 合并：`C = (C1/127.499 + C2/(127.499²×2)) × rowmax + Σx⊗offset`，再 ×scale + bias

代码锚点：`weight_quant_batch_matmul_v2_msd_multicore.h` 中 `EXPAND_FACTOR_1 = 127.499`、`EXPAND_FACTOR_2 = 127.499*127.499*2`。

## 2. 为什么有效（收益来源排序）

1. **HBM 搬运主导项 3nk → nk（~3×）**：普通流程读 INT8 权重 nk + 写 BF16 权重 2nk + Cube 再读 2nk；MSD 权重只读一次 nk，新增激活分解 4mk 和输出 2mn 都是小项（decode m 很小）
2. **Vector 反量化 O(nk) → O(n+m) 消停顿**：反量化计算从"整个权重矩阵"变成"逐行 scale 合并"
3. **双倍 GEMM 被 INT8 2× 吞吐吸收**：两次 INT8 MMAD 的时间 ≈ 一次 BF16 MMAD（INT8 吞吐 2×；MXFP4 下 2×FP4@4× = 1×FP8@2×）
4. 实测口径：带宽利用率 40-50% → 80%+（GQA decode），隐含 e2e ~1.6-2×；INT4 理论上限 ~5×，Amdahl 摊薄后实际区间 **1.5-2.3×**

## 3. 精度为什么不下降（易懂版 + 严格版）

- **易懂版**：两遍 INT8 ≈ 16 个有效 bit，BF16 尾数只有 7 bit、FP16 10 bit——MSD 的"分辨率"反而是浮点的 32 倍以上；且 INT8×INT8 的乘积在 INT32 中**精确无舍入**，浮点累加每步都在丢低位
- **严格版**：误差界 M/64516 ≈ M/2¹⁶（fractional scaling 口径 M/65015）；端到端 L2 误差 0.003% vs 基线 0.60%（**200× 改善**）
- **和 FP16 比的边界**：MSD 误差是相对行最大值 M 的绝对量；当某元素 < M/32 时，其相对精度不如 FP16（浮点对小数有相对精度优势）。但 attention/FFN 输出由大行主导，端到端 L2 影响可忽略

## 4. 127×254 的由来（为什么第二项分母不是 127×127）

- 第一遍量化后残差 |r| ≤ α/2，即残差分布**宽度恰好为 α**
- 第二遍用 INT8 铺满这个宽度：INT8 对称区间有 254 步（−127~127 去掉端点约定）→ β = α/254
- 若用 127 步铺满宽度 α，量程会浪费一倍（≈ 浪费 1 bit）
- 解析上界 β=α/254 使第二遍**免做 max 归约**（不需要再扫一遍残差求范围），这是工程上的关键收益

## 5. C 部分和绕 HBM 的开销（mmad 后 C1/C2 写 workspace 再读回 Vector 合并）

- **根因**：老架构 AIC/AIV 物理分离、无片上共享通路，GM 是唯一汇合点；950 的 CV 融合通路 + 论文 FixPipe 方案正是为消掉这笔开销
- 定量：C 绕 workspace 额外 16mn 字节（int32 C1+C2 写读），相对权重主导项 nk 的比值 = **16m/k**
  - m=1 → 0.4%；m=16 → 6.3%；m=64 → 25%（k=4096）
- 实际开销被 L2 命中（写读紧邻）和 N 向分块流水（WaitForCube 双缓冲）摊薄
- FixPipe 随路合并（L0C→UB 时直接做 C1/C2 合成）可再省 Vector 30-50%

## 5.5 Attention 场景的收益与边界

- **KV cache 读取 5Md → 2Md（2.5×）**：decode attention 中 K/V 的反量化同样可走 MSD 路线（d 为 head dim）
- **拐点 N\***：上下文长度超过 N* 才划算（双倍 GEMM 的开销需要足够长的 KV 读取来摊销）——**d=128 时 N\*≈20，d=576（MLA）时 N\*≈31**；低于拐点收益为负
- **仅 decode 小 batch 适用**；prefill 的 attention/GEMM 是 compute-bound，MSD 无收益甚至负收益
- 实测口径：GQA decode 带宽利用率 40-50% → **80%+**，隐含端到端 ~1.6-2×

## 6. MSD 模板工程约束（tiling 代码核实）

- `MSD_M_LIMIT = 64`（仅 decode 小 m，prefill 不推荐）
- k % 32 == 0、n % 32 == 0
- INT8 perchannel 切 K 条件：k > 13696 或 n > 32000（`MSD_PERCHANNEL_MAX_K/N`）
- INT4 perchannel 需 trans_b + ND + n ≥ 2m、order_ = 3、必切 K
- pergroup 仅 INT4 非转置，groupsize 64/128，m ≤ groupsize/8，n % 64 == 0
- 禁用：transA / uint64 / int64 scale / int8 输出 / pertensor
- 模板优先级：`MSD_GROUP_PRIORITY = 1`、`MSD_PRIORITY = 2`（老模板族最前），ASW_PRIORITY = 11 兜底
- resident-tile 融合 kernel 是 3× 收益的前提，退化为 two-read 时只有 ~1.5×

## 7. MXFP4 实例化

- α = 2^⌈log2(Mb/1.859375)⌉（E8M0 约束 scale 为 2 的幂、α 松弛超 1.75）
- β = α/16，误差界 α/64，~6.6 有效 bit 超过单遍 MXFP8 的 5.24

## 8. 面试一句话口径

> "MSD 的本质是**把反量化从权重侧搬到激活侧**：激活分解为两个 INT8 后和 INT8 权重直接做两次 MMAD，权重全程不离开量化格式。收益排序：HBM 往返 3nk→nk（~3×）> Vector 反量化 O(nk)→O(n+m) 消停顿；前提是双倍 GEMM 被 INT8 2× 吞吐吸收；附带精度净赚（两遍 INT8≈16bit vs BF16 7bit）；边界是 decode only、需要 resident-tile 融合、attention 场景存在 N* 拐点（d=128 时 ≈20）。"
