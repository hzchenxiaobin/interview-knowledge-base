# 推理系统与 vLLM

## 1. KV Cache：为什么 K/V 可以复用，Q 不能

**一句话**：自回归解码里，每个 token 的 K/V 要被之后生成的**每一个** token 各用一次；Q 只在生成下一个 token 那一步用一次，用完即弃。

公式视角，位置 i 的 attention：`o_i = Σ_{j≤i} softmax(q_iᵀk_j/√d) v_j`

- **q 的下标是 i**：只有 q_i 参与，算"位置 i 想从过去读什么"
- **k/v 的下标是 j**：所有历史位置都要参与

时间线视角：

```
第 t 步：q_t × (k₁...k_t) → 生成 x_{t+1}
```

- 第 j 个 token 的 k_j, v_j 在第 j、j+1、…、T 步各被读一次 → 不缓存则每步重算前缀，O(T) 爆炸到 O(T²)
- q_t 只在第 t 步出现，之后永远不再使用 → 无缓存价值

**这个不对称性决定了推理系统的形态**：
1. decode 是 GEMV 形态 memory-bound：每步只需一次矩阵-向量乘，但要读全部历史 K/V + 全部权重
2. PagedAttention 的分页/CoW 全部围绕 K/V 设计——Q 是瞬时量，连进 cache 的资格都没有
3. 与 wqbmm 同构：Q 一行可塞寄存器/L0，瓶颈全在 K/V 搬运 → KV cache 量化（FP16→INT8/INT4）就是 attention 版的"减少搬运"

## 2. PagedAttention 论文精读（SOSP'23）

### 问题
KV cache 显存浪费限制 batch size，吞吐 = batch size。13B 模型 A100-40G：参数 65%、KV cache ~30%。OPT-13B 单 token KV = 2×5120×40层×2B = **800KB**，2048 序列最多 1.6GB。现有系统按最大长度静态预分配连续显存，产生三种浪费（reserved / 内部碎片 / 外部碎片），**实测只有 20.4%–38.2% 的 KV cache 显存真存了 token 状态**。

### 核心思想：OS 虚拟内存 → KV cache

| OS 虚拟内存 | PagedAttention |
|---|---|
| 页 | KV block（默认 16 token） |
| 字节 / 进程 | token / 请求 |
| 逻辑页→物理页 | 逻辑 block→物理 block（block table） |
| fork 的 copy-on-write | 物理块引用计数>1 时写入触发块级 CoW |
| swap 到磁盘 | KV 块换出到 CPU RAM |
| 共享库 | 共享前缀（system prompt 只存一份） |

attention 改成分块形式：kernel 按 block table 取非连续物理块边取边算。一个请求的最大浪费压到**一个 block 以内**，外部碎片消失。

### 三个共享场景
1. **Parallel sampling**：n 个序列共享 prompt 物理块，省 6.1%–30.5%
2. **Beam search**：动态共享前缀块，省 **37.6%–66.3%**（共享最多收益最大）
3. **Shared prefix**：system prompt 预置物理块像共享库，5-shot 前缀吞吐 **3.58×** Orca(Oracle)

### 调度与抢占
FCFS + 抢占最晚到达；恢复二选一：swapping（换出 CPU，小块时大量小传输打不满 PCIe）vs recomputation（重算 KV，已生成 token 拼进 prompt 一次 prefill 出来，开销不超 swap 的 20%）；块 16–64 两者相当。同请求多序列 gang-scheduled。

### 代价（必答"有什么 overhead"）
- PagedAttention kernel 比 FasterTransformer attention **慢 20–26%**（block table 间接寻址 + 变长分支），但只影响 attention 算子，端到端被内存收益碾压
- 块大小权衡：太小并行度不足，太大碎片回升/共享率降——**16 是甜点**（vLLM 默认值由来）
- 只对"动态长度 + 内存容量 bound"负载成立；compute-bound 负载是负优化

### 端到端数字
vs Orca(Oracle) 请求率 **1.7×–2.7×**；vs Orca(Max) 2.7×–8×；vs FasterTransformer 最高 **22×**；整体口径同等延迟下吞吐 **2–4×**。

## 3. GQA 论文精读（Ainslie et al. 2023）

**一句话**：MHA 和 MQA 的广义插值——Q 头分 G 组、组内共享一份 K/V 头，并用 mean pooling + 5% 算力 uptrain 从 MHA checkpoint 低成本转换。

- MHA：H 个 Q 头各配独立 K/V → 质量好但 KV cache 大、decode 慢
- MQA（2019）：全部 Q 头共享一份 K/V → KV cache 缩 H 倍但质量降、训练不稳
- **GQA-G**：KV cache 和 decode K/V 读取降为 **G/H**；GQA-1=MQA、GQA-H=MHA
- **Uptraining**：组内 K/V 投影头 **mean pooling** 合并（优于随机选头）+ ~5% 预训练算力继续训适配
- 结论（T5-Large/XL/XXL）：**GQA-8 质量追平 MHA、decode 速度接近 MQA**；模型越大性价比越高
- 边界：实验基于 encoder-decoder T5；decoder-only（Llama 2 70B 用 GQA-8）进一步验证，但小模型或 G 压太狠仍有可见损失——别说"完全无损"
- 技术链：MQA（省得狠伤质量）→ GQA（折中甜点）→ MLA（DeepSeek，低秩潜向量压缩，压缩率更高但结构复杂、与 RoPE 有兼容性成本）
- 算子视角：GQA 不减少计算、只减少搬运（K/V 头数 H→G），与 MSD 减权重搬运同哲学；Qwen3-4B 即 GQA（Q 32 头 / KV 8 头），压测时 KV cache 若换 MHA 涨 4×

## 4. vLLM 压测实验项目（GitHub: hzchenxiaobin/vllm-benchmark-lab）

环境：AutoDL 4090 + Qwen3-4B（ModelScope 下载）+ vllm bench serve。

四组对照实验（`run_experiments.sh`，核心函数 `launch_and_bench`）：

| 实验 | 变量 | 设计意图 |
|---|---|---|
| exp1 prefix_off/on | prefix caching 开关 + `--random-prefix-len 1024` | 验证共享前缀收益（论文 §4.4 的现代演进：V1 block 哈希全前缀缓存） |
| exp2 chunk_512/2048/8192 | `--max-num-batched-tokens`，输入 4096 | chunked prefill 对 TTFT/TPOT 的权衡 |
| exp3 graph_on/off | `--enforce-eager`，低并发长输出 | CUDA Graph 消除小 batch 下的 launch 开销 |
| exp4 seqs_32/128/512 | `--max-num-seqs`，request-rate inf | 调度并发上限与吞吐/尾延迟 |

配套：`collect_results.py` 汇总 JSON → Markdown 表格（TTFT/TPOT/E2EL 均值与 P99）；`实验报告模板.md` 预留"面试弹药句"。

## 5. V1 源码精读要点（后续深入方向）

- 异步调度（scheduler 与 model runner 重叠，"0 开销调度"）
- block 哈希前缀缓存（替代论文时期的预留前缀块）
- KVConnector / PD 分离（prefill-decode 解耦）
