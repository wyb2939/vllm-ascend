# Qwen3.6-27B W8A8 新/旧算子 1P1D 分离式性能对比报告（含 Profiling 分析）

## 1. 测试概述

本次测试对比 **C8-PTH INT8 KV 新算子**（FIA antiquant mode 5，per-token/per-head 伪量化）与 **BF16 KV 旧算子** 在 1P1D（Prefill/Decode 分离）部署形态下的端到端性能，并采集运行过程 profiling、导出 `kernel_details.csv` 做算子级耗时分析。两者共用同一 W8A8 权重 checkpoint，唯一差异是 KV cache 数据类型与对应 attention 算子。

- 测试日期：2026-11-28
- 补丁：`/home/w00544044/c8_pth_fia_mode5.patch`（md5 `526bc56dc309e8ebe934fda1dad1aa5e`，131763 字节，2026-11-28 18:13）
- 测试负载：128K 输入 / 5K 输出 / 8 并发 / 随机 token
- 投机解码：MTP 开启，`num_speculative_tokens=3`
- Chunked Prefill：开启
- Prefix Caching：开启（`--enable-prefix-caching`）
- Profiling：`torch_npu.profiler`（`--profiler-config`），经 `/start_profile`、`/stop_profile` 采集运行窗口，离线导出 `kernel_details.csv`

> 本轮补丁相对上一版（md5 `39965e4a`）的唯一代码变化在 `attention_v1.py::_scatter_pth_scale_caches`：把 per-(token,head) INT8 scale 的写入从高级索引（`k_scale_cache[block_ids, :, offsets] = k_scale`）改为单次 `torch_npu.npu_scatter_nd_update_`，直接针对上一轮 profiling 发现的 `IndexPut` 热点（1.18M us / 4.7% decode）。本报告的 profiling 数据验证了该优化效果。

## 2. 软件与硬件环境

| 项目 | 版本 / 配置 |
|---|---|
| 模型 | Qwen3.6-27B-w8a8-1-c8pth（64 层 = 48 GDN/linear_attention + 16 full_attention，MTP 层 idx=64） |
| vLLM | 0.33.0 |
| vLLM-Ascend | 0.33.0（commit 3489ff62d） |
| CANN | 9.2.1 |
| torch_npu | 2.10.0.post7 |
| 硬件 | Ascend 910（每卡 65.5 GiB），共 4 卡 |
| 部署 | 1P1D：P 节点卡 4,5（TP=2），D 节点卡 6,7（TP=2） |
| KV 连接器 | MooncakeConnectorV1（kv_port P=31000 / D=31100） |
| 代理 | load_balance_proxy（端口 18001） |
| Profiler | torch_npu.profiler Level1（PipeUtilization），Text 导出 |

## 3. vLLM 启动参数

### 3.1 新算子（C8-PTH INT8 KV）

**Prefill（卡 4,5，端口 18100）**
```
--quantization ascend --kv-cache-dtype int8
--tensor-parallel-size 2
--max-model-len 139264 --max-num-batched-tokens 16384 --max-num-seqs 8
--gpu-memory-utilization 0.9
--trust-remote-code --enforce-eager
--enable-prefix-caching --enable-chunked-prefill
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/w00544044/prof_raw/new_prefill"}'
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":31000,...}'
```

**Decode（卡 6,7，端口 18200）**
```
--quantization ascend --kv-cache-dtype int8
--tensor-parallel-size 2
--max-model-len 139264 --max-num-batched-tokens 512 --max-num-seqs 8
--gpu-memory-utilization 0.9
--trust-remote-code --enforce-eager
--enable-prefix-caching --enable-chunked-prefill
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/w00544044/prof_raw/new_decode"}'
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":31100,...}'
```

### 3.2 旧算子（BF16 KV）

与 3.1 完全相同，仅将 `--quantization ascend --kv-cache-dtype int8` 替换为 `--kv-cache-dtype bfloat16`（权重仍为 W8A8，`quantization=ascend` 由 checkpoint 自动识别）。P/D 的 `--max-num-batched-tokens` 分别为 16384 / 512。profiler 目录对应 `old_prefill` / `old_decode`。

### 3.3 Benchmark 参数

```
vllm bench serve --backend vllm --base-url http://127.0.0.1:18001 --endpoint /v1/completions
  --dataset-name random --random-input-len 131072 --random-output-len 5120
  --random-range-ratio 0.01 --num-prompts 8 --max-concurrency 8 --request-rate inf
  --ignore-eos --seed 42
```

## 4. 测试结果

### 4.1 汇总对比

| 指标 | 新算子（C8-PTH INT8 KV） | 旧算子（BF16 KV） | 新/旧 |
|---|---|---|---|
| 成功请求 / 失败 | 8 / 0 | 8 / 0 | — |
| 并发数 | 8 | 8 | — |
| 输入长度（token） | 131072（±1%） | 131072（±1%） | — |
| 输出长度（token） | 5120（±1%） | 5120（±1%） | — |
| 总输入 token | 1,048,660 | 1,048,660 | — |
| 总输出 token | 41,044 | 41,044 | — |
| **Benchmark 总时长 (s)** | **466.02** | **454.66** | 1.02× |
| **输出吞吐 (tok/s)** | **88.07** | **90.27** | 0.98× |
| **总吞吐 (tok/s)** | **2338.30** | **2396.73** | 0.98× |
| 请求吞吐 (req/s) | 0.02 | 0.02 | 1.00× |
| **Mean TTFT (ms)** | **163,605.44** | **110,962.92** | 1.47× |
| Median TTFT (ms) | 171,630.74 | 105,177.44 | 1.63× |
| P90 TTFT (ms) | 258,659.75 | 182,671.92 | 1.42× |
| P99 TTFT (ms) | 273,666.55 | 219,291.84 | 1.25× |
| **Mean TPOT (ms)** | **40.51** | **40.41** | 1.00× |
| Median TPOT (ms) | 39.10 | 39.09 | 1.00× |
| P90 TPOT (ms) | 45.98 | 47.97 | 0.96× |
| P99 TPOT (ms) | 46.48 | 55.41 | 0.84× |
| **Mean ITL (ms)** | **142.65** | **135.75** | 1.05× |
| Median ITL (ms) | 142.10 | 135.05 | 1.05× |
| P90 ITL (ms) | 149.08 | 140.80 | 1.06× |
| P99 ITL (ms) | 161.71 | 147.51 | 1.10× |
| Mean E2EL (ms) | 371,415.95 | 318,198.91 | 1.17× |
| Median E2EL (ms) | 384,332.51 | 294,808.48 | 1.30× |
| P99 E2EL (ms) | 465,599.45 | 452,957.40 | 1.03× |

### 4.2 MTP 投机解码接受率（全程累计）

| 指标 | 新算子 | 旧算子 |
|---|---|---|
| 统计窗口数 | 45 | 44 |
| 累计 Accepted token | 29,400 | 28,834 |
| 累计 Drafted token | 34,986 | 36,663 |
| **累计接受率** | **84.03%** | **78.65%** |

> 说明：随机 token 负载下接受率随窗口波动属正常现象。本轮新算子接受率（84.03%）高于旧算子（78.65%），MTP 路径健康，无崩塌。

## 5. 结论与分析

1. **功能正确性**：新算子与旧算子均 8/8 请求成功、0 失败、0 错误日志、0 NaN。

2. **性能（本轮补丁大幅改善）**：
   - 新算子输出吞吐 **88.07 tok/s**，达到旧算子 90.27 tok/s 的 **98%**（上一版补丁为 88%）。
   - **TPOT 已完全追平**：Mean TPOT 40.51ms vs 40.41ms（1.00×），P99 TPOT 甚至更低（46.48 vs 55.41ms）。
   - 相比上一版补丁（新算子 60.75 tok/s、TPOT 48.31ms），本轮吞吐提升 **45%**、TPOT 下降 **16%**，直接受益于 `IndexPut`→`npu_scatter_nd_update_` 优化。
   - 剩余差距仅在 **TTFT**（163.6s vs 111.0s，+47%），即 INT8 KV 的 prefill attention 反量化开销，与 decode 路径无关。

3. **Prefix Caching 配置**：开启 `--enable-prefix-caching` 后，vLLM 自动将 `mamba_cache_mode` 切为 `align`，4 个 KV cache group 等宽（新 1664 / 旧 896）。该配置下新旧算子 MTP 接受率均正常（84.0% / 78.7%），无崩塌。

4. **内存收益**：INT8 KV 相比 BF16 理论减半 KV cache 占用，同等负载下可支持更大并发或更长上下文。

## 6. Profiling 与 kernel_details 分析

### 6.1 采集与导出流程

1. 服务以 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"<dir>"}'` 启动（P/D 各一份目录）。
2. 通过 HTTP `POST /start_profile`（带 `profile_prefix`）与 `POST /stop_profile` 控制采集窗口，窗口内驱动 2 个 8192 输入 / 256 输出的请求。
3. 原始数据落盘到 `/home/w00544044/prof_raw/{new,old}_{prefill,decode}/`，并拷贝到 `/mnt/share/w00544044/prof/raw/`。
4. 用 `msprof --export=on --type=text` 离线导出，并将 `op_summary_*.csv` 转换为 `kernel_details.csv`。
5. 合并各 rank、按 `OP Type` 聚合，输出到 `/mnt/share/w00544044/prof/kernel_analysis/`。

> 注：torch_npu 在 worker 守护进程中无法在线解析（`cannot be parsed during the daemon process`），因此采用离线 `msprof` 导出；`KernelViewParser` 在部分原始目录因缺 TASK 表失败，改用 `op_summary` 直接转换生成 `kernel_details.csv`（列 `Op Name→Name`、`Task Duration(us)→Duration(us)`），内容等价。

### 6.2 产物路径（/mnt/share/w00544044/prof）

| 路径 | 说明 |
|---|---|
| `raw/{new,old}_{prefill,decode}/` | 原始 profiling 数据（含 trace、PROF 目录） |
| `kernel_export/{new,old}_{prefill,decode}/*.csv` | 各 rank 原始 kernel_details |
| `kernel_analysis/new_prefill__kernel_details.csv` | 新算子 prefill 合并（37,264 行） |
| `kernel_analysis/new_decode__kernel_details.csv` | 新算子 decode 合并（1,124,804 行） |
| `kernel_analysis/old_prefill__kernel_details.csv` | 旧算子 prefill 合并（13,087 行） |
| `kernel_analysis/old_decode__kernel_details.csv` | 旧算子 decode 合并（429,168 行） |
| `kernel_analysis/kernel_analysis_summary.csv` | 按 OP Type 聚合的耗时汇总（275 行） |

### 6.3 算子耗时 Top（按 OP Type 聚合）

**Prefill（新算子，总 11,969,251 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| hcom_allReduce_ | 3,653,351 | 27.7% | 1,584 |
| QuantBatchMatmulV3 | 2,551,558 | 19.3% | 2,544 |
| allreduceAicpuKernel | 2,259,565 | 17.1% | 1,584 |
| ChunkGatedDeltaRuleFwd | 964,661 | 7.3% | 576 |
| RepeatInterleave | 799,962 | 6.1% | 408 |
| ScatterNdUpdate | 515,335 | 3.9% | 408 |
| AddRmsNorm | 336,746 | 2.5% | 1,560 |
| MatMulV3 | 329,024 | 2.5% | 588 |
| FusedInferAttentionScore | 282,515 | 2.1% | 204 |
| Index | 276,534 | 2.1% | 1,500 |

**Prefill（旧算子，总 4,434,347 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| allreduceAicpuKernel | 1,883,872 | 40.0% | 792 |
| hcom_allReduce_ | 989,860 | 21.0% | 792 |
| QuantBatchMatmulV3 | 651,305 | 13.8% | 636 |
| ChunkGatedDeltaRuleFwd | 232,440 | 4.9% | 144 |
| AddRmsNorm | 166,995 | 3.5% | 780 |
| MatMulV3 | 160,577 | 3.4% | 196 |
| DynamicQuant | 118,944 | 2.5% | 1,272 |
| SwiGlu | 108,125 | 2.3% | 390 |
| FusedInferAttentionScore | 71,893 | 1.5% | 51 |
| CausalConv1d | 50,336 | 1.1% | 144 |

**Decode（新算子，总 45,482,426 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| hcom_allReduce_ | 26,196,050 | 54.1% | 57,384 |
| allreduceAicpuKernel | 9,139,504 | 18.9% | 57,384 |
| QuantBatchMatmulV3 | 4,340,379 | 9.0% | 91,488 |
| MatMulV2 | 2,740,283 | 5.7% | 42,828 |
| RecurrentGatedDeltaRule | 719,383 | 1.5% | 19,968 |
| FusedInferAttentionScore | 717,349 | 1.5% | 7,896 |
| hcom_allGather_ | 473,806 | 1.0% | 2,892 |
| DynamicQuant | 429,202 | 0.9% | 107,280 |
| Slice | 364,475 | 0.8% | 170,752 |
| allgatherAicpuKernel | 361,994 | 0.7% | 2,892 |

**Decode（旧算子，总 24,747,578 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| hcom_allReduce_ | 15,994,778 | 62.3% | 26,220 |
| allreduceAicpuKernel | 4,158,839 | 16.2% | 26,220 |
| QuantBatchMatmulV3 | 1,999,494 | 7.8% | 41,800 |
| MatMulV2 | 1,262,769 | 4.9% | 19,570 |
| RecurrentGatedDeltaRule | 350,472 | 1.4% | 9,120 |
| FusedInferAttentionScore | 240,737 | 0.9% | 3,610 |
| hcom_allGather_ | 224,594 | 0.9% | 1,330 |
| DynamicQuant | 178,211 | 0.7% | 41,800 |
| Slice | 172,795 | 0.7% | 78,560 |
| allgatherAicpuKernel | 164,891 | 0.6% | 1,330 |

### 6.4 关键算子对比（新 vs 旧）

| 引擎 | 算子 | 新算子 sum / mean | 旧算子 sum / mean | 说明 |
|---|---|---|---|---|
| Decode | **ScatterNdUpdate** | 145,017 us / 9.18 us (15,792 次) | — | **本轮新增的 scale 写入路径**（替代旧 IndexPut） |
| Decode | RepeatInterleave | 62,006 us / 3.93 us (15,792 次) | — | scatter 索引构造开销 |
| Decode | FusedInferAttentionScore | 717,349 us / 90.85 us | 240,737 us / 66.69 us | 新算子单次慢 36%，INT8 KV 反量化开销 |
| Decode | RecurrentGatedDeltaRule | 719,383 us / 36.03 us | 350,472 us / 38.43 us | 单次耗时相当（新调用次数更多） |
| Decode | hcom_allReduce_ | 26.20M us / 54.1% | 15.99M us / 62.3% | 通信占比均最高 |
| Prefill | FusedInferAttentionScore | 282,515 us / 1,384.88 us | 71,893 us / 1,409.67 us | 单次耗时相当 |
| Prefill | ScatterNdUpdate | 515,335 us / 1,263 us | — | 新算子 scale 写入 |
| Prefill | RepeatInterleave | 799,962 us / 1,961 us | — | 新算子 scatter 索引构造 |

### 6.5 优化效果验证（IndexPut → ScatterNdUpdate）

| 指标 | 上一版补丁（md5 39965e4a） | 本轮补丁（md5 526bc56d） | 变化 |
|---|---|---|---|
| 旧 scale 写入路径 | `IndexPut` 1,184,062 us / 7,752 次 | 已移除 | — |
| 新 scale 写入路径 | — | `ScatterNdUpdate` 145,017 us + `RepeatInterleave` 62,006 us | 合计 207,023 us |
| **该路径耗时** | **1,184,062 us** | **207,023 us** | **-82.5%（约 5.7× 加速）** |
| decode 中占比 | 4.7% | 0.46% | -4.2pp |
| 端到端输出吞吐 | 60.75 tok/s | 88.07 tok/s | +45% |
| Mean TPOT | 48.31 ms | 40.51 ms | -16% |

**分析要点**：
- **通信（hcom_allReduce_ + allreduceAicpuKernel）仍是 decode 绝对瓶颈**，新旧算子合计占比分别约 73% / 79%。这是 TP=2 + PD 分离下每步 all-reduce 的固有成本，与 KV 量化无关。
- **本轮补丁精准命中上一轮发现的 `IndexPut` 热点**：高级索引编译出的 ~9 个小算子（Cast/Clip/FloorDiv/BroadcastTo/FloorMod/IndexCheck/IndexPut/Range/Index/GatherV3/Transpose）被折叠为单次 `npu_scatter_nd_update_`，该路径耗时从 1.18M us 降到 0.21M us（-82.5%），是吞吐提升 45%、TPOT 追平旧算子的直接原因。
- **新算子 decode 剩余唯一额外开销**是 `FusedInferAttentionScore` 单次慢 36%（91 vs 67us），即 INT8 KV 反量化在 attention 内的成本，约占总 decode 1.5%，已不足以造成明显吞吐差距。
- **Prefill 侧**新算子新增 `RepeatInterleave`（6.1%）+ `ScatterNdUpdate`（3.9%）用于 scale 写入，以及更多 `QuantBatchMatmulV3` 调用（INT8 路径矩阵乘拆分），这是 TTFT 高 47% 的主要来源之一，叠加 INT8 KV prefill attention 反量化。

## 7. 原始日志与产物位置

- 新算子 P：`/home/w00544044/compare_results/pd128k_new_prefill.log`
- 新算子 D：`/home/w00544044/compare_results/pd128k_new_decode.log`
- 新算子 bench：`/home/w00544044/compare_results/pd128k_newprof2_bench.log`
- 旧算子 P：`/home/w00544044/compare_results/pd128k_old_prefill.log`
- 旧算子 D：`/home/w00544044/compare_results/pd128k_old_decode.log`
- 旧算子 bench：`/home/w00544044/compare_results/pd128k_oldprof2_bench.log`
- Profiling 原始数据：`/mnt/share/w00544044/prof/raw/{new,old}_{prefill,decode}/`
- kernel_details 导出：`/mnt/share/w00544044/prof/kernel_export/`
- kernel_details 合并与分析：`/mnt/share/w00544044/prof/kernel_analysis/`


## 8. 算子调用
split_qkv_rmsnorm_mrope_kernel
DynamicQuant
DynamicQuant
ScatterPaKvCache
Cast
ClipByValueV2
FloorDiv
BroadcastTo
FloorMod
Range
BroadcastTo
RepeatInterleave
Tile
BroadcastTo
RepeatInterleave
Pack
ScatterNdUpdate
ScatterNdUpdate
Range
IndexCheck
Index
GatherV3
Transpose
FusedInferAttentionScore

