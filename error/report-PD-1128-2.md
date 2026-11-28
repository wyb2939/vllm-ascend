# Qwen3.6-27B W8A8 新/旧算子 1P1D 分离式性能对比报告（含 Profiling 分析）

## 1. 测试概述

本次测试对比 **C8-PTH INT8 KV 新算子**（FIA antiquant mode 5，per-token/per-head 伪量化）与 **BF16 KV 旧算子** 在 1P1D（Prefill/Decode 分离）部署形态下的端到端性能，并额外采集运行过程 profiling、导出 `kernel_details.csv` 做算子级耗时分析。两者共用同一 W8A8 权重 checkpoint，唯一差异是 KV cache 数据类型与对应 attention 算子。

- 测试日期：2026-11-28
- 补丁：`/home/w00544044/c8_pth_fia_mode5.patch`（md5 `39965e4acb93693b3c3aa63bf69c66f8`，129839 字节，2026-11-28 11:50）
- 测试负载：128K 输入 / 5K 输出 / 8 并发 / 随机 token
- 投机解码：MTP 开启，`num_speculative_tokens=3`
- Chunked Prefill：开启
- **Prefix Caching：开启（`--enable-prefix-caching`，本次新增要求）**
- Profiling：`torch_npu.profiler`（`--profiler-config`），经 `/start_profile`、`/stop_profile` 采集运行窗口，离线导出 `kernel_details.csv`

> 说明：本轮相比上一轮唯一的功能性配置变化是开启 prefix caching。该开关会触发 vLLM 将 `mamba_cache_mode` 自动切换为 `align`（见第 6 节），对 MTP 接受率与性能有实质影响，详见第 5 节。

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
| **Benchmark 总时长 (s)** | **675.66** | **594.68** | 1.14× |
| **输出吞吐 (tok/s)** | **60.75** | **69.02** | 0.88× |
| **总吞吐 (tok/s)** | **1612.80** | **1832.41** | 0.88× |
| 请求吞吐 (req/s) | 0.01 | 0.01 | 1.00× |
| **Mean TTFT (ms)** | **142,060.22** | **117,154.89** | 1.21× |
| Median TTFT (ms) | 146,574.98 | 108,687.87 | 1.35× |
| P90 TTFT (ms) | 225,517.74 | 195,533.87 | 1.15× |
| P99 TTFT (ms) | 242,144.99 | 251,570.12 | 0.96× |
| **Mean TPOT (ms)** | **48.31** | **47.96** | 1.01× |
| Median TPOT (ms) | 37.54 | 38.24 | 0.98× |
| P90 TPOT (ms) | 64.58 | 69.99 | 0.92× |
| P99 TPOT (ms) | 115.80 | 107.08 | 1.08× |
| **Mean ITL (ms)** | **147.54** | **135.12** | 1.09× |
| Median ITL (ms) | 146.82 | 134.15 | 1.09× |
| P90 ITL (ms) | 153.48 | 141.43 | 1.09× |
| P99 ITL (ms) | 164.92 | 150.99 | 1.09× |
| Mean E2EL (ms) | 389,742.30 | 362,805.67 | 1.07× |
| Median E2EL (ms) | 366,239.96 | 316,394.22 | 1.16× |
| P99 E2EL (ms) | 658,733.25 | 583,720.12 | 1.13× |

### 4.2 MTP 投机解码接受率（全程累计）

| 指标 | 新算子 | 旧算子 |
|---|---|---|
| 统计窗口数 | 71 | 57 |
| 累计 Accepted token | 28,033 | 26,502 |
| 累计 Drafted token | 40,995 | 43,656 |
| **累计接受率** | **68.38%** | **60.71%** |

> 说明：压测运行期间（前 ~50 个窗口）新算子稳定在 53%–87%、旧算子稳定在 50%–80%，两者量级相当；累计值被压测结束后的空载窗口（接受率 <10%）拉低。随机 token 负载下接受率随窗口波动属正常现象。**在开启 prefix caching 的配置下，新算子 MTP 接受率与旧算子相当，未出现上一轮（关闭 prefix caching 时）的崩塌。**

## 5. 结论与分析

1. **功能正确性**：新算子与旧算子均 8/8 请求成功、0 失败、0 错误日志、0 NaN。

2. **性能**：新算子输出吞吐 60.75 tok/s，为旧算子 69.02 tok/s 的 **88%**；总时长 1.14×。TPOT 基本持平（48.31 vs 47.96ms），差距主要来自 **TTFT 高 21%**（142.1s vs 117.2s），即 INT8 KV 的 prefill attention 反量化开销。这与上一轮（关闭 prefix caching）的结论一致：TTFT 是主要差距来源。

3. **Prefix Caching 的影响（关键发现）**：
   - 开启 `--enable-prefix-caching` 后，vLLM 对 Qwen3.5 混合模型自动将 `mamba_cache_mode` 由 `none` 切换为 `align`（日志：`Mamba cache mode is set to 'align' ... when prefix caching is enabled`）。
   - `align` 模式下 mamba block_size 被对齐到 attention block（新算子 1664 / 旧算子 896），4 个 KV cache group 全部等宽（新 `(1664,1664,1664,1664)`，旧 `(896,896,896,896)`）。
   - 该配置下**新旧算子 MTP 接受率均正常**（68.4% / 60.7%），未复现上一轮关闭 prefix caching 时的接受率崩塌，说明 align 路径的状态块索引是正确的。
   - 代价：align 模式引入额外的块对齐与状态块管理开销，使本轮绝对吞吐低于上一轮（新算子 60.75 vs 92.91 tok/s），但新旧相对差距（88% vs 91%）基本一致。

4. **内存收益**：INT8 KV 相比 BF16 理论减半 KV cache 占用，同等负载下可支持更大并发或更长上下文。

## 6. Profiling 与 kernel_details 分析

### 6.1 采集与导出流程

1. 服务以 `--profiler-config '{"profiler":"torch","torch_profiler_dir":"<dir>"}'` 启动（P/D 各一份目录）。
2. 通过 HTTP `POST /start_profile`（带 `profile_prefix`）与 `POST /stop_profile` 控制采集窗口，窗口内驱动 2 个 8192 输入 / 256 输出的请求。
3. 原始数据落盘到 `/home/w00544044/prof_raw/{new,old}_{prefill,decode}/`。
4. 用 `msprof --export=on --type=text` 离线导出，并将 `op_summary_*.csv` 转换为 `kernel_details.csv`。
5. 合并各 rank、按 `OP Type` 聚合，输出到 `/mnt/share/w00544044/prof/kernel_analysis/`。

> 注：torch_npu 在 worker 守护进程中无法在线解析（`cannot be parsed during the daemon process`），因此采用离线 `msprof` 导出；`KernelViewParser` 在部分原始目录因缺 TASK 表失败，改用 `op_summary` 直接转换生成 `kernel_details.csv`（列 `Op Name→Name`、`Task Duration(us)→Duration(us)`），内容等价。

### 6.2 产物路径

| 文件 | 说明 |
|---|---|
| `/mnt/share/w00544044/prof/kernel_export/{new,old}_{prefill,decode}/*.csv` | 各 rank 原始 kernel_details |
| `/mnt/share/w00544044/prof/kernel_analysis/new_prefill__kernel_details.csv` | 新算子 prefill 合并（29,427 行） |
| `/mnt/share/w00544044/prof/kernel_analysis/new_decode__kernel_details.csv` | 新算子 decode 合并（531,190 行） |
| `/mnt/share/w00544044/prof/kernel_analysis/old_prefill__kernel_details.csv` | 旧算子 prefill 合并（14,062 行） |
| `/mnt/share/w00544044/prof/kernel_analysis/old_decode__kernel_details.csv` | 旧算子 decode 合并（441,356 行） |
| `/mnt/share/w00544044/prof/kernel_analysis/kernel_analysis_summary.csv` | 按 OP Type 聚合的耗时汇总（269 行） |

### 6.3 算子耗时 Top（按 OP Type 聚合）

**Prefill（新算子，总 4,728,937 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| QuantBatchMatmulV3 | 1,575,056 | 29.6% | 1,908 |
| hcom_allReduce_ | 1,197,190 | 22.5% | 1,188 |
| ChunkGatedDeltaRuleFwd | 552,242 | 10.4% | 432 |
| Index | 273,649 | 5.2% | 1,584 |
| allreduceAicpuKernel | 247,904 | 4.7% | 1,188 |
| AddRmsNorm | 208,314 | 3.9% | 1,170 |
| MatMulV3 | 205,805 | 3.9% | 441 |
| FusedInferAttentionScore | 201,953 | 3.8% | 153 |
| DynamicQuant | 144,785 | 2.7% | 2,214 |
| CausalConv1d | 122,038 | 2.3% | 432 |

**Prefill（旧算子，总 4,539,828 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| allreduceAicpuKernel | 1,442,816 | 29.9% | 792 |
| QuantBatchMatmulV3 | 1,300,931 | 27.0% | 1,272 |
| hcom_allReduce_ | 528,721 | 11.0% | 792 |
| ChunkGatedDeltaRuleFwd | 468,773 | 9.7% | 288 |
| AddRmsNorm | 167,188 | 3.5% | 780 |
| MatMulV3 | 160,392 | 3.3% | 196 |
| FusedInferAttentionScore | 144,937 | 3.0% | 102 |
| DynamicQuant | 119,100 | 2.5% | 1,272 |
| SwiGlu | 106,324 | 2.2% | 390 |
| CausalConv1d | 100,647 | 2.1% | 288 |

**Decode（新算子，总 23,689,777 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| hcom_allReduce_ | 12,829,758 | 51.0% | 28,152 |
| allreduceAicpuKernel | 4,820,853 | 19.2% | 28,152 |
| QuantBatchMatmulV3 | 2,174,671 | 8.7% | 44,880 |
| MatMulV2 | 1,343,598 | 5.3% | 21,012 |
| IndexPut | 1,184,062 | 4.7% | 7,752 |
| FusedInferAttentionScore | 351,527 | 1.4% | 3,876 |
| RecurrentGatedDeltaRule | 350,249 | 1.4% | 9,792 |
| hcom_allGather_ | 232,866 | 0.9% | 1,428 |
| DynamicQuant | 207,306 | 0.8% | 52,632 |

**Decode（旧算子，总 25,495,461 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| hcom_allReduce_ | 16,393,899 | 62.1% | 27,048 |
| allreduceAicpuKernel | 4,390,130 | 16.6% | 27,048 |
| QuantBatchMatmulV3 | 2,078,449 | 7.9% | 43,120 |
| MatMulV2 | 1,292,856 | 4.9% | 20,188 |
| RecurrentGatedDeltaRule | 346,114 | 1.3% | 9,408 |
| FusedInferAttentionScore | 245,267 | 0.9% | 3,724 |
| hcom_allGather_ | 221,453 | 0.8% | 1,372 |
| DynamicQuant | 180,784 | 0.7% | 43,120 |

### 6.4 关键算子对比（新 vs 旧）

| 引擎 | 算子 | 新算子 sum / mean | 旧算子 sum / mean | 说明 |
|---|---|---|---|---|
| Prefill | FusedInferAttentionScore | 201,953 us / 1,320 us | 144,937 us / 1,421 us | 单次耗时相近；新算子调用次数更多（153 vs 102，与 chunk 数相关） |
| Prefill | QuantBatchMatmulV3 | 1,575,056 us / 826 us | 1,300,931 us / 1,023 us | 新算子单次更快，总耗时高因调用次数多（1,908 vs 1,272） |
| Prefill | DynamicQuant | 144,785 us / 65 us | 119,100 us / 94 us | INT8 KV 路径新增量化开销，单次更轻 |
| Decode | FusedInferAttentionScore | 351,527 us / 91 us | 245,267 us / 66 us | **新算子单次慢 38%**，INT8 KV 反量化开销的直接体现 |
| Decode | IndexPut | 1,184,062 us / 153 us | 约 0（仅 IndexPutV2 498 us） | **新算子独有大头**：INT8 KV 的 per-(token,head) scale 写入（7,752 次） |
| Decode | RecurrentGatedDeltaRule | 350,249 us / 36 us | 346,114 us / 37 us | 基本持平 |
| Decode | hcom_allReduce_ | 12.83M us / 51.0% | 16.39M us / 62.1% | 通信占比均最高，新算子占比更低 |

**分析要点**：
- **通信（hcom_allReduce_ + allreduceAicpuKernel）是 decode 绝对瓶颈**，新旧算子合计占比分别约 70% / 79%。这是 TP=2 + PD 分离下每步 all-reduce 的固有成本，与 KV 量化无关。
- **新算子 decode 的两处额外开销**：
  1. `FusedInferAttentionScore` 单次 91us vs 66us（+38%），即 INT8 KV 反量化在 attention 内的成本；
  2. `IndexPut` 1.18M us（4.7%），用于写入 per-(token,head) 的 INT8 scale，旧算子无此项。
  这两项合计约占新算子 decode 的 6.1%，是相对旧算子吞吐差距（12%）的主要构成，其余差距来自通信与调度抖动。
- **Prefill 侧**新算子 `DynamicQuant`（2,214 次）与更多 `QuantBatchMatmulV3` 调用反映 INT8 路径的量化/矩阵乘拆分，但单次耗时均低于旧算子，整体 prefill 总耗时相近（4.73M vs 4.54M us）。

## 7. 原始日志与产物位置

- 新算子 P：`/home/w00544044/compare_results/pd128k_new_prefill.log`
- 新算子 D：`/home/w00544044/compare_results/pd128k_new_decode.log`
- 新算子 bench：`/home/w00544044/compare_results/pd128k_newprof_bench.log`
- 旧算子 P：`/home/w00544044/compare_results/pd128k_old_prefill.log`
- 旧算子 D：`/home/w00544044/compare_results/pd128k_old_decode.log`
- 旧算子 bench：`/home/w00544044/compare_results/pd128k_oldprof_bench.log`
- Profiling 原始数据：`/home/w00544044/prof_raw/{new,old}_{prefill,decode}/`
- kernel_details 导出：`/mnt/share/w00544044/prof/kernel_export/`
- kernel_details 合并与分析：`/mnt/share/w00544044/prof/kernel_analysis/`

## 8. 优化：算子替换

split_qkv_rmsnorm_mrope_kernel
DynamicQuant
DynamicQuant
ScatterPaKvCache
Cast
ClipByValueV2
FloorDiv
BroadcastTo
FloorMod
IndexCheck
IndexPut
IndexCheck
IndexPut
Range
IndexCheck
Index
GatherV3
Transpose
FusedInferAttentionScore

中间的需要替换成：scatterNdUpdate (k scale)  scatterNdUpdate (v scale)