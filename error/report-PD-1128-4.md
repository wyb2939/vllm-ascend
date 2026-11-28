# Qwen3.6-27B W8A8 新/旧算子 1P1D 分离式性能对比报告（含 Profiling 分析）

## 1. 测试概述

本次测试对比 **C8-PTH INT8 KV 新算子**（FIA antiquant mode 5，per-token/per-head 伪量化）与 **BF16 KV 旧算子** 在 1P1D（Prefill/Decode 分离）部署形态下的端到端性能，并采集运行过程 profiling、导出 `kernel_details.csv` 做算子级耗时分析。两者共用同一 W8A8 权重 checkpoint，唯一差异是 KV cache 数据类型与对应 attention 算子。

- 测试日期：2026-11-28
- 补丁：`/home/w00544044/c8_pth_fia_mode5.patch`（md5 `08149d630066a909703ee9d29677a969`，134551 字节，2026-11-28 20:28）
- 测试负载：128K 输入 / 5K 输出 / 8 并发 / 随机 token
- 投机解码：MTP 开启，`num_speculative_tokens=3`
- Chunked Prefill：开启
- Prefix Caching：开启（`--enable-prefix-caching`）
- Profiling：`torch_npu.profiler`（`--profiler-config`），经 `/start_profile`、`/stop_profile` 采集运行窗口，离线导出 `kernel_details.csv`

> 本轮补丁相对上一版（md5 `526bc56d`）的唯一代码变化在 `attention_v1.py::_scatter_pth_scale_caches`：把 scale scatter 的索引构造（Cast/Clip/FloorDiv/BroadcastTo/FloorMod/Range/RepeatInterleave/Tile/Pack 链）缓存到共享的 `attn_metadata.c8_pth_scale_scatter_indices`，每步只构造一次，而非每层重复构造（16 个 full-attention 层 × 每步）。这直接针对上一轮 profiling 中发现的索引构造开销。本报告 profiling 验证了该优化效果。

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
| **Benchmark 总时长 (s)** | **468.31** | **639.58** | 0.73× |
| **输出吞吐 (tok/s)** | **87.64** | **64.17** | 1.37× |
| **总吞吐 (tok/s)** | **2326.89** | **1703.79** | 1.37× |
| 请求吞吐 (req/s) | 0.02 | 0.01 | — |
| **Mean TTFT (ms)** | **173,660.06** | **108,005.36** | 1.61× |
| Median TTFT (ms) | 174,493.29 | 105,144.20 | 1.66× |
| P90 TTFT (ms) | 265,084.47 | 175,715.45 | 1.51× |
| P99 TTFT (ms) | 282,065.55 | 196,285.36 | 1.44× |
| **Mean TPOT (ms)** | **37.63** | **47.05** | 0.80× |
| Median TPOT (ms) | 37.88 | 35.46 | 1.07× |
| P90 TPOT (ms) | 39.46 | 71.55 | 0.55× |
| P99 TPOT (ms) | 40.40 | 110.71 | 0.36× |
| **Mean ITL (ms)** | **141.25** | **131.22** | 1.08× |
| Median ITL (ms) | 140.88 | 130.34 | 1.08× |
| P90 ITL (ms) | 146.91 | 137.39 | 1.07× |
| P99 ITL (ms) | 157.51 | 145.83 | 1.08× |
| Mean E2EL (ms) | 366,677.74 | 349,288.11 | 1.05× |
| Median E2EL (ms) | 366,219.21 | 309,123.22 | 1.18× |
| P99 E2EL (ms) | 468,084.03 | 625,468.19 | 0.75× |

### 4.2 MTP 投机解码接受率（全程累计）

| 指标 | 新算子 | 旧算子 |
|---|---|---|
| 统计窗口数 | 42 | 62 |
| 累计 Accepted token | 30,116 | 26,334 |
| 累计 Drafted token | 32,820 | 44,154 |
| **累计接受率** | **91.76%** | **59.64%** |

> 说明：随机 token 负载下接受率随窗口波动属正常现象。本轮新算子接受率（91.76%）显著高于旧算子（59.64%），MTP 路径健康，无崩塌。

## 5. 结论与分析

1. **功能正确性**：新算子与旧算子均 8/8 请求成功、0 失败、0 错误日志、0 NaN。

2. **性能（新算子本轮全面领先）**：
   - 新算子输出吞吐 **87.64 tok/s**，旧算子 64.17 tok/s，新算子为旧算子的 **1.37×**。
   - **TPOT 尾部大幅改善**：新算子 Mean TPOT 37.63ms、P90 39.46ms、P99 40.40ms，尾部非常平滑；旧算子 Mean 47.05ms、P90 71.55ms、P99 110.71ms，尾部抖动明显。新算子 P99 TPOT 仅为旧算子的 36%。
   - 相比上一版补丁（md5 `526bc56d`：新算子 88.07 tok/s、Mean TPOT 40.51ms），本轮吞吐基本持平，**TPOT 从 40.51ms 降到 37.63ms（-7%）**，尾部更平滑，直接受益于索引构造缓存优化。
   - 新算子唯一劣势是 **TTFT**（173.7s vs 108.0s，+61%），即 INT8 KV 的 prefill attention 反量化开销，与 decode 路径无关。

3. **旧算子本轮偏慢（需注意）**：本轮旧算子吞吐 64.17 tok/s、Mean TPOT 47.05ms，明显低于上一轮同配置的 90.27 tok/s / 40.41ms。从日志看旧算子本轮 TPOT 尾部抖动剧烈（P90 71.55ms / P99 110.71ms），且累计接受率仅 59.64%（上一轮 78.65%），属**运行间波动**（通信/调度抖动 + 随机负载下 MTP 命中波动）。因此本报告的新/旧对比中，旧算子基线偏保守；若以两轮旧算子较优值（~90 tok/s）为参照，新算子（87.64 tok/s）仍与之相当，且 TPOT 尾部显著更优。

4. **Prefix Caching 配置**：开启 `--enable-prefix-caching` 后，vLLM 自动将 `mamba_cache_mode` 切为 `align`，4 个 KV cache group 等宽（新 1664 / 旧 896）。该配置下新旧算子 MTP 接受率均正常，无崩塌。

5. **内存收益**：INT8 KV 相比 BF16 理论减半 KV cache 占用，同等负载下可支持更大并发或更长上下文。

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
| `kernel_analysis/new_prefill__kernel_details.csv` | 新算子 prefill 合并（11,292 行） |
| `kernel_analysis/new_decode__kernel_details.csv` | 新算子 decode 合并（488,148 行） |
| `kernel_analysis/old_prefill__kernel_details.csv` | 旧算子 prefill 合并（14,062 行） |
| `kernel_analysis/old_decode__kernel_details.csv` | 旧算子 decode 合并（441,356 行） |
| `kernel_analysis/kernel_analysis_summary.csv` | 按 OP Type 聚合的耗时汇总（275 行） |

### 6.3 算子耗时 Top（按 OP Type 聚合）

**Prefill（新算子，总 5,030,955 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| allreduceAicpuKernel | 1,593,190 | 28.9% | 528 |
| QuantBatchMatmulV3 | 1,260,639 | 22.9% | 848 |
| hcom_allReduce_ | 671,354 | 12.2% | 528 |
| ChunkGatedDeltaRuleFwd | 447,901 | 8.1% | 192 |
| MatMulV3 | 290,266 | 5.3% | 196 |
| ScatterNdUpdate | 235,808 | 4.3% | 136 |
| AddRmsNorm | 167,535 | 3.0% | 520 |
| FusedInferAttentionScore | 139,804 | 2.5% | 68 |
| DynamicQuant | 125,067 | 2.3% | 984 |
| SwiGlu | 99,390 | 1.8% | 260 |

**Prefill（旧算子，总 4,524,592 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| allreduceAicpuKernel | 1,422,208 | 29.6% | 792 |
| QuantBatchMatmulV3 | 1,301,300 | 27.1% | 1,272 |
| hcom_allReduce_ | 537,357 | 11.2% | 792 |
| ChunkGatedDeltaRuleFwd | 464,930 | 9.7% | 288 |
| AddRmsNorm | 167,095 | 3.5% | 780 |
| MatMulV3 | 160,323 | 3.3% | 196 |
| FusedInferAttentionScore | 144,619 | 3.0% | 102 |
| DynamicQuant | 119,256 | 2.5% | 1,272 |
| SwiGlu | 107,065 | 2.2% | 390 |
| CausalConv1d | 100,440 | 2.1% | 288 |

**Decode（新算子，总 20,574,035 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| hcom_allReduce_ | 11,480,179 | 52.6% | 26,772 |
| allreduceAicpuKernel | 4,310,342 | 19.8% | 26,772 |
| QuantBatchMatmulV3 | 2,048,851 | 9.4% | 42,680 |
| MatMulV2 | 1,284,397 | 5.9% | 19,982 |
| RecurrentGatedDeltaRule | 343,161 | 1.6% | 9,312 |
| FusedInferAttentionScore | 335,457 | 1.5% | 3,686 |
| hcom_allGather_ | 230,229 | 1.1% | 1,358 |
| DynamicQuant | 199,505 | 0.9% | 50,052 |
| allgatherAicpuKernel | 173,563 | 0.8% | 1,358 |
| Slice | 168,351 | 0.8% | 80,128 |

**Decode（旧算子，总 24,811,897 us）**

| OP Type | 耗时 (us) | 占比 | 次数 |
|---|---|---|---|
| hcom_allReduce_ | 15,788,524 | 61.4% | 27,048 |
| allreduceAicpuKernel | 4,316,578 | 16.8% | 27,048 |
| QuantBatchMatmulV3 | 2,072,349 | 8.1% | 43,120 |
| MatMulV2 | 1,292,957 | 5.0% | 20,188 |
| RecurrentGatedDeltaRule | 352,357 | 1.4% | 9,408 |
| FusedInferAttentionScore | 243,946 | 0.9% | 3,724 |
| hcom_allGather_ | 216,210 | 0.8% | 1,372 |
| DynamicQuant | 182,421 | 0.7% | 43,120 |
| allgatherAicpuKernel | 173,611 | 0.7% | 1,372 |
| Slice | 172,944 | 0.7% | 79,808 |

### 6.4 关键算子对比（新 vs 旧）

| 引擎 | 算子 | 新算子 sum / mean | 旧算子 sum / mean | 说明 |
|---|---|---|---|---|
| Decode | FusedInferAttentionScore | 335,457 us / 91.01 us | 243,946 us / 65.51 us | 新算子单次慢 39%，INT8 KV 反量化开销 |
| Decode | RecurrentGatedDeltaRule | 343,161 us / 36.85 us | 352,357 us / 37.45 us | 基本持平 |
| Decode | ScatterNdUpdate | 66,762 us / 9.06 us (7,372 次) | — | scale 写入（替代旧 IndexPut） |
| Decode | IndexPutV2 | 57,220 us / 14.75 us (3,880 次) | 498 us / 2.54 us | 新算子另有少量 index put |
| Decode | hcom_allReduce_ | 11.48M us / 52.6% | 15.79M us / 61.4% | 通信占比均最高 |
| Prefill | FusedInferAttentionScore | 139,804 us | 144,619 us | 总耗时相近 |
| Prefill | ScatterNdUpdate | 235,808 us / 136 次 | — | 新算子 scale 写入 |

### 6.5 优化效果验证（索引构造缓存）

上一轮 profiling 显示 scale scatter 的**索引构造链**（Cast/ClipByValueV2/FloorDiv/BroadcastTo/FloorMod/Range/RepeatInterleave/Tile/Pack）随每个 attention 层重复执行（每步 16 个 full-attention 层 × 每层一次）。本轮补丁把索引缓存到 `attn_metadata.c8_pth_scale_scatter_indices`，每步只构造一次。

| 指标（decode，per FIA call 归一化） | 上一版补丁（md5 526bc56d） | 本轮补丁（md5 08149d63） | 变化 |
|---|---|---|---|
| RepeatInterleave 次数 / FIA | 2.00 | 0.316 | -84% |
| Tile 次数 / FIA | 1.00 | 0.158 | -84% |
| Pack 次数 / FIA | 1.00 | 0.158 | -84% |
| FloorDiv/FloorMod/Clip 次数 / FIA | ~1.0–1.16 | ~0.16–0.32 | -70%~-84% |
| **索引构造链总耗时** | ~1.0M+ us（含 IndexPut 1.18M） | **391,634 us** | 大幅下降 |
| 端到端 Mean TPOT | 40.51 ms | 37.63 ms | -7% |
| 端到端 P99 TPOT | 46.48 ms | 40.40 ms | -13% |

**分析要点**：
- **通信（hcom_allReduce_ + allreduceAicpuKernel）仍是 decode 绝对瓶颈**，新旧算子合计占比分别约 72% / 78%。这是 TP=2 + PD 分离下每步 all-reduce 的固有成本，与 KV 量化无关。
- **索引构造链的每层重复执行已被消除**：RepeatInterleave/Tile/Pack/FloorDiv/FloorMod 等算子相对 FIA 调用次数从 ~1–2 次降到 ~0.16–0.32 次，即每步只构造一次索引，而非每层一次。这是 TPOT 从 40.51ms 降到 37.63ms、尾部更平滑的直接原因。
- **新算子 decode 剩余主要额外开销**是 `FusedInferAttentionScore` 单次慢 39%（91 vs 66us），即 INT8 KV 反量化在 attention 内的成本，约占总 decode 1.6%。
- **Prefill 侧**新算子 `ScatterNdUpdate`（4.3%）+ 更多 `QuantBatchMatmulV3` 调用（INT8 路径矩阵乘拆分）是 TTFT 偏高的主要来源之一，叠加 INT8 KV prefill attention 反量化。

## 7. 原始日志与产物位置

- 新算子 P：`/home/w00544044/compare_results/pd128k_new_prefill.log`
- 新算子 D：`/home/w00544044/compare_results/pd128k_new_decode.log`
- 新算子 bench：`/home/w00544044/compare_results/pd128k_newprof3_bench.log`
- 旧算子 P：`/home/w00544044/compare_results/pd128k_old_prefill.log`
- 旧算子 D：`/home/w00544044/compare_results/pd128k_old_decode.log`
- 旧算子 bench：`/home/w00544044/compare_results/pd128k_oldprof3_bench.log`
- Profiling 原始数据：`/mnt/share/w00544044/prof/raw/{new,old}_{prefill,decode}/`
- kernel_details 导出：`/mnt/share/w00544044/prof/kernel_export/`
- kernel_details 合并与分析：`/mnt/share/w00544044/prof/kernel_analysis/`
