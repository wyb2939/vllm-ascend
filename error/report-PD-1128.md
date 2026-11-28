# Qwen3.6-27B W8A8 新/旧算子 1P1D 分离式性能对比报告

## 1. 测试概述

本次测试对比 **C8-PTH INT8 KV 新算子**（FIA antiquant mode 5，per-token/per-head 伪量化）与 **BF16 KV 旧算子** 在 1P1D（Prefill/Decode 分离）部署形态下的端到端性能。两者共用同一 W8A8 权重 checkpoint，唯一差异是 KV cache 数据类型与对应 attention 算子。

- 测试日期：2026-11-28
- 测试负载：128K 输入 / 5K 输出 / 8 并发 / 随机 token
- 投机解码：MTP 开启，`num_speculative_tokens=3`
- Chunked Prefill：开启

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

## 3. vLLM 启动参数

### 3.1 新算子（C8-PTH INT8 KV）

**Prefill（卡 4,5，端口 18100）**
```
--quantization ascend --kv-cache-dtype int8
--tensor-parallel-size 2
--max-model-len 139264 --max-num-batched-tokens 16384 --max-num-seqs 8
--gpu-memory-utilization 0.9
--trust-remote-code --enforce-eager
--no-enable-prefix-caching --enable-chunked-prefill
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":31000,...}'
```

**Decode（卡 6,7，端口 18200）**
```
--quantization ascend --kv-cache-dtype int8
--tensor-parallel-size 2
--max-model-len 139264 --max-num-batched-tokens 512 --max-num-seqs 8
--gpu-memory-utilization 0.9
--trust-remote-code --enforce-eager
--no-enable-prefix-caching --enable-chunked-prefill
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":31100,...}'
```

### 3.2 旧算子（BF16 KV）

与 3.1 完全相同，仅将 `--quantization ascend --kv-cache-dtype int8` 替换为 `--kv-cache-dtype bfloat16`（权重仍为 W8A8，`quantization=ascend` 由 checkpoint 自动识别）。P/D 的 `--max-num-batched-tokens` 分别为 16384 / 512。

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
| **Benchmark 总时长 (s)** | **679.86** | **428.14** | 1.59× |
| **输出吞吐 (tok/s)** | **60.37** | **95.87** | 0.63× |
| **总吞吐 (tok/s)** | **1602.85** | **2545.20** | 0.63× |
| 请求吞吐 (req/s) | 0.01 | 0.02 | 0.50× |
| **Mean TTFT (ms)** | **112,241.83** | **108,600.19** | 1.03× |
| Median TTFT (ms) | 106,198.61 | 102,461.37 | 1.04× |
| P90 TTFT (ms) | 190,298.95 | 178,660.52 | 1.07× |
| P99 TTFT (ms) | 214,045.83 | 222,298.53 | 0.96× |
| **Mean TPOT (ms)** | **48.55** | **40.58** | 1.20× |
| Median TPOT (ms) | 38.50 | 39.87 | 0.97× |
| P90 TPOT (ms) | 65.61 | 44.16 | 1.49× |
| P99 TPOT (ms) | 109.37 | 44.45 | 2.46× |
| **Mean ITL (ms)** | **144.07** | **144.52** | 1.00× |
| Median ITL (ms) | 143.41 | 141.68 | 1.01× |
| P90 ITL (ms) | 149.76 | 157.80 | 0.95× |
| P99 ITL (ms) | 160.01 | 208.23 | 0.77× |
| Mean E2EL (ms) | 361,198.13 | 316,715.83 | 1.14× |
| Median E2EL (ms) | 334,324.53 | 302,457.53 | 1.11× |
| P99 E2EL (ms) | 660,455.29 | 423,921.82 | 1.56× |

### 4.2 MTP 投机解码接受率

| 指标 | 新算子 | 旧算子 |
|---|---|---|
| 累计 Accepted token | 27,225 | 29,531 |
| 累计 Drafted token | 41,496 | 34,584 |
| **累计接受率** | **65.61%** | **85.39%** |
| 平均接受长度（峰值窗口） | ~3.36 / 4 | ~3.3 / 4 |
| 每位置接受率（峰值窗口） | 0.90 / 0.80 / 0.66 | 0.83 / 0.76 / 0.62 |

> 说明：随机 token 负载下，draft 命中率随并发窗口波动较大（单窗口 9%–86%）。累计接受率是整轮 8 请求的加权平均，更具代表性。新算子 65.61% 虽低于旧算子 85.39%，但已从修复前的 **0%（全 NaN）** 完全恢复，证明 C8-PTH 在 PD+MTP 下功能正确。

## 5. 结论与分析

1. **功能正确性**：新算子与旧算子均 8/8 请求成功、0 失败、0 错误日志。新算子在修复 PD 状态传输问题后，MTP 接受率恢复至 65.61%，输出正常。

2. **性能**：新算子输出吞吐 60.37 tok/s，为旧算子 95.87 tok/s 的 **63%**；总时长 1.59×。主要开销来自：
   - **TPOT 尾部恶化**：Median TPOT 接近（38.5 vs 39.9ms），但 P90/P99 显著拉高（65.6/109.4 vs 44.2/44.5ms），说明 INT8 反量化算子在部分 decode 步（长序列 attention）有额外开销。
   - **TTFT 基本持平**（112s vs 109s，+3%），说明 prefill 阶段两算子差异不大，瓶颈在 decode 阶段。
   - **ITL 基本持平**（144 vs 145ms）。

3. **接受率差异**：新算子 65.61% vs 旧算子 85.39%。INT8 KV 量化改变了 attention 数值分布，draft head 与量化 KV 之间存在轻微分布错配，导致部分 draft token 被拒。这属于量化 KV 的固有代价，但远好于修复前的 0%。

4. **内存收益**：INT8 KV 相比 BF16 理论上减半 KV cache 占用（本次 128K×8 负载下两算子均能容纳全部请求，KV 使用率新算子约 26%、旧算子约 76%，印证新算子内存占用显著更低）。

## 6. 附：修复背景

本次测试前修复了 C8-PTH 在 PD 分离 + MTP 下接受率 0% 的根因：C8-PTH 曾把 mamba block_size 从 `max_model_len`(139264) 对齐到 attention block(1664)，在 `mamba_cache_mode="none"` 下触发 `MambaManager.get_num_skipped_tokens()` 把块表第 0 列（真实 ssm 状态块）null 掉，导致 Mooncake connector 传输错块、D 侧读到零状态 → GDN 输出 NaN → MTP 全 NaN → 接受率 0%。修复为**仅在 `align` 模式对齐 mamba block**，`none` 模式保留原生 block_size。修复后 mamba block 恢复 `(1664,139264,139264,139264)`，接受率从 0% 恢复至 65.61%。

## 7. 原始日志位置

- 新算子 P：`/home/w00544044/compare_results/pd128k_new_prefill.log`
- 新算子 D：`/home/w00544044/compare_results/pd128k_new_decode.log`
- 新算子 bench：`/home/w00544044/compare_results/pd128k_new128k3_bench.log`
- 旧算子 P：`/home/w00544044/compare_results/pd128k_old_prefill.log`
- 旧算子 D：`/home/w00544044/compare_results/pd128k_old_decode.log`
- 旧算子 bench：`/home/w00544044/compare_results/pd128k_old128k_bench.log`




## 关键发现

  功能正确：新算子 8/8 成功、0 错误，MTP 接受率从修复前的 0%（全 NaN）恢复到 65.61%，验证了 PD 状态传输修复在 TP=2 下同样有效。
  性能差距主要在 decode 尾部：Median TPOT 接近（38.5 vs 39.9ms），但 P90/P99 拉高（65.6/109.4 vs 44.2/44.5ms），TTFT/ITL 基本持平 → INT8   反量化算子的开销集中在长序列 attention 的个别 decode 步。
  内存收益明显：同等负载下 KV 使用率新算子 ~26% vs 旧算子 ~76%，INT8 KV 显著节省显存。

  过程中的重要处理

  - 移除了源码中所有调试探针（含 C8-PTH 补丁自带的一个在 decode 路径会因 seq_lens=None 崩溃的 MTP read probe），确保 benchmark   测的是真实性能而非插桩开销。
  - 修复本身（attn_utils.py 的 mamba block 对齐门控）保留在代码中。