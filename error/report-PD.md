# C8_PTH（FIA antiquant mode 5）1P1D PD 分离 + MTP(3) + ChunkPrefill 128K 长输入对比报告（最新补丁）

- **测试日期**：2026-11-25
- **环境**：Ascend 910（16 chip），容器 `w00544044_OP`，vLLM v0.33.0（V1 engine，MRV2 worker）
- **模型**：`Qwen3.6-27B-w8a8-1-c8pth`（64 层 = 48 GDN/linear_attention + 16 full_attention，hidden 5120，heads 24，num_kv_heads 4，head_dim 256，full_attention_interval=4）
- **补丁**：`/home/w00544044/c8_pth_fia_mode5.patch`（**128895 字节，md5 `e271d117e14a6bd0117f953d54a3845d`**，最新版，含 `mooncake_hybrid_connector.py` 的 MTP 地址切片修复）
- **部署形态**：单机 1P1D PD 分离
  - **Prefill**：卡 10,11（TP=2）→ 端口 18100（`kv_producer`，KV 端口 31000）
  - **Decode**：卡 12,13（TP=2）→ 端口 18200（`kv_consumer`，KV 端口 31100）
  - **Load-balance proxy**：端口 18001
  - **KV 连接器**：**`MooncakeConnectorV1`**（新补丁在 `attention_v1.py` 中新增硬性校验：C8-PTH PD 仅支持 MooncakeConnectorV1，因其按块级 raw block 传输每层全部 cache 张量——INT8 K/V + 两个 fp32 scale cache）
- **测试顺序**：先新算子（C8_PTH，int8 KV），跑完停掉，再旧算子（BF16 KV）

---

## 1. 关键结论（先看这里）

| 维度 | 新算子 C8_PTH | 旧算子 BF16 | 差异 |
|---|---|---|---|
| 成功请求 | 8/8 | 8/8 | — |
| Benchmark 总时长 | 1561.91 s | 642.76 s | **+143.0%** |
| Output 吞吐 | 26.28 tok/s | 63.86 tok/s | **-58.8%** |
| Total 吞吐 | 697.68 tok/s | 1695.36 tok/s | **-58.8%** |
| Mean TTFT | 183505.22 ms | 114834.90 ms | **+59.8%** |
| Median TTFT | 107019.67 ms | 101773.37 ms | +5.2% |
| Mean TPOT | 151.01 ms | 56.24 ms | **+168.5%** |
| Median TPOT | 151.30 ms | 37.12 ms | +307.6% |
| Mean ITL | 151.01 ms | 130.79 ms | +15.5% |
| Mean E2EL | 958117.81 ms | 402753.32 ms | +137.9% |
| **MTP 接受率（加权）** | **0.00%**（1/123,141） | **44.33%** | **严重劣化** |
| **MTP 平均接受长度** | **1.002** | **1.936** | **严重劣化** |

> **核心结论：应用最新补丁（含 `mooncake_hybrid_connector.py` 的 MTP 地址切片修复）后，在 MooncakeConnectorV1 的 1P1D PD 分离下，新算子 C8_PTH 的 MTP 接受率仍为 0.00%（123,141 个 draft token 仅 1 个被接受），MTP 依然完全失效。** 该补丁的 hybrid connector 修复**未覆盖** C8-PTH PD 实际使用的 MooncakeConnectorV1 路径，因此 PD 分离下的 MTP 崩塌问题**未解决**。

---

## 2. 最新补丁分析（关键背景）

### 2.1 补丁内容

最新补丁（128895 字节，md5 `e271d117...`）相比上一版（122313 字节，md5 `a4ddfe48...`）主要变化：

1. **`attention_v1.py`（+762 行）**：大幅扩充 C8-PTH 注意力实现，新增 **PD 分离连接器硬性校验**：
   ```python
   if self.vllm_config.kv_transfer_config is not None and not uses_mooncake_connector(...):
       raise NotImplementedError(
           "C8 per-token-head v1 PD disaggregation (kv_transfer) is only "
           "supported with MooncakeConnectorV1, whose block-level transfer "
           "registers and moves every per-layer cache tensor (INT8 K/V plus "
           "both fp32 scale caches) as raw blocks."
       )
   ```
   即 **C8-PTH PD 必须用 MooncakeConnectorV1**（其他连接器会丢弃 fp32 scale cache，导致反量化出垃圾）。

2. **`mooncake_hybrid_connector.py`（+30 行）**：修复 MTP 层范围传输的地址切片——按**层边界**（而非统一 per-layer 计数）切片扁平地址列表，因为不同 cache 类型的每层组件数不同（mamba conv/ssm=2，C8-PTH attention k/v/k_scale/v_scale=4）。原逻辑用统一计数会导致首次类型切换后所有层地址错位，MTP 接受率崩到 ~0。

3. 其他：`kv_cache_config_builder.py`、`attn_utils.py`、`modelslim_config.py`、`utils.py`（新增 `is_c8_pth_kv_quant`）、`patch_model_runner.py`、`hardware_profile.py` 等。

### 2.2 关键矛盾

- **补丁的 MTP 修复在 `mooncake_hybrid_connector.py`（MooncakeHybridConnector）**；
- **但 C8-PTH PD 的硬性校验强制要求 `MooncakeConnectorV1`（`mooncake_connector.py`）**；
- **`mooncake_connector.py`（V1）本次补丁未修改**（`git diff` 显示该文件无改动，`layer_component_counts` 计数为 0）。

**因此：本次补丁的 MTP 修复路径（hybrid connector）与 C8-PTH PD 实际可用的连接器（V1）不匹配，MTP 崩塌问题在 V1 路径下未修复。** 实测验证：新算子 MTP 接受率 0.00%。

---

## 3. vLLM 启动参数（两算子除 KV dtype 外完全一致）

### 公共环境变量

| 变量 | 值 |
|---|---|
| `MODEL` | `/mnt/share/weight/Qwen3.6-27B-w8a8-1-c8pth` |
| `LD_LIBRARY_PATH` | 追加 `/usr/local/Ascend/cann-9.2.1/opp/vendors/custom_transformer/op_api/lib/` 与 mooncake 包目录 |
| `PYTORCH_NPU_ALLOC_CONF` | `max_split_size_mb:256` |
| `VLLM_HOST_IP` / `HCCL_IF_IP` | `80.5.17.122` |
| `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` / `HCCL_SOCKET_IFNAME` | `enp48s3u1u1` |
| `HCCL_EXEC_TIMEOUT` / `HCCL_CONNECT_TIMEOUT` | 204 / 120 |
| `VLLM_MOONCAKE_CONNECTOR_TIMEOUT` | 1800 |
| `VLLM_RPC_TIMEOUT` | 3600000 |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 30000 |
| `OMP_PROC_BIND` / `OMP_NUM_THREADS` | false / 1 |

### Prefill 引擎（卡 10,11，TP=2，端口 18100）

| 参数 | 新算子 C8_PTH | 旧算子 BF16 |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | 10,11 | 10,11 |
| `--quantization` | `ascend` | （无） |
| `--kv-cache-dtype` | `int8` | `bfloat16` |
| `--tensor-parallel-size` | 2 | 2 |
| `--max-model-len` | 139264 | 139264 |
| `--max-num-batched-tokens` | 16384 | 16384 |
| `--max-num-seqs` | 8 | 8 |
| `--gpu-memory-utilization` | 0.9 | 0.9 |
| `--trust-remote-code` / `--enforce-eager` | ✓ / ✓ | ✓ / ✓ |
| `--no-enable-prefix-caching` | ✓ | ✓ |
| `--enable-chunked-prefill` | ✓ | ✓ |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3}` | 同左 |
| `--kv-transfer-config` | `MooncakeConnectorV1` / `kv_producer` / port 31000 / engine_id 0 / prefill tp=2, decode tp=2 | 同左 |

### Decode 引擎（卡 12,13，TP=2，端口 18200）

| 参数 | 新算子 C8_PTH | 旧算子 BF16 |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | 12,13 | 12,13 |
| `--quantization` | `ascend` | （无） |
| `--kv-cache-dtype` | `int8` | `bfloat16` |
| `--tensor-parallel-size` | 2 | 2 |
| `--max-model-len` | 139264 | 139264 |
| `--max-num-batched-tokens` | 512 | 512 |
| `--max-num-seqs` | 8 | 8 |
| `--gpu-memory-utilization` | 0.9 | 0.9 |
| `--enable-chunked-prefill` | ✓ | ✓ |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":3}` | 同左 |
| `--kv-transfer-config` | `MooncakeConnectorV1` / `kv_consumer` / port 31100 / engine_id 1 | 同左 |

### Proxy

```
python3 examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --port 18001 --host 0.0.0.0 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 18100 \
  --decoder-hosts 127.0.0.1 --decoder-ports 18200
```

---

## 4. 压测负载参数（128K 长输入）

| 项 | 值 |
|---|---|
| 数据集 | `random`（随机 token） |
| 输入长度 | 131072（`--random-range-ratio 0.01` → 实际采样 [129761, 132383]） |
| 输出长度 | 5120（ratio 0.01 → 实际采样 [5068, 5172]） |
| `--num-prompts` | 8 |
| `--max-concurrency` | 8（满并发） |
| `--request-rate` | inf（一次性打满） |
| `--ignore-eos` | ✓（固定输出长度） |
| `--seed` | 42 |
| 实际总输入 token | 1,048,660（两算子完全一致） |
| 实际总输出 token | 41,044（两算子完全一致） |
| 峰值并发 | 8 |

> 说明：因 `--ignore-eos`，两算子生成 token 总数完全一致（41,044），保证吞吐对比公平。

---

## 5. 性能对比详细结果

### 5.1 吞吐与延迟总表

| 指标 | 新 C8_PTH | 旧 BF16 | 差异 |
|---|---|---|---|
| Successful requests | 8 | 8 | — |
| Failed requests | 0 | 0 | — |
| Benchmark duration (s) | 1561.91 | 642.76 | +143.0% |
| Output token throughput (tok/s) | 26.28 | 63.86 | -58.8% |
| Peak output token throughput (tok/s) | 49.00 | 56.00 | -12.5% |
| Total token throughput (tok/s) | 697.68 | 1695.36 | -58.8% |

### 5.2 TTFT（Time to First Token，含 P→D KV 传输）

| 指标 | 新 C8_PTH | 旧 BF16 | 差异 |
|---|---|---|---|
| Mean TTFT (ms) | 183505.22 | 114834.90 | +59.8% |
| Median / P50 TTFT (ms) | 107019.67 | 101773.37 | +5.2% |
| P90 TTFT (ms) | 354551.15 | 194605.29 | +82.2% |
| P99 TTFT (ms) | 751353.08 | 272738.19 | +175.5% |

> TTFT 解读：P50 TTFT 两算子接近（107s vs 102s），说明单请求 128K prefill 本身差异不大。新算子 P90/P99 显著更高（355s/751s vs 195s/273s），是因为 decode 极慢（MTP 失效）导致 P→D 流水阻塞、后续请求排队加剧。

### 5.3 TPOT（Time per Output Token，除首 token）

| 指标 | 新 C8_PTH | 旧 BF16 | 差异 |
|---|---|---|---|
| Mean TPOT (ms) | 151.01 | 56.24 | **+168.5%** |
| Median / P50 TPOT (ms) | 151.30 | 37.12 | +307.6% |
| P90 TPOT (ms) | 151.53 | 116.39 | +30.2% |
| P99 TPOT (ms) | 151.56 | 116.42 | +30.2% |

> TPOT 是本次劣化核心。新算子 TPOT 几乎恒定在 151ms（P50≈P90≈P99），符合"MTP 全部被拒、每步固定跑 1+3 个 draft 前向"的特征。旧算子中位 TPOT 仅 37ms（MTP 接受率 44%，每步平均出 1.94 token）。

### 5.4 ITL（Inter-Token Latency）

| 指标 | 新 C8_PTH | 旧 BF16 | 差异 |
|---|---|---|---|
| Mean ITL (ms) | 151.01 | 130.79 | +15.5% |
| Median / P50 ITL (ms) | 150.36 | 130.25 | +15.5% |
| P90 ITL (ms) | 156.86 | 136.22 | +15.2% |
| P99 ITL (ms) | 165.95 | 144.11 | +15.2% |

### 5.5 E2E 延迟

| 指标 | 新 C8_PTH | 旧 BF16 | 差异 |
|---|---|---|---|
| Mean E2EL (ms) | 958117.81 | 402753.32 | +137.9% |
| Median / P50 E2EL (ms) | 888131.92 | 320288.74 | +177.3% |
| P90 E2EL (ms) | 1127562.11 | 624188.23 | +80.6% |
| P99 E2EL (ms) | 1518453.12 | 640891.00 | +136.9% |

---

## 6. MTP 投机解码统计（本次最关键发现）

| 指标 | 新 C8_PTH | 旧 BF16 |
|---|---|---|
| 统计样本数（10s 窗口） | 156 | 64 |
| 累计 Accepted tokens | 1 | 23,434 |
| 累计 Drafted tokens | 123,141 | 52,866 |
| **加权接受率** | **0.00%** | **44.33%** |
| **平均接受长度（逐样本均值）** | **1.002** | **1.936** |

### 结论

- 旧算子 MTP 正常工作（44.33%，平均接受长度 1.94）。
- 新算子 MTP **完全失效**（0.00%，123,141 个 draft token 仅 1 个被接受）。
- **最新补丁未能修复 PD 分离下的 MTP 崩塌**——因为补丁的 MTP 地址修复在 `MooncakeHybridConnector`，而 C8-PTH PD 强制使用 `MooncakeConnectorV1`（该文件本次未修改）。

### 历史对照（累积证据链）

| 部署模式 | 连接器 | 新算子接受率 | 旧算子接受率 | 补丁版本 |
|---|---|---|---|---|
| 单实例 num_spec=1 | 无 | 77% | 76% | 旧 |
| PD 分离 8K | V1 | 0.50% | 59.45% | 旧 |
| PD 分离 128K | V1 | ~0.00% | 66.19% | 旧 |
| **混部 128K** | 无 | **57.43%** | 82.45% | 旧 |
| **PD 分离 128K（本轮）** | **V1** | **0.00%** | **44.33%** | **最新** |

> **证据链完整结论**：新算子 MTP 在混部（无跨节点 KV 传输）下正常（57.43%），在 PD 分离下（无论新旧补丁、V1 连接器）均崩塌到 ~0%。**根因确定在 PD 分离的 KV 传输路径（MooncakeConnectorV1 + C8 多视图 KV 的 MTP 层范围传输），而非 C8 int8 KV 量化算子本身。** 最新补丁的 hybrid connector 修复未覆盖 V1 路径。

### 排查建议（P0）

1. **将 `mooncake_hybrid_connector.py` 的层边界地址切片修复移植到 `mooncake_connector.py`（V1）**：V1 的 MTP 层范围传输同样需要按层边界（heterogeneous per-layer 组件数）切片，而非统一 per-layer 计数。这是最可能的修复点。
2. 或**解除 `attention_v1.py` 中对 MooncakeHybridConnector 的禁用**，让 C8-PTH PD 走已修复的 hybrid connector（需确认 hybrid connector 支持 C8 的 4 视图 raw block 传输）。
3. 对比 V1 与 hybrid connector 的 `register_kv_caches` / MTP 层范围传输逻辑，定位 V1 下 C8 多视图 KV 的地址错位。

---

## 7. KV Cache 容量

| 项 | 新 C8_PTH | 旧 BF16 |
|---|---|---|
| NPU KV cache size（prefill） | 1,012,565 tokens | 1,060,230 tokens |
| NPU KV cache size（decode） | 1,016,917 tokens | 1,064,374 tokens |
| 139K 单请求最大并发 | 7.27x / 7.30x | 7.61x / 7.64x |

> 8 请求 × ~133K ≈ 1.06M 略超 decode 容量，部分请求排队，两算子一致，不影响相对对比。

---

## 8. 过程问题与解决方案

| # | 问题 | 现象 | 根因 | 解决方案 |
|---|---|---|---|---|
| 1 | 补丁应用冲突 | `test_attn_utils_v2.py` patch 不适用 | UT 文件上下文与当前 HEAD 不符 | `git apply --exclude='tests/ut/worker/test_attn_utils_v2.py'` 跳过（不影响运行） |
| 2 | `compare_results` 目录丢失 | 启动脚本报 "No such file or directory" | 目录被清理（磁盘清理） | `mkdir -p /home/w00544044/compare_results` 重建 |
| 3 | MooncakeHybridConnector 启动即崩 | `NotImplementedError: C8 per-token-head v1 PD ... only supported with MooncakeConnectorV1` | 新补丁在 `attention_v1.py` 新增硬性校验，C8-PTH PD 仅允许 V1 | 改用 `MooncakeConnectorV1` |
| 4 | MTP 接受率仍 0% | 新补丁 + V1 下接受率 0.00% | 补丁的 MTP 修复在 hybrid connector，V1 未修复 | 见第 6 节排查建议（P0，待修） |

---

## 9. 待关注事项

1. **PD 分离 MTP 崩塌未修复（P0）**：最新补丁的 MTP 地址切片修复在 `MooncakeHybridConnector`，但 C8-PTH PD 强制用 `MooncakeConnectorV1`（未修复）。实测新算子接受率仍 0.00%。**建议将层边界切片修复移植到 V1，或解除对 hybrid connector 的禁用。**
2. **TPOT 恒定 151ms**：新算子每步 decode 时间稳定但极慢，符合 MTP 全拒 + 固定 3 draft 前向特征。
3. **吞吐 -58.8%、TTFT +59.8%**：均由 MTP 失效引起；混部已验证（57.43% 接受率、TPOT 持平）C8 算子本身正常。

---

## 附：原始日志路径（容器内）

| 内容 | 路径 |
|---|---|
| 新算子 prefill 日志 | `/home/w00544044/compare_results/pdfix_new_prefill.log` |
| 新算子 decode 日志 | `/home/w00544044/compare_results/pdfix_new_decode.log` |
| 新算子 proxy 日志 | `/home/w00544044/compare_results/pdfix_new_proxy.log` |
| 新算子压测结果 | `/home/w00544044/compare_results/pdfix_new_bench.log` |
| 旧算子 prefill 日志 | `/home/w00544044/compare_results/pdfix_old_prefill.log` |
| 旧算子 decode 日志 | `/home/w00544044/compare_results/pdfix_old_decode.log` |
| 旧算子 proxy 日志 | `/home/w00544044/compare_results/pdfix_old_proxy.log` |
| 旧算子压测结果 | `/home/w00544044/compare_results/pdfix_old_bench.log` |
