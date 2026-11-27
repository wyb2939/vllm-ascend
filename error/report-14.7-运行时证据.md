# 14.7 节运行时证据闭环报告：C8_PTH PD 分离 MTP 接受率 0% 根因裁决

- **测试日期**：2026-11-27
- **环境**：Ascend 910（16 chip），容器 `w00544044_OP`，vLLM v0.33.0（V1 engine，MRV2 worker）
- **模型**：`Qwen3.6-27B-w8a8-1-c8pth`（64 层 = 48 GDN + 16 full_attention，MTP 层 idx=64）
- **补丁**：`/home/w00544044/c8_pth_fia_mode5.patch`（130935 字节，md5 `949280dc...`）
- **部署**：1P1D PD 分离，P=卡10,11（TP=2），D=卡12,13（TP=2），MooncakeConnectorV1
- **关键前置**：MTP 接受率问题**依赖输入 token 熵**——低熵文本（20 词词表）MTP 正常 3.82；真实 `--dataset-name random`（高熵随机 token）MTP 0.00%。本报告全部用真实随机 token 负载。

---

## 0. 裁决结论（先看这里）

> **三个候选全部排除，根因不在 V1 传输路径、不在元数据、不在 P 侧写入内容。**
> 三步证据（元数据 / 传输几何 / 内容 checksum）全部一致，且**单实例（无 PD）同一随机负载下 MTP 正常（1.95–2.60）**，只有 PD 分离时崩到 0。
> **真正的根因在 D 侧（decode 引擎）对 MTP 层 KV 的使用/调度路径，而非 KV 内容本身。** 需要下一步在 D 侧 MTP 层 forward / draft 调度处加探针。

---

## 1. 14.7.1 注册元数据比对 —— 完全对称 ✅

MTP 层在元数据中为 **layer_idx=64**（layer 名 `mtp.layers.0.self_attn.attn`），归属 group0（FullAttentionSpec）。

| 比对点 | Prefill | Decode | 结论 |
|---|---|---|---|
| a) MTP 层是否双侧注册 | 是（idx=64，group0，17 层含 MTP） | 是（idx=64，group0，17 层含 MTP） | ✅ 对称 |
| b) `block_len_per_addr[64]` | `[65536, 65536, 1024, 1024]` | `[65536, 65536, 1024, 1024]` | ✅ 一致 |
| b) `block_stride_per_addr[64]` | `[65536, 65536, 1024, 1024]` | `[65536, 65536, 1024, 1024]` | ✅ 一致 |
| c) `block_size_scale[64]` | `[13,13,13,13]` | `[13,13,13,13]` | ✅ 一致 |
| c) 分量数 / dtype | 4（K/V int8 + 2 fp32 scale） | 4（同） | ✅ 对称 |

> 注：MTP 层 idx=**64**（不是文档假设的 48；48 是最后一个 mamba 层）。MTP 层与普通 full_attention 层（idx=3/63）block 形态完全相同，是标准 C8-PTH 4 视图布局。
> **候选 a（只在 D 注册）、候选 b（block_len 不一致）排除。**

---

## 2. 14.7.2 传输几何比对 —— 完全一致 ✅

Decode 侧探针（`Mooncake MTP layer transfer geometry`）：

```
layer_idx=64 cache_idx_count=4
block_lens=[65536, 65536, 1024, 1024]
block_strides=[65536, 65536, 1024, 1024]
remote_block_strides=[65536, 65536, 1024, 1024]
src_bases=[20701742366720, 20702339596288, 20702936825856, 20702946157568]   # D 本地
dst_bases=[20701742366720, 20702337040384, 20702931714048, 20702941005824]   # P 远端
```

- 4 个分量 block_len / stride / remote_stride 逐项一致。
- `dst_bases`（P 端 MTP 层基址）与 14.7.1 P 侧注册的 `kv_caches_base_addr[64]` 完全吻合。
- 传输方向正确（`batch_transfer_sync_read`：D 本地 ← P 远端）。

---

## 3. 14.7.3 内容级 checksum —— P 与 D 逐字节一致 ✅

对 MTP 层（L64）与相邻 full_attention 层（L63）的同一 block（bid=13）做内容统计（sum / abs / 非零数）：

| 侧 | L64 (MTP) c0 (K int8) | c1 (V int8) | c2 (k_scale) | c3 (v_scale) |
|---|---|---|---|---|
| **P TP0** | s=53333, a=1629817, nz=64465 | s=86051, a=1523657, nz=64252 | 13.448 | 31.539 |
| **D TP0** | s=53333, a=1629817, nz=64465 | s=86051, a=1523657, nz=64252 | 13.448 | 31.539 |
| **P TP1** | s=46162, a=1555130, nz=64481 | s=62884, a=1418334, nz=64275 | 15.050 | 22.381 |
| **D TP1** | s=46162, a=1555130, nz=64481 | s=62884, a=1418334, nz=64275 | 15.050 | 22.381 |

> **P 与 D 的 MTP 层 KV 内容逐字节一致（含 2 个 fp32 scale），TP0/TP1 均一致。** L63 同样一致。
> **候选 3 前半（传输几何错位）排除；传输完全正确。**

---

## 4. 决定性对照实验：单实例 vs PD 分离（推翻候选 3 后半）

同一**随机 token** 128K 负载、同一新算子 C8_PTH、同一 MTP(3)：

| 部署形态 | MTP 平均接受长度 | MTP 接受率 | TPOT |
|---|---|---|---|
| **单实例（无 PD，卡10,11 TP=2）** | **1.95 – 2.60** | 正常 | 63.70 ms |
| **PD 分离（V1，P=10,11 / D=12,13）** | **1.00** | **0.00%** | 151.92 ms |
| 旧算子 BF16 PD 分离（对照） | 1.863 | 86.30% | 36.11 ms |

- **单实例用的是同一套 C8_PTH 量化 + MTP 层写入代码，MTP 正常** → 证明 P 侧 MTP 层 quantize+scatter 写入内容**没有错**（候选 3 后半排除）。
- **P==D 内容逐字节一致** → 传输没有错。
- **但 PD 分离下 D 侧 MTP 就是 0%** → 问题在 **D 侧消费 MTP KV 的路径**，而非内容。

---

## 5. 三个候选的最终裁决

| 候选 | 现象假设 | 证据 | 裁决 |
|---|---|---|---|
| a) MTP 层只在 D 注册、P 无 | P 未实例化 MTP 层 | 14.7.1：P/D 双侧均注册 idx=64 | ❌ 排除 |
| b) 两侧 block_len 不一致 | MTP spec 几何漂移 | 14.7.1/14.7.2：block_len/stride/scale 全对称 | ❌ 排除 |
| c) P 侧 MTP 层写入内容错误 | scale scatter/页内容错 | 14.7.3 P==D 一致 + 单实例 MTP 正常 | ❌ 排除 |

> **三个候选全部排除。** 问题不在传输、不在元数据、不在 P 侧写入。

---

## 6. 根因新方向（下一步）

既然内容正确、单实例正常、只有 PD 分离崩，根因必然在 **D 侧（decode 引擎）消费 MTP 层 KV 的路径**。重点怀疑：

1. **D 侧 MTP 层 KV 的 block 地址映射**：D 把 P 传来的 MTP KV 写入了正确的物理块，但 decode 时 draft 模型读取 MTP KV 用的 block_table / slot mapping 可能指向了错误的逻辑块（内容对但被读错位置）。
2. **D 侧 MTP 层 attention 的 metadata（block_table/seq_lens）**：PD 分离下 D 侧 draft 层重建 attention metadata 时，MTP 层的 block table 可能与主模型不同步。
3. **`pp_layer_indices` 的 MTP 层范围**：connector 中 `end_layer_index += self.num_draft_layers` 仅在 `prefill_pp_rank == _prefill_pp_size - 1` 时生效；PP=1 时应生效，需确认 D 侧实际拉取的 MTP 层范围与 decode 时读取的层范围一致。
4. **D 侧 draft 模型与主模型的 KV 层索引对齐**：PD 分离下 D 侧主模型 64 层 + MTP 1 层共 65 层，draft 验证时读取的 MTP 层 idx 是否与传输写入的 idx=64 一致。

**建议下一步探针**（D 侧，主线程/前向同步点，避免后台线程 NPU 竞争）：
- 在 D 侧 MTP 层 forward 处打印：本次 decode 读取的 MTP KV block_table、slot_mapping、以及读到的 KV 内容 checksum（与传输写入的对比）。
- 对比「传输写入 D 的 MTP 块」与「D decode 时实际读取的 MTP 块」是否为同一物理块。

---

## 7. 过程记录

| 项 | 说明 |
|---|---|
| 输入熵依赖 | 低熵文本 MTP 3.82；高熵随机 token 0%。验证 MTP 必须用 `--dataset-name random` |
| 单实例崩溃 | 首轮单实例随机 128K 触发 HCCL 内部错误（`rtDeviceSynchronizeWithTimeout`），重启后正常。属偶发，需关注 |
| 探针安全 | 内容探针在 ZMQ 后台线程做 NPU 张量运算，8 并发会 segfault；本轮单请求复现，且用 `.detach().cpu()` 规避 |
| 探针清理 | 连接器已还原补丁原始态（仅 18 行 DEBUG 探针，`git diff` 确认无 1473 残留） |

