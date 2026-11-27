# 14.7 运行时证据闭环报告：C8_PTH PD 分离 MTP 接受率 0% 根因裁决（最终版）

- **测试日期**：2026-11-27
- **环境**：Ascend 910（16 chip），容器 `w00544044_OP`，vLLM v0.33.0（V1 engine，MRV2 worker）
- **模型**：`Qwen3.6-27B-w8a8-1-c8pth`（64 层 = 48 GDN + 16 full_attention，MTP 层 idx=64）
- **补丁**：`/home/w00544044/c8_pth_fia_mode5.patch`（130935 字节，md5 `6e206ace...`，含 `C8PTH MTP read probe`）
- **部署**：1P1D PD 分离，P=chips10,11（TP=2），D=chips4,5（TP=2），MooncakeConnectorV1
- **负载**：`--dataset-name random`（高熵随机 token），40K 输入（P max-batched=16384 → **3 chunk**），200 输出

---

## 0. 最终裁决（一句话）

> **根因不在 KV 传输、不在元数据、不在 P 侧写入、不在量化/scatter。**
> **D 侧 MTP 层 forward 的输入 K/V 本身在 draft 验证第 2 次迭代（step 3）起全 NaN**（key_nan=2048=4×512，absmax=nan），而**单实例同一负载全程 key_nan=0**。
> 即：**PD 分离下 D 侧喂给 MTP draft 层的 hidden states / 跨层输入变 NaN**，是 decode 引擎侧的数据流问题，与 C8-PTH 量化算子、Mooncake 传输均无关。

---

## 1. 三步证据（全部一致，排除传输/元数据/写入）

### 14.7.1 注册元数据 —— 完全对称 ✅
- MTP 层 = **idx 64**（`mtp.layers.0.self_attn.attn`，group0），P/D 双侧注册。
- `block_len_per_addr[64] = [65536,65536,1024,1024]`、`block_size_scale=[13,13,13,13]`，两侧一致。
- → 候选 a（只在 D 注册）、候选 b（block_len 不一致）**排除**。

### 14.7.2 传输几何 —— 完全一致 ✅
- `block_lens/strides/remote_strides` 全 `[65536,65536,1024,1024]`，4 分量。
- `dst_bases`(P) 与 P 注册基址吻合，方向正确（P→D 拉取）。

### 14.7.3 内容 checksum —— P 与 D 逐字节一致 ✅
- MTP 层 L64 block13：P 与 D 完全一致（TP0/TP1 均一致，含 2 个 fp32 scale）。
- → **传输完全正确**，候选 3 前半（传输错位）排除。

---

## 2. 关键运行时证据：NaN 来自 MTP 层输入（非传输/量化）

### 2.1 read probe（新补丁内置，D 侧 MTP 读取）
| step | q_len | NaN 数（使用块内） | 最后块有限 |
|---|---|---|---|
| 0–2 | 1 | 0 | ✅ |
| **3** | **4（draft 验证）** | **8（=2 slot×4 head）** | 最后块仍有限 |
| 4+ | 1/4 | 8（持续） | ✅ |

NaN 在 **D 侧第一次 q_len=4（MTP draft 验证）步产生**，位于实际使用块内，此后永久 NaN。

### 2.2 scatter probe（量化后 scale）
| step | ntok | k_scale_nan_rows |
|---|---|---|
| 0 | 4 | 0 |
| 1,2 | 1 | 0 |
| **3** | **4** | **4（全 NaN）** |
| 4+ | 1/4 | 1/4（持续） |

### 2.3 kvinput probe（**量化前的原始 K/V 输入**）—— 决定性
| step | ntok | key_nan | key_absmax |
|---|---|---|---|
| 0 | 4 | **0** | 13.4375 |
| 1 | 1 | **0** | 8.9375 |
| 2 | 1 | **0** | 7.0312 |
| **3** | **4** | **2048（全 NaN）** | **nan** |
| 4+ | 1/4 | 512/2048 | nan |

> **NaN 在量化之前就已存在**（原始 K/V 输入全 NaN，2048 = 4 token × 512 hidden）。quantize / scatter / 传输都是无辜的——它们只是忠实地把 NaN 传下去。

---

## 3. 决定性对照：单实例 vs PD 分离（同一 40K 3-chunk 随机负载）

| 部署 | kvinput key_nan | MTP 接受长度 | TPOT |
|---|---|---|---|
| **单实例（无 PD，chips10,11）** | **全程 0**（24 次采样全 0） | 1.76–2.00 | 88.19 ms |
| **PD 分离（V1，P=10,11 / D=4,5）** | **step 3 起 2048（全 NaN）** | 1.00 | 150 ms |

> **同一 MTP forward 代码、同一量化算子、同一负载：单实例输入 K/V 全程有限，PD 分离 step 3 起全 NaN。**
> 这彻底证明：**问题不在 MTP 模型本身、不在 C8-PTH 量化、不在 Mooncake 传输**，而在 **PD 分离下 D 侧喂给 MTP draft 层的输入数据**。

---

## 4. 三个候选的最终裁决

| 候选 | 证据 | 裁决 |
|---|---|---|
| a) MTP 层只在 D 注册 | 14.7.1 双侧注册 idx=64 | ❌ 排除 |
| b) 两侧 block_len 不一致 | 14.7.1/14.7.2 全对称 | ❌ 排除 |
| c) P 侧 MTP 层写入内容错误 | 14.7.3 P==D 一致 + 单实例输入有限 | ❌ 排除 |

> **三个候选全部排除。** 真正的故障点在 **D 侧 MTP draft 层的输入数据流**（PD 分离特有）。

---

## 5. 根因定位与下一步修复方向

### 现象本质
PD 分离下，D 侧 decode 引擎在 **MTP draft 验证循环的第 2 次迭代（step 3，q_len=4）** 开始，喂给 MTP 层 attention 的 **K/V 投影输入（hidden states）全 NaN**。step 0 的首次 q_len=4 正常，说明首次 draft 验证 OK，是**循环复用/状态累积后**变 NaN。

### 高优先级怀疑点（D 侧 decode，按可能性排序）
1. **MTP draft 模型的 hidden states 在 draft 循环中累积 NaN**：draft 模型每步接收主模型的 hidden state + 上一步 draft 输出，PD 分离下主模型 hidden state 的来源（是否经过 KV 传输/重算）与单实例不同，可能引入了 NaN。
2. **D 侧 MTP 层 attention 读取主模型 KV 时的 dequant**：MTP 层 attention 若读取了主模型 64 层的 dequantized KV（跨层共享），该读取路径在 PD 分离下可能读到未初始化/NaN 的块。
3. **draft 模型的 RMSNorm / RoPE 输入**：若 draft 输入 hidden 含 NaN，经过 norm/rope 后全 NaN。需在 MTP draft 模型入口（attention 之前）加探针，定位 NaN 首次出现的位置（是 draft 模型输入就 NaN，还是 draft 内部某算子产生）。

### 建议下一步探针（D 侧，主线程前向同步点）
- 在 **MTP draft 模型 forward 入口**（layer 0 之前）打印 input hidden states 的 NaN 数 / absmax。
- 在 **MTP 层 attention 的 q/k/v 投影输出**处打印 NaN 数（本次已在 k/v 输入处确认 NaN，需再往上游追一层到 qkv_proj 之前）。
- 对比单实例与 PD 分离在**同一 step（step 3）**的 draft 输入 hidden 是否一致。

---

## 6. 过程记录

| 项 | 说明 |
|---|---|
| 输入熵依赖 | 低熵文本 MTP 正常；高熵随机 token 才复现 0%。验证必须用 `--dataset-name random` |
| 新补丁 read probe bug | 补丁自带 read probe 在 `seq_lens=None` 时 `seq_lens[0]` 崩溃，已修复（用 `attn_metadata.seq_lens_list` 兜底） |
| 探针清理 | attention_v1.py 已还原到补丁原始态 + read probe bugfix，无 scatter/kvinput 残留 |
| 关键结论 | NaN 在量化前（原始 K/V 输入）就存在，单实例无此问题 → 根因在 D 侧 MTP draft 输入数据流 |

---

## 附：证据日志路径（容器内）

| 内容 | 路径 |
|---|---|
| PD 分离 D 侧（含 read/scatter/kvinput 探针） | `/home/w00544044/compare_results/serve_pd_d.log` |
| PD 分离 P 侧 | `/home/w00544044/compare_results/serve_pd_p.log` |
| 单实例对照（含 kvinput 探针） | `/home/w00544044/compare_results/single_new.log` |
| 启动脚本（PD，P=10,11/D=4,5） | `/tmp/pd_new_45.sh` |
| 启动脚本（单实例，10,11） | `/tmp/single_new.sh` |
