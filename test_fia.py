# -*- coding: utf-8 -*-
"""FIA 伪量化（antiquant mode 5）的精度和性能。

同一份 INT8 KV 和 scale：
  1. CPU：反量化到 FP32，再做因果 attention
  2. 原算子：先反量化成 BF16，再调 npu_fused_infer_attention_score（不带 antiquant）
  3. 改后算子：INT8 KV + scale 直接走 antiquant mode 5

attention 的误差是测量结果，这里不设阈值。某条调用失败才算 FAIL。

    python test_fia.py
    python test_fia.py --bench
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def metrics(ref: torch.Tensor, got: torch.Tensor) -> dict[str, float]:
    g = ref.detach().float().cpu().reshape(-1)
    x = got.detach().float().cpu().reshape(-1)
    if g.shape != x.shape:
        raise RuntimeError(f"shape {tuple(got.shape)} != {tuple(ref.shape)}")
    if g.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0, "cosine": 1.0}
    diff = (x - g).abs()
    denom = g.abs().clamp(min=1e-3)
    cosine = torch.dot(g, x) / (g.norm() * x.norm()).clamp(min=1e-12)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "cosine": float(cosine),
    }


def fill_pairs(row: dict, cpu: torch.Tensor | None, orig: torch.Tensor | None, new: torch.Tensor | None) -> None:
    if cpu is not None and new is not None:
        row["new_cpu"] = metrics(cpu, new)
    if cpu is not None and orig is not None:
        row["orig_cpu"] = metrics(cpu, orig)
    if orig is not None and new is not None:
        row["new_orig"] = metrics(orig, new)


def mark_nan(row: dict) -> None:
    for key in ("new_cpu", "orig_cpu", "new_orig"):
        stats = row.get(key)
        if stats is not None and not math.isfinite(stats["max_abs"]):
            row["fail"] = True
            row["error"] = row.get("error") or "结果里有 NaN 或 Inf"
            return


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _pair(stats: dict | None) -> str:
    if not stats:
        return f"{'-':>20}"
    return f"{stats['max_abs']:.3e}/{stats['cosine']:.6f}"


def print_precision(rows: list[dict]) -> None:
    print(f"{'case':<24} {'改后vsCPU':>20} {'原算子vsCPU':>20} {'改后vs原算子':>20}")
    print("格子是 max_abs/cosine")
    print("-" * 88)
    for row in rows:
        line = f"{row['name']:<24} {_pair(row.get('new_cpu'))} {_pair(row.get('orig_cpu'))} {_pair(row.get('new_orig'))}"
        if row.get("error"):
            line += f"  FAIL {row['error']}"
        print(line)


def print_bench(rows: list[dict]) -> None:
    timed = [row for row in rows if "orig_ms" in row and "new_ms" in row]
    if not timed:
        return
    print(f"\n{'case':<24} {'原算子ms':>10} {'改后ms':>10} {'改后/原':>10}")
    print("-" * 58)
    for row in timed:
        ratio = row["new_ms"] / row["orig_ms"] if row["orig_ms"] else float("inf")
        print(f"{row['name']:<24} {row['orig_ms']:10.3f} {row['new_ms']:10.3f} {ratio:10.3f}")


def save_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def count_failed(rows: list[dict]) -> int:
    return sum(1 for row in rows if row.get("fail"))

OUT_DIR = Path(__file__).resolve().parent / "out"
BLOCK_SIZE = 128
MASK_SIZE = 2048


@dataclass(frozen=True)
class Case:
    name: str
    layout: str
    batch: int
    q_s: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    kv_lens: tuple[int, ...]
    inner_precise: int


def _cases(seed: int) -> list[Case]:
    """Generate 500 distinct configurations within the current Sq=2..4 limits.

    Balance BNSD, BSH and BSND across KV lengths up to 32 * 1024. BSND with small
    B*Nkv supplies FD candidates; actual dispatch depends on device cores.
    Bound the expanded CPU golden KV to keep each case practical.
    """
    rng = random.Random(seed)
    cases = []
    seen = set()
    boundary_lengths = (127, 128, 129, 255, 256, 257, 1023, 1024,
                        1025, 2047, 2048, 2049, 4095, 4096, 8191, 8192,
                        16383, 16384, 32767, 32768)
    while len(cases) < 500:
        category = len(cases) % 4
        layout = ("BNSD", "BSH", "BSND", "BSND")[category]
        batch = rng.randint(1, 32)
        q_s = rng.randint(2, 4)
        head_dim = rng.choice((128, 256))
        num_kv_heads = rng.choice((1, 2, 4, 8))
        group = rng.choice((1, 2, 3, 4, 8))
        num_heads = num_kv_heads * group
        if num_heads > 256 or num_kv_heads * head_dim >= 65536:
            continue
        # Long BSND candidates keep all query rows in one S1 tile and BN small.
        if category == 3 and (batch * num_kv_heads > 4 or group * q_s > 128):
            continue
        max_length = 32 * 1024
        lengths = [
            rng.choice(boundary_lengths) if rng.random() < 0.5
            else rng.randint(q_s, max_length)
            for _ in range(batch)
        ]
        if any(length > max_length for length in lengths):
            continue
        if category == 3:
            lengths[rng.randrange(batch)] = rng.randint(2048, max_length)
        kv_lens = tuple(lengths)
        if batch * num_heads * max(kv_lens) * head_dim > 8 * 1024 * 1024:
            continue
        inner_precise = rng.choice((0, 1))
        signature = (layout, batch, q_s, num_heads, num_kv_heads,
                     head_dim, kv_lens, inner_precise)
        if signature in seen:
            continue
        seen.add(signature)
        name = (f"random_{len(cases):03d}_{layout.lower()}_b{batch}"
                f"_s{q_s}_n{num_heads}_kv{num_kv_heads}_d{head_dim}")
        cases.append(Case(name, *signature))
    return cases


def _seed(name: str) -> int:
    acc = 0
    for ch in name:
        acc = (acc * 131 + ord(ch)) % (2**31 - 1)
    return acc or 1


def _build_query(case: Case, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(
        case.batch, case.num_heads, case.q_s, case.head_dim, generator=gen, dtype=torch.float32
    )
    q = q.to(torch.bfloat16)
    if case.layout == "BNSD":
        return q
    if case.layout == "BSND":
        return q.permute(0, 2, 1, 3).contiguous()
    if case.layout == "BSH":
        return q.permute(0, 2, 1, 3).reshape(case.batch, case.q_s, case.num_heads * case.head_dim).contiguous()
    raise RuntimeError(case.layout)


def _pack_kv(case: Case, gen: torch.Generator):
    """逻辑 KV 是 BNSD。量化后按页存放：(num_blocks, block_size, Nkv*D)。"""
    max_len = max(case.kv_lens)
    logical = torch.randn(
        case.batch, case.num_kv_heads, max_len, case.head_dim, generator=gen, dtype=torch.float32
    )
    for b, length in enumerate(case.kv_lens):
        if length < max_len:
            logical[b, :, length:, :] = 0

    amax = logical.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale_logical = amax / 127.0
    qmax = torch.iinfo(torch.int8).max
    int8_logical = torch.clamp(torch.round(logical / scale_logical), -qmax, qmax).to(torch.int8)

    max_blocks = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = torch.full((case.batch, max_blocks), -1, dtype=torch.int32)
    counts = [(length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in case.kv_lens]
    order = torch.randperm(sum(counts), generator=gen)
    cursor = 0
    for b, count in enumerate(counts):
        block_table[b, :count] = order[cursor : cursor + count].to(torch.int32)
        cursor += count
    num_blocks = int(order.numel())

    cache = torch.zeros(num_blocks, BLOCK_SIZE, case.num_kv_heads * case.head_dim, dtype=torch.int8)
    k_scale = torch.ones(num_blocks, case.num_kv_heads, BLOCK_SIZE, dtype=torch.float32)
    for b, length in enumerate(case.kv_lens):
        for t in range(length):
            blk = int(block_table[b, t // BLOCK_SIZE])
            off = t % BLOCK_SIZE
            cache[blk, off, :] = int8_logical[b, :, t, :].reshape(-1)
            k_scale[blk, :, off] = scale_logical[b, :, t, 0]
    return cache, k_scale.clone(), block_table, int8_logical, scale_logical.squeeze(-1)


def _bf16_cache(cache: torch.Tensor, scale: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """INT8 cache 乘 per-(token, head) scale，得到原算子要的 BF16 分页 KV。"""
    num_blocks, block_size, hidden = cache.shape
    head_dim = hidden // num_kv_heads
    values = cache.view(num_blocks, block_size, num_kv_heads, head_dim).float()
    scales = scale.permute(0, 2, 1).unsqueeze(-1)
    return (values * scales).to(torch.bfloat16).view(num_blocks, block_size, hidden).contiguous()


def _query_bnsd(case: Case, query: torch.Tensor) -> torch.Tensor:
    if case.layout == "BNSD":
        return query
    if case.layout == "BSND":
        return query.permute(0, 2, 1, 3).contiguous()
    if case.layout == "BSH":
        return query.reshape(case.batch, case.q_s, case.num_heads, case.head_dim).permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(case.layout)


def _golden(case: Case, query: torch.Tensor, int8_logical: torch.Tensor, scale_logical: torch.Tensor) -> torch.Tensor:
    """scale_logical: (B, Nkv, S)。反量化后在 FP32 里做因果 attention。"""
    q = _query_bnsd(case, query).float()
    kv = int8_logical.float() * scale_logical.unsqueeze(-1)
    groups = case.num_heads // case.num_kv_heads
    if groups > 1:
        kv = kv.repeat_interleave(groups, dim=1)
    scale = case.head_dim**-0.5
    outs = []
    for b, length in enumerate(case.kv_lens):
        qb = q[b, :, : case.q_s, :]
        kb = kv[b, :, :length, :]
        score = torch.matmul(qb, kb.transpose(-1, -2)) * scale
        if case.q_s > 1:
            qi = torch.arange(case.q_s)[:, None]
            kj = torch.arange(length)[None, :]
            allow = kj <= (length - case.q_s + qi)
            score = score.masked_fill(~allow.unsqueeze(0), torch.finfo(torch.float32).min)
        prob = torch.softmax(score, dim=-1)
        outs.append(torch.matmul(prob, kb))
    out = torch.stack(outs, dim=0)
    if case.layout == "BNSD":
        return out
    if case.layout == "BSND":
        return out.permute(0, 2, 1, 3).contiguous()
    return out.permute(0, 2, 1, 3).reshape(case.batch, case.q_s, case.num_heads * case.head_dim).contiguous()


def _call_fia(case: Case, query, key, value, block_table, k_scale=None, v_scale=None, mask=None):
    import torch_npu

    kwargs = dict(
        num_heads=case.num_heads,
        num_key_value_heads=case.num_kv_heads,
        scale=case.head_dim**-0.5,
        block_table=block_table,
        block_size=BLOCK_SIZE,
        input_layout=case.layout,
        actual_seq_lengths_kv=list(case.kv_lens),
        inner_precise=case.inner_precise,
    )
    if mask is not None:
        kwargs["atten_mask"] = mask
        kwargs["sparse_mode"] = 3
        kwargs["actual_seq_lengths"] = [case.q_s] * case.batch
    if k_scale is not None:
        kwargs.update(
            key_antiquant_scale=k_scale,
            value_antiquant_scale=v_scale,
            key_antiquant_mode=5,
            value_antiquant_mode=5,
        )
    out, _ = torch_npu.npu_fused_infer_attention_score(query, key, value, **kwargs)
    return out


def _note(row: dict, text: str) -> None:
    row["fail"] = True
    row["error"] = f"{row['error']}；{text}" if row.get("error") else text


def run_case(case: Case, verbose: bool) -> dict:
    row = {"name": case.name, "fail": False, **asdict(case)}
    row["kv_lens"] = list(case.kv_lens)
    gen = torch.Generator().manual_seed(_seed(case.name))
    query = _build_query(case, gen)
    key, k_scale, block_table, int8_logical, scale_logical = _pack_kv(case, gen)
    value = key.clone()
    v_scale = k_scale.clone()
    cpu = _golden(case, query, int8_logical, scale_logical)

    device = "npu:0"
    query_n = query.to(device)
    key_n = key.to(device)
    value_n = value.to(device)
    k_scale_n = k_scale.to(device)
    v_scale_n = v_scale.to(device)
    table_n = block_table.to(device)
    bf16_k = _bf16_cache(key, k_scale, case.num_kv_heads).to(device)
    bf16_v = _bf16_cache(value, v_scale, case.num_kv_heads).to(device)
    mask = None
    if case.q_s > 1:
        mask = torch.triu(torch.ones(MASK_SIZE, MASK_SIZE, dtype=torch.int8, device=device), diagonal=1)

    def run_orig():
        return _call_fia(case, query_n, bf16_k, bf16_v, table_n, mask=mask)

    def run_new():
        return _call_fia(case, query_n, key_n, value_n, table_n, k_scale_n, v_scale_n, mask)
    
    # def run_new():
        # return _call_fia(case, query_n, bf16_k, bf16_v, table_n, mask=mask)


    orig = new = None
    try:
        orig = run_orig()
        torch.npu.synchronize()
    except Exception as exc:
        _note(row, f"原算子 {type(exc).__name__}: {exc}")
        if verbose:
            traceback.print_exc()
    try:
        new = run_new()
        torch.npu.synchronize()
    except Exception as exc:
        _note(row, f"改后算子 {type(exc).__name__}: {exc}")
        if verbose:
            traceback.print_exc()

    fill_pairs(row, cpu, None if orig is None else orig, None if new is None else new)
    mark_nan(row)
    row["_bench"] = None if orig is None or new is None else (run_orig, run_new)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="FIA 伪量化：CPU、原算子、改后算子")
    parser.add_argument("--bench", action="store_true", help="精度之后计时原算子和改后算子")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random case seed for reproducible runs")
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    print("FIA 伪量化")
    print("  CPU    FP32 反量化后做因果 attention")
    print("  原算子 BF16 KV + npu_fused_infer_attention_score（不带 antiquant）")
    print("  改后   INT8 KV + scale，antiquant mode 5")
    print()

    case_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    cases = _cases(case_seed)
    print(f"Random cases: {len(cases)}, seed={case_seed}", flush=True)
    rows = []
    for case in cases:
        row = run_case(case, args.verbose)
        row["case_seed"] = case_seed
        bench_fn = row.pop("_bench", None)
        if args.bench and bench_fn is not None and not row.get("fail"):
            orig_fn, new_fn = bench_fn
            row["orig_ms"] = time_ms(orig_fn, args.warmup, args.iters)
            row["new_ms"] = time_ms(new_fn, args.warmup, args.iters)
        rows.append(row)
        print(f"{case.name}: {'FAIL' if row.get('fail') else 'ok'}", flush=True)

    print()
    # print_precision(rows)
    if args.bench:
        print(f"\n性能  warmup={args.warmup} iters={args.iters}")
        print_bench(rows)
    save_json(OUT_DIR / "summary.json", rows)
    failed = count_failed(rows)
    print(f"\n{len(rows) - failed} 组跑完，{failed} 组失败。attention 误差未设阈值，上表即结果。")
    print(f"结果目录: {OUT_DIR}")
    return 1 if failed else 0

#python test_fia.py
if __name__ == "__main__":
    sys.exit(main())
