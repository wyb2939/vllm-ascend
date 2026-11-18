# -*- coding: utf-8 -*-
"""FIA NPU workload for external msprof profiling.

Runs BF16 KV and INT8 KV + antiquant mode 5 in that order for each case.
Each variant runs --warmup calls, synchronizes, then runs --iters calls
and synchronizes again. msprof collects performance data externally.
External whole-process profiling also captures setup and warmup operations.
summary.json records case configuration and execution status only.

    msprof --application="python test_fia_msprof.py --warmup 10 --iters 100"
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def run_iterations(fn, warmup: int, iters: int) -> None:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()


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


def _cases() -> list[Case]:
    specs = [
        # ("bsnd_d256_s1", "BSND", 32, 4, 3, 1, 256, (131072, ) * 32, 0),
        # ("bsnd_d256_s1", "BSND", 30, 3, 3, 1, 256, (131072, ) * 30, 0),
        # ("bsnd_d256_s1", "BSND", 16, 4, 3, 1, 256, (102400, ) * 16, 0),
        # ("bsnd_d256_s1", "BSND", 30, 3, 3, 1, 256, (65536, ) * 30, 0),
        # ("bsnd_d256_s1", "BSND", 16, 2, 3, 1, 256, (65536, ) * 16, 0),
        # ("bsnd_d256_s1", "BSND", 32, 3, 3, 1, 256, (32768, ) * 32, 0),
        # ("bsnd_d256_s1", "BSND", 16, 3, 3, 1, 256, (32768, ) * 16, 0),
        # ("bsnd_d256_s1", "BSND", 32, 2, 3, 1, 256, (16384, ) * 32, 0),
        # ("bsnd_d256_s1", "BSND", 8, 4, 3, 1, 256, (16384, ) * 8, 0),
        # ("bsnd_d256_s1", "BSND", 2, 3, 3, 1, 256, (16384, ) * 2, 0),
        # ("bsnd_d256_s1", "BSND", 24, 4, 3, 1, 256, (8192, ) * 24, 0),
        # ("bsnd_d256_s1", "BSND", 12, 2, 3, 1, 256, (8192, ) * 12, 0),
        # ("bsnd_d256_s1", "BSND", 16, 3, 3, 1, 256, (4096, ) * 16, 0),
        # ("bsnd_d256_s1", "BSND", 4, 2, 3, 1, 256, (4096, ) * 4, 0),
        # ("bsnd_d256_s1", "BSND", 16, 3, 3, 1, 256, (2048, ) * 16, 0),
        ("bsnd_d256_s1", "BSND", 8, 4, 3, 1, 256, (1024, ) * 8, 0),
    ]
    cases = []
    for spec in specs:
        case = Case(*spec)
        if len(case.kv_lens) != case.batch:
            raise RuntimeError(f"{case.name}: kv_lens {len(case.kv_lens)} != batch {case.batch}")
        if case.num_heads % case.num_kv_heads != 0:
            raise RuntimeError(f"{case.name}: Nq not divisible by Nkv")
        cases.append(case)
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
    return cache, k_scale, block_table


def _bf16_cache(cache: torch.Tensor, scale: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """INT8 cache 乘 per-(token, head) scale，得到原算子要的 BF16 分页 KV。"""
    num_blocks, block_size, hidden = cache.shape
    head_dim = hidden // num_kv_heads
    values = cache.view(num_blocks, block_size, num_kv_heads, head_dim).float()
    scales = scale.permute(0, 2, 1).unsqueeze(-1)
    return (values * scales).to(torch.bfloat16).view(num_blocks, block_size, hidden).contiguous()


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


def run_case(case: Case, verbose: bool, warmup: int, iters: int) -> dict:
    row = {"name": case.name, "fail": False, **asdict(case), "warmup": warmup, "iters": iters}
    row["kv_lens"] = list(case.kv_lens)
    gen = torch.Generator().manual_seed(_seed(case.name))
    query = _build_query(case, gen)
    key, k_scale, block_table = _pack_kv(case, gen)
    value = key.clone()
    v_scale = k_scale.clone()

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
    
    for label, fn in (("BF16 KV", run_orig), ("INT8 KV", run_new)):
        try:
            print(f"{case.name}: {label}, warmup={warmup}, iters={iters}", flush=True)
            run_iterations(fn, warmup, iters)
        except Exception as exc:
            _note(row, f"{label} {type(exc).__name__}: {exc}")
            if verbose:
                traceback.print_exc()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="FIA NPU workload for external msprof profiling")
    parser.add_argument("--bench", action="store_true", help="Compatibility flag; workload execution is always enabled")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be >= 0 and --iters must be > 0")

    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    print("FIA 伪量化")
    print("  原算子 BF16 KV + npu_fused_infer_attention_score（不带 antiquant）")
    print("  改后   INT8 KV + scale，antiquant mode 5")
    print()

    rows = []
    for case in _cases():
        try:
            row = run_case(case, args.verbose, args.warmup, args.iters)
        except Exception as exc:
            row = {**asdict(case), "fail": True, "error": f"Setup {type(exc).__name__}: {exc}"}
            if args.verbose:
                traceback.print_exc()
        rows.append(row)
        print(f"{case.name}: {'FAIL' if row.get('fail') else 'ok'}", flush=True)

    print()
    print(f"msprof workload completed: warmup={args.warmup} iters={args.iters}")
    for row in rows:
        if row.get("error"):
            print(f"{row['name']}: FAIL {row['error']}")
    save_json(OUT_DIR / "summary.json", rows)
    failed = count_failed(rows)
    print(f"\n{len(rows) - failed} 组跑完，{failed} 组失败。")
    print(f"结果目录: {OUT_DIR}")
    return 1 if failed else 0

# msprof --application="python test_fia_msprof.py --warmup 2 --iters 5"
if __name__ == "__main__":
    sys.exit(main())
