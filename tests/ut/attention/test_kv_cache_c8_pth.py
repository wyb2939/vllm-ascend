# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Unit tests for the C8 per-(token, head) KV cache layout primitives."""

import math
from dataclasses import replace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.attention_c8_pth import (
    C8_PTH_SCALE_DTYPE,
    c8_pth_cache_spec,
    c8_pth_cache_views_for_spec,
    c8_pth_section_bytes,
    c8_pth_state_content_bytes,
)


def _spec(ratio=1, heads=2, head_size=128, head_size_v=128, padding=0):
    spec = c8_pth_cache_spec(
        FullAttentionSpec(
            block_size=128 * ratio,
            num_kv_heads=heads,
            head_size=head_size,
            head_size_v=head_size_v,
            dtype=torch.int8,
        )
    )
    return replace(spec, page_size_padded=spec.page_size_bytes + padding * ratio)


def test_scales_are_fp32():
    assert C8_PTH_SCALE_DTYPE is torch.float32


def test_state_content_bytes_adds_both_scales():
    # 128 (K int8) + 128 (V int8) + 2 * 4 (K/V fp32 scales).
    assert c8_pth_state_content_bytes(128, 128) == 264


def test_spec_keeps_real_head_dimensions_and_metadata():
    original = FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=128, head_size_v=128, dtype=torch.int8)
    spec = c8_pth_cache_spec(original)
    assert type(spec) is FullAttentionSpec
    assert (spec.head_size, spec.head_size_v) == (128, 128)
    assert spec.dtype is torch.int8
    assert spec.state_content_bytes == 264
    # page = block_size * num_kv_heads * state_content_bytes.
    assert spec.page_size_bytes == 128 * 2 * 264
    assert original.state_content_bytes is None


def test_spec_rejects_non_int8_dtype():
    with pytest.raises(ValueError, match="requires --kv-cache-dtype int8"):
        c8_pth_cache_spec(FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=128, dtype=torch.bfloat16))


@pytest.mark.parametrize("ratio", [1, 2, 8])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("head_size_v", [64, 128])
@pytest.mark.parametrize("padding", [0, 256])
def test_page_views_partition_payload_and_preserve_padding(ratio, heads, head_size_v, padding):
    spec = _spec(ratio, heads, head_size_v=head_size_v, padding=padding)
    # Even base offset: the allocator's 2 MiB alignment guarantees it in
    # production; the scale sections additionally require it for their
    # 4-byte element view.
    offset, num_pages, guard = 128, 2, 97
    backing = torch.full((offset + num_pages * spec.page_size_bytes + guard,), 93, dtype=torch.int8)
    raw = backing[offset:-guard]
    k, v, ks, vs = c8_pth_cache_views_for_spec(raw, spec, 128, C8_PTH_SCALE_DTYPE)
    kernel_blocks = num_pages * ratio
    k_bytes, v_bytes, ks_bytes, vs_bytes = c8_pth_section_bytes(kernel_blocks, 128, heads, 128, head_size_v)
    assert k.shape == (kernel_blocks, 128, heads, 128)
    assert v.shape == (kernel_blocks, 128, heads, head_size_v)
    # FIA's paged antiquant-scale contract is 3-D (blocks, heads, block).
    assert ks.shape == (kernel_blocks, heads, 128)
    assert vs.shape == (kernel_blocks, heads, 128)
    assert k.dtype is torch.int8 and v.dtype is torch.int8
    assert ks.dtype is torch.float32 and vs.dtype is torch.float32
    # Section-major layout: every view is fully contiguous (required by the
    # FIA tiling path — get_max_workspace rejects per-page-interleaved
    # stride[0] = page pitch with 561002 during graph capture).
    for view in (k, v, ks, vs):
        assert view.is_contiguous()
    # Partition: the four sections own consecutive byte ranges across the
    # whole allocation (K-all | V-all | K_scale-all | V_scale-all), with the
    # per-page padding bytes of every kernel block inside the K/V runs.
    # Fill each view's bytes with a distinct marker so untouched guard bytes
    # stay distinguishable from written payload bytes.
    for marker, view in enumerate((k, v, ks, vs), start=1):
        view.view(torch.uint8).fill_(marker)
    start = 0
    for marker, size in zip((1, 2, 3, 4), (k_bytes, v_bytes, ks_bytes, vs_bytes), strict=True):
        assert (raw.view(torch.uint8)[start : start + size] == marker).all()
        start += size
    # Section payloads end at `start`; any allocation tail beyond the four
    # sections (a padded page budget has none, the unpadded one none either)
    # and the offset/guard bands stay untouched.
    assert (raw.view(torch.uint8)[start:] == 93).all()
    assert (backing[:offset] == 93).all()
    assert (backing[-guard:] == 93).all()


def test_views_are_mutually_consistent_with_scale_roundtrip():
    """Quantize a known tensor, scatter into the views, dequantize back and
    check the error bound of the per-(token, head) scheme."""
    spec = _spec()
    num_blocks, heads, head_dim = spec.block_size, 2, 128
    raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8)
    k, v, ks, vs = c8_pth_cache_views_for_spec(raw, spec, 128, C8_PTH_SCALE_DTYPE)

    tokens = 4
    kv_float = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16) * 3
    # Mirror npu_dynamic_quant: fp32 absmax scale over the head dim.
    absmax = kv_float.float().abs().amax(dim=-1, keepdim=True)
    scale = (absmax / 127).clamp(min=torch.finfo(torch.float32).tiny)
    int8 = torch.round(kv_float.float() / scale).clamp(-127, 127).to(torch.int8)

    # Write through the views at token slots 0..3 of block 0. The scale
    # cache is (block, head, token): a [tokens, heads] scale lands
    # transposed across the head-major axes.
    k[0, :tokens] = int8
    ks[0, :, :tokens] = scale.squeeze(-1).transpose(0, 1)

    # fp32 multiply mirrors the operator's in-kernel dequantization.
    dequant = k[0, :tokens].to(torch.float32) * ks[0, :, :tokens].transpose(0, 1).unsqueeze(-1)
    max_err = (dequant - kv_float.float()).abs().max().item()
    # torch.round is half-to-even: a .5 quotient can round either way, so
    # bound at one full ULP of the scale.
    assert max_err <= scale.max().item() + 1e-6


@pytest.mark.parametrize("case", ["bad_ratio", "raw_short"])
def test_invalid_views_fail_before_as_strided(case):
    if case == "bad_ratio":
        spec = _spec(ratio=2)
        raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.int8)
        with pytest.raises(ValueError, match="whole kernel blocks"):
            c8_pth_cache_views_for_spec(raw, spec, 96, C8_PTH_SCALE_DTYPE)
        return
    spec = _spec()
    raw = torch.zeros(2 * spec.page_size_bytes, dtype=torch.int8)
    with pytest.raises(ValueError, match="whole scheduler pages"):
        c8_pth_cache_views_for_spec(raw[:-1], spec, 128, C8_PTH_SCALE_DTYPE)


def test_views_are_independent_of_hybrid_padded_pages():
    """Hybrid pools pad both groups' pages to a common size; the section
    geometry must derive from the payload budget, not from the padded page,
    so the padded tail never leaks into the views (the 1664-threshold
    non-deterministic corruption family)."""
    from dataclasses import replace as dc_replace

    spec = _spec()
    padded = dc_replace(spec, page_size_padded=spec.page_size_bytes + 173056)
    assert padded.page_size_bytes > spec.page_size_bytes
    nb_blocks = 2
    k_pad, v_pad, ks_pad, vs_pad = c8_pth_cache_views_for_spec(
        torch.zeros(nb_blocks * padded.page_size_bytes, dtype=torch.int8), padded, 128, C8_PTH_SCALE_DTYPE
    )
    k_unpad, v_unpad, ks_unpad, vs_unpad = c8_pth_cache_views_for_spec(
        torch.zeros(nb_blocks * spec.page_size_bytes, dtype=torch.int8), spec, 128, C8_PTH_SCALE_DTYPE
    )
    # Identical geometry whether or not the scheduler padded the page.
    assert k_pad.shape == k_unpad.shape
    assert ks_pad.shape == ks_unpad.shape
    assert all(v.is_contiguous() for v in (k_pad, v_pad, ks_pad, vs_pad))
    # The padded allocation's tail bytes are unused payload-wise.
    payload = sum(
        math.prod(s) * (4 if t is torch.float32 else 1)
        for s, t in zip(
            (k_pad.shape, v_pad.shape, ks_pad.shape, vs_pad.shape),
            (torch.int8, torch.int8, torch.float32, torch.float32),
        )
    )
    tail = torch.zeros(nb_blocks * padded.page_size_bytes, dtype=torch.int8)
    k_p2, v_p2, ks_p2, vs_p2 = c8_pth_cache_views_for_spec(tail, padded, 128, C8_PTH_SCALE_DTYPE)
    del k_p2, v_p2, ks_p2, vs_p2
    assert tail[payload:].count_nonzero() == 0
