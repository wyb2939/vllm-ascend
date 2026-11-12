# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.attention_c8a16 import (
    AscendC8A16AttentionBackendImpl,
    c8a16_cache_spec,
    c8a16_cache_views,
    quantize_c8a16,
)


def _spec(head_size=8, head_size_v=8, num_kv_heads=2, block_size=4):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size_v,
        dtype=torch.bfloat16,
    )


def _impl():
    impl = object.__new__(AscendC8A16AttentionBackendImpl)
    impl.num_heads = 4
    impl.num_kv_heads = 2
    impl.head_size = 8
    impl.scale = 8**-0.5
    impl.kv_sharing_target_layer_name = None
    impl.attn_type = None
    return impl


def test_cache_spec_budgets_int8_kv_and_two_float32_scales():
    spec = c8a16_cache_spec(_spec(head_size=128, head_size_v=256))
    assert spec.dtype == torch.int8
    # One INT8 K byte + one INT8 V byte + two float32 scales per (token, head).
    assert spec.state_content_bytes == 128 + 256 + 8
    assert spec.page_size_bytes == 4 * 2 * (128 + 256 + 8)


def test_cache_views_partition_backing_into_four_contiguous_regions():
    spec = c8a16_cache_spec(_spec())
    num_blocks = 3
    raw = torch.arange(num_blocks * spec.page_size_bytes, dtype=torch.int32).to(torch.int8)
    k, k_scale, v, v_scale = c8a16_cache_views(raw, spec, num_blocks)

    assert k.shape == (num_blocks, 4, 2, 8)
    assert k_scale.shape == (num_blocks, 2, 4)
    assert v.shape == (num_blocks, 4, 2, 8)
    assert v_scale.shape == (num_blocks, 2, 4)
    for view in (k, k_scale, v, v_scale):
        assert view.is_contiguous()

    # Layout is [K][K_scale][V][V_scale]; storage offsets are in each view's
    # own element units, so compare byte offsets instead.
    k_bytes = num_blocks * 4 * 2 * 8
    scale_bytes = num_blocks * 2 * 4 * 4
    v_bytes = num_blocks * 4 * 2 * 8
    expected_bytes = [0, k_bytes, k_bytes + scale_bytes, k_bytes + scale_bytes + v_bytes]
    # Each scale region is (num_blocks, Nkv, block_size) float32.
    actual_bytes = [t.storage_offset() * t.element_size() for t in (k, k_scale, v, v_scale)]
    assert actual_bytes == expected_bytes
    assert k_scale.dtype == torch.float32 and v_scale.dtype == torch.float32


def test_cache_views_reject_wrong_allocation_size():
    spec = c8a16_cache_spec(_spec())
    with pytest.raises(ValueError, match="expected"):
        c8a16_cache_views(torch.zeros(8, dtype=torch.int8), spec, 1)


def test_quantize_uses_per_token_per_head_scale():
    torch.manual_seed(0)
    key = torch.randn(5, 2, 8, dtype=torch.bfloat16)
    value = torch.randn(5, 2, 8, dtype=torch.bfloat16)

    k_int8, k_scale, v_int8, v_scale = quantize_c8a16(key, value)

    assert k_int8.dtype == torch.int8 and v_int8.dtype == torch.int8
    assert k_scale.shape == (5, 2) and v_scale.shape == (5, 2)
    assert k_scale.dtype == torch.float32
    # Symmetric scale: amax / 127, and the max-magnitude element maps to +-127.
    expected = key.float().abs().amax(dim=-1) / 127.0
    torch.testing.assert_close(k_scale, expected)
    assert k_int8.abs().max() <= 127
    # Dequantization error stays within one quantization step.
    dequant = k_int8.float() * k_scale.unsqueeze(-1)
    assert (dequant - key.float()).abs().max() <= k_scale.max() + 1e-6


def test_quantize_guards_zero_amax_token():
    key = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    value = torch.zeros(1, 1, 4, dtype=torch.bfloat16)
    k_int8, k_scale, _, _ = quantize_c8a16(key, value)
    assert torch.isfinite(k_scale).all()
    assert (k_int8 == 0).all()


def test_scatter_scales_writes_paged_slots_and_skips_padding():
    impl = _impl()
    scale_cache = torch.zeros(2, 2, 4, dtype=torch.float32)
    scales = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # slot 0 -> block 0 offset 0; slot 5 -> block 1 offset 1; -1 is padding.
    slots = torch.tensor([0, 5, -1], dtype=torch.int64)

    impl._scatter_scales(scale_cache, scales, slots)

    torch.testing.assert_close(scale_cache[0, :, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(scale_cache[1, :, 1], torch.tensor([3.0, 4.0]))
    # Padding row must not be written anywhere.
    assert scale_cache.sum() == 1.0 + 2.0 + 3.0 + 4.0


def test_scatter_scales_noop_on_empty_slots():
    impl = _impl()
    scale_cache = torch.zeros(1, 1, 4, dtype=torch.float32)
    impl._scatter_scales(scale_cache, torch.zeros(0, 1), torch.zeros(0, dtype=torch.int64))
    assert scale_cache.sum() == 0


def test_gather_dequant_restores_token_major_values():
    impl = _impl()
    num_blocks, block_size = 2, 4
    impl.key_cache = torch.zeros(num_blocks, block_size, 2, 8, dtype=torch.int8)
    impl.value_cache = torch.zeros(num_blocks, block_size, 2, 8, dtype=torch.int8)
    impl.key_scale_cache = torch.ones(num_blocks, 2, block_size, dtype=torch.float32)
    impl.value_scale_cache = torch.ones(num_blocks, 2, block_size, dtype=torch.float32)

    impl.key_cache[0, 0] = 7
    impl.key_cache[0, 1] = 3
    impl.key_scale_cache[0, :, 0] = 2.0
    impl.key_scale_cache[0, :, 1] = 0.5
    impl.value_cache[0, 0] = -4
    impl.value_scale_cache[0, :, 0] = 1.5

    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    dense_k, dense_v = impl._gather_dequant_kv(block_table, [2], torch.float32)

    assert dense_k.shape == (2, 2, 8)
    # Token 0: 7 * 2.0; token 1: 3 * 0.5.
    assert (dense_k[0] == 14.0).all()
    assert (dense_k[1] == 1.5).all()
    assert (dense_v[0] == -6.0).all()


def test_gather_dequant_clamps_padding_block_ids():
    impl = _impl()
    impl.key_cache = torch.ones(1, 4, 1, 8, dtype=torch.int8)
    impl.value_cache = torch.ones(1, 4, 1, 8, dtype=torch.int8)
    impl.key_scale_cache = torch.ones(1, 1, 4, dtype=torch.float32)
    impl.value_scale_cache = torch.ones(1, 1, 4, dtype=torch.float32)
    # A -1 padding block id must not raise; its rows are dropped by the mask.
    block_table = torch.tensor([[0, -1]], dtype=torch.int32)
    dense_k, _ = impl._gather_dequant_kv(block_table, [3], torch.float32)
    assert dense_k.shape[0] == 3


def test_forward_decode_uses_antiquant_mode_5_for_single_token():
    impl = _impl()
    impl.key_cache = torch.zeros(2, 4, 2, 8, dtype=torch.int8)
    impl.value_cache = torch.zeros(2, 4, 2, 8, dtype=torch.int8)
    impl.key_scale_cache = torch.ones(2, 2, 4, dtype=torch.float32)
    impl.value_scale_cache = torch.ones(2, 2, 4, dtype=torch.float32)

    attn_metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        block_tables=torch.zeros(2, 2, dtype=torch.int32),
        seq_lens_list=[4, 4],
        attn_mask=None,
    )
    query = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    output = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    captured = {}

    def fake_fia(q, k, v, **kwargs):
        captured.update(kwargs)
        captured["query_shape"] = tuple(q.shape)
        return torch.zeros(2, 4, 1, 8, dtype=torch.bfloat16), None

    with patch(
        "vllm_ascend.attention.attention_c8a16.torch_npu.npu_fused_infer_attention_score",
        side_effect=fake_fia,
    ):
        impl._forward_decode(query, attn_metadata, output, 1)

    assert captured["key_antiquant_mode"] == 5
    assert captured["value_antiquant_mode"] == 5
    assert captured["input_layout"] == "BNSD"
    assert captured["sparse_mode"] == 0
    assert "atten_mask" not in captured
    # Single-token decode keeps BNSD with S=1.
    assert captured["query_shape"] == (2, 4, 1, 8)


def test_forward_decode_mtp_verify_passes_causal_mask():
    impl = _impl()
    impl.key_cache = torch.zeros(2, 4, 2, 8, dtype=torch.int8)
    impl.value_cache = torch.zeros(2, 4, 2, 8, dtype=torch.int8)
    impl.key_scale_cache = torch.ones(2, 2, 4, dtype=torch.float32)
    impl.value_scale_cache = torch.ones(2, 2, 4, dtype=torch.float32)
    mask = torch.zeros(2048, 2048, dtype=torch.int8)

    attn_metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=6,
        block_tables=torch.zeros(2, 2, dtype=torch.int32),
        seq_lens_list=[4, 4],
        attn_mask=mask,
    )
    query = torch.zeros(6, 4, 8, dtype=torch.bfloat16)
    output = torch.zeros(6, 4, 8, dtype=torch.bfloat16)
    captured = {}

    def fake_fia(q, k, v, **kwargs):
        captured.update(kwargs)
        return torch.zeros(2, 4, 3, 8, dtype=torch.bfloat16), None

    with patch(
        "vllm_ascend.attention.attention_c8a16.torch_npu.npu_fused_infer_attention_score",
        side_effect=fake_fia,
    ):
        impl._forward_decode(query, attn_metadata, output, 3)

    assert captured["sparse_mode"] == 3
    assert captured["atten_mask"] is mask
    assert captured["actual_seq_lengths"] == [3, 3]


def test_uniform_decode_qlen_detects_non_uniform_verify_lengths():
    impl = _impl()
    uniform = SimpleNamespace(actual_seq_lengths_q=[3, 6, 9])
    assert impl._uniform_decode_qlen(uniform, 3) == 3
    mixed = SimpleNamespace(actual_seq_lengths_q=[2, 5, 9])
    assert impl._uniform_decode_qlen(mixed, 3) is None
    assert impl._uniform_decode_qlen(uniform, 0) is None


def test_bind_cache_requires_four_tensors():
    impl = _impl()
    with pytest.raises(RuntimeError, match="KV cache tuple"):
        impl._bind_cache((torch.zeros(1), torch.zeros(1)))
    impl._bind_cache((torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)))
    assert impl.key_scale_cache is not None
