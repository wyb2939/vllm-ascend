# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Unit tests for the FIA antiquant-mode-5 (per-token-per-head) C8 backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionState,
    AscendC8PerTokenHeadAttentionBackendImpl,
)
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.utils import AscendDeviceType

_BLOCK = 128
_HEADS = 4
_Q_HEADS = 8
_HEAD_DIM = 64


class TestAscendC8PerTokenHeadBackendImpl(TestBase):
    """Dispatch tests: antiquant mode 5 parameters travel to FIA, the paged
    scale caches go whole, and the per-(token, head) cache write quantizes
    and scatters K/V plus both scale rows."""

    def setUp(self):
        super().setUp()
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.mock_vllm_config.cache_config.cache_dtype = "int8"
        self.mock_vllm_config.cache_config.block_size = _BLOCK
        self.mock_vllm_config.kv_transfer_config = None

        self.config_patcher = patch(
            "vllm_ascend.attention.attention_v1.get_current_vllm_config", return_value=self.mock_vllm_config
        )
        self.utils_config_patcher = patch(
            "vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.mock_vllm_config
        )
        # Pin the supported device generation (A3) so the hardware gate is
        # deterministic on the CPU runner.
        self.profile_patcher = patch(
            "vllm_ascend.attention.attention_v1.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A3),
        )
        self.config_patcher.start()
        self.utils_config_patcher.start()
        self.profile_patcher.start()
        self.addCleanup(self.utils_config_patcher.stop)
        self.addCleanup(self.config_patcher.stop)
        self.addCleanup(self.profile_patcher.stop)

        self.layer = MagicMock()
        self.layer.layer_name = "test_layer"
        self.layer._k_scale_float = 1.0
        self.layer._v_scale_float = 1.0

    def _make_impl(self):
        return AscendC8PerTokenHeadAttentionBackendImpl(
            num_heads=_Q_HEADS,
            head_size=_HEAD_DIM,
            scale=1.0,
            num_kv_heads=_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="int8",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
        )

    def _make_kv_cache(self, num_blocks=4):
        # (k, v, k_scale, v_scale): fp32 scale caches shaped (blocks, heads, bs, 1).
        k = torch.zeros(num_blocks, _BLOCK, _HEADS, _HEAD_DIM, dtype=torch.int8)
        v = torch.zeros_like(k)
        ks = torch.ones(num_blocks, _HEADS, _BLOCK, dtype=torch.float32)
        vs = torch.ones_like(ks)
        return (k, v, ks, vs)

    def _make_metadata(self, state, batch=2):
        metadata = MagicMock()
        metadata.attn_state = state
        metadata.seq_lens_list = [10, 20]
        metadata.block_tables = torch.zeros(batch, 2, dtype=torch.long)
        metadata.slot_mapping = torch.arange(batch)
        metadata.num_actual_tokens = batch
        metadata.num_decode_tokens = batch if state is not AscendAttentionState.ChunkedPrefill else 1
        metadata.num_decodes = batch if state is not AscendAttentionState.ChunkedPrefill else 1
        metadata.num_prefills = 0 if state is not AscendAttentionState.ChunkedPrefill else 1
        metadata.actual_seq_lengths_q = [batch] if state is not AscendAttentionState.ChunkedPrefill else [1, 3]
        metadata.causal = True
        metadata.model_runner_type = None
        metadata.attn_mask = None
        return metadata

    @staticmethod
    def _mock_dynamic_quant(tokens, heads, head_dim):
        """Patch DeviceOperator.npu_dynamic_quant with a per-(token, head)
        symmetric int8 quantizer emitting fp32 scales, mirroring the real
        operator's contract for a (T, H, D) input."""

        def _quant(x, dynamic_scale=None, *, act_quant_type=torch.float32, **kwargs):
            del dynamic_scale, kwargs
            assert act_quant_type == torch.int8
            absmax = x.float().abs().amax(dim=-1, keepdim=True)
            scale = (absmax / 127).clamp(min=torch.finfo(torch.float32).tiny)
            int8 = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
            return int8, scale.squeeze(-1)

        return patch("vllm_ascend.attention.attention_v1.DeviceOperator.npu_dynamic_quant", side_effect=_quant)

    @patch("vllm_ascend.attention.attention_v1.notify_kv_cache_written")
    @patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache")
    def test_cache_write_quantizes_per_token_head(self, mock_device_scatter, mock_notify):
        tokens = 2
        key = torch.randn(tokens, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        quant_patcher = self._mock_dynamic_quant(tokens, _HEADS, _HEAD_DIM)
        with quant_patcher as mock_dynquant:
            impl = self._make_impl()
            kv_cache = self._make_kv_cache()
            query = torch.randn(tokens, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
            output = torch.empty_like(query)
            metadata = self._make_metadata(AscendAttentionState.DecodeOnly)

            impl.forward(self.layer, query, key, value, kv_cache, metadata, output)

        # The dynamic quantization operator ran once per K and V.
        self.assertEqual(mock_dynquant.call_count, 2)
        # The impl bound the fp32 scale caches and scattered INT8 K/V via the
        # device operator.
        self.assertIs(impl.k_scale_cache, kv_cache[2])
        self.assertEqual(kv_cache[2].dtype, torch.float32)
        mock_device_scatter.assert_called_once()
        scattered_key = mock_device_scatter.call_args.kwargs["key"]
        self.assertEqual(scattered_key.dtype, torch.int8)
        mock_notify.assert_called_once()

    @patch("vllm_ascend.attention.attention_v1.torch_npu.npu_fused_infer_attention_score")
    def test_decode_passes_antiquant_mode_5_with_paged_scales(self, mock_fia):
        impl = self._make_impl()
        kv_cache = self._make_kv_cache()
        batch = 2
        query = torch.randn(batch, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        output = torch.empty_like(query)
        metadata = self._make_metadata(AscendAttentionState.DecodeOnly)
        mock_fia.return_value = (torch.ones(batch, _Q_HEADS, 1, _HEAD_DIM), None)

        impl.forward(self.layer, query, None, None, kv_cache, metadata, output)

        mock_fia.assert_called_once()
        kwargs = mock_fia.call_args.kwargs
        self.assertEqual(kwargs["antiquant_mode"], 5)
        self.assertEqual(kwargs["key_antiquant_mode"], 5)
        self.assertEqual(kwargs["value_antiquant_mode"], 5)
        # FIA's paged antiquant-scale contract is 3-D (blocks, heads, block):
        # the cache's trailing singleton dim is squeezed off (zero-copy view,
        # same storage). No separated dequant_scale_* parameters are sent.
        self.assertEqual(tuple(kwargs["key_antiquant_scale"].shape), (4, _HEADS, _BLOCK))
        self.assertEqual(tuple(kwargs["value_antiquant_scale"].shape), (4, _HEADS, _BLOCK))
        self.assertIs(kwargs["key_antiquant_scale"].untyped_storage(), kv_cache[2].untyped_storage())
        self.assertIs(kwargs["value_antiquant_scale"].untyped_storage(), kv_cache[3].untyped_storage())
        self.assertNotIn("dequant_scale_key", kwargs)
        self.assertNotIn("dequant_scale_value", kwargs)
        self.assertEqual(kwargs["input_layout"], "BNSD")

    @patch("vllm_ascend.attention.attention_v1.torch_npu.npu_fused_infer_attention_score")
    def test_prefill_cache_hit_gathers_dequant_to_dense_float(self, mock_fia):
        """TND carries NO KV antiquant (BNSD-only): a cache-hit prefill
        gathers the paged INT8 cache, dequantizes with the per-(token, head)
        scales, and calls TND with float K/V and no antiquant parameters."""
        impl = self._make_impl()
        kv_cache = self._make_kv_cache(num_blocks=2)
        # Fill block 0 with a known pattern and scales; block_table row 0
        # points at block 0 (FIA gathers via block_table, TND via dense).
        tokens, heads, head_dim = 3, _HEADS, _HEAD_DIM
        kv_cache[0].copy_(torch.randint(-32, 32, (2, _BLOCK, heads, head_dim), dtype=torch.int8))
        kv_cache[2].fill_(2.0)
        kv_cache[3].fill_(3.0)
        query = torch.randn(tokens, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        output = torch.empty_like(query)
        metadata = self._make_metadata(AscendAttentionState.PrefillCacheHit)
        metadata.actual_seq_lengths_q = [3]
        metadata.seq_lens_list = [3]
        metadata.seq_lens = torch.tensor([3], dtype=torch.int32)
        metadata.block_tables = torch.tensor([[0]], dtype=torch.long)
        mock_fia.return_value = (torch.ones(tokens, _Q_HEADS, 1, _HEAD_DIM), None)

        impl.forward(self.layer, query, None, None, kv_cache, metadata, output)

        mock_fia.assert_called_once()
        kwargs = mock_fia.call_args.kwargs
        self.assertEqual(kwargs["input_layout"], "TND")
        self.assertIsNone(kwargs["block_table"])
        # No antiquant parameters travel with TND.
        self.assertNotIn("antiquant_mode", kwargs)
        self.assertNotIn("key_antiquant_mode", kwargs)
        self.assertNotIn("key_antiquant_scale", kwargs)
        self.assertNotIn("dequant_scale_key", kwargs)
        # K/V are dense float, dequantized with the cached scales.
        torch.testing.assert_close(kwargs["key"].dtype, torch.bfloat16)
        torch.testing.assert_close(kwargs["key"], (kv_cache[0][0, :tokens] * 2.0).to(torch.bfloat16))
        torch.testing.assert_close(kwargs["value"], (kv_cache[1][0, :tokens] * 3.0).to(torch.bfloat16))

    @patch("vllm_ascend.attention.attention_v1.torch_npu.npu_fused_infer_attention_score")
    def test_chunked_prefill_decode_bnsd_mode5_prefill_tnd_float(self, mock_fia):
        """Chunked prefill: decode rows take BNSD mode 5; prefill rows take
        TND with float K/V and no antiquant parameters."""
        impl = self._make_impl()
        kv_cache = self._make_kv_cache(num_blocks=2)
        kv_cache[2].fill_(2.0)
        kv_cache[3].fill_(3.0)
        tokens = 3
        query = torch.randn(tokens, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        output = torch.empty_like(query)
        metadata = self._make_metadata(AscendAttentionState.ChunkedPrefill)
        metadata.seq_lens_list = [10, 20]
        metadata.seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        metadata.block_tables = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        mock_fia.return_value = (torch.ones(tokens, _Q_HEADS, 1, _HEAD_DIM), None)

        impl.forward(self.layer, query, None, None, kv_cache, metadata, output)

        # One decode call (BNSD, mode 5) + one prefill call (TND, float K/V).
        self.assertEqual(mock_fia.call_count, 2)
        decode_kwargs = mock_fia.call_args_list[0].kwargs
        prefill_kwargs = mock_fia.call_args_list[1].kwargs
        self.assertEqual(decode_kwargs["input_layout"], "BNSD")
        self.assertEqual(decode_kwargs["antiquant_mode"], 5)
        # 3-D squeezed view over the scale cache storage (FIA contract).
        self.assertEqual(tuple(decode_kwargs["key_antiquant_scale"].shape), (4, _HEADS, _BLOCK))
        self.assertIs(decode_kwargs["key_antiquant_scale"].untyped_storage(), kv_cache[2].untyped_storage())
        self.assertEqual(prefill_kwargs["input_layout"], "TND")
        self.assertNotIn("antiquant_mode", prefill_kwargs)
        self.assertNotIn("key_antiquant_scale", prefill_kwargs)
        self.assertIsNone(prefill_kwargs["block_table"])
        torch.testing.assert_close(prefill_kwargs["key"].dtype, torch.bfloat16)

    @patch("vllm_ascend.attention.attention_v1.torch_npu.npu_fused_infer_attention_score")
    def test_chunked_prefill_continuing_chunk_gathers_full_history(self, mock_fia):
        """Continuing chunk (seq_len 2557 > q_len 893, the 1664-threshold
        garbled-output bug): forward must NOT treat this chunk's own K/V as
        the attended keys — the prefill TND call gathers + dequantizes the
        FULL cached history (kvlen = seq_len cumsum, not the chunk rows)."""
        impl = self._make_impl()
        num_blocks = 4
        kv_cache = self._make_kv_cache(num_blocks=num_blocks)
        kv_cache[2].fill_(2.0)
        kv_cache[3].fill_(3.0)
        # Fill block 0 (history) and block 1 (new rows' slots) distinctly.
        kv_cache[0].copy_(torch.randint(-64, 64, kv_cache[0].shape, dtype=torch.int8))
        kv_cache[1].copy_(torch.randint(-64, 64, kv_cache[1].shape, dtype=torch.int8))

        chunk_tokens = 3  # the continuing chunk's new rows
        seq_len = 2 * _BLOCK + chunk_tokens  # > one scheduler block
        query = torch.randn(chunk_tokens, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        key = torch.randn(chunk_tokens, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        output = torch.empty_like(query)
        metadata = self._make_metadata(AscendAttentionState.ChunkedPrefill)
        metadata.num_actual_tokens = chunk_tokens
        metadata.seq_lens_list = [seq_len]
        metadata.seq_lens = torch.tensor([seq_len], dtype=torch.int32)
        metadata.block_tables = torch.tensor([[0, 1]], dtype=torch.long)
        metadata.slot_mapping = torch.arange(_BLOCK, _BLOCK + chunk_tokens)
        mock_fia.return_value = (torch.ones(chunk_tokens, _Q_HEADS, 1, _HEAD_DIM), None)

        impl.forward(self.layer, query, key, value, kv_cache, metadata, output)

        mock_fia.assert_called_once()
        kwargs = mock_fia.call_args.kwargs
        self.assertEqual(kwargs["input_layout"], "TND")
        self.assertNotIn("antiquant_mode", kwargs)
        # The attended keys are the FULL cached history (seq_len rows), not
        # the chunk's own rows.
        self.assertEqual(kwargs["key"].shape[0], seq_len)
        self.assertEqual(kwargs["actual_seq_lengths_kv"].tolist(), [seq_len])
        # Gathered rows follow the block table: row 0 of block 0, row bs of
        # block 1, and the chunk's own rows landed in block 1 (slot >= bs).
        torch.testing.assert_close(kwargs["key"][:_BLOCK], (kv_cache[0][0] * 2.0).to(torch.bfloat16))
        self.assertGreater(kwargs["key"].shape[0], _BLOCK)

    def test_quantization_matches_dynamic_quant_operator_semantics(self):
        impl = self._make_impl()
        key = torch.randn(5, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        with patch(
            "vllm_ascend.attention.attention_v1.DeviceOperator.npu_dynamic_quant",
            side_effect=lambda x, dynamic_scale=None, *, act_quant_type=torch.float32, **kw: (
                torch.zeros_like(x, dtype=torch.int8),
                torch.ones_like(x[..., 0], dtype=torch.float32),
            ),
        ) as mock_dynquant:
            impl._quantize_kv_per_token_head(key, value, 5)

        # npu_dynamic_quant invoked once per K and per V, in int8 mode.
        self.assertEqual(mock_dynquant.call_count, 2)
        first_call = mock_dynquant.call_args_list[0]
        self.assertEqual(first_call.kwargs.get("act_quant_type"), torch.int8)
        torch.testing.assert_close(first_call.args[0], key)

    @patch("vllm_ascend.attention.attention_v1.notify_kv_cache_written")
    @patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache")
    def test_scale_cache_scatter_writes_slot_rows(self, mock_device_scatter, mock_notify):
        tokens = 2
        key = torch.randn(tokens, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        query = torch.randn(tokens, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        output = torch.empty_like(query)
        quant_patcher = self._mock_dynamic_quant(tokens, _HEADS, _HEAD_DIM)
        with quant_patcher:
            impl = self._make_impl()
            kv_cache = self._make_kv_cache(num_blocks=4)
            # Slots in different blocks to exercise the block/offset decomposition.
            metadata = self._make_metadata(AscendAttentionState.DecodeOnly)
            metadata.slot_mapping = torch.tensor([5, _BLOCK + 7])

            impl.forward(self.layer, query, key, value, kv_cache, metadata, output)

            k_scale, v_scale = impl._quantize_kv_per_token_head(key, value, tokens)[2:]

        # Slots 5 -> block 0 offset 5, slot 128+7 -> block 1 offset 7.
        torch.testing.assert_close(kv_cache[2][0, :, 5, 0], k_scale[0])
        torch.testing.assert_close(kv_cache[2][1, :, 7, 0], k_scale[1])
        torch.testing.assert_close(kv_cache[3][0, :, 5, 0], v_scale[0])
        torch.testing.assert_close(kv_cache[3][1, :, 7, 0], v_scale[1])


class TestAscendC8PerTokenHeadHardwareGate(TestBase):
    """The mode-5 path is A2/A3 only; A5 must refuse it at construction and
    at forward time instead of silently running the unsupported kernel."""

    def setUp(self):
        super().setUp()
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.mock_vllm_config.cache_config.cache_dtype = "int8"
        self.mock_vllm_config.cache_config.block_size = _BLOCK
        self.mock_vllm_config.kv_transfer_config = None
        self.mock_vllm_config.model_config.dtype = torch.bfloat16

        self.config_patcher = patch(
            "vllm_ascend.attention.attention_v1.get_current_vllm_config", return_value=self.mock_vllm_config
        )
        self.config_patcher.start()
        self.addCleanup(self.config_patcher.stop)

        self.layer = MagicMock()
        self.layer.layer_name = "test_layer"
        self.layer._k_scale_float = 1.0
        self.layer._v_scale_float = 1.0

    def _patch_a5(self):
        return patch(
            "vllm_ascend.attention.attention_v1.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A5),
        )

    def test_forward_refuses_on_a5(self):
        with self._patch_a5():
            impl = AscendC8PerTokenHeadAttentionBackendImpl(
                num_heads=_Q_HEADS,
                head_size=_HEAD_DIM,
                scale=1.0,
                num_kv_heads=_HEADS,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="int8",
                logits_soft_cap=None,
                attn_type="decoder",
                kv_sharing_target_layer_name=None,
            )
            kv_cache = (
                torch.zeros(2, _BLOCK, _HEADS, _HEAD_DIM, dtype=torch.int8),
                torch.zeros_like(torch.zeros(2, _BLOCK, _HEADS, _HEAD_DIM, dtype=torch.int8)),
                torch.ones(2, _HEADS, _BLOCK, dtype=torch.float32),
                torch.ones(2, _HEADS, _BLOCK, dtype=torch.float32),
            )
            query = torch.randn(1, _Q_HEADS, _HEAD_DIM, dtype=torch.bfloat16)
            output = torch.empty_like(query)
            metadata = MagicMock()
            metadata.attn_state = AscendAttentionState.DecodeOnly
            metadata.model_runner_type = None

            with self.assertRaisesRegex(ValueError, "only supported on A2/A3"):
                impl.forward(self.layer, query, None, None, kv_cache, metadata, output)

    def test_a5_capability_absent_and_a2_a3_present(self):
        from vllm_ascend.device.hardware_profile import HardwareCapability

        self.assertFalse(
            get_hardware_profile(AscendDeviceType.A5).supports(HardwareCapability.FIA_ANTIQ_PER_TOKEN_HEAD)
        )
        self.assertTrue(get_hardware_profile(AscendDeviceType.A2).supports(HardwareCapability.FIA_ANTIQ_PER_TOKEN_HEAD))
        self.assertTrue(get_hardware_profile(AscendDeviceType.A3).supports(HardwareCapability.FIA_ANTIQ_PER_TOKEN_HEAD))

    def test_is_c8_pth_kv_quant_false_on_a5(self):
        from vllm_ascend.utils import is_c8_pth_kv_quant

        with patch(
            "vllm_ascend.utils.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A5),
        ):
            quant_config = SimpleNamespace(_has_c8_pth_recipe=True)
            vllm_config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="int8"), quant_config=quant_config)
            self.assertFalse(is_c8_pth_kv_quant(vllm_config))
