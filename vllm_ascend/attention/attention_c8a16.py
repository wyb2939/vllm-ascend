#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""C8A16 (INT8 KV cache, BF16 activation) attention backend.

KV is stored as INT8 with a dynamic per-(token, head) scale. The scale is
computed at scatter time from the token's own K/V values, so no checkpoint
scale is required. Decode and MTP verify steps read the paged INT8 cache
directly through ``npu_fused_infer_attention_score`` with
``key/value_antiquant_mode=5`` (per-token x per-head antiquant on
PagedAttention). Prefill falls back to gathering and dequantizing the paged
cache so the same INT8 storage serves every phase.

Layout: one contiguous backing allocation per layer, carved into four
contiguous regions ``[K | K_scale | V | V_scale]``. All four views are
contiguous, which keeps the scale scatter a plain indexed write. The backing
size is ``num_blocks * page_size_bytes`` where the spec budgets
``head_size + head_size_v + 8`` bytes per (token, head): one INT8 K byte, one
INT8 V byte, and two float32 scales.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch_npu
from vllm.v1.attention.backend import AttentionLayer, AttentionType  # type: ignore

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import FullAttentionSpec

# ``--kv-cache-dtype`` value that carries the INT8 KV payload.
C8A16_KV_CACHE_DTYPE = "int8"
# ModelSlim recipe that selects dynamic per-token x per-head INT8 KV
# quantization on top of the int8 cache dtype.
C8A16_KV_CACHE_TYPE = "C8A16"
# Bytes of scale metadata per (token, head): float32 K scale + float32 V scale.
C8A16_SCALE_BYTES_PER_HEAD = 8
# Symmetric INT8 quantization range.
INT8_QMAX = 127
INT8_QMIN = -128
# Guard against a zero-amax token turning the reciprocal into inf.
MIN_AMAX = 1e-6
# Workspace cache key for the graph-captured decode FIA call.
_C8A16_FIA_WORKSPACE_KEY = "c8a16.npu_fused_infer_attention_score.workspace"


@dataclass(frozen=True, slots=True)
class C8A16FIAParamProvider:
    """Resolve the per-step lengths the captured decode call re-reads.

    ACL-graph replay re-executes only device work, so the lengths must come
    from the runner's persistent buffers rather than the CPU lists captured
    into the graph.
    """

    layer_name: str | None

    def resolve(self, attn_metadata) -> dict[str, Any]:
        metadata = attn_metadata[self.layer_name]
        return {
            "actual_seq_lengths": metadata.actual_seq_lengths_q,
            "actual_seq_lengths_kv": metadata.seq_lens_list,
            "block_table": metadata.block_tables,
        }


def c8a16_scale_shape(num_blocks: int, block_size: int, num_kv_heads: int) -> tuple[int, int, int]:
    """Per-(token, head) scale cache shape for one of K/V.

    This is the layout ``npu_fused_infer_attention_score`` expects for
    ``key/value_antiquant_scale`` under PagedAttention with antiquant mode 5:
    ``(num_blocks, num_kv_heads, block_size)``.
    """
    return (num_blocks, num_kv_heads, block_size)


def c8a16_cache_spec(spec: "FullAttentionSpec") -> "FullAttentionSpec":
    """Budget INT8 K/V plus two float32 per-(token, head) scales."""
    k_dim, v_dim = spec.head_size, spec.head_size_v
    return replace(
        spec,
        dtype=torch.int8,
        state_content_bytes=k_dim + v_dim + C8A16_SCALE_BYTES_PER_HEAD,
    )


def c8a16_cache_views(
    raw_tensor: torch.Tensor,
    spec: "FullAttentionSpec",
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carve one raw backing into contiguous ``(K, K_scale, V, V_scale)`` views.

    ``raw_tensor`` is the per-layer int8 allocation of
    ``num_blocks * page_size_bytes`` bytes. The regions are laid out
    ``[all K][all K_scale][all V][all V_scale]``; each view is contiguous so
    the scale scatter is a simple indexed assignment.
    """
    if raw_tensor.dtype != torch.int8 or raw_tensor.ndim != 1:
        raise ValueError("C8A16 raw cache must be a contiguous 1-D int8 tensor.")
    k_dim, v_dim = spec.head_size, spec.head_size_v
    num_kv_heads, block_size = spec.num_kv_heads, spec.block_size

    k_elems = num_blocks * num_kv_heads * block_size * k_dim
    scale_elems = num_blocks * num_kv_heads * block_size
    v_elems = num_blocks * num_kv_heads * block_size * v_dim
    scale_bytes = scale_elems * 4
    expected = k_elems + 2 * scale_bytes + v_elems
    if raw_tensor.numel() != expected:
        raise ValueError(
            f"C8A16 cache allocation has {raw_tensor.numel()} bytes; expected {expected} "
            f"(num_blocks={num_blocks}, Nkv={num_kv_heads}, block={block_size}, "
            f"k_dim={k_dim}, v_dim={v_dim})."
        )

    # The float32 scale views start at an absolute byte offset of
    # ``storage_offset + k_elems`` (the raw tensor may be a slice of an aligned
    # KV-transfer allocation), which must be float32-aligned for
    # ``view(torch.float32)`` to be legal.
    base_bytes = raw_tensor.storage_offset() * raw_tensor.element_size()
    if k_elems % 4 or v_elems % 4 or (base_bytes + k_elems) % 4:
        raise ValueError(
            f"C8A16 scale offsets must be float32-aligned (base={base_bytes}, "
            f"k_elems={k_elems}, v_elems={v_elems})."
        )

    k_end = k_elems
    ks_end = k_end + scale_bytes
    v_end = ks_end + v_elems
    # K/V keep the ND paged layout the scatter operator and FIA both consume:
    # (num_blocks, block_size, num_kv_heads, head_dim).
    k_cache = raw_tensor[:k_end].view(num_blocks, block_size, num_kv_heads, k_dim)
    k_scale = raw_tensor[k_end:ks_end].view(torch.float32).view(num_blocks, num_kv_heads, block_size)
    v_cache = raw_tensor[ks_end:v_end].view(num_blocks, block_size, num_kv_heads, v_dim)
    v_scale = raw_tensor[v_end:].view(torch.float32).view(num_blocks, num_kv_heads, block_size)
    return k_cache, k_scale, v_cache, v_scale


def quantize_c8a16(
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantize K/V with a per-(token, head) scale.

    ``key``/``value`` are ``(num_tokens, num_kv_heads, head_dim)`` in the
    activation dtype. Returns int8 K/V and their float32 scales shaped
    ``(num_tokens, num_kv_heads)``.
    """
    k_float = key.float()
    v_float = value.float()
    k_scale = (k_float.abs().amax(dim=-1, keepdim=True) / INT8_QMAX).clamp(min=MIN_AMAX)
    v_scale = (v_float.abs().amax(dim=-1, keepdim=True) / INT8_QMAX).clamp(min=MIN_AMAX)
    k_int8 = torch.clamp(torch.round(k_float / k_scale), INT8_QMIN, INT8_QMAX).to(torch.int8)
    v_int8 = torch.clamp(torch.round(v_float / v_scale), INT8_QMIN, INT8_QMAX).to(torch.int8)
    return k_int8, k_scale.squeeze(-1), v_int8, v_scale.squeeze(-1)


class AscendC8A16AttentionBackendImpl(AscendAttentionBackendImpl):
    """INT8 KV cache with dynamic per-(token, head) scales.

    Installed per layer via ``layer.impl.__class__`` assignment from
    :class:`AscendC8A16KVCacheAttentionMethod`, so this subclass's
    ``__init__`` never runs. Only the forward path and the cache binding are
    overridden; everything else is inherited.
    """

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        # The base implementation scatters float K/V into the cache. C8A16
        # must quantize first, so the write stays inside forward().
        raise NotImplementedError("C8A16 KV cache update is only supported via forward().")

    def _bind_cache(self, kv_cache) -> None:
        if kv_cache is None or len(kv_cache) < 4:
            raise RuntimeError(
                "C8A16 attention requires a (k, k_scale, v, v_scale) KV cache tuple, "
                f"got {type(kv_cache)} of length {0 if kv_cache is None else len(kv_cache)}."
            )
        self.key_cache, self.key_scale_cache, self.value_cache, self.value_scale_cache = (
            kv_cache[0],
            kv_cache[1],
            kv_cache[2],
            kv_cache[3],
        )

    def _scatter_scales(
        self,
        scale_cache: torch.Tensor,
        scales: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write ``(num_tokens, Nkv)`` scales into their paged slots."""
        if slot_mapping.numel() == 0:
            return
        # Negative slots are padding sentinels; the KV scatter skips them, so
        # the scale write must skip the same rows. Mask unconditionally to keep
        # this a pure device-side op (no host sync in the hot path).
        valid = slot_mapping >= 0
        slot_mapping = slot_mapping[valid]
        scales = scales[valid]
        if slot_mapping.numel() == 0:
            return
        block_size = scale_cache.shape[2]
        block_ids = torch.div(slot_mapping, block_size, rounding_mode="floor")
        offsets = slot_mapping % block_size
        # Advanced-index assignment handles the strided paged layout; the
        # scale cache itself is contiguous per block.
        scale_cache[block_ids, :, offsets] = scales

    def _quantize_and_scatter(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
    ) -> None:
        num_actual_tokens = attn_metadata.num_actual_tokens
        slots = attn_metadata.slot_mapping[:num_actual_tokens]
        k_int8, k_scale, v_int8, v_scale = quantize_c8a16(
            key[:num_actual_tokens], value[:num_actual_tokens]
        )
        torch_npu.npu_scatter_pa_kv_cache(
            key=k_int8,
            value=v_int8,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slots,
            cache_mode="Norm",
        )
        self._scatter_scales(self.key_scale_cache, k_scale, slots)
        self._scatter_scales(self.value_scale_cache, v_scale, slots)
        notify_kv_cache_written()

    def _decode_query_bnsd(self, query: torch.Tensor, num_tokens: int, batch_size: int, q_s: int) -> torch.Tensor:
        # TND input -> BNSD; the operator requires a contiguous query.
        return (
            query[:num_tokens]
            .view(batch_size, q_s, self.num_heads, self.head_size)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _uniform_decode_qlen(self, attn_metadata: AscendMetadata, batch_size: int) -> int | None:
        """Return the shared per-request query length, or None if not uniform.

        The BNSD decode path needs one ``q_s`` for the whole batch. Adaptive
        verification can compact requests with different verify lengths, so
        callers fall back to the dequantized TND path when this returns None.
        """
        if batch_size <= 0:
            return None
        qlens = attn_metadata.actual_seq_lengths_q
        if qlens is None or len(qlens) < batch_size:
            return None
        lengths = [qlens[0]] + [qlens[i] - qlens[i - 1] for i in range(1, batch_size)]
        first = lengths[0]
        if first <= 0 or any(length != first for length in lengths):
            return None
        return first

    def _forward_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        q_s: int,
    ) -> torch.Tensor:
        """Decode / MTP verify: read paged INT8 KV with antiquant mode 5."""
        batch_size = attn_metadata.num_decodes
        num_tokens = batch_size * q_s
        block_size = self.key_cache.shape[1]

        query_bnsd = self._decode_query_bnsd(query, num_tokens, batch_size, q_s)
        kwargs = dict(
            key_antiquant_scale=self.key_scale_cache,
            value_antiquant_scale=self.value_scale_cache,
            block_table=attn_metadata.block_tables[:batch_size],
            actual_seq_lengths_kv=attn_metadata.seq_lens_list[:batch_size],
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            key_antiquant_mode=5,
            value_antiquant_mode=5,
            inner_precise=1,
        )
        if q_s > 1:
            # MTP verify: each query row sees its own causal prefix.
            kwargs["atten_mask"] = attn_metadata.attn_mask
            kwargs["sparse_mode"] = 3
            kwargs["actual_seq_lengths"] = [q_s] * batch_size
        else:
            kwargs["sparse_mode"] = 0

        # Paged FIA consumes the cache flattened to (num_blocks, block_size, Nkv*D).
        key_flat = self.key_cache.reshape(self.key_cache.shape[0], block_size, -1)
        value_flat = self.value_cache.reshape(self.value_cache.shape[0], block_size, -1)
        output_bnsd = output[:num_tokens].view(batch_size, q_s, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        if _EXTRA_CTX.capturing:
            # The task writes straight into ``output_bnsd``, a view of
            # ``output``'s storage, so the caller already sees the result.
            self._capture_decode_fia(query_bnsd, key_flat, value_flat, output_bnsd, kwargs)
            return output

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query_bnsd,
            key_flat,
            value_flat,
            **kwargs,
        )
        output[:num_tokens] = attn_output.permute(0, 2, 1, 3).reshape(num_tokens, self.num_heads, self.head_size)
        return output

    def _capture_decode_fia(
        self,
        query_bnsd: torch.Tensor,
        key_flat: torch.Tensor,
        value_flat: torch.Tensor,
        output_bnsd: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> None:
        """Register the decode FIA call as an updatable graph task.

        The lengths and block table come from the runner's persistent buffers
        via ``C8A16FIAParamProvider``, so every replay re-reads the current
        step instead of the values captured into the graph.
        """
        softmax_lse = torch.empty(1, dtype=query_bnsd.dtype, device=query_bnsd.device)
        workspace = get_capture_resource(
            _C8A16_FIA_WORKSPACE_KEY,
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query_bnsd,
                key=key_flat,
                value=value_flat,
                block_table=kwargs["block_table"],
                input_layout="BNSD",
                block_size=kwargs["block_size"],
                actual_seq_lengths=kwargs.get("actual_seq_lengths"),
                actual_seq_lengths_kv=kwargs["actual_seq_lengths_kv"],
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=kwargs["sparse_mode"],
                atten_mask=kwargs.get("atten_mask"),
                key_antiquant_scale=kwargs["key_antiquant_scale"],
                value_antiquant_scale=kwargs["value_antiquant_scale"],
                key_antiquant_mode=5,
                value_antiquant_mode=5,
                inner_precise=1,
            ),
        )
        register_task(
            torch_npu.npu_fused_infer_attention_score.out,
            {
                "query": query_bnsd,
                "key": key_flat,
                "value": value_flat,
                "block_table": kwargs["block_table"],
                "input_layout": "BNSD",
                "block_size": kwargs["block_size"],
                "actual_seq_lengths": kwargs.get("actual_seq_lengths"),
                "actual_seq_lengths_kv": kwargs["actual_seq_lengths_kv"],
                "num_key_value_heads": self.num_kv_heads,
                "num_heads": self.num_heads,
                "scale": self.scale,
                "sparse_mode": kwargs["sparse_mode"],
                "atten_mask": kwargs.get("atten_mask"),
                "key_antiquant_scale": kwargs["key_antiquant_scale"],
                "value_antiquant_scale": kwargs["value_antiquant_scale"],
                "key_antiquant_mode": 5,
                "value_antiquant_mode": 5,
                "inner_precise": 1,
                "workspace": workspace,
                "out": [output_bnsd, softmax_lse],
            },
            C8A16FIAParamProvider(self._layer_name),
        )

    def _gather_dequant_kv(
        self,
        block_table: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 K/V for prefill and dequantize to ``target_dtype``."""
        block_size = self.key_cache.shape[1]
        max_blocks = block_table.shape[1]
        max_tokens = max_blocks * block_size

        # Padding rows may carry a sentinel block id; clamp so they index block 0
        # and are dropped by the length mask below.
        safe_table = block_table.clamp(min=0)

        def _gather(cache: torch.Tensor, scale_cache: torch.Tensor) -> torch.Tensor:
            gathered = cache[safe_table]  # (B, max_blocks, Bs, Nkv, D)
            gathered = gathered.reshape(block_table.shape[0], max_tokens, self.num_kv_heads, -1)
            # Scale cache is (num_blocks, Nkv, block_size); reorder to the
            # token-major (B, max_blocks, block_size, Nkv) the K/V gather uses.
            scales = scale_cache[safe_table].permute(0, 1, 3, 2).reshape(
                block_table.shape[0], max_tokens, self.num_kv_heads, 1
            )
            return gathered.to(target_dtype) * scales.to(target_dtype)

        dense_k = _gather(self.key_cache, self.key_scale_cache)
        dense_v = _gather(self.value_cache, self.value_scale_cache)

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=dense_k.device)
        positions = torch.arange(max_tokens, dtype=torch.long, device=dense_k.device)
        valid = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).reshape(-1)
        return dense_k.reshape(-1, self.num_kv_heads, dense_k.shape[-1])[valid], dense_v.reshape(
            -1, self.num_kv_heads, dense_v.shape[-1]
        )[valid]

    def _forward_prefill_dequant(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill fallback: dequantize the paged cache and run FIA TND."""
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        seq_lens = attn_metadata.seq_lens_list

        key, value = self._gather_dequant_kv(attn_metadata.block_tables, seq_lens, query.dtype)
        cumulative_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[:num_tokens],
            key=key.contiguous(),
            value=value.contiguous(),
            atten_mask=attn_metadata.attn_mask,
            block_table=None,
            input_layout="TND",
            block_size=self.key_cache.shape[1],
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=cumulative_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        output[:num_tokens] = attn_output.view(num_tokens, self.num_heads, self.head_size)
        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        self._layer_name = layer.layer_name

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for AscendC8A16AttentionBackendImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)

        self._bind_cache(kv_cache)
        if (
            key is not None
            and value is not None
            and self.kv_sharing_target_layer_name is None
            and self.attn_type != AttentionType.ENCODER_ONLY
        ):
            self._quantize_and_scatter(key, value, attn_metadata)

        if attn_metadata.model_runner_type == "pooling":
            raise NotImplementedError("C8A16 attention does not support pooling models.")

        # Decode and MTP verify have no prefill requests; everything else
        # (chunked prefill, prefill-cache-hit) takes the dequant fallback.
        if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes > 0:
            q_s = self._uniform_decode_qlen(attn_metadata, attn_metadata.num_decodes)
            if q_s is not None:
                return self._forward_decode(query, attn_metadata, output, q_s)
            # Adaptive verification can compact requests with different verify
            # lengths; the BNSD decode call needs one q_s, so dequantize instead.
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            # No cached KV exists yet; the current tokens are the whole KV.
            return self._forward_prefill_nocache(query, key, value, attn_metadata, output)
        return self._forward_prefill_dequant(query, attn_metadata, output)

    def _forward_prefill_nocache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[:num_tokens],
            key=key[:num_tokens].contiguous(),
            value=value[:num_tokens].contiguous(),
            atten_mask=attn_metadata.attn_mask,
            block_table=None,
            input_layout="TND",
            block_size=self.key_cache.shape[1],
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_qlen,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        output[:num_tokens] = attn_output.view(num_tokens, self.num_heads, self.head_size)
        return output
