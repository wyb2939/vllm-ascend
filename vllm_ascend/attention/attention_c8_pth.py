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
"""C8 per-token-per-head KV cache layout primitives (FIA antiquant mode 5).

Recipe: ``kv_cache_type: "C8_PER_TOKEN_HEAD"`` (ModelSlim). K/V are cached as
INT8 with DYNAMIC per-(token, head) scales computed at cache-write time; there
are no checkpoint scale tensors. Q stays BF16/FP16 and
``npu_fused_infer_attention_score`` dequantizes the paged INT8 KV in-kernel
(C8A16 pseudo-quantization, ``key/value_antiquant_mode=5``) for both the
Q_S == 1 decode path (BNSD) and the new Q_S > 1 path (TND + block_table),
replacing the gather + dequant-to-dense prefill of the static per-channel C8
path.

Operator parameter contract (FIA antiquant): when the antiquant (伪量化)
parameters and the separated KV dequant scales (KV分离量化参数,
``dequant_scale_key`` / ``dequant_scale_value``) are both supplied, the
separated parameters take precedence. This integration passes ONLY the
antiquant parameters, so the two can never conflict.

Physical layout: one combined raw allocation per layer, laid out in
section-major order ``[K | V | K_scale | V_scale]`` — each section is one
contiguous run across ALL kernel blocks (NOT per-page interleaved: the FIA
tiling path behind ``_npu_fused_infer_attention_score_get_max_workspace``
rejects stride[0] != per-block payload with 561002, "Non-contiguous cache
is supported only by the arch22 regular FAInfer or D512 MLA PA routes"):

- K / V: ``[num_kernel_blocks, block_size, num_kv_heads, head_dim]`` INT8,
  each contiguous. The attention backend passes them to the FIA calls as
  the 3-D ND views ``(num_blocks, block_size, num_kv_heads * head_dim)`` —
  the operator's paged-INT8 (C8A16) layout is ND regardless of
  ``VLLM_KV_CACHE_LAYOUT``.
- K_scale / V_scale: ``[num_blocks, num_kv_heads, block_size]`` FLOAT32
  (one scale per (block, head, token), symmetric, no offset), contiguous —
  the exact dtype and per-(token, head) granularity ``npu_dynamic_quant``
  emits for a ``(T, H, D)`` input and the 3-D shape FIA antiquant mode 5
  consumes (test_fia.py contract); no squeeze needed at the call boundary.
  The head-major order matches the ND K/V pages and serves the BNSD decode
  call unchanged. Note: TND does NOT support KV antiquant — mode 5 travels
  only on the BNSD decode calls; prefill runs float-KV TND (continuing
  chunks dequantize the paged cache to dense float first).

The spec page budget includes the scale bytes via ``state_content_bytes``
(same mechanism as the C8-MXFP packet spec), so the scheduler never
over-commits HBM for the scale caches. Section byte sizes are given by
``c8_pth_section_bytes``.

Hardware scope: this path is A2/A3 only (``HardwareCapability.
FIA_ANTIQ_PER_TOKEN_HEAD``, a member of ``_STANDARD_CAPABILITIES`` and
therefore absent from the A5 profile). The gate is enforced in three places:
``is_c8_pth_kv_quant`` (spec/allocate/view construction), the quant-config
``get_quant_method`` dispatch (loud error instead of a silent fallthrough),
and the attention impl's forward (defense in depth).
"""

import math
from dataclasses import replace

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec

# One scale element per (token, head) per K/V. The dynamic quantization
# operator (``npu_dynamic_quant``) emits FLOAT32 scales, and the operator
# consumes them as FLOAT32 per (token, head): 4 bytes each.
C8_PTH_SCALE_ELEM_BYTES = 4
C8_PTH_SCALE_DTYPE = torch.float32
# Scales per token per head: one K scale + one V scale.
C8_PTH_SCALES_PER_TOKEN_HEAD = 2
# Symmetric per-token-per-head quantization: int8 range [-127, 127].
C8_PTH_QUANT_MAX = 127


def c8_pth_state_content_bytes(head_size: int, head_size_v: int) -> int:
    """Storage bytes per (token, kv-head): INT8 K + INT8 V + both scales."""
    return head_size + head_size_v + C8_PTH_SCALES_PER_TOKEN_HEAD * C8_PTH_SCALE_ELEM_BYTES


def c8_pth_cache_spec(spec: FullAttentionSpec) -> FullAttentionSpec:
    """Extend a plain INT8 full-attention spec with the scale-cache budget.

    Same mechanism as ``mxfp_cache_spec``: the page size grows by the scale
    bytes so ``num_blocks`` derived by the scheduler accounts for them, while
    block ids and prefix caching stay unchanged.
    """
    if spec.dtype != torch.int8:
        raise ValueError(f"C8 per-token-head KV cache requires --kv-cache-dtype int8, got {spec.dtype}.")
    result = replace(
        spec,
        dtype=torch.int8,
        state_content_bytes=c8_pth_state_content_bytes(spec.head_size, spec.head_size_v),
    )
    if result.page_size_padded is not None and result.page_size_padded < result.unpadded_page_size_bytes:
        raise ValueError("C8 per-token-head padded page is smaller than its K/V and scale payload.")
    return result


def c8_pth_section_bytes(
    num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, head_size_v: int
) -> tuple[int, int, int, int]:
    """Byte sizes of the four sections over ``num_blocks`` kernel pages:
    ``(K, V, K_scale, V_scale)``."""
    tokens = num_blocks * block_size * num_kv_heads
    k_bytes = tokens * head_size
    v_bytes = tokens * head_size_v
    scale_bytes = tokens * C8_PTH_SCALE_ELEM_BYTES
    return k_bytes, v_bytes, scale_bytes, scale_bytes


def c8_pth_cache_views_for_spec(
    raw_tensor: torch.Tensor,
    spec: FullAttentionSpec,
    kernel_block_size: int,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the ``(k, v, k_scale, v_scale)`` views over one raw allocation.

    ``raw_tensor`` is the contiguous INT8 allocation sized
    ``num_scheduler_blocks * spec.page_size_bytes``; a scheduler page spans
    ``ratio`` kernel blocks, so all four sections scale with the kernel
    count. The four sections sit back to back across the WHOLE allocation
    (section-major order: ``[K-all | V-all | K_scale-all | V_scale-all]``),
    NOT interleaved per page: K and V then become one contiguous run each,
    which the FIA tiling path requires —
    ``_npu_fused_infer_attention_score_get_max_workspace`` rejects the
    per-page-interleaved (stride[0] = whole-page pitch) layout with 561002
    ("Non-contiguous cache is supported only by the arch22 regular FAInfer
    or D512 MLA PA routes"), so the attention backend can hand the views
    straight to both the eager calls and the graph-capture workspace query.

    - K: ``[num_kernel_blocks, block_size, num_kv_heads, head_size]`` INT8,
      contiguous; the backend passes it to FIA as the 3-D ND view
      ``(num_blocks, block_size, num_kv_heads * head_size)`` (the operator's
      paged-INT8 C8A16 layout is ND regardless of ``VLLM_KV_CACHE_LAYOUT``).
    - V: same, ``head_size_v`` wide, contiguous.
    - K_scale / V_scale: ``[num_blocks, num_kv_heads, kernel_block_size]``
      FLOAT32 (one scale per (block, head, token), symmetric, no offset) —
      the exact dtype and per-(token, head) granularity ``npu_dynamic_quant``
      emits for a ``(T, H, D)`` input and the 3-D shape FIA antiquant mode 5
      consumes (test_fia.py contract). Head-major order matches the K/V
      pages; contiguous, so no squeeze is needed at the call boundary.

    The spec page budget still includes the scale bytes via
    ``state_content_bytes`` (same mechanism as the C8-MXFP packet spec), so
    the scheduler never over-commits HBM for the scale caches.

    Hybrid-page independence: the kernel-block count is derived from the raw
    allocation's PAYLOAD budget (``c8_pth_section_bytes``), never from
    ``spec.page_size_bytes``. In hybrid Attention/Mamba pools the scheduler
    pads both groups' pages up to a common size, so the padded page can exceed
    the C8 payload; deriving geometry from the padded page would bake foreign
    padding ratios into the views (the old per-page-interleaved layout did
    exactly that and mis-addressed every page beyond the first). Here the
    sections are pure payload runs and the padded tail bytes of the raw
    allocation are simply left unused.
    """
    if spec.block_size % kernel_block_size:
        raise ValueError(
            "C8 per-token-head scheduler block must contain whole kernel blocks: "
            f"{spec.block_size} % {kernel_block_size} != 0."
        )
    scheduler_blocks = raw_tensor.numel() // spec.page_size_bytes
    if raw_tensor.numel() % spec.page_size_bytes:
        raise ValueError("C8 per-token-head raw allocation must contain whole scheduler pages.")
    num_kernel_blocks = scheduler_blocks * (spec.block_size // kernel_block_size)

    num_kv_heads = spec.num_kv_heads
    k_bytes, v_bytes, scale_bytes, _ = c8_pth_section_bytes(
        num_kernel_blocks, kernel_block_size, num_kv_heads, spec.head_size, spec.head_size_v
    )
    ks_offset = k_bytes + v_bytes
    vs_offset = ks_offset + scale_bytes
    payload_bytes = vs_offset + scale_bytes
    if payload_bytes > raw_tensor.numel():
        raise ValueError("C8 per-token-head payload exceeds the raw allocation.")

    base_offset = raw_tensor.storage_offset()
    if (base_offset + ks_offset) % C8_PTH_SCALE_ELEM_BYTES:
        raise ValueError(
            "C8 per-token-head scale sections require 4-byte-aligned offsets; "
            "head dims producing odd K/V byte counts are unsupported."
        )

    def _section_view(shape, offset_bytes, dtype):
        # Whole-section contiguous run: narrow the raw int8 bytes to the
        # section's byte range, then reinterpret and reshape (zero-copy).
        dtype_size = get_dtype_size(dtype)
        section_bytes = math.prod(shape) * dtype_size
        section = raw_tensor[offset_bytes : offset_bytes + section_bytes]
        return section.view(dtype).view(*shape)

    k_view = _section_view((num_kernel_blocks, kernel_block_size, num_kv_heads, spec.head_size), 0, torch.int8)
    v_view = _section_view((num_kernel_blocks, kernel_block_size, num_kv_heads, spec.head_size_v), k_bytes, torch.int8)
    k_scale_view = _section_view((num_kernel_blocks, num_kv_heads, kernel_block_size), ks_offset, scale_dtype)
    v_scale_view = _section_view((num_kernel_blocks, num_kv_heads, kernel_block_size), vs_offset, scale_dtype)
    return k_view, v_view, k_scale_view, v_scale_view
