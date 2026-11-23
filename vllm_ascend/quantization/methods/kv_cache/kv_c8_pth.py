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
"""C8 per-token-per-head (antiquant mode 5) attention quantization scheme.

Recipe: ``kv_cache_type: "C8_PER_TOKEN_HEAD"``. Unlike the static per-channel
C8 recipe (``kv_cache_type: "C8"``) this one ships no checkpoint scale/offset
tensors: K/V scales are computed dynamically per (token, kv-head) at cache
write time and stored in dedicated scale caches next to the INT8 K/V pages.
``npu_fused_infer_attention_score`` then dequantizes the paged INT8 KV
in-kernel (C8A16 pseudo-quantization, antiquant mode 5) for Q_S == 1 and the
newly supported Q_S > 1 paths, with Q kept in BF16/FP16.
"""

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import logger

from ..base import AscendAttentionScheme
from .kv_c8 import _c8_kv_scale_weight_loader


class AscendC8PerTokenHeadKVCacheAttentionMethod(AscendAttentionScheme):
    """FIA antiquant-mode-5 KV cache for dense-attention GQA models.

    K/V are cached as INT8 with dynamic per-(token, head) scales; the scales
    live in per-layer cache views (see ``attention_c8_pth.py``), so this
    method creates only neutral placeholder parameters for checkpoints that
    still carry legacy scale keys (they are loaded but never consumed).
    """

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix
        vllm_config = get_current_vllm_config()
        self.is_kv_producer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer

    def create_weights(self, layer: torch.nn.Module) -> None:
        from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

        if not get_current_hardware_profile().supports(HardwareCapability.FIA_ANTIQ_PER_TOKEN_HEAD):
            raise ValueError(
                "[vllm-ascend/C8_PER_TOKEN_HEAD] The per-(token, head) INT8 KV cache "
                "(FIA antiquant mode 5) is only supported on A2/A3 devices."
            )
        if not self.is_kv_producer:
            logger.info_once(
                "[vllm-ascend/C8_PER_TOKEN_HEAD] KV cache producer is disabled; "
                "setting kv_cache_torch_dtype to torch.int8."
            )
            layer.kv_cache_torch_dtype = torch.int8
        # Upgrade impl to the per-token-head C8 subclass so its forward path
        # (antiquant mode 5 for both Q_S == 1 and Q_S > 1) is always used.
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_v1 import AscendC8PerTokenHeadAttentionBackendImpl

            layer.impl.__class__ = AscendC8PerTokenHeadAttentionBackendImpl
        # Dynamic quantization needs no checkpoint scales. Some ModelSlim
        # exporters still emit the static C8 keys; register neutral
        # parameters so AutoWeightsLoader accepts them without consuming them.
        dtype = torch.get_default_dtype()
        layer.k_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        layer.k_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        layer.k_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        layer.k_cache_offset.weight_loader = _c8_kv_scale_weight_loader
        layer.v_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        layer.v_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        layer.v_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        layer.v_cache_offset.weight_loader = _c8_kv_scale_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.k_cache_scale.data = layer.k_cache_scale.data.flatten()
        layer.k_cache_offset.data = layer.k_cache_offset.data.flatten()
        layer.v_cache_scale.data = layer.v_cache_scale.data.flatten()
        layer.v_cache_offset.data = layer.v_cache_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "[vllm-ascend/C8_PER_TOKEN_HEAD] apply should not be called. "
            "Per-token-head C8 KV cache quantization is handled by the "
            "attention backend."
        )
