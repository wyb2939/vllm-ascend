#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""C8A16 KV cache quantization method (dynamic per-token x per-head INT8)."""

import torch
from vllm.config import get_current_vllm_config

from ..base import AscendAttentionScheme


class AscendC8A16KVCacheAttentionMethod(AscendAttentionScheme):
    """INT8 KV cache with a runtime per-(token, head) scale.

    The scale is derived from each token's own K/V values at scatter time, so
    this method registers no checkpoint parameters. Its only job is to install
    the C8A16 attention implementation on the layer; the forward path owns
    quantization, cache writes and the antiquant-mode-5 decode call.
    """

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix
        vllm_config = get_current_vllm_config()
        # The KV payload is INT8 regardless of PD role; the producer flag only
        # decides who writes the cache, which the attention backend handles.
        self.is_kv_producer = (
            vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_producer
        )

    def create_weights(self, layer: torch.nn.Module) -> None:
        if not self.is_kv_producer:
            layer.kv_cache_torch_dtype = torch.int8
        # Class surgery: the subclass constructor never runs, so the backend
        # must not rely on instance state set there.
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_c8a16 import AscendC8A16AttentionBackendImpl

            layer.impl.__class__ = AscendC8A16AttentionBackendImpl

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

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
            "[vllm-ascend/C8A16] AscendC8A16KVCacheAttentionMethod.apply should not "
            "be called. C8A16 KV cache quantization is handled by the attention backend."
        )
