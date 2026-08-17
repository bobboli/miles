# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron-Bridge mappings for the Miles DeepSeek-V4 model implementation."""

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ColumnParallelMapping, ReplicatedMapping
from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel


_AUTO_MAPPINGS: dict[str, str] = {
    "decoder.layers.*.self_attention.wq_a.weight": "layers.*.attn.wq_a.weight",
    "decoder.layers.*.self_attention.q_norm.weight": "layers.*.attn.q_norm.weight",
    "decoder.layers.*.self_attention.wq_b.weight": "layers.*.attn.wq_b.weight",
    "decoder.layers.*.self_attention.wkv.weight": "layers.*.attn.wkv.weight",
    "decoder.layers.*.self_attention.kv_norm.weight": "layers.*.attn.kv_norm.weight",
    "decoder.layers.*.self_attention.wo_a.weight": "layers.*.attn.wo_a.weight",
    "decoder.layers.*.self_attention.wo_b.weight": "layers.*.attn.wo_b.weight",
}

_REPLICATED_MAPPINGS: dict[str, str] = {
    "decoder.hc_head_params.hc_head_fn": "hc_head_fn",
    "decoder.hc_head_params.hc_head_base": "hc_head_base",
    "decoder.hc_head_params.hc_head_scale": "hc_head_scale",
    "decoder.layers.*.hc_attn_fn": "layers.*.hc_attn_fn",
    "decoder.layers.*.hc_attn_base": "layers.*.hc_attn_base",
    "decoder.layers.*.hc_attn_scale": "layers.*.hc_attn_scale",
    "decoder.layers.*.hc_ffn_fn": "layers.*.hc_ffn_fn",
    "decoder.layers.*.hc_ffn_base": "layers.*.hc_ffn_base",
    "decoder.layers.*.hc_ffn_scale": "layers.*.hc_ffn_scale",
    "decoder.layers.*.self_attention.compressor.wkv.weight": "layers.*.attn.compressor.wkv.weight",
    "decoder.layers.*.self_attention.compressor.wgate.weight": "layers.*.attn.compressor.wgate.weight",
    "decoder.layers.*.self_attention.compressor.ape": "layers.*.attn.compressor.ape",
    "decoder.layers.*.self_attention.compressor.norm.weight": "layers.*.attn.compressor.norm.weight",
    "decoder.layers.*.self_attention.indexer.linear_wq_b.weight": "layers.*.attn.indexer.wq_b.weight",
    "decoder.layers.*.self_attention.indexer.linear_weights_proj.weight": "layers.*.attn.indexer.weights_proj.weight",
    "decoder.layers.*.self_attention.indexer.compressor.wkv.weight": "layers.*.attn.indexer.compressor.wkv.weight",
    "decoder.layers.*.self_attention.indexer.compressor.wgate.weight": "layers.*.attn.indexer.compressor.wgate.weight",
    "decoder.layers.*.self_attention.indexer.compressor.ape": "layers.*.attn.indexer.compressor.ape",
    "decoder.layers.*.self_attention.indexer.compressor.norm.weight": "layers.*.attn.indexer.compressor.norm.weight",
}


@MegatronModelBridge.register_bridge(
    source="DeepseekV4ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v4",
)
class MilesDeepSeekV4Bridge(DeepSeekV4Bridge):
    """DeepSeek-V4 bridge extended for checkpoint-native Miles parameter names."""

    def mapping_registry(self) -> MegatronMappingRegistry:
        registry = super().mapping_registry()
        base = list(registry.mappings if hasattr(registry, "mappings") else registry._mappings)
        extras = [AutoMapping(megatron_param=name, hf_param=hf_name) for name, hf_name in _AUTO_MAPPINGS.items()]
        extras += [ReplicatedMapping(megatron_param=name, hf_param=hf_name) for name, hf_name in _REPLICATED_MAPPINGS.items()]
        extras.append(
            ColumnParallelMapping(
                megatron_param="decoder.layers.*.self_attention.attn_sink",
                hf_param="layers.*.attn.attn_sink",
            )
        )
        return MegatronMappingRegistry(*base, *extras)
