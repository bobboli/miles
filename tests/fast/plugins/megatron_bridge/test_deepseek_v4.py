from types import SimpleNamespace

from megatron.bridge.models.conversion.param_mapping import AutoMapping, ColumnParallelMapping, ReplicatedMapping

from miles_plugins.megatron_bridge.deepseek_v4 import MilesDeepSeekV4Bridge


def _by_megatron(registry):
    return {mapping.megatron_param: mapping for mapping in registry.mappings}


def test_miles_deepseek_v4_parameter_names_are_registered():
    bridge = MilesDeepSeekV4Bridge()
    bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=0)

    mappings = _by_megatron(bridge.mapping_registry())

    assert isinstance(mappings["decoder.layers.*.self_attention.wq_a.weight"], AutoMapping)
    assert mappings["decoder.layers.*.self_attention.wq_a.weight"].hf_param == "layers.*.attn.wq_a.weight"
    assert isinstance(mappings["decoder.layers.*.self_attention.attn_sink"], ColumnParallelMapping)
    assert mappings["decoder.layers.*.self_attention.attn_sink"].hf_param == "layers.*.attn.attn_sink"
    assert isinstance(mappings["decoder.layers.*.hc_attn_scale"], ReplicatedMapping)
    assert mappings["decoder.layers.*.hc_attn_scale"].hf_param == "layers.*.hc_attn_scale"
    assert isinstance(mappings["decoder.hc_head_params.hc_head_fn"], ReplicatedMapping)
    assert mappings["decoder.hc_head_params.hc_head_fn"].hf_param == "hc_head_fn"
