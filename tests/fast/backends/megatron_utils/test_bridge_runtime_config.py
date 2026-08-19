from types import SimpleNamespace
from unittest.mock import Mock, patch

from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config


def _runtime_args(**overrides):
    values = {
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "sequence_parallel": True,
        "context_parallel_size": 1,
        "calculate_per_token_loss": False,
        "variable_seq_lengths": False,
        "attention_softmax_in_fp32": True,
        "fp32_residual_connection": False,
        "deterministic_mode": True,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": None,
        "cpu_offloading_num_layers": 0,
        "distribute_saved_activations": False,
        "tp_comm_overlap": False,
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "attention_backend": "auto",
        "spec": "miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec",
        "experimental_attention_variant": "dsv4",
        "mtp_num_layers": None,
        "moe_token_dispatcher_type": "alltoall",
        "decoder_first_pipeline_num_layers": None,
        "decoder_last_pipeline_num_layers": None,
        "moe_router_bias_update_rate": None,
        "moe_aux_loss_coeff": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_explicit_model_spec_overrides_bridge_layer_spec():
    provider = SimpleNamespace(transformer_layer_spec="bridge-default", experimental_attention_variant="dsv4_hybrid")
    args = _runtime_args()
    custom_spec = Mock(return_value="custom-layer-spec")

    with patch("miles.backends.megatron_utils.model_provider.import_module", return_value=custom_spec):
        _apply_bridge_runtime_config(provider, args)

    assert provider.transformer_layer_spec("config", vp_stage=3) == "custom-layer-spec"
    custom_spec.assert_called_once_with(args, "config", vp_stage=3)
    assert provider.experimental_attention_variant == "dsv4"


def test_bridge_layer_spec_is_preserved_without_explicit_override():
    provider = SimpleNamespace(transformer_layer_spec="bridge-default", mtp_num_layers=1)

    _apply_bridge_runtime_config(provider, _runtime_args(spec=None))

    assert provider.transformer_layer_spec == "bridge-default"
    assert provider.mtp_num_layers is None
