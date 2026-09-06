from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs):
    """Map replay streams to Megatron MoE layers using the local model layout."""
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)

    if len(replay_list) != len(layer_indices):
        raise AssertionError(
            f"registered {len(replay_list)} routing replays for {len(layer_indices)} local MoE layers"
        )

    for replay_idx, fallback_layer_idx in enumerate(layer_indices):
        replay = replay_list[replay_idx]
        layer_idx = replay.stream_idx if replay.stream_idx is not None else fallback_layer_idx
        if not 0 <= layer_idx < replay_data.shape[1]:
            raise AssertionError(
                f"routing replay stream_idx {layer_idx} out of range "
                f"(replay_data has {replay_data.shape[1]} streams)"
            )
        replay.stream_idx = layer_idx
        replay.record(replay_data[:, layer_idx])
