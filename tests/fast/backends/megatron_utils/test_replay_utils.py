from types import SimpleNamespace

import torch

from miles.backends.megatron_utils import replay_utils


class _Replay:
    def __init__(self, stream_idx=None):
        self.stream_idx = stream_idx
        self.recorded = []

    def record(self, data):
        self.recorded.append(data)


def _model(num_layers):
    config = SimpleNamespace(num_layers=num_layers, moe_layer_freq=[1] * num_layers)
    return SimpleNamespace(module=SimpleNamespace(config=config))


def test_register_replay_list_moe_prefers_explicit_global_stream_indices(monkeypatch):
    monkeypatch.setattr(replay_utils, "get_num_layers_to_build", lambda config, vp_stage: 4)
    monkeypatch.setattr(replay_utils, "get_transformer_layer_offset", lambda config, vp_stage: 0)

    replays = [_Replay(stream_idx=3), _Replay(stream_idx=0), _Replay(stream_idx=1), _Replay(stream_idx=2)]
    replay_data = torch.arange(8).reshape(2, 4, 1)

    replay_utils.register_replay_list_moe(replays, replay_data, models=[_model(num_layers=4)])

    assert [replay.stream_idx for replay in replays] == [3, 0, 1, 2]
    for replay in replays:
        torch.testing.assert_close(replay.recorded[0], replay_data[:, replay.stream_idx])


def test_register_replay_list_moe_falls_back_to_local_layout(monkeypatch):
    monkeypatch.setattr(replay_utils, "get_num_layers_to_build", lambda config, vp_stage: 2)
    monkeypatch.setattr(replay_utils, "get_transformer_layer_offset", lambda config, vp_stage: 4)

    replays = [_Replay(), _Replay()]
    replay_data = torch.arange(12).reshape(2, 6, 1)

    replay_utils.register_replay_list_moe(replays, replay_data, models=[_model(num_layers=6)])

    assert [replay.stream_idx for replay in replays] == [4, 5]
    for replay in replays:
        torch.testing.assert_close(replay.recorded[0], replay_data[:, replay.stream_idx])
