from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.update_weight.common import AtomicUpdateGroup
from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import (
    _process_conversion_tasks,
    _select_bridge_checkpoint,
    _stream_atomic_units,
)


def test_bridge_uses_direct_hf_trainer_seed(tmp_path):
    trainer_seed = tmp_path / "trainer"
    trainer_seed.mkdir()
    (trainer_seed / "model.safetensors.index.json").write_text("{}")
    rollout_schema = tmp_path / "rollout-schema"
    rollout_schema.mkdir()

    selected = _select_bridge_checkpoint(
        Namespace(load=str(trainer_seed), ref_load=str(trainer_seed), hf_checkpoint=str(rollout_schema))
    )

    assert selected == str(trainer_seed)


def test_bridge_keeps_rollout_checkpoint_for_megatron_seed(tmp_path):
    trainer_seed = tmp_path / "torch-dist"
    trainer_seed.mkdir()
    (trainer_seed / "latest_checkpointed_iteration.txt").write_text("1")
    rollout_checkpoint = tmp_path / "rollout"

    selected = _select_bridge_checkpoint(
        Namespace(load=str(trainer_seed), ref_load=str(trainer_seed), hf_checkpoint=str(rollout_checkpoint))
    )

    assert selected == str(rollout_checkpoint)


def test_bridge_uses_hf_reference_after_resuming_training_checkpoint(tmp_path):
    training_checkpoint = tmp_path / "torch-dist"
    training_checkpoint.mkdir()
    (training_checkpoint / "latest_checkpointed_iteration.txt").write_text("1")
    trainer_seed = tmp_path / "trainer"
    trainer_seed.mkdir()
    (trainer_seed / "model.safetensors.index.json").write_text("{}")
    rollout_schema = tmp_path / "rollout-schema"
    rollout_schema.mkdir()

    selected = _select_bridge_checkpoint(
        Namespace(
            load=str(training_checkpoint),
            ref_load=str(trainer_seed),
            hf_checkpoint=str(rollout_schema),
        )
    )

    assert selected == str(trainer_seed)


def test_stream_atomic_units_keeps_derived_tensors_with_cross_parameter_group():
    items = [
        ("model.layers.0.self_attn.wq_a.weight", "q-weight", "layer.self_attention.wq_a.weight"),
        ("model.layers.0.self_attn.wq_a.weight_scale_inv", "q-scale", "layer.self_attention.wq_a.weight"),
        ("model.layers.0.self_attn.wkv.weight", "kv-weight", "layer.self_attention.wkv.weight"),
        ("model.layers.0.self_attn.wkv.weight_scale_inv", "kv-scale", "layer.self_attention.wkv.weight"),
    ]
    groups = [
        AtomicUpdateGroup(
            "wqkv_a",
            (".self_attention.wq_a.weight", ".self_attention.wkv.weight"),
        )
    ]

    assert list(_stream_atomic_units(items, groups)) == [
        [
            ("model.layers.0.self_attn.wq_a.weight", "q-weight"),
            ("model.layers.0.self_attn.wq_a.weight_scale_inv", "q-scale"),
            ("model.layers.0.self_attn.wkv.weight", "kv-weight"),
            ("model.layers.0.self_attn.wkv.weight_scale_inv", "kv-scale"),
        ]
    ]


def test_stream_atomic_units_keeps_derived_tensors_without_named_group():
    items = [
        ("model.layers.0.mlp.weight", "weight", "layer.mlp.weight"),
        ("model.layers.0.mlp.weight_scale_inv", "scale", "layer.mlp.weight"),
    ]

    assert list(_stream_atomic_units(items, [])) == [
        [
            ("model.layers.0.mlp.weight", "weight"),
            ("model.layers.0.mlp.weight_scale_inv", "scale"),
        ]
    ]


def test_stream_atomic_units_rejects_non_consecutive_source_tensors():
    items = [
        ("q.weight", "q-weight", "layer.self_attention.wq_a.weight"),
        ("other.weight", "other", "layer.other.weight"),
        ("q.weight_scale_inv", "q-scale", "layer.self_attention.wq_a.weight"),
    ]
    groups = [
        AtomicUpdateGroup(
            "wqkv_a",
            (".self_attention.wq_a.weight", ".self_attention.wkv.weight"),
        )
    ]

    with pytest.raises(AssertionError, match="Non-consecutive tensors"):
        list(_stream_atomic_units(items, groups))


def test_bridge_export_rejects_missing_backed_up_parameter():
    task = SimpleNamespace(
        param_weight=torch.nn.Parameter(torch.ones(1)),
        vp_stage=0,
        param_name="decoder.weight",
    )

    with pytest.raises(KeyError, match="defeat train offload"):
        list(_process_conversion_tasks([task], {}))


def test_bridge_export_allows_model_resident_buffer():
    task = SimpleNamespace(
        param_weight=torch.ones(1),
        vp_stage=0,
        param_name="decoder.buffer",
    )

    assert list(_process_conversion_tasks([task], {})) == [task]
