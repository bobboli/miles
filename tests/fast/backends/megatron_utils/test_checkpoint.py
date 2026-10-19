from argparse import Namespace
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import checkpoint
from miles.backends.training_utils.model_companion import ModelCompanion


class TestCheckpointTrainingProvenance:
    @pytest.mark.parametrize(
        ("source", "finetune", "ckpt_step", "local_iteration", "expected"),
        [
            ("0", False, None, None, True),
            ("release", False, None, None, True),
            ("0", True, None, None, False),
            ("iter_0000000", False, None, None, True),
            ("release", False, 0, None, True),
            ("release", False, None, -1, True),
            ("release", False, None, 0, True),
        ],
    )
    def test_native_load_uses_finetune_for_rollout_numbering(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        source: str,
        finetune: bool,
        ckpt_step: int | None,
        local_iteration: int | None,
        expected: bool,
    ) -> None:
        """Native loading preserves the existing finetune-based rollout numbering."""
        load_path = tmp_path / (source if source.startswith("iter_") else "checkpoint")
        load_path.mkdir()
        if source.startswith("iter_"):
            (load_path / "metadata.json").write_text("{}")
        else:
            (load_path / "latest_checkpointed_iteration.txt").write_text(source)
        args = Namespace(load=str(load_path), finetune=finetune, ckpt_step=ckpt_step, lora_rank=0)
        monkeypatch.setattr(checkpoint, "get_args", lambda: args)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **_kwargs: (0, 123))
        checkpointing_context = (
            None
            if local_iteration is None
            else {"local_checkpoint_manager": SimpleNamespace(find_latest=lambda: local_iteration)}
        )

        result = checkpoint.load_checkpoint(
            ddp_model=[],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=False,
        )

        assert result == (0, expected, False)

    def test_loading_hf_weights_preserves_non_finetune_rollout_numbering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HF loading preserves the existing non-finetune rollout numbering."""
        (tmp_path / "config.json").write_text("{}")
        args = Namespace(load=str(tmp_path), finetune=False, ckpt_step=None, lora_rank=0)
        monkeypatch.setattr(checkpoint, "get_args", lambda: args)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **_kwargs: (0, 0))

        result = checkpoint.load_checkpoint(
            ddp_model=[],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
        )

        assert result == (0, True, False)

    @pytest.mark.parametrize(
        ("adapter_result", "expected"),
        [
            ((True, 0, True), True),
            ((True, 0, False), True),
            ((True, None, False), False),
            ((False, None, False), False),
        ],
    )
    def test_lora_restore_preserves_the_difference_between_zero_and_missing_training_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        adapter_result: tuple[bool, int | None, bool],
        expected: bool,
    ) -> None:
        """Adapter weights alone cannot be mistaken for a saved iteration-zero training state."""
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
        args = Namespace(
            load=str(tmp_path),
            finetune=True,
            ckpt_step=None,
            lora_rank=8,
            lora_adapter_path=str(tmp_path / "adapter"),
            no_load_optim=False,
        )
        monkeypatch.setattr(checkpoint, "get_args", lambda: args)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **_kwargs: (0, 123))
        monkeypatch.setattr(checkpoint, "load_lora_adapter", lambda *_args, **_kwargs: adapter_result)

        result = checkpoint.load_checkpoint(
            ddp_model=[],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
        )

        assert result == (0, expected, adapter_result[2])


class TestRawMbridgeCompanionIsolation:
    @pytest.mark.parametrize("fail", [False, True])
    def test_hf_loading_excludes_and_restores_companion_state(
        self, raw_mbridge_loader: Callable[[bool], list[set[str]]], fail: bool
    ) -> None:
        """HF loading must not map training metadata or discard it when loading fails."""
        observed = raw_mbridge_loader(fail)
        models = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        companions = [ModelCompanion(pipeline_rank=0, chunk_index=i, replica_id=(0, 0, 0)) for i in range(2)]
        for model, companion in zip(models, companions, strict=True):
            companion.weight_version.fill_(7)
            companion.sample_consumptions.data = torch.tensor([[1, 2, 3, 4, 0]])
            model.add_module("model_companion", companion)

        with pytest.raises(RuntimeError, match="HF loading failed") if fail else nullcontext():
            checkpoint._load_hf_weights_with_mbridge(
                ddp_model=models, args=Namespace(offload_train=False), load_path="/unused/hf"
            )

        assert observed == [{"weight", "bias"}, {"weight", "bias"}]
        for model, companion in zip(models, companions, strict=True):
            assert model.model_companion is companion
            assert companion.weight_version.item() == 7
            assert companion.sample_consumptions.tolist() == [[1, 2, 3, 4, 0]]
            torch.testing.assert_close(model.weight, torch.full_like(model.weight, 5))


class TestPruneUnretainedCheckpoints:
    @staticmethod
    def _make_checkpoints(save_dir: Path, iterations: list[int], *, latest: int | None) -> None:
        for iteration in iterations:
            (save_dir / f"iter_{iteration:07d}").mkdir(parents=True)
        if latest is not None:
            (save_dir / "latest_checkpointed_iteration.txt").write_text(str(latest))

    @staticmethod
    def _remaining(save_dir: Path) -> list[str]:
        return sorted(path.name for path in save_dir.iterdir() if path.is_dir())

    def test_keeps_every_retained_rollout_step_and_the_latest(self, tmp_path: Path) -> None:
        """Rollout r is saved as iteration r, so a 60-rollout interval keeps 59 and 119, not 60 and 120."""
        self._make_checkpoints(tmp_path, [19, 39, 59, 79, 99, 119, 139], latest=139)

        checkpoint.prune_unretained_checkpoints(str(tmp_path), retain_interval=60)

        assert self._remaining(tmp_path) == ["iter_0000059", "iter_0000119", "iter_0000139"]

    def test_a_directory_newer_than_the_tracker_is_left_alone(self, tmp_path: Path) -> None:
        """An asynchronous save writes its directory before the tracker names it."""
        self._make_checkpoints(tmp_path, [19, 39, 59], latest=39)

        checkpoint.prune_unretained_checkpoints(str(tmp_path), retain_interval=60)

        assert self._remaining(tmp_path) == ["iter_0000039", "iter_0000059"]

    def test_nothing_is_deleted_without_a_tracker(self, tmp_path: Path) -> None:
        """Without a tracker no checkpoint is known to be complete, so none may go."""
        self._make_checkpoints(tmp_path, [19, 39], latest=None)

        checkpoint.prune_unretained_checkpoints(str(tmp_path), retain_interval=60)

        assert self._remaining(tmp_path) == ["iter_0000019", "iter_0000039"]

    def test_other_entries_in_the_save_directory_are_untouched(self, tmp_path: Path) -> None:
        """Rollout state and a released conversion live next to the iteration directories."""
        self._make_checkpoints(tmp_path, [19, 39], latest=39)
        (tmp_path / "rollout" / "19").mkdir(parents=True)
        (tmp_path / "release").mkdir()
        (tmp_path / "iter_notanumber").mkdir()

        checkpoint.prune_unretained_checkpoints(str(tmp_path), retain_interval=60)

        assert self._remaining(tmp_path) == ["iter_0000039", "iter_notanumber", "release", "rollout"]
