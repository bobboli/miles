from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.utils.profile_utils import TrainProfiler, _should_record_memory_history


def _profiler_args(ranks):
    return SimpleNamespace(
        memory_snapshot_ranks=ranks,
        profile_target=["train_overall"],
        record_memory_history=True,
        use_pytorch_profiler=False,
    )


@pytest.mark.parametrize(
    ("ranks", "rank", "expected"),
    [
        (None, 7, True),
        ([7], 7, True),
        ([1, 7], 7, True),
        ([1, 3], 7, False),
    ],
)
def test_should_record_memory_history(ranks, rank, expected):
    args = SimpleNamespace(memory_snapshot_ranks=ranks)

    with patch("torch.distributed.get_rank", return_value=rank):
        assert _should_record_memory_history(args) is expected


def test_should_record_memory_history_supports_legacy_args():
    with patch("torch.distributed.get_rank", return_value=7):
        assert _should_record_memory_history(SimpleNamespace())


def test_train_profiler_skips_memory_history_on_unselected_rank():
    with (
        patch("torch.distributed.get_rank", return_value=7),
        patch("miles.utils.profile_utils._BaseMemoryProfiler.create") as create,
    ):
        profiler = TrainProfiler(_profiler_args([3]))

    assert profiler._memory_profiler_overall is None
    create.assert_not_called()


def test_train_profiler_starts_memory_history_on_selected_rank():
    with (
        patch("torch.distributed.get_rank", return_value=7),
        patch("miles.utils.profile_utils._BaseMemoryProfiler.create") as create,
    ):
        profiler = TrainProfiler(_profiler_args([7]))

    assert profiler._memory_profiler_overall is create.return_value
    create.return_value.start.assert_called_once_with()
