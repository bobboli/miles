from argparse import Namespace
from unittest.mock import MagicMock, patch


def _scheduler_args(num_rollout: int) -> Namespace:
    return Namespace(
        num_rollout=num_rollout,
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        global_batch_size=256,
        lr_decay_iters=None,
        lr_wsd_decay_iters=None,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_warmup_init=0.0,
        lr=1e-6,
        min_lr=0.0,
        lr_decay_style="constant",
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        weight_decay_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        lr_wsd_decay_style="linear",
    )


@patch("miles.backends.megatron_utils.model.OptimizerParamScheduler")
def test_eval_only_run_uses_positive_scheduler_horizon(mock_scheduler):
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler

    args = _scheduler_args(num_rollout=0)
    optimizer = MagicMock()

    get_optimizer_param_scheduler(args, optimizer)

    assert args.train_iters == 1
    assert args.lr_decay_iters == 1
    assert mock_scheduler.call_args.kwargs["lr_decay_steps"] == args.global_batch_size
    assert mock_scheduler.call_args.kwargs["wd_incr_steps"] == args.global_batch_size


@patch("miles.backends.megatron_utils.model.OptimizerParamScheduler")
def test_training_run_preserves_computed_scheduler_horizon(mock_scheduler):
    from miles.backends.megatron_utils.model import get_optimizer_param_scheduler

    args = _scheduler_args(num_rollout=20)

    get_optimizer_param_scheduler(args, MagicMock())

    assert args.train_iters == 20
    assert args.lr_decay_iters == 20
    assert mock_scheduler.call_args.kwargs["lr_decay_steps"] == 20 * args.global_batch_size
    assert mock_scheduler.call_args.kwargs["wd_incr_steps"] == 20 * args.global_batch_size
