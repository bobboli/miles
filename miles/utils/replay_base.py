import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


class Replay:
    def __init__(self, stream_idx: int | None = None):
        self.stream_idx = stream_idx
        self.debug_source = None
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []

    def record(self, top_indices: torch.Tensor):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self) -> torch.Tensor:
        if self.forward_index >= len(self.top_indices_list):
            shapes = [
                t.shape if isinstance(t, torch.Tensor) else f"non-tensor({type(t)})" for t in self.top_indices_list
            ]
            raise IndexError(
                f"pop_forward out of range: forward_index={self.forward_index}, "
                f"len(top_indices_list)={len(self.top_indices_list)}, shapes={shapes}"
            )
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def pop_backward(self) -> torch.Tensor:
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0


class BaseReplayManager:
    name: str = ""
    filename: str = ""
    replay_check_min_overlap_ratio = 0.0  # 0.0 = mismatch only on zero overlap
    # True when replayed indices are token/KV positions (indexer): rebase the
    # per-sample 0-based rollout indices onto the packed training sequence.
    replay_indices_are_token_positions = False

    def __init__(self):
        self.replays: list[Replay] = []
        self.current: Replay | None = None
        self.enabled = False
        self.stage = "fallthrough"
        self.register_replay_list_func = None

    def create_replay(self, stream_idx: int | None = None) -> Replay:
        replay = Replay(stream_idx=stream_idx)
        self.replays.append(replay)
        return replay

    def set_current(self, replay: Replay):
        self.current = replay

    def get_current(self) -> Replay | None:
        return self.current

    def clear_all(self):
        for replay in self.replays:
            replay.clear()

    def clear_all_forward(self):
        for replay in self.replays:
            replay.clear_forward()

    def get_topk_fn(self, old_topk_fn, return_probs):
        manager = self

        def _get_replay_result(replay, top_indices, scores, topk, *args, **kwargs):
            assert (
                top_indices.shape[0] == scores.shape[0]
            ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {scores.shape[0]}"

            assert (
                top_indices.shape[1] == topk
            ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"

            if self.enable_check_replay_result:
                self.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)

            # fill padding tokens with arange to avoid invalid reading
            all_invalid = (top_indices == -1).all(dim=-1)
            if all_invalid.any():
                ar = (
                    torch.arange(top_indices.shape[1], device=top_indices.device, dtype=top_indices.dtype)
                    % scores.shape[1]
                )
                top_indices = torch.where(all_invalid.unsqueeze(-1), ar, top_indices)

            manager.validate_replay_indices(replay, top_indices, scores.shape[1])

            if return_probs:
                return scores.gather(1, top_indices), top_indices
            else:
                return top_indices

        def new_topk_fn(scores, topk, *args, **kwargs):
            if not manager.enabled:
                return old_topk_fn(scores, topk, *args, **kwargs)

            stage = manager.stage
            replay = manager.get_current()

            if stage == "fallthrough":
                return old_topk_fn(scores, topk, *args, **kwargs)

            elif stage == "record":
                result = old_topk_fn(scores, topk, *args, **kwargs)
                if return_probs:
                    probs, top_indices = result
                else:
                    top_indices = result
                replay.record(top_indices)
                return result

            elif stage == "replay_forward":
                return _get_replay_result(replay, replay.pop_forward(), scores, topk, *args, **kwargs)

            elif stage == "replay_backward":
                return _get_replay_result(replay, replay.pop_backward(), scores, topk, *args, **kwargs)

            else:
                return old_topk_fn(scores, topk, *args, **kwargs)

        return new_topk_fn

    def validate_replay_indices(self, replay: Replay, top_indices: torch.Tensor, num_choices: int) -> None:
        """Optionally validate manager-specific replay invariants before use."""

    def register_to_module(self, module, attr_name: str, stream_idx: int | None = None):
        if not self.enabled:
            return
        replay = self.create_replay(stream_idx=stream_idx)
        if os.environ.get("MILES_VALIDATE_ROUTING_REPLAY") == "1":
            config = getattr(module, "config", None)
            replay.debug_source = {
                "module": type(module).__name__,
                "layer_number": getattr(module, "layer_number", None),
                "is_hash_layer": getattr(module, "is_hash_layer", None),
                "moe_n_hash_layers": getattr(config, "moe_n_hash_layers", None),
            }
        setattr(module, attr_name, replay)
        manager = self

        def pre_forward_hook(*args, **kwargs):
            manager.set_current(replay)

        module.register_forward_pre_hook(pre_forward_hook)

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, *args, **kwargs):
        """
        CI checker for R3. Only enable when enable_check_replay_result=True.
        Per token, measure the overlap between the training engine's recomputed
        topk and the replayed topk (ignoring -1 padding). A token is mismatched
        when overlap < replay_check_min_overlap_ratio of its valid picks (ratio 0.0 =
        only zero overlap). Raise when the mismatched fraction exceeds
        replay_check_max_mismatch_fraction.
        """
        orig_top_indices = old_topk_fn(scores, topk, *args, **kwargs)
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices

        orig_flat = orig_top_indices.view(-1, orig_top_indices.shape[-1])  # [n_tokens, topk]
        replay_flat = top_indices.view(-1, top_indices.shape[-1])
        valid_orig = orig_flat != -1
        valid_replay = replay_flat != -1

        # token-wise set overlap via a membership mask, avoiding the O(n*topk^2)
        # all-pairs tensor; -1 padding is routed to a sentinel column
        n_kv = int(torch.maximum(orig_flat.max(), replay_flat.max()).clamp_min(0)) + 1
        membership = orig_flat.new_zeros((orig_flat.shape[0], n_kv + 1), dtype=torch.bool)
        membership.scatter_(1, torch.where(valid_orig, orig_flat, n_kv).long(), True)
        replay_idx = torch.where(valid_replay, replay_flat, n_kv).long()
        hit = membership.gather(1, replay_idx) & valid_replay  # [n_tokens, topk]

        overlap = hit.sum(dim=1)
        valid_count = valid_replay.sum(dim=1)
        is_padding = valid_count == 0
        if self.replay_check_min_overlap_ratio == 0.0:
            required = torch.ones_like(overlap)  # legacy: mismatch only on zero overlap
        else:
            required = (valid_count * self.replay_check_min_overlap_ratio).ceil().to(overlap.dtype)
        is_mismatch = (overlap < required) & ~is_padding

        mismatch_count = is_mismatch.sum().item()
        threshold = float(os.environ.get("MILES_TEST_R3_THRESHOLD", self.replay_check_max_mismatch_fraction))
        mismatch_threshold = threshold * orig_flat.shape[0]
        logger.info(
            f"Replay check (rank {_get_rank()}, stage {self.stage}): "
            f"mismatch {mismatch_count}/{orig_flat.shape[0]} tokens, threshold {mismatch_threshold:.0f}"
        )
        if mismatch_count == 0:
            return

        mismatch_indices = is_mismatch.nonzero(as_tuple=False).squeeze(1)
        for idx in mismatch_indices[:10]:
            i = idx.item()
            lines = []
            for j in range(max(0, i - 3), min(len(orig_flat), i + 4)):
                marker = " <<<" if j == i else ""
                lines.append(f"  token {j}: orig={orig_flat[j].tolist()}, replay={replay_flat[j].tolist()}{marker}")
            logger.warning(
                f"Replay check (rank {_get_rank()}, stage {self.stage}): "
                f"token {i} overlap {overlap[i].item()}/{valid_count[i].item()}, topk={topk}\n" + "\n".join(lines)
            )

        if mismatch_count > mismatch_threshold:
            raise AssertionError(f"R3 mismatch tokens ({mismatch_count}) > threshold ({mismatch_threshold:.0f})")


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2

    def validate_replay_indices(self, replay: Replay, top_indices: torch.Tensor, num_choices: int) -> None:
        if os.environ.get("MILES_VALIDATE_ROUTING_REPLAY") != "1":
            return

        flat = top_indices.reshape(-1, top_indices.shape[-1])
        invalid = ((flat < 0) | (flat >= num_choices)).any(dim=-1)
        ordered = flat.sort(dim=-1).values
        duplicate = (ordered[:, 1:] == ordered[:, :-1]).any(dim=-1)
        bad = invalid | duplicate
        if not bad.any():
            return

        bad_rows = bad.nonzero(as_tuple=False).squeeze(-1)[:10]
        examples = [(int(row), flat[row].tolist()) for row in bad_rows]
        raise RuntimeError(
            "Invalid routing replay indices before MoE dispatch: "
            f"stream={replay.stream_idx}, rows={flat.shape[0]}, "
            f"invalid={int(invalid.sum())}, duplicate={int(duplicate.sum())}, "
            f"source={getattr(replay, 'debug_source', None)}, "
            f"examples={examples}"
        )


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    if_sp_region = False
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2
    replay_check_min_overlap_ratio = 0.8
    replay_indices_are_token_positions = True


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]
