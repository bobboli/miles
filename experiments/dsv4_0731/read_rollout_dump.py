"""Print what a rollout actually generated, from a `--save-debug-rollout-data` dump.

Aggregate metrics say a rollout answered nothing and ran to the token limit, but
they cannot say what it produced, and the shape of a degenerate output narrows
the cause more than any of them: a repeated token points at routing or the MoE
output, noise points at weights read under the wrong interpretation, and fluent
text that never stops points at the stop condition rather than the weights.

    python experiments/dsv4_0731/read_rollout_dump.py <rollout_0.pt> [--count 3]
"""

from __future__ import annotations

import argparse
import collections

import torch

PREVIEW_HEAD = 400
PREVIEW_TAIL = 200


def repetition(text: str) -> str:
    """The most frequent token-ish chunk and how much of the text it covers."""
    pieces = text.split()
    if not pieces:
        return "empty"
    top, count = collections.Counter(pieces).most_common(1)[0]
    return f"{top!r} x{count} of {len(pieces)} ({count / len(pieces):.0%})"


def routing_summary(routed) -> str:
    """How concentrated the expert routing is, when the rollout returned it.

    Degenerate text can come from a router that sends every token to the same
    few experts, or from experts whose outputs are wrong while routing is
    healthy. The two need different fixes and this tells them apart.
    """
    if routed is None:
        return "not returned"
    flat = torch.as_tensor(routed).flatten()
    if flat.numel() == 0:
        return "empty"
    counts = torch.bincount(flat.to(torch.int64))
    used = int((counts > 0).sum())
    top = counts.topk(min(5, counts.numel()))
    share = float(top.values.sum()) / float(flat.numel())
    return (
        f"{flat.numel()} picks over {used} experts, "
        f"top5={top.indices.tolist()} covering {share:.0%}"
    )


def _is_sample(value) -> bool:
    """A sample is whatever carries a response, dict or object."""
    if isinstance(value, dict):
        return "response" in value
    return hasattr(value, "response")


def find_samples(loaded, depth: int = 0):
    """Dig out the sample objects, whatever container the dump wrapped them in.

    The format has moved around — a bare list, a dict keyed by group, a dict with
    the samples under one key — so recognize samples by the field that matters
    rather than by the shape around them.
    """
    if isinstance(loaded, dict):
        if "response" in loaded:
            return [loaded]
        print(f"{'  ' * depth}dict keys: {list(loaded)[:8]}")
        for value in loaded.values():
            found = find_samples(value, depth + 1)
            if found:
                return found
        return []
    if isinstance(loaded, (list, tuple)):
        if loaded and _is_sample(loaded[0]):
            return list(loaded)
        for value in loaded[:4]:
            found = find_samples(value, depth + 1)
            if found:
                return found
        return []
    return [loaded] if _is_sample(loaded) else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    loaded = torch.load(args.dump, map_location="cpu", weights_only=False)
    samples = find_samples(loaded)
    print(f"{len(samples)} samples")
    if not samples:
        return

    first = samples[0]
    print("fields:", sorted(first) if isinstance(first, dict) else sorted(vars(first)))

    for index, sample in enumerate(samples[: args.count]):
        get = sample.get if isinstance(sample, dict) else (lambda k, d=None: getattr(sample, k, d))
        response = get("response")
        reward = get("reward")
        tokens = get("tokens")
        print("=" * 72)
        print(
            f"[{index}] reward={reward} "
            f"tokens={len(tokens) if tokens is not None else '?'} "
            f"chars={len(response) if response else 0}"
        )
        if not response:
            print("  no response text on this sample")
            continue
        print("  repetition:", repetition(response))
        print("  routing:", routing_summary(get("rollout_routed_experts")))
        print("  HEAD:", repr(response[:PREVIEW_HEAD]))
        print("  TAIL:", repr(response[-PREVIEW_TAIL:]))


if __name__ == "__main__":
    main()
