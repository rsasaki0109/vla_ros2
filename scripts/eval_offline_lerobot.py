#!/usr/bin/env python3
"""Open-loop evaluation of a VLA checkpoint on recorded LeRobot episodes.

At every ``--stride``-th frame the policy predicts an action chunk from the
recorded cameras + state, and the chunk is compared with the demonstrator's next
``--horizon`` actions. A hold-state baseline (repeat the current joints) is
reported beside it. This measures imitation error, not task success, and is only
an honest held-out number for episodes the checkpoint was not trained on.

    pip install -e ".[smolvla]"
    python scripts/eval_offline_lerobot.py --device cuda --episodes 0 10 20
    python scripts/eval_offline_lerobot.py --pretrained <your-finetuned-checkpoint>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vla_ros2 import load_model
from vla_ros2.eval_offline import evaluate_offline, iter_lerobot_frames

DEFAULT_DATASET = "lerobot/svla_so100_stacking"
DEFAULT_PRETRAINED = "lerobot/smolvla_base"


def _fmt(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.2f}" for value in values) + "]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="smolvla")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--stride", type=int, default=60, help="evaluate every Nth frame")
    parser.add_argument("--horizon", type=int, default=50, help="recorded actions to compare")
    parser.add_argument("--max-frames", type=int, default=None, help="per episode")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    model = load_model(
        args.model,
        pretrained=args.pretrained,
        device=args.device,
        return_action_chunk=True,
    )
    load_sec = time.perf_counter() - start
    frames = iter_lerobot_frames(
        args.dataset,
        args.episodes,
        stride=args.stride,
        horizon=args.horizon,
        max_frames_per_episode=args.max_frames,
    )
    result = evaluate_offline(model, frames)

    errors = result.model_errors
    print(f"model={args.pretrained} frames={result.num_frames} episodes={args.episodes}")
    print(f"load={load_sec:.1f}s infer={result.infer_ms_mean:.0f}ms/chunk horizon={errors.horizon}")
    print(f"policy      MAE={errors.mae:8.3f}  first-step MAE={errors.first_step_mae:8.3f}")
    print(f"            per-dim MAE={_fmt(errors.mae_per_dim)}")
    if result.baseline_errors is not None:
        base = result.baseline_errors
        print(f"hold-state  MAE={base.mae:8.3f}  first-step MAE={base.first_step_mae:8.3f}")
        print(f"            per-dim MAE={_fmt(base.mae_per_dim)}")
    if args.out:
        payload = {
            "dataset": args.dataset,
            "pretrained": args.pretrained,
            "episodes": args.episodes,
            "stride": args.stride,
            "device": args.device,
            "load_sec": load_sec,
            **result.to_dict(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
