#!/usr/bin/env python3
"""Compare vla_ros2 adapters on the same instruction/image/state.

Examples:

    # Always-available baselines (no GPU)
    python scripts/compare_vla_models.py --models dummy random scripted

    # Include SmolVLA (needs smolvla extra + GPU)
    python scripts/compare_vla_models.py --models dummy smolvla \\
        --pretrained smolvla=checkpoints/smolvla_so100_stacking_20k/.../pretrained_model

    # Write JSON for a report
    python scripts/compare_vla_models.py --models dummy random smolvla --out /tmp/compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from vla_ros2 import list_models
from vla_ros2.compare import compare_models, compare_results_to_json
from vla_ros2.core.types import VLAObservation

DEFAULT_INSTRUCTION = "stack the red block on the blue block"


def _parse_pretrained_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            msg = f"expected NAME=CHECKPOINT, got {pair!r}"
            raise argparse.ArgumentTypeError(msg)
        name, checkpoint = pair.split("=", 1)
        overrides[name.strip()] = checkpoint.strip()
    return overrides


def _load_image(path: Path | None) -> dict[str, Any]:
    if path is None:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        return {"camera1": frame}
    from PIL import Image

    return {"camera1": Image.open(path).convert("RGB")}


def _load_state(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = "state JSON must be an object"
        raise ValueError(msg)
    return parsed


def main() -> None:
    available = [info.name for info in list_models()]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy", "random", "scripted"],
        choices=available,
        metavar="MODEL",
        help=f"Adapters to compare (choices: {', '.join(available)})",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--image", type=Path, default=None, help="RGB image path (optional)")
    parser.add_argument(
        "--state-json",
        default="",
        help='Proprio JSON, e.g. {"state":[0,45,90,0,0,0.5]}',
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--pretrained",
        action="append",
        default=[],
        metavar="MODEL=CHECKPOINT",
        help="Per-model checkpoint override (repeatable)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write full JSON results here")
    args = parser.parse_args()

    observation = VLAObservation(
        instruction=args.instruction,
        images=_load_image(args.image),
        state=_load_state(args.state_json),
        metadata={"source": "compare_vla_models"},
    )
    overrides = _parse_pretrained_overrides(args.pretrained)
    results = compare_models(
        args.models,
        observation,
        device=args.device,
        pretrained_overrides=overrides,
    )

    header = f"{'model':<10} {'ok':<4} {'load_ms':>8} {'infer_ms':>9}  action_preview"
    print(header)
    print("-" * len(header))
    for item in results:
        load_ms = f"{item.load_ms:.1f}" if item.load_ms is not None else "-"
        infer_ms = f"{item.infer_ms:.1f}" if item.infer_ms is not None else "-"
        preview = item.action_preview or (item.error or "")
        print(f"{item.name:<10} {str(item.ok):<4} {load_ms:>8} {infer_ms:>9}  {preview}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(compare_results_to_json(results) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    if not all(item.ok for item in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
