#!/usr/bin/env python3
"""Calibrate the VLM success threshold on real frames from a LeRobot dataset.

For each episode the last frame is labeled "done" (this assumes the recorded
demonstrations succeed) and frames at ``--negative-fractions`` of the episode
"not done" (0.0 = first frame; 0.3 on svla_so100_stacking = cube in the gripper,
not yet stacked). ``--compare-with-start`` sends the first frame as a reference
alongside each judged frame. The VLM's P(yes) is
collected for both and the threshold with the best balanced accuracy is chosen;
AUROC says whether the VLM separates them at all (0.5 = chance).

    pip install -e ".[smolvla,vlm]"
    python scripts/vlm_calibrate_lerobot.py --device cuda --episodes 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vla_ros2.vlm import (
    DEFAULT_PROMPT_TEMPLATE,
    SuccessDetector,
    calibrate_threshold,
    load_vlm,
)

DEFAULT_DATASET = "lerobot/svla_so100_stacking"


def _to_uint8_hwc(tensor: Any) -> np.ndarray:
    array = np.asarray(tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm", default="smolvlm")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--camera", default="observation.images.top")
    parser.add_argument("--no-image-splitting", action="store_true")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="must contain {instruction}",
    )
    parser.add_argument(
        "--negative-fractions",
        type=float,
        nargs="+",
        default=[0.0],
        help="episode positions labeled not done (0.0 = first frame)",
    )
    parser.add_argument(
        "--compare-with-start",
        action="store_true",
        help="judge each frame against the episode's first frame (two-image prompt)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args(argv)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs: dict[str, object] = {"device": args.device}
    if args.pretrained:
        kwargs["pretrained"] = args.pretrained
    if args.no_image_splitting:
        kwargs["image_splitting"] = False
    vlm = load_vlm(args.vlm, **kwargs)
    detector = SuccessDetector(vlm, prompt_template=args.prompt_template)

    rows: list[dict[str, Any]] = []
    for episode in args.episodes:
        dataset = LeRobotDataset(args.dataset, episodes=[episode])
        last = len(dataset) - 1
        samples = [(False, int(round(fraction * last))) for fraction in args.negative_fractions]
        samples.append((True, last))
        reference = _to_uint8_hwc(dataset[0][args.camera]) if args.compare_with_start else None
        for label, index in samples:
            item = dataset[index]
            image = _to_uint8_hwc(item[args.camera])
            start = time.perf_counter()
            judgement = detector.judge(str(item["task"]), [image], reference=reference)
            latency_ms = (time.perf_counter() - start) * 1000.0
            rows.append(
                {
                    "episode": episode,
                    "frame": index,
                    "done": label,
                    "p_yes": judgement.confidence,
                    "latency_ms": latency_ms,
                }
            )
            print(
                f"episode {episode:3d} frame {index:4d} done={label!s:5} "
                f"p_yes={judgement.confidence:.3f} ({latency_ms:.0f} ms)"
            )

    scores = [float(row["p_yes"]) for row in rows]
    labels = [bool(row["done"]) for row in rows]
    calibration = calibrate_threshold(scores, labels)
    done = [s for s, label in zip(scores, labels, strict=True) if label]
    not_done = [s for s, label in zip(scores, labels, strict=True) if not label]
    print(
        f"AUROC={calibration.auroc:.3f} threshold={calibration.threshold:.3f} "
        f"balanced_accuracy={calibration.balanced_accuracy:.3f} "
        f"mean p_yes done={np.mean(done):.3f} not_done={np.mean(not_done):.3f}"
    )
    if args.out:
        payload = {
            "dataset": args.dataset,
            "camera": args.camera,
            "vlm": vlm.name,
            "model_id": vlm.model_id,
            "device": args.device,
            "image_splitting": not args.no_image_splitting,
            "prompt_template": (
                detector.compare_prompt_template
                if args.compare_with_start
                else args.prompt_template
            ),
            "compare_with_start": args.compare_with_start,
            "negative_fractions": args.negative_fractions,
            "auroc": calibration.auroc,
            "threshold": calibration.threshold,
            "balanced_accuracy": calibration.balanced_accuracy,
            "rows": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
