#!/usr/bin/env python3
"""Off-robot check of the VLM success detector (no ROS2 required).

Without --image, judges two synthetic scenes (red cube stacked on the blue cube
vs. beside it) so you can see whether a VLM separates success from failure.

    pip install -e ".[vlm]"
    python scripts/vlm_success_check.py --device cuda
    python scripts/vlm_success_check.py --image frame.png --instruction "open the drawer"
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from PIL import Image, ImageDraw

from vla_ros2.vlm import SuccessDetector, load_vlm

DEFAULT_INSTRUCTION = "put the red cube on top of the blue cube"


def synthetic_scene(*, stacked: bool, size: int = 384) -> np.ndarray:
    image = Image.new("RGB", (size, size), (200, 190, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 280, size, size], fill=(120, 100, 80))
    draw.rectangle([150, 220, 230, 300], fill=(30, 60, 220))
    if stacked:
        draw.rectangle([155, 145, 225, 220], fill=(220, 30, 30))
    else:
        draw.rectangle([280, 225, 350, 300], fill=(220, 30, 30))
    return np.asarray(image)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm", default="smolvlm")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image", action="append", default=[], help="RGB image to judge")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-image-splitting",
        action="store_true",
        help="SmolVLM: one downscaled view instead of tiles (faster, less accurate)",
    )
    parser.add_argument("--out", default=None, help="write results as JSON")
    args = parser.parse_args(argv)

    kwargs: dict[str, object] = {"device": args.device}
    if args.pretrained:
        kwargs["pretrained"] = args.pretrained
    if args.no_image_splitting:
        kwargs["image_splitting"] = False
    start = time.perf_counter()
    vlm = load_vlm(args.vlm, **kwargs)
    load_sec = time.perf_counter() - start
    detector = SuccessDetector(vlm, threshold=args.threshold)

    if args.image:
        cases = [(path, np.asarray(Image.open(path).convert("RGB"))) for path in args.image]
    else:
        cases = [
            ("synthetic:stacked", synthetic_scene(stacked=True)),
            ("synthetic:not_stacked", synthetic_scene(stacked=False)),
        ]

    results = []
    print(f"vlm={vlm.name} model={vlm.model_id} load={load_sec:.1f}s")
    for label, image in cases:
        judgement = detector.judge(args.instruction, [image])
        p_yes = "n/a" if judgement.confidence is None else f"{judgement.confidence:.3f}"
        print(
            f"{label}: success={judgement.success} p_yes={p_yes} "
            f"latency={judgement.latency_ms:.0f}ms"
        )
        results.append(
            {
                "case": label,
                "success": judgement.success,
                "p_yes": judgement.confidence,
                "latency_ms": judgement.latency_ms,
            }
        )
    if args.out:
        payload = {
            "vlm": vlm.name,
            "model_id": vlm.model_id,
            "device": args.device,
            "instruction": args.instruction,
            "load_sec": load_sec,
            "results": results,
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
