"""Measure a real OpenVLA-7b local-runtime profile through the public adapter.

This is a runtime-evidence capture, not a benchmark. It loads OpenVLA-7b on the local GPU
(4-bit by default so it fits a 16 GB consumer card), runs a few inferences on a synthetic
camera frame, and records the measured load time, VRAM, and per-inference latency to a JSON
artifact.

It makes **no task-success claim**: the input is a synthetic frame, so the output action is
a real inference result, not evidence of robot-skill quality. It exists to substantiate the
``local_runtime`` / ``gpu_inference`` evidence cells with measured numbers.

Run (needs the openvla extra + a CUDA GPU; OpenVLA's remote code needs ``timm<1.0``):

    PYTHONPATH=src python3 scripts/measure_openvla_runtime.py \
        --runs 5 --out docs/assets/openvla_local_runtime.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from vla_ros2 import load_model

DEFAULT_OUT = Path("docs/assets/openvla_local_runtime.json")
DEFAULT_INSTRUCTION = "pick up the red block"


def _synthetic_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)


def _gpu_info() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import torch

    load_start = perf_counter()
    model = load_model(
        "openvla",
        device=args.device,
        pretrained=args.pretrained,
        unnorm_key=args.unnorm_key,
        attn_implementation="eager",
        load_in_4bit=args.load_in_4bit,
    )
    load_sec = perf_counter() - load_start
    vram_after_load_gb = torch.cuda.memory_allocated() / 1e9

    latencies_ms: list[float] = []
    last_action: list[float] = []
    for i in range(args.runs):
        frame = _synthetic_frame(seed=i)
        torch.cuda.synchronize()
        start = perf_counter()
        action = model.predict(image=frame, instruction=args.instruction)
        torch.cuda.synchronize()
        latencies_ms.append((perf_counter() - start) * 1000.0)
        last_action = [float(x) for x in np.asarray(action.to_numpy()).reshape(-1)]

    ordered = sorted(latencies_ms)
    record = {
        "model": "openvla",
        "pretrained": args.pretrained,
        "quantization": "nf4-4bit" if args.load_in_4bit else "bf16",
        "instruction": args.instruction,
        "image_source": "synthetic-random-rgb-224",
        "runs": args.runs,
        "load_sec": round(load_sec, 2),
        "vram_after_load_gb": round(vram_after_load_gb, 3),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "latency_ms_min": round(min(latencies_ms), 1),
        "latency_ms_p50": round(ordered[len(ordered) // 2], 1),
        "latency_ms_max": round(max(latencies_ms), 1),
        "action_dim": len(last_action),
        "sample_action": [round(x, 5) for x in last_action],
        "platform": platform.platform(),
        "note": (
            "Runtime evidence on a synthetic frame. Measures load/VRAM/latency only; "
            "it is not a task-success or policy-quality claim."
        ),
        **_gpu_info(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
