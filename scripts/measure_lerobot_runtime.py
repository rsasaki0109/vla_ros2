"""Measure a real LeRobot policy (SmolVLA / pi0) local-runtime profile via the adapter.

Runtime-evidence capture, not a benchmark. Loads a LeRobot policy on the local GPU through
the public adapter, runs a few inferences on a synthetic camera frame, and records measured
load time, VRAM, and per-inference latency to a JSON artifact.

It makes **no task-success claim**: the input is a synthetic frame, so the output action is
a real inference result, not evidence of robot-skill quality. It substantiates the
``local_runtime`` / ``gpu_inference`` evidence cells with measured numbers.

Run (needs the smolvla/openpi extra + a CUDA GPU):

    PYTHONPATH=src python3 scripts/measure_lerobot_runtime.py \
        --model smolvla --runs 5 --out docs/assets/smolvla_local_runtime.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from vla_ros2 import load_model

DEFAULTS = {
    "smolvla": "lerobot/smolvla_base",
    "pi0": "lerobot/pi0",
}
DEFAULT_INSTRUCTION = "pick up the cube"


def _synthetic_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(DEFAULTS), default="smolvla")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import torch

    pretrained = args.pretrained or DEFAULTS[args.model]
    out = args.out or Path(f"docs/assets/{args.model}_local_runtime.json")

    load_kwargs: dict[str, object] = {
        "device": args.device,
        "pretrained": pretrained,
        "local_files_only": True,
    }
    if args.model == "pi0":
        load_kwargs["enable_local"] = True

    load_start = perf_counter()
    model = load_model(args.model, **load_kwargs)
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

    # Drop the first run from the steady-state stats: it carries one-time CUDA/graph warmup.
    steady = latencies_ms[1:] or latencies_ms
    ordered = sorted(steady)
    record = {
        "model": args.model,
        "pretrained": pretrained,
        "instruction": args.instruction,
        "image_source": "synthetic-random-rgb-256",
        "runs": args.runs,
        "load_sec": round(load_sec, 2),
        "vram_after_load_gb": round(vram_after_load_gb, 3),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "latency_ms_first": round(latencies_ms[0], 1),
        "latency_ms_steady_min": round(min(ordered), 1),
        "latency_ms_steady_p50": round(ordered[len(ordered) // 2], 1),
        "latency_ms_steady_max": round(max(ordered), 1),
        "action_dim": len(last_action),
        "sample_action": [round(x, 5) for x in last_action],
        "platform": platform.platform(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "note": (
            "Runtime evidence on a synthetic frame. Measures load/VRAM/latency only; "
            "it is not a task-success or policy-quality claim."
        ),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
