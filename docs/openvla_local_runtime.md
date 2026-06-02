# OpenVLA Local Runtime Evidence

This page records a **real, measured** OpenVLA-7b run on a local consumer GPU through the
public `vla_zoo` adapter. It is runtime evidence, not a benchmark and **not a task-success
claim**: the input is a synthetic camera frame, so the output action is a genuine inference
result, not evidence of robot-skill quality.

## What was measured

OpenVLA-7b loaded through `load_model("openvla", load_in_4bit=True, ...)` and predicted a
7-DoF `eef_delta` action on a synthetic 224×224 RGB frame, repeated over several runs.

| Metric | Value |
|---|---|
| Model | `openvla/openvla-7b` |
| Quantization | 4-bit (nf4, double-quant, bf16 compute) |
| GPU | NVIDIA GeForce GPU |
| torch / CUDA | 2.10.0+cu128 / 12.8 |
| Load time | ~20 s (warm cache) |
| VRAM after load | 4.35 GB |
| VRAM peak | **4.63 GB** (fits a 16 GB card with wide margin) |
| Latency (min / p50 / max) | 1135 / 1575 / 2654 ms |
| Action shape | `(7,)` |

The measured numbers are checked in at
[`docs/assets/openvla_local_runtime.json`](assets/openvla_local_runtime.json).

## Why 4-bit

bf16 weights are ~15 GB and do not fit alongside activations on a 16 GB GPU. 4-bit (nf4)
loading via bitsandbytes brings the footprint to ~4.6 GB peak, so the full 7B model runs on
a consumer card. The adapter exposes this directly:

```python
from vla_zoo import load_model

model = load_model(
    "openvla",
    device="cuda:0",
    unnorm_key="bridge_orig",
    load_in_4bit=True,        # fits a 16 GB consumer GPU
)
action = model.predict(image=frame, instruction="pick up the red block")
```

## Reproduce

OpenVLA's `trust_remote_code` modeling requires `timm>=0.9.10,<1.0.0`. If your environment
has a newer timm (e.g. for other models), isolate it in a venv that inherits the system
packages and shadows only timm:

```bash
python3 -m venv --system-site-packages /tmp/openvla_venv
/tmp/openvla_venv/bin/pip install "timm>=0.9.10,<1.0.0"

PYTHONPATH=src /tmp/openvla_venv/bin/python scripts/measure_openvla_runtime.py \
    --runs 5 --out docs/assets/openvla_local_runtime.json
```

The capture script is [`scripts/measure_openvla_runtime.py`](../scripts/measure_openvla_runtime.py).

## Scope and limitations

- The input is a synthetic random frame. This verifies the **runtime path** (load →
  inference → typed action, with latency and VRAM), not policy quality or task success.
- transformers 4.48.3 emits a version-mismatch warning (OpenVLA targets 4.40.1); the remote
  code still runs, but exact action values can differ from the reference stack.
- For deployment, prefer the remote-server split (see [deployment](deployment.md)); on-device
  4-bit inference at ~1–2 s/step is an outer-loop policy rate, not a control rate.
