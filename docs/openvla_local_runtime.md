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

## Remote serving (verified end-to-end)

The same 4-bit path also runs behind the real FastAPI server, which is the recommended
deployment split (heavy model on a GPU box, lightweight client on the robot — see
[deployment](deployment.md)). The `serve` command now exposes `--load-in-4bit` so the 7B
model fits a 16 GB card on the server side too:

```bash
# server (GPU box), in the OpenVLA venv (timm<1.0)
HF_HUB_OFFLINE=1 vla-zoo serve --model openvla --pretrained openvla/openvla-7b \
    --device cuda:0 --unnorm-key bridge_orig --load-in-4bit --host 127.0.0.1 --port 8012

# client (robot side): health-first probe records one /v1/predict response
vla-zoo remote-probe --model openvla --remote-url http://127.0.0.1:8012 \
    --out docs/assets/sample_task_verification/openvla_remote_probe.json \
    --markdown-out docs/assets/sample_task_verification/openvla_remote_probe.md --strict
```

The health check returned `ready: true` and `/v1/predict` returned a typed 7-DoF action; the
recorded result is checked in at
[`openvla_remote_probe.md`](assets/sample_task_verification/openvla_remote_probe.md).

## ROS2 remote trace (verified end-to-end)

The real `VLARuntimeNode` was driven in `runtime=remote` mode against the live OpenVLA-7b
(4-bit) server, recording its action/status/diagnostics streams:

```bash
# server (OpenVLA venv, 4-bit) as above, then with the ROS2 overlay sourced:
python3 scripts/record_ros2_remote_trace.py --model openvla \
    --remote-url http://127.0.0.1:8014 --duration 35 --output-dir results/ros2_remote_openvla
vla-zoo ros remote-smoke-check --output-dir results/ros2_remote_openvla --model openvla \
    --remote-url http://127.0.0.1:8014
```

The check passed with 7 `RemoteVLAClient` actions and 143 status/diagnostics records, 0
inference errors. The recorded result is checked in at
[`sample_ros2_remote_openvla/remote_smoke_check.md`](assets/sample_ros2_remote_openvla/remote_smoke_check.md).
The latest `vla-zoo-diagnostics/v1` snapshot reconstructed from that run's `/diagnostics`
stream (`vla-zoo diag-report --from-ros-log .../vla_diagnostics.jsonl`) is at
[`runtime_diagnostics_snapshot.md`](assets/sample_ros2_remote_openvla/runtime_diagnostics_snapshot.md),
and the time-series reduction over the whole log (`--summary`: latency p50/max, drop/clip
peaks, worst-severity record) is at
[`runtime_diagnostics_summary.md`](assets/sample_ros2_remote_openvla/runtime_diagnostics_summary.md).
See the [SmolVLA evidence page](smolvla_local_runtime.md) for the loopback-multicast note on
why this uses the single-process harness on this host.

## Real-scene action probe (verified runtime path)

The measurements above run on a synthetic random frame. To exercise the *real* image
preprocessing/encode path, `vla-zoo demo action-probe` drives OpenVLA-7b (4-bit) through the
PyBullet pick-and-place rollout and records the action stream from genuinely rendered frames:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src /tmp/openvla_venv/bin/python -m vla_zoo.cli.main demo action-probe \
  --model openvla --runtime local --allow-local-heavy \
  --pretrained openvla/openvla-7b --device cuda:0 \
  --adapter-kwarg load_in_4bit=true --adapter-kwarg unnorm_key=bridge_orig \
  --out docs/assets/sample_pybullet_openvla/openvla_action_probe.jsonl \
  --summary-md docs/assets/sample_pybullet_openvla/runtime_action_probe.md
```

The recorded run (21 adapter queries, action dim 7, latency p50 ~2.0 s) is at
[`runtime_action_probe.md`](assets/sample_pybullet_openvla/runtime_action_probe.md); the
canonical `vla_actions.jsonl` log
([`openvla_action_probe.jsonl`](assets/sample_pybullet_openvla/openvla_action_probe.jsonl))
replays through `vla-zoo bench-replay` (`success=None`). This upgrades only the **input** — from
synthetic noise to a real scene render — and is still **not** a task-success or policy-quality
claim (`policy_quality=not_verified`). The probe needs PyBullet *inside* the timm<1.0 OpenVLA
venv (`/tmp/openvla_venv`).

## Scope and limitations

- The input is a synthetic random frame. This verifies the **runtime path** (load →
  inference → typed action, with latency and VRAM), not policy quality or task success.
- transformers 4.48.3 emits a version-mismatch warning (OpenVLA targets 4.40.1); the remote
  code still runs, but exact action values can differ from the reference stack.
- For deployment, prefer the remote-server split (see [deployment](deployment.md)); on-device
  4-bit inference at ~1–2 s/step is an outer-loop policy rate, not a control rate.
