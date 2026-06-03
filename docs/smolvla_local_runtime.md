# SmolVLA Local Runtime Evidence

This page records a **real, measured** SmolVLA run on a local consumer GPU through the
public `vla_zoo` adapter. It is runtime evidence, not a benchmark and **not a task-success
claim**: the input is a synthetic camera frame, so the output action is a genuine inference
result, not evidence of robot-skill quality.

## What was measured

`lerobot/smolvla_base` loaded through `load_model("smolvla", ...)` and predicted a 6-DoF
action on a synthetic 256×256 RGB frame, repeated over several runs. SmolVLA is built on a
~500M VLM backbone, so it is small and fast.

| Metric | Value |
|---|---|
| Model | `lerobot/smolvla_base` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (16 GB) |
| torch / CUDA | 2.10.0+cu128 / 12.8 |
| Load time | ~23 s (warm cache) |
| VRAM peak | **0.97 GB** (small footprint) |
| Latency, first run | 802 ms (one-time warmup) |
| Latency, steady (min / p50 / max) | 60 / 121 / 134 ms |
| Action shape | `(6,)` |

The measured numbers are checked in at
[`docs/assets/smolvla_local_runtime.json`](assets/smolvla_local_runtime.json).

At ~60-130 ms steady-state, SmolVLA is fast enough to run as a real-time outer-loop policy
on this consumer GPU, in contrast to OpenVLA-7b's ~1-2 s/step (see
[OpenVLA local runtime evidence](openvla_local_runtime.md)).

## Reproduce

SmolVLA needs the LeRobot dependencies (`pip install "vla_zoo[smolvla]"`). They are heavy and
can shift `numpy` / `huggingface_hub`, so isolate them in a venv that inherits the system
packages:

```bash
python3 -m venv --system-site-packages /tmp/lerobot_venv
/tmp/lerobot_venv/bin/pip install "lerobot[smolvla]>=0.5.1,<0.6.0"

PYTHONPATH=src /tmp/lerobot_venv/bin/python scripts/measure_lerobot_runtime.py \
    --model smolvla --runs 6 --out docs/assets/smolvla_local_runtime.json
```

The capture script is [`scripts/measure_lerobot_runtime.py`](../scripts/measure_lerobot_runtime.py).

## Remote serving (verified end-to-end)

The same checkpoint was also served through the real FastAPI server and exercised over HTTP,
which is the recommended deployment split (see [deployment](deployment.md)):

```bash
# server (GPU box / this machine), in the LeRobot venv
HF_HUB_OFFLINE=1 vla-zoo serve --model smolvla --pretrained lerobot/smolvla_base \
    --device cuda --host 127.0.0.1 --port 8011

# client (robot side): health-first probe records one /v1/predict response
vla-zoo remote-probe --model smolvla --remote-url http://127.0.0.1:8011 \
    --out docs/assets/sample_task_verification/smolvla_remote_probe.json \
    --markdown-out docs/assets/sample_task_verification/smolvla_remote_probe.md --strict
```

The health check returned `ready: true` and `/v1/predict` returned a typed 6-DoF action; the
recorded result is checked in at
[`smolvla_remote_probe.md`](assets/sample_task_verification/smolvla_remote_probe.md). This is
the first real-model (non-dummy) remote `/v1/predict` recording in the repo.

## ROS2 remote trace (verified end-to-end)

The real `VLARuntimeNode` was also driven in `runtime=remote` mode against the live SmolVLA
server, recording its `/vla/action`, `/vla/status`, and `/diagnostics` streams:

```bash
# server in the LeRobot venv (as above), then, with the ROS2 overlay sourced:
python3 scripts/record_ros2_remote_trace.py --model smolvla \
    --remote-url http://127.0.0.1:8013 --output-dir results/ros2_remote_smolvla
vla-zoo ros remote-smoke-check --output-dir results/ros2_remote_smolvla --model smolvla \
    --remote-url http://127.0.0.1:8013
```

The check passed with 14 `RemoteVLAClient` actions and 106 status/diagnostics records, 0
inference errors. The recorded result is checked in at
[`sample_ros2_remote_smolvla/remote_smoke_check.md`](assets/sample_ros2_remote_smolvla/remote_smoke_check.md).
The latest `vla-zoo-diagnostics/v1` snapshot reconstructed from that run's `/diagnostics`
stream (`vla-zoo diag-report --from-ros-log .../vla_diagnostics.jsonl`) is at
[`runtime_diagnostics_snapshot.md`](assets/sample_ros2_remote_smolvla/runtime_diagnostics_snapshot.md),
and the time-series reduction over the whole log (`--summary`: latency p50/max, drop/clip
peaks, worst-severity record) is at
[`runtime_diagnostics_summary.md`](assets/sample_ros2_remote_smolvla/runtime_diagnostics_summary.md).

> **Loopback note:** the standard flow is the 3-process `smoke_record.launch.py`, which
> needs cross-process DDS discovery (multicast). If `ip link show lo` shows no `MULTICAST`
> flag, cross-process discovery fails; `scripts/record_ros2_remote_trace.py` runs the same
> real node, input, and recorder in one process to sidestep that, recording the identical
> node → RemoteVLAClient → server path.

## Real-scene action probe (verified runtime path)

Every measurement above runs on a synthetic random frame. To exercise the *real* image
preprocessing/encode path, `vla-zoo demo action-probe` drives SmolVLA through the PyBullet
pick-and-place rollout and records the full action stream from genuinely rendered camera
frames:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src python -m vla_zoo.cli.main demo action-probe \
  --model smolvla --runtime local --allow-local-heavy \
  --pretrained lerobot/smolvla_base --device cuda --local-files-only --return-action-chunk \
  --out docs/assets/sample_pybullet_smolvla/smolvla_action_probe.jsonl \
  --summary-md docs/assets/sample_pybullet_smolvla/runtime_action_probe.md
```

The recorded run (21 adapter queries, action dim 6) is at
[`runtime_action_probe.md`](assets/sample_pybullet_smolvla/runtime_action_probe.md); the
canonical `vla_actions.jsonl` log
([`smolvla_action_probe.jsonl`](assets/sample_pybullet_smolvla/smolvla_action_probe.jsonl))
replays through `vla-zoo bench-replay` (which records `success=None`). This upgrades only the
**input** — from synthetic noise to a real scene render. It is still **not** a task-success
or policy-quality claim (`policy_quality=not_verified`); the chunk-per-query latency (p50
~382 ms) is higher than the cached-queue `select_action` steady state because each query
forces a fresh encode. The SmolVLA and OpenVLA probes are compared side by side (latency /
action rate, no task-success claim) at
[`runtime_probe_comparison.md`](assets/sample_pybullet_compare/runtime_probe_comparison.md).

## pi0 status

The same script targets pi0 with `--model pi0`. The version-matched checkpoint is now
settled: the old `lerobot/pi0` config schema is rejected by LeRobot 0.5.1, but
`lerobot/pi0_base` decodes cleanly (`PI0Config`, 32D) and its bf16 model fits a 16 GB GPU
(~8.9 GB) via the adapter's new `dtype` override. Local inference stays blocked only on the
gated `google/paligemma-3b-pt-224` tokenizer the pi0 processor requires
(`GatedRepoError 401` — manual license acceptance + token). See the
[pi0 compatibility probe](assets/sample_task_verification/pi0_compatibility_probe.md) for the
full version matrix and the reproduce command.

## Scope and limitations

- The input is a synthetic random frame. This verifies the **runtime path** (load →
  inference → typed action, with latency and VRAM), not policy quality or task success.
- SmolVLA base still needs robot/task-specific fine-tuning and calibration before any
  skill claim.
