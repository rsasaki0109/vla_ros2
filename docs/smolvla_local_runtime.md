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
| GPU | NVIDIA GeForce GPU |
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

## pi0 status

The same script targets pi0 with `--model pi0`, but local pi0 loading currently fails on a
concrete config-schema mismatch: the cached `lerobot/pi0` checkpoint carries `PI0Config`
fields (`resize_imgs_with_padding`, `adapt_to_pi_aloha`, `num_steps`, …) that LeRobot 0.5.1
rejects. A version-matched checkpoint or serving environment is needed. See the
[pi0 compatibility probe](assets/sample_task_verification/pi0_compatibility_probe.md).

## Scope and limitations

- The input is a synthetic random frame. This verifies the **runtime path** (load →
  inference → typed action, with latency and VRAM), not policy quality or task success.
- SmolVLA base still needs robot/task-specific fine-tuning and calibration before any
  skill claim.
