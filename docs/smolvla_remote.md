# SmolVLA Remote Serving

SmolVLA is the most feasible real-model path in vla_zoo, and its most credible
deployment shape is **remote serving**: run the heavyweight LeRobot policy on a
GPU box, and keep the robot-side process (Python API, ROS2 node, or PyBullet
comparison) on the `runtime=remote` client. This page documents environment
isolation and the reproducible bring-up plan.

This is integration-shape and operations guidance. It is **not** a claim that
SmolVLA reaches any task-success quality; the base checkpoint still needs
robot/task-specific fine-tuning and calibration.

## Why a dedicated environment

The `smolvla` extra installs `lerobot[smolvla]`, which pins specific
`transformers` and `torch` versions. The `openvla` extra pins a *different*
`transformers` (`4.40.1`) plus `tokenizers`/`timm`. Installing both in one
environment leads to dependency conflicts, so SmolVLA serving should live in its
own virtual environment:

```bash
python3 -m venv .venv-smolvla
.venv-smolvla/bin/pip install -e ".[cli,server,smolvla]"
```

The base vla_zoo install (`pip install -e ".[dev,cli,server,sim]"`) stays free of
heavyweight model dependencies, so CI and the dry-run runtime keep working
without LeRobot or GPU weights.

## Generate the plan

```bash
vla-zoo smolvla-remote-plan \
  --public-host gpu-box \
  --port 8000 \
  --device cuda:0 \
  --markdown-out docs/assets/smolvla_remote_smoke_plan.md \
  --out docs/assets/smolvla_remote_smoke_plan.json
```

The command emits the env-isolation, server, health-check, and robot-side
consumption commands as a single reproducible plan. See
[`docs/assets/smolvla_remote_smoke_plan.md`](assets/smolvla_remote_smoke_plan.md).

## Bring-up sequence

1. **Server (GPU box)** — start the FastAPI inference server inside the isolated
   environment:

   ```bash
   vla-zoo serve --model smolvla --pretrained lerobot/smolvla_base \
     --host 0.0.0.0 --port 8000 --device cuda:0
   ```

2. **Health check** — confirm readiness before sending observations:

   ```bash
   curl -fsS http://gpu-box:8000/health
   ```

3. **Robot-side consumption** — drive the server through the remote runtime:

   ```bash
   python examples/python/smolvla_remote_client.py \
     --remote-url http://gpu-box:8000 --instruction "pick up the red block"

   vla-zoo compare pybullet --models smolvla --runtime remote \
     --remote-map smolvla=http://gpu-box:8000
   ```

4. **ROS2 remote smoke** — generate a matching ROS2 recording plan (dry-run safe
   by default):

   ```bash
   vla-zoo ros remote-smoke-plan --model smolvla --remote-url http://gpu-box:8000
   ```

## Evidence status

The `remote_server` cell for SmolVLA in the
[evidence matrix](assets/vla_model_evidence_matrix.html) stays `planned`: the
server command and plan are reproducible, but a recorded SmolVLA `/v1/predict`
response is not yet checked in. It moves to `partial`/`verified` only after a
real recorded response from an actual SmolVLA server.
