# SmolVLA Remote Serving Plan

Run LeRobot SmolVLA as a remote inference server in an isolated environment,
then consume it from the robot-side runtime through `runtime=remote`. This file
is a reproducible bring-up plan, not a recorded run, and makes no claim about
SmolVLA task-success quality.

## 1. Isolated Environment

`lerobot[smolvla]` pins specific `transformers`/`torch` versions that clash with
the `openvla` extra, so install it in a dedicated virtual environment:

```bash
python3 -m venv .venv-smolvla
.venv-smolvla/bin/pip install -e '.[cli,server,smolvla]'
```

## 2. SmolVLA Server (GPU box)

```bash
vla-zoo serve --model smolvla --host 0.0.0.0 --port 8000 --pretrained lerobot/smolvla_base --device cuda:0
```

Confirm readiness before sending requests:

```bash
curl -fsS http://gpu-box:8000/health
```

## 3. Robot-Side Consumption

```bash
.venv-smolvla/bin/python examples/python/smolvla_remote_client.py --remote-url http://gpu-box:8000 --instruction 'pick up the red block'
vla-zoo compare pybullet --models smolvla --runtime remote --remote-map smolvla=http://gpu-box:8000
```

Generate a matching ROS2 remote smoke recording plan:

```bash
vla-zoo ros remote-smoke-report --model smolvla --remote-url http://gpu-box:8000 --output-dir results/ros2_remote_smoke --duration-sec 30
```

## Settings

- model: `smolvla`
- pretrained: `lerobot/smolvla_base`
- public_url: `http://gpu-box:8000`
- device: `cuda:0`
- dtype: `-`
- venv_dir: `.venv-smolvla`
- extras: `cli,server,smolvla`

## Caveat

SmolVLA needs the pinned lerobot[smolvla] stack, which conflicts with the openvla extra; keep it in a dedicated virtual environment. This plan records no /v1/predict response and makes no policy-quality or task-success claim.
