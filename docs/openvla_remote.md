# OpenVLA Remote GPU Path

OpenVLA 7B is important for credibility, but local inference was blocked by free
VRAM (see the [OpenVLA prompt probe](assets/sample_task_verification/openvla_prompt_probe.md)).
The correct path is a **remote GPU server** with enough memory, consumed from the
robot/client side through `runtime=remote`.

This page documents the remote bring-up and a health-first probe. It is
integration and operations guidance, **not** a task-success claim: no robot-skill
quality is asserted for OpenVLA in this repository.

## Server (GPU box)

```bash
vla-zoo serve --model openvla \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda:0 \
  --pretrained openvla/openvla-7b \
  --unnorm-key bridge_orig
```

OpenVLA weights and optional dependencies are external. Install them on the GPU
box only (`pip install -e ".[cli,server,openvla]"`); the base/robot environment
stays free of heavyweight model dependencies.

## Health-first probe (robot/client side)

`vla-zoo remote-probe` checks `/health` **before** sending a single `/v1/predict`
request, so an unreachable or not-ready server fails fast with a recorded reason
instead of a confusing client traceback:

```bash
vla-zoo remote-probe --model openvla --remote-url http://gpu-box:8000 \
  --instruction "pick up the red block" \
  --out results/openvla_remote_probe.json \
  --markdown-out results/openvla_remote_probe.md \
  --strict
```

`--strict` exits non-zero unless the probe completes. The Python entry point is
[`examples/python/openvla_remote_probe.py`](../examples/python/openvla_remote_probe.py).

The probe result has one of three statuses:

| status | meaning |
|---|---|
| `ok` | `/health` ready and `/v1/predict` returned a typed action (recorded) |
| `unreachable` | `/health` failed or reported not-ready; predict was not attempted |
| `predict_failed` | server was healthy but `/v1/predict` raised |

## Tooling demonstration

The probe is exercised end-to-end against the in-repo `dummy` server so the tool
itself is verified without any model download:
[`remote_probe_dummy.md`](assets/sample_task_verification/remote_probe_dummy.md).
This is a tooling sample, not OpenVLA evidence.

## Evidence status

The OpenVLA `remote_server` cell in the
[evidence matrix](assets/vla_model_evidence_matrix.html) stays `planned`: the
server command and health-first probe are reproducible, but a real OpenVLA
`/v1/predict` response on a GPU box is not yet checked in. It moves to `verified`
only after a real recorded OpenVLA response.
