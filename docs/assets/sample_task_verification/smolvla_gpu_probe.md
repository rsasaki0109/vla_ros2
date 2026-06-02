# SmolVLA GPU Probe

Date: 2026-06-02

This probe ran the real LeRobot SmolVLA checkpoint through the `vla_zoo`
adapter boundary:

```python
from vla_zoo import load_model

model = load_model("smolvla", pretrained="lerobot/smolvla_base", device="cuda")
action = model.predict(
    image=zero_rgb_image,
    instruction="pick up the red block",
    state=zero_state_6d,
)
```

Result:

| Field | Value |
|---|---|
| Model | `lerobot/smolvla_base` |
| Dependency env | `.venv-smolvla`, `lerobot[smolvla]==0.5.1` |
| GPU | 16 GB VRAM GPU |
| Action spec | `custom`, shape `(6,)` |
| Load time | 38.015 s |
| CUDA forward time | 2064.141 ms |
| Peak CUDA memory | 926.3 MB |
| Output action | `[-0.111041, 0.305288, 0.123189, 0.145649, 0.281270, -0.229793]` |

The adapter mapped the single `primary` image to SmolVLA's three declared camera
inputs: `observation.images.camera1`, `observation.images.camera2`, and
`observation.images.camera3`. It used a provided zero 6D state vector.

This is a real model inference-path probe, not a robot task-success benchmark.
SmolVLA base still needs robot/task-specific fine-tuning and calibrated
observations/actions before any meaningful skill claim.
