# pi0 Compatibility Probe

Date: 2026-06-02

This probe checked whether public LeRobot pi0 checkpoints could be used as a
local `vla_zoo` action probe in the current environment.

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Local package env: `.venv-smolvla`
- LeRobot: `0.5.1`
- Torch: `2.10.0+cu128`

## Checked Checkpoints

| Checkpoint | Result |
|---|---|
| `lerobot/pi0` | Did not load with LeRobot 0.5.1 config decoding. The checkpoint config includes fields not accepted by the installed `PI0Config`. |
| `lerobot/pi0_base` | Local load was started but did not complete within the probe window; it is treated as heavy/local-uncleared, not verified. |

## Runtime Position

`vla_zoo` keeps `load_model("pi0")` lightweight and remote-first by default.
Local loading must be explicit:

```python
from vla_zoo import load_model

model = load_model(
    "pi0",
    enable_local=True,
    pretrained="lerobot/pi0_base",
    device="cuda",
)
```

This does not claim pi0 local inference success. It records the compatibility
boundary so users can run pi0 in a dedicated server or a version-pinned LeRobot
environment without surprising the base runtime.
