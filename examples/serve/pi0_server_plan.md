# VLA GPU Server Plan

Run one model server per heavyweight adapter, then compare them from the robot-side
runtime through `runtime=remote`. This file is a deployment plan, not a claim that
all listed external checkpoints have been locally verified.

## Servers

| Model | Endpoint | Install | Command | Notes |
|---|---|---|---|---|
| `pi0` | `http://gpu-box:8001` | `pip install -e ".[cli,server,openpi]"` | `vla-zoo serve --model pi0 --host 0.0.0.0 --port 8001 --pretrained lerobot/pi0_base --device cuda:0` | Uses explicit LeRobot checkpoint selection; checkpoint compatibility is version-sensitive. |

## Robot-Side Comparison

```bash
vla-zoo compare pybullet --models pi0 --runtime remote --remote-map pi0=http://gpu-box:8001
```

Remote map:

```text
pi0=http://gpu-box:8001
```
