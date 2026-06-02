# VLA GPU Server Plan

Run one model server per heavyweight adapter, then compare them from the robot-side
runtime through `runtime=remote`. This file is a deployment plan, not a claim that
all listed external checkpoints have been locally verified.

## Servers

| Model | Endpoint | Install | Command | Notes |
|---|---|---|---|---|
| `openvla` | `http://gpu-box:8001` | `pip install -e ".[cli,server,openvla]"` | `vla-zoo serve --model openvla --host 0.0.0.0 --port 8001 --pretrained openvla/openvla-7b --device cuda:0 --dtype bfloat16 --unnorm-key bridge_orig` | Requires external OpenVLA weights and enough GPU memory. |
| `pi0` | `http://gpu-box:8002` | `pip install -e ".[cli,server,openpi]"` | `vla-zoo serve --model pi0 --host 0.0.0.0 --port 8002 --pretrained lerobot/pi0_base --device cuda:0` | Uses explicit LeRobot checkpoint selection; checkpoint compatibility is version-sensitive. |
| `smolvla` | `http://gpu-box:8003` | `pip install -e ".[cli,server,smolvla]"` | `vla-zoo serve --model smolvla --host 0.0.0.0 --port 8003 --pretrained lerobot/smolvla_base --device cuda:0 --dtype bfloat16` | Uses LeRobot policy loading with multi-camera/state observations. |
| `groot` | `http://gpu-box:8004` | `Install the external GR00T stack, then pip install -e ".[cli,server,groot]"` | `vla-zoo serve --model groot --host 0.0.0.0 --port 8004` | Experimental placeholder until an external GR00T runtime is wired in. |

## Robot-Side Comparison

```bash
vla-zoo compare pybullet --models openvla,pi0,smolvla,groot --runtime remote --remote-map openvla=http://gpu-box:8001,pi0=http://gpu-box:8002,smolvla=http://gpu-box:8003,groot=http://gpu-box:8004
```

Remote map:

```text
openvla=http://gpu-box:8001,pi0=http://gpu-box:8002,smolvla=http://gpu-box:8003,groot=http://gpu-box:8004
```
