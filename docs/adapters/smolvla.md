# smolvla Adapter Card

| Field | Value |
|---|---|
| Source | built-in |
| Target | `vla_zoo.adapters.smolvla:SmolVLAAdapter` |
| Upstream project | LeRobot SmolVLA |
| Default checkpoint | `lerobot/smolvla_base` |
| Aliases | lerobot-smolvla |
| Status | missing optional deps: pip install "vla_zoo[smolvla]" |
| Experimental | true |
| Domain | - |
| Install hint | `pip install "vla_zoo[smolvla]"` |

## Runtime Contract

| Field | Value |
|---|---|
| Family | LeRobot policy |
| Role | multi-camera/state/action-chunk compact VLA target |
| Action space | `custom` |
| Action shape | `checkpoint-specific; lerobot/smolvla_base is (6,)` |
| Output | policy-specific continuous action |
| Control Hz | policy/robot dependent |
| Action chunks | internal queue; chunk output optional |
| Proprioception | required by typical deployments |
| Local runtime | supported with optional LeRobot dependencies |
| Remote runtime | recommended |
| Dependencies | lerobot[smolvla], torch, HF weights |
| License caveat | external project, dataset, and checkpoint licenses apply |

## Inputs

- multi-camera images
- natural language instruction
- robot state

## Verification Status

Local CUDA inference-path probe completed with lerobot/smolvla_base, including a PyBullet-rendered multi-camera/state observation path. This is not a robot task-success claim.

## Operational Caveats

- This card describes the vla_zoo runtime contract, not task success on a robot.
- External model weights, datasets, and upstream licenses are not redistributed by vla_zoo.
- Real robot deployment still needs calibrated cameras, state mapping, action clipping, watchdogs, and a hardware-specific bridge.

## Useful Commands

```bash
vla-zoo info smolvla
vla-zoo serve-plan --models smolvla
```
