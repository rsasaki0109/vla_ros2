# openvla Adapter Card

| Field | Value |
|---|---|
| Source | built-in |
| Target | `vla_zoo.adapters.openvla:OpenVLAAdapter` |
| Upstream project | OpenVLA |
| Default checkpoint | `openvla/openvla-7b` |
| Aliases | - |
| Status | available |
| Experimental | false |
| Domain | - |
| Install hint | `pip install "vla_zoo[openvla]"` |

## Runtime Contract

| Field | Value |
|---|---|
| Family | VLA foundation model |
| Role | single-image VLA reference adapter |
| Action space | `eef_delta` |
| Action shape | `(7,)` |
| Output | OpenVLA-style 7-DoF action |
| Control Hz | model/robot dependent |
| Action chunks | no |
| Proprioception | not required by default adapter |
| Local runtime | supported with optional ML dependencies |
| Remote runtime | recommended for robot-side ROS2 |
| Dependencies | torch, transformers, HF weights |
| License caveat | external project and model license apply |

## Inputs

- single RGB image
- natural language instruction
- optional unnormalization key

## Verification Status

Adapter scaffold and prompt path are documented. In the latest local run, the CUDA prompt probe did not complete because free GPU memory was insufficient.

## Operational Caveats

- This card describes the vla_zoo runtime contract, not task success on a robot.
- External model weights, datasets, and upstream licenses are not redistributed by vla_zoo.
- Real robot deployment still needs calibrated cameras, state mapping, action clipping, watchdogs, and a hardware-specific bridge.

## Useful Commands

```bash
vla-zoo info openvla
vla-zoo serve-plan --models openvla
```
