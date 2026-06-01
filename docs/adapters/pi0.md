# pi0 Adapter Card

| Field | Value |
|---|---|
| Source | built-in |
| Target | `vla_zoo.adapters.pi0:Pi0Adapter` |
| Upstream project | openpi / LeRobot |
| Default checkpoint | `lerobot/pi0_base` |
| Aliases | openpi, pi0-fast, pi05 |
| Status | experimental |
| Experimental | true |
| Domain | - |
| Install hint | `pip install "vla_zoo[openpi]"` |

## Runtime Contract

| Field | Value |
|---|---|
| Family | pi-family VLA |
| Role | remote-first action-chunk VLA target |
| Action space | `custom` |
| Action shape | `checkpoint-specific; lerobot/pi0 is (6,), lerobot/pi0_base is (32,)` |
| Output | policy-specific continuous manipulation action |
| Control Hz | policy/server dependent |
| Action chunks | expected |
| Proprioception | expected |
| Local runtime | disabled by default; enable_local=True in a dedicated GPU env |
| Remote runtime | recommended |
| Dependencies | LeRobot/openpi stack in serving environment |
| License caveat | external project and checkpoint license apply |

## Inputs

- images per policy config
- natural language instruction
- robot state expected

## Verification Status

Remote-first adapter path is implemented. Local real-model action probe has not completed in this repository; use a dedicated GPU serving environment.

## Operational Caveats

- This card describes the vla_zoo runtime contract, not task success on a robot.
- External model weights, datasets, and upstream licenses are not redistributed by vla_zoo.
- Real robot deployment still needs calibrated cameras, state mapping, action clipping, watchdogs, and a hardware-specific bridge.

## Useful Commands

```bash
vla-zoo info pi0
vla-zoo serve-plan --models pi0
```
