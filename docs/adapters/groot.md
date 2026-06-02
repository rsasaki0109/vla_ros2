# groot Adapter Card

| Field | Value |
|---|---|
| Source | built-in |
| Target | `vla_zoo.adapters.groot:GR00TAdapter` |
| Upstream project | Isaac GR00T |
| Default checkpoint | `adapter-specific` |
| Aliases | gr00t, isaac-groot |
| Status | experimental |
| Experimental | true |
| Domain | humanoid/generalist |
| Install hint | `Install Isaac GR00T dependencies in the serving environment.` |

## Runtime Contract

| Field | Value |
|---|---|
| Family | humanoid/generalist foundation model |
| Role | experimental humanoid/generalist adapter target |
| Action space | `custom` |
| Action shape | `adapter-specific` |
| Output | humanoid/generalist action interface |
| Control Hz | stack dependent |
| Action chunks | adapter-specific |
| Proprioception | expected |
| Local runtime | experimental placeholder |
| Remote runtime | recommended |
| Dependencies | Isaac GR00T stack in serving environment |
| License caveat | external NVIDIA project and model license apply |

## Inputs

- multimodal observations
- instruction/task context
- robot state expected

## Verification Status

Experimental placeholder only; real GR00T inference is not verified in this repository yet.

## Operational Caveats

- This card describes the vla_zoo runtime contract, not task success on a robot.
- External model weights, datasets, and upstream licenses are not redistributed by vla_zoo.
- Real robot deployment still needs calibrated cameras, state mapping, action clipping, watchdogs, and a hardware-specific bridge.

## Useful Commands

```bash
vla-zoo info groot
vla-zoo serve-plan --models groot
```
