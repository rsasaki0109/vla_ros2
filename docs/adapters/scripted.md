# scripted Adapter Card

| Field | Value |
|---|---|
| Source | built-in |
| Target | `vla_zoo.adapters.baselines:ScriptedAdapter` |
| Upstream project | vla_zoo |
| Default checkpoint | `none` |
| Aliases | heuristic, rule-based |
| Status | available |
| Experimental | false |
| Domain | - |
| Install hint | `-` |

## Runtime Contract

| Field | Value |
|---|---|
| Family | rule-based baseline |
| Role | upper-bound sanity check for the scripted smoke scene |
| Action space | `eef_delta` |
| Action shape | `(7,)` |
| Output | phase-aware 7-DoF end-effector delta |
| Control Hz | 5 |
| Action chunks | no |
| Proprioception | not required |
| Local runtime | supported |
| Remote runtime | supported |
| Dependencies | base install |
| License caveat | none |

## Inputs

- phase metadata
- instruction optional
- image optional

## Verification Status

Exercised in deterministic PyBullet comparison reports as a scripted smoke baseline.

## Operational Caveats

- This card describes the vla_zoo runtime contract, not task success on a robot.
- External model weights, datasets, and upstream licenses are not redistributed by vla_zoo.
- Real robot deployment still needs calibrated cameras, state mapping, action clipping, watchdogs, and a hardware-specific bridge.

## Useful Commands

```bash
vla-zoo info scripted
vla-zoo predict --model scripted --instruction "pick up the red block"
```
