# random Adapter Card

| Field | Value |
|---|---|
| Source | built-in |
| Target | `vla_zoo.adapters.baselines:RandomAdapter` |
| Upstream project | vla_zoo |
| Default checkpoint | `none` |
| Aliases | random-baseline |
| Status | available |
| Experimental | false |
| Domain | - |
| Install hint | `-` |

## Runtime Contract

| Field | Value |
|---|---|
| Family | stochastic baseline |
| Role | action plumbing and visualization stress check |
| Action space | `eef_delta` |
| Action shape | `(7,)` |
| Output | seeded random 7-DoF end-effector delta |
| Control Hz | 5 |
| Action chunks | no |
| Proprioception | not required |
| Local runtime | supported |
| Remote runtime | supported |
| Dependencies | base install |
| License caveat | none |

## Inputs

- instruction optional
- image optional
- state optional

## Verification Status

Exercised in deterministic PyBullet comparison reports as a seeded baseline.

## Operational Caveats

- This card describes the vla_zoo runtime contract, not task success on a robot.
- External model weights, datasets, and upstream licenses are not redistributed by vla_zoo.
- Real robot deployment still needs calibrated cameras, state mapping, action clipping, watchdogs, and a hardware-specific bridge.

## Useful Commands

```bash
vla-zoo info random
vla-zoo predict --model random --instruction "pick up the red block"
```
