# dummy Adapter Card

| Field | Value |
|---|---|
| Source | built-in |
| Target | `vla_zoo.adapters.dummy:DummyAdapter` |
| Upstream project | vla_zoo |
| Default checkpoint | `none` |
| Aliases | - |
| Status | available |
| Experimental | false |
| Domain | - |
| Install hint | `-` |

## Runtime Contract

| Field | Value |
|---|---|
| Family | dry-run baseline |
| Role | CI/runtime smoke sanity check |
| Action space | `eef_delta` |
| Action shape | `(7,)` |
| Output | zero 7-DoF end-effector delta |
| Control Hz | 5 |
| Action chunks | optional via chunk_size |
| Proprioception | not required |
| Local runtime | supported |
| Remote runtime | supported |
| Dependencies | base install |
| License caveat | none |

## Inputs

- image optional
- instruction optional
- state optional

## Verification Status

Exercised by unit tests, CLI predict, ROS2 dry-run launch, and PyBullet smoke reports.

## Operational Caveats

- This card describes the vla_zoo runtime contract, not task success on a robot.
- External model weights, datasets, and upstream licenses are not redistributed by vla_zoo.
- Real robot deployment still needs calibrated cameras, state mapping, action clipping, watchdogs, and a hardware-specific bridge.

## Useful Commands

```bash
vla-zoo info dummy
vla-zoo predict --model dummy --instruction "pick up the red block"
```
