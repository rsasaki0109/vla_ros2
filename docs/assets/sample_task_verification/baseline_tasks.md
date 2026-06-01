## PyBullet Multi-Task VLA Runtime Verification

| Task | Instruction | Model | Runtime | Endpoint | OK | Frames | Queries | Errors | Scene | Lifted | Goal dist m | Cube moved m | Phase | Mean latency ms | Mean abs action | Note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `pick_red_block` | pick up the red block | `dummy` | `local` | - | true | 10 | 1 | 0 | true | true | 0.012 | 0.371 | 1.00 | 0.04 | 0.000 | - |
| `pick_red_block` | pick up the red block | `scripted` | `local` | - | true | 10 | 1 | 0 | true | true | 0.012 | 0.371 | 1.00 | 0.15 | 0.375 | - |
| `pick_red_block` | pick up the red block | `random` | `local` | - | true | 10 | 1 | 0 | true | true | 0.012 | 0.371 | 1.00 | 0.09 | 0.117 | - |
| `move_red_block_left` | move the red block to the left target zone | `dummy` | `local` | - | true | 10 | 1 | 0 | true | true | 0.128 | 0.270 | 1.00 | 0.02 | 0.000 | - |
| `move_red_block_left` | move the red block to the left target zone | `scripted` | `local` | - | true | 10 | 1 | 0 | true | true | 0.128 | 0.270 | 1.00 | 0.11 | 0.375 | - |
| `move_red_block_left` | move the red block to the left target zone | `random` | `local` | - | true | 10 | 1 | 0 | true | true | 0.128 | 0.270 | 1.00 | 0.06 | 0.117 | - |
| `move_red_block_right` | move the red block to the right target zone | `dummy` | `local` | - | true | 10 | 1 | 0 | true | true | 0.002 | 0.256 | 1.00 | 0.02 | 0.000 | - |
| `move_red_block_right` | move the red block to the right target zone | `scripted` | `local` | - | true | 10 | 1 | 0 | true | true | 0.002 | 0.256 | 1.00 | 0.10 | 0.375 | - |
| `move_red_block_right` | move the red block to the right target zone | `random` | `local` | - | true | 10 | 1 | 0 | true | true | 0.002 | 0.256 | 1.00 | 0.05 | 0.117 | - |

This is a runtime smoke comparison on the same deterministic PyBullet scene. It measures adapter availability, query behavior, errors, latency, action magnitude, and scripted-scene task telemetry; it is not a model-quality benchmark.
