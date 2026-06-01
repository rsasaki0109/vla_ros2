## PyBullet VLA Runtime Comparison

| Task | Instruction | Model | Runtime | Endpoint | OK | Frames | Queries | Errors | Scene | Lifted | Goal dist m | Cube moved m | Phase | Mean latency ms | Mean abs action | Note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `pick_red_block` | pick up the red block | `dummy` | `local` | - | true | 10 | 1 | 0 | true | true | 0.012 | 0.371 | 1.00 | 0.02 | 0.000 | - |
| `pick_red_block` | pick up the red block | `scripted` | `local` | - | true | 10 | 1 | 0 | true | true | 0.012 | 0.371 | 1.00 | 0.09 | 0.375 | - |
| `pick_red_block` | pick up the red block | `random` | `local` | - | true | 10 | 1 | 0 | true | true | 0.012 | 0.371 | 1.00 | 0.07 | 0.117 | - |

This is a runtime smoke comparison on the same deterministic PyBullet scene. It measures adapter availability, query behavior, errors, latency, action magnitude, and scripted-scene task telemetry; it is not a model-quality benchmark.
