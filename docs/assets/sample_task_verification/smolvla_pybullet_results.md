## PyBullet VLA Runtime Comparison

| Task | Instruction | Model | Runtime | Endpoint | OK | Frames | Queries | Errors | Scene | Lifted | Goal dist m | Cube moved m | Phase | Mean latency ms | Mean abs action | Note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `pick_red_block` | pick up the red block | `smolvla` | `local` | - | true | 10 | 3 | 0 | true | true | 0.012 | 0.371 | 1.00 | 1024.77 | 0.360 | - |

This is a runtime smoke comparison on the same deterministic PyBullet scene. It measures adapter availability, query behavior, errors, latency, action magnitude, and scripted-scene task telemetry; it is not a model-quality benchmark.
