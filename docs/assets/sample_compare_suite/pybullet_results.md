## PyBullet VLA Runtime Comparison

| Model | Runtime | Endpoint | OK | Frames | Queries | Errors | Task | Lifted | Goal dist m | Cube moved m | Phase | Mean latency ms | Mean abs action | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dummy` | `local` | - | true | 10 | 1 | 0 | true | true | 0.136 | 0.471 | 1.00 | 0.04 | 0.000 | - |
| `scripted` | `local` | - | true | 10 | 1 | 0 | true | true | 0.136 | 0.471 | 1.00 | 0.17 | 0.375 | - |
| `random` | `local` | - | true | 10 | 1 | 0 | true | true | 0.136 | 0.471 | 1.00 | 0.09 | 0.117 | - |

This is a runtime smoke comparison on the same deterministic PyBullet scene. It measures adapter availability, query behavior, errors, latency, action magnitude, and scripted-scene task telemetry; it is not a model-quality benchmark.
