# Real-Scene Action Probe Runtime Comparison

- Schema: `vla-zoo-benchmark/v1`

| Model | Source | Samples | Success rate | Latency p50 (ms) | Latency p95 (ms) | Latency mean (ms) | Action rate (Hz) | Exceptions |
|---|---|---|---|---|---|---|---|---|
| smolvla | pybullet-action-probe | 21 | - | 381.92 | 571.64 | 436.77 | 1.07 | 0 |
| openvla | pybullet-action-probe | 21 | - | 1996.76 | 3675.92 | 2185.37 | 1.07 | 0 |

Runtime-centric benchmark comparison. It measures latency and action throughput, not robot task-success quality. A blank success rate means the source made no task-success claim (for example, replayed action logs).

