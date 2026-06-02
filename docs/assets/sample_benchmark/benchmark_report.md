# vla_zoo Benchmark Comparison

- Schema: `vla-zoo-benchmark/v1`

| Model | Source | Samples | Success rate | Latency p50 (ms) | Latency p95 (ms) | Latency mean (ms) | Action rate (Hz) | Exceptions |
|---|---|---|---|---|---|---|---|---|
| dummy | ros2-action-replay | 69 | - | 0.02 | 0.03 | 0.02 | 2.50 | 0 |

Runtime-centric benchmark comparison. It measures latency and action throughput, not robot task-success quality. A blank success rate means the source made no task-success claim (for example, replayed action logs).

