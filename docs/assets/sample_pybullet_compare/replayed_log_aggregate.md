# Replayed Action-Log Aggregate (Ranked)

- Schema: `vla-zoo-benchmark-aggregate/v1`
- Ranked by: `latency_ms_p50` (lower is better)
- Entries: 2

| Rank | Model | Source | Samples | Success rate | Latency p50 (ms) | Latency p95 (ms) | Latency mean (ms) | Action rate (Hz) | Exceptions |
|---|---|---|---|---|---|---|---|---|---|
| 1 | smolvla | ros2-action-replay | 21 | - | 381.92 | 571.64 | 436.77 | 1.07 | 0 |
| 2 | openvla | ros2-action-replay | 21 | - | 1996.76 | 3675.92 | 2185.37 | 1.07 | 0 |

## Per-model roll-up

Best is the lowest `latency_ms_p50` across each model's runs.

| Model | Runs | Best | Median | Worst |
|---|---|---|---|---|
| smolvla | 1 | 381.92 | 381.92 | 381.92 |
| openvla | 1 | 1996.76 | 1996.76 | 1996.76 |

Runtime-centric aggregate. Rank is by the selected latency / action-rate metric, not by robot task-success quality. A blank success rate means the source made no task-success claim (for example, replayed action logs).

