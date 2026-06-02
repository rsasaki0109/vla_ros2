# VLA Runtime Leaderboard

- Schema: `vla-zoo-leaderboard/v1`
- Ranked by: `latency_ms_p50` (lower is better)
- Adapters: 4

| Rank | Model | Status | Latency p50 (ms) | Latency p95 (ms) | Action rate (Hz) | Memory | Evidence |
|---|---|---|---|---|---|---|---|
| 🥇 1 | smolvla | verified | 381.9 | 571.6 | 1.07 | 0.97 GB | [probe](../sample_pybullet_smolvla/runtime_action_probe.md) |
| 🥈 2 | openvla | verified | 1996.8 | 3675.9 | 1.07 | 4.60 GB | [probe](../sample_pybullet_openvla/runtime_action_probe.md) |
| — | pi0 | blocked | - | - | - | 8.90 GB | [probe](../sample_task_verification/pi0_compatibility_probe.md) |
| — | groot | blocked | - | - | - | - | [probe](../sample_task_verification/groot_block_probe.md) |

Runtime leaderboard. Rank is by the selected latency / action-rate metric, measured on recorded probes — NOT by robot task-success or policy quality (policy_quality stays not_verified for every model). Memory is a measured runtime footprint. Blocked rows are adapters whose local runtime path is currently gated, shown for completeness rather than omitted; they carry no fabricated numbers.

