# Real-Scene Action Probe (Runtime Path)

- Model: `openvla` (runtime `local`)
- Task: `pick_red_block` — "pick up the red block"
- Input: **pybullet-render** (real rendered scene frames, not synthetic noise)
- Adapter queries recorded: 21
- Action dim: 7

| Metric | Value |
| --- | --- |
| Latency min | 1540.0 ms |
| Latency p50 | 1996.8 ms |
| Latency max | 3803.0 ms |
| Latency mean | 2185.4 ms |
| Mean abs action | 0.1218 |
| Max abs action | 0.9961 |
| Sample action | -0.0013, -0.0020, -0.0151, -0.0040, -0.0195, 0.0193, 0.9961 |
| Policy quality | `not_verified` |

> Runtime evidence: a real adapter driven on real PyBullet-rendered scene frames, recording latency and action magnitude. It exercises the real image preprocessing path that synthetic-frame probes skip. It is NOT a task-success or policy-quality claim (policy_quality=not_verified): the action stream is not evidence that the robot task was completed.
