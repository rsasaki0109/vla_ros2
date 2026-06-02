# Real-Scene Action Probe (Runtime Path)

- Model: `smolvla` (runtime `local`)
- Task: `pick_red_block` — "pick up the red block"
- Input: **pybullet-render** (real rendered scene frames, not synthetic noise)
- Adapter queries recorded: 21
- Action dim: 6

| Metric | Value |
| --- | --- |
| Latency min | 347.9 ms |
| Latency p50 | 381.9 ms |
| Latency max | 923.0 ms |
| Latency mean | 436.8 ms |
| Mean abs action | 0.4281 |
| Max abs action | 1.1269 |
| Sample action | 0.7276, 0.2253, 0.8765, 0.4632, 0.1706, 1.1082 |
| Policy quality | `not_verified` |

> Runtime evidence: a real adapter driven on real PyBullet-rendered scene frames, recording latency and action magnitude. It exercises the real image preprocessing path that synthetic-frame probes skip. It is NOT a task-success or policy-quality claim (policy_quality=not_verified): the action stream is not evidence that the robot task was completed.
