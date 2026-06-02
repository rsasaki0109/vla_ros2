# ROS2 Action Replay Latency / Action-Rate Summary

- Schema: `vla-zoo-benchmark/v1`
- Source: `ros2-action-replay`
- Model: `dummy`

| Metric | Value |
|---|---|
| Samples | 69 |
| Success rate | - |
| Latency p50 | 0.02 ms |
| Latency p95 | 0.03 ms |
| Latency mean | 0.02 ms |
| Action rate | 2.50 Hz |
| Exceptions | 0 |

ROS bag replay stub: replays recorded vla_zoo JSONL action logs (vla_actions.jsonl) for latency/action-rate analysis. Native rosbag2 (.db3/.mcap) decoding is not yet implemented and is gated on the ROS2 stack. No task-success claim is made.

