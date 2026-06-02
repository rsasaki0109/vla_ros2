# ROS2 Remote Runtime Smoke Check (SmolVLA)

This report validates the ROS2 remote runtime path from recorded JSONL logs:
`vla_runtime_node` status, diagnostics, and published `VLAAction` messages.

It proves transport and runtime wiring, not VLA policy quality or hardware safety.

## Summary

| Field | Value |
|---|---:|
| overall | ok |
| status_count | 106 |
| action_count | 14 |
| diagnostics_count | 106 |
| ready_count | 14 |
| dry_run_count | 106 |
| remote_status_count | 106 |
| remote_action_count | 14 |
| remote_diagnostics_count | 106 |
| inference_error_count | 0 |
| diagnostic_error_count | 0 |
| mean_latency_ms | 914.28 |
| max_latency_ms | 5193.75 |

## Runtime Evidence

| Field | Value |
|---|---|
| expected_model | `smolvla` |
| expected_remote_url | `http://127.0.0.1:8013` |
| models_seen | `smolvla` |
| adapters_seen | `RemoteVLAClient` |
| action_spaces | `custom` |

## Scope

- `remote_status_count` means status metadata reported `runtime=remote` and the expected URL.
- `remote_action_count` means recorded actions came from `RemoteVLAClient`.
- `remote_diagnostics_count` means diagnostics reported the remote runtime and URL.
- `dry_run=true` keeps this path on the typed-action publication boundary.
- This does not command hardware and does not validate model task success.
