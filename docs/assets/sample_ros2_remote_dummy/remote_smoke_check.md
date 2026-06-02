# vla_zoo ROS2 Remote Dummy Smoke: Remote Runtime Check

This report validates the ROS2 remote runtime path from recorded JSONL logs:
`vla_runtime_node` status, diagnostics, and published `VLAAction` messages.

It proves transport and runtime wiring, not VLA policy quality or hardware safety.

## Summary

| Field | Value |
|---|---:|
| overall | ok |
| status_count | 149 |
| action_count | 69 |
| diagnostics_count | 149 |
| ready_count | 69 |
| dry_run_count | 149 |
| remote_status_count | 149 |
| remote_action_count | 69 |
| remote_diagnostics_count | 149 |
| inference_error_count | 0 |
| diagnostic_error_count | 0 |
| mean_latency_ms | 67.54 |
| max_latency_ms | 159.16 |

## Runtime Evidence

| Field | Value |
|---|---|
| expected_model | `dummy` |
| expected_remote_url | `http://127.0.0.1:8766` |
| models_seen | `dummy` |
| adapters_seen | `RemoteVLAClient` |
| action_spaces | `eef_delta` |

## Scope

- `remote_status_count` means status metadata reported `runtime=remote` and the expected URL.
- `remote_action_count` means recorded actions came from `RemoteVLAClient`.
- `remote_diagnostics_count` means diagnostics reported the remote runtime and URL.
- `dry_run=true` keeps this path on the typed-action publication boundary.
- This does not command hardware and does not validate model task success.
