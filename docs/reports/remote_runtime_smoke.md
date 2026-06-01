# Remote Runtime Smoke Report

This report validates recorded PyBullet Action Playground traces. It checks
that the trace contains expected model/task records, enough frames, referenced
GIF assets, adapter query counts, and adapter error summaries.

It is a runtime-path report, not a claim of real robot task success.

## Summary

| Field | Value |
|---|---|
| Overall | ok |
| Records | 3/3 ok |
| Models | dummy |
| Tasks | move_red_block_left, move_red_block_right, pick_red_block |
| Trace files | docs/assets/action_playground_remote_dummy.json |

## Task Matrix

| Task | `dummy` |
|---|---|
| pick_red_block | ok (remote) |
| move_red_block_left | ok (remote) |
| move_red_block_right | ok (remote) |

## Record Details

| Model | Runtime | Task | Status | Frames | GIF | Queries | Errors | Goal m | Latency ms |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| `dummy` | remote | pick_red_block | ok | 60 | yes | 8 | 0 | 0.012 | 87.10 |
| `dummy` | remote | move_red_block_left | ok | 60 | yes | 8 | 0 | 0.128 | 81.66 |
| `dummy` | remote | move_red_block_right | ok | 60 | yes | 8 | 0 | 0.002 | 79.02 |

## Scope

- `ok` means the runtime trace record was present and internally consistent.
- `missing` means no trace record was provided for that model/task pair.
- Heavy VLA adapters such as OpenVLA, pi0, SmolVLA, and GR00T require explicit
  local GPU or remote-server traces before they should be described as verified.
- This report does not validate hardware actuation, policy quality, safety bridges,
  checkpoint licenses, or zero-shot robot performance.
