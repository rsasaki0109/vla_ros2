# Action Playground Verification Report

This report validates recorded PyBullet Action Playground traces. It checks
that the trace contains expected model/task records, enough frames, referenced
GIF assets, adapter query counts, and adapter error summaries.

It is a runtime-path report, not a claim of real robot task success.

## Summary

| Field | Value |
|---|---|
| Overall | ok |
| Records | 9/9 ok |
| Models | dummy, random, scripted |
| Tasks | move_red_block_left, move_red_block_right, pick_red_block |
| Trace files | docs/assets/action_playground.json |

## Task Matrix

| Task | `dummy` | `scripted` | `random` |
|---|---|---|---|
| pick_red_block | ok (local) | ok (local) | ok (local) |
| move_red_block_left | ok (local) | ok (local) | ok (local) |
| move_red_block_right | ok (local) | ok (local) | ok (local) |

## Record Details

| Model | Runtime | Task | Status | Frames | GIF | Queries | Errors | Goal m | Latency ms |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| `dummy` | local | pick_red_block | ok | 60 | yes | 8 | 0 | 0.012 | 0.02 |
| `scripted` | local | pick_red_block | ok | 60 | yes | 8 | 0 | 0.012 | 0.10 |
| `random` | local | pick_red_block | ok | 60 | yes | 8 | 0 | 0.012 | 0.05 |
| `dummy` | local | move_red_block_left | ok | 60 | yes | 8 | 0 | 0.128 | 0.02 |
| `scripted` | local | move_red_block_left | ok | 60 | yes | 8 | 0 | 0.128 | 0.09 |
| `random` | local | move_red_block_left | ok | 60 | yes | 8 | 0 | 0.128 | 0.04 |
| `dummy` | local | move_red_block_right | ok | 60 | yes | 8 | 0 | 0.002 | 0.02 |
| `scripted` | local | move_red_block_right | ok | 60 | yes | 8 | 0 | 0.002 | 0.09 |
| `random` | local | move_red_block_right | ok | 60 | yes | 8 | 0 | 0.002 | 0.05 |

## Scope

- `ok` means the runtime trace record was present and internally consistent.
- `missing` means no trace record was provided for that model/task pair.
- Heavy VLA adapters such as OpenVLA, pi0, SmolVLA, and GR00T require explicit
  local GPU or remote-server traces before they should be described as verified.
- This report does not validate hardware actuation, policy quality, safety bridges,
  checkpoint licenses, or zero-shot robot performance.
