# Benchmark Result Schema and ROS Bag Replay

vla_zoo benchmarks emit a **versioned, machine-readable JSONL result schema** so that
latency and action-rate summaries are reproducible and reviewable. This is runtime
credibility plumbing, not a model-quality claim: a row records what the runtime did,
and `success` is `null` whenever the source cannot honestly assert task success.

See also [benchmark design](benchmark_design.md) for the broader benchmark roadmap.

## Result schema (`vla-zoo-benchmark/v1`)

Each line of a results JSONL file is one `EpisodeRecord`:

| Field | Meaning |
|---|---|
| `schema_version` | Always `vla-zoo-benchmark/v1`; readers reject other versions. |
| `model` | Adapter/model name. |
| `source` | `smoke-benchmark`, `ros2-action-replay`, etc. |
| `index` | Sample index within the run. |
| `task_id` | Task identifier, if any. |
| `success` | `true`/`false` for a real task, or `null` when no success claim is made. |
| `latency_ms` | Per-sample action latency, or `null`. |
| `num_actions` | Number of actions in the sample (chunk length for chunked outputs). |
| `error` | Error string if the sample raised, else `null`. |
| `note` | Optional free-text note. |

A latency / action-rate summary (`BenchmarkSummary`) is computed from the records:
sample count, success rate (when present), latency p50/p95/mean, action rate (Hz), and
exception count.

## Run the smoke benchmark with schema output

```bash
vla-zoo bench --model dummy --episodes 5 \
  --jsonl-out out/smoke_results.jsonl \
  --summary-md out/smoke_summary.md \
  --summary-out out/smoke_summary.json
```

## ROS bag replay (stub)

The replay path is a deliberately scoped **stub**: it replays vla_zoo's own recorded
JSONL action logs (`vla_actions.jsonl`) for latency/action-rate analysis. Native
rosbag2 (`.db3`/`.mcap`) decoding is **not yet implemented** and is gated on the ROS2
stack, so the module stays importable without ROS2 installed. A replay makes **no
task-success claim** — `success` is recorded as `null`.

```bash
vla-zoo bench-replay \
  --action-log docs/assets/sample_ros2_remote_dummy/vla_actions.jsonl \
  --jsonl-out docs/assets/sample_benchmark/ros2_replay_results.jsonl \
  --summary-md docs/assets/sample_benchmark/ros2_replay_summary.md \
  --summary-out docs/assets/sample_benchmark/ros2_replay_summary.json
```

The checked-in sample
[ROS2 action replay summary](assets/sample_benchmark/ros2_replay_summary.md) is
generated from the dummy remote ROS2 smoke log. It reports latency and a ~2.5 Hz action
rate for the recorded `dummy` stream; it is a runtime-throughput measurement, not a
robot-skill result.

## Comparison report (HTML / Markdown)

`bench-report` renders one or more summary JSON files into a comparison table that sits
alongside the evidence matrix and artifact index on the Pages surface:

```bash
vla-zoo bench-report \
  --summaries docs/assets/sample_benchmark/ros2_replay_summary.json \
  --html-out docs/assets/sample_benchmark/benchmark_report.html \
  --markdown-out docs/assets/sample_benchmark/benchmark_report.md
```

See the generated
[benchmark comparison report](assets/sample_benchmark/benchmark_report.html). Pass several
`--summaries` to compare models/sources side by side; a blank success rate means that
source made no task-success claim.

## What is still stubbed

- Native rosbag2 (`.db3`/`.mcap`) decoding (gated on ROS2).
- LIBERO / SimplerEnv smoke runners (gated on their dependencies).

These stay explicit stubs until their dependencies and a recorded run exist.
