# Comparisons

| Project | Primary value | vla_zoo relationship |
|---|---|---|
| LeRobot | datasets, policies, training, hardware workflows | vla_zoo should run LeRobot/SmolVLA policies through ROS2 |
| OpenVLA | open VLA model and fine-tuning code | vla_zoo wraps it behind a stable runtime and ROS2 topics |
| openpi | pi-family model code and checkpoints | vla_zoo provides deployment and runtime boundary |
| Isaac GR00T | humanoid/generalist foundation model stack | vla_zoo provides experimental adapter and ROS2-facing interface |
| SimplerEnv | real-to-sim policy evaluation benchmark | vla_zoo can run adapters against it with unified metrics |
| LIBERO | manipulation benchmark suites | vla_zoo can expose LIBERO tasks through the same model API |
| Genesis | scalable simulation platform | vla_zoo treats Genesis as a benchmark/runtime backend |

`vla_zoo` is intentionally infrastructure-focused. It is the ROS2-native runtime boundary around model projects and benchmark projects, not a replacement for them.

## Runtime Comparison Workflow

Use the adapter table first. It does not load model weights:

```bash
vla-zoo doctor --no-ros
vla-zoo compare adapters
```

Then inspect method profiles. This is the lightweight way to compare integration shape
before running heavyweight policies:

```bash
vla-zoo compare methods
vla-zoo compare methods --markdown-out results/vla_method_profiles.md
```

The method profile output covers input requirements, action space, action shape, action
chunks, proprioception expectations, runtime support, dependency profile, and license caveats.
It is useful for deciding which models can be compared locally, which should be served
remotely, and which require robot-specific adapter work.

For issue reports, release notes, and README snippets, generate a complete artifact directory:

```bash
vla-zoo compare suite --out-dir results/vla_compare_suite
```

The suite writes method profiles, PyBullet result tables, a self-contained PyBullet HTML report,
an interactive dashboard, and an index `README.md`. Use `--no-pybullet` to generate only the
lightweight method profile artifacts.

Then run the same deterministic PyBullet smoke scene for baseline methods and runtime paths:

```bash
vla-zoo compare pybullet --models dummy,scripted,random,openvla,pi0,smolvla,groot
```

The local comparison skips heavy OpenVLA loading by default to avoid accidental downloads. The `dummy`, `scripted`, and `random` adapters are CPU smoke baselines for validating the runtime and metrics pipeline before comparing heavyweight VLA policies. For real model-to-model checks, run each VLA behind a remote GPU server and compare from the robot-side environment:

```bash
vla-zoo compare pybullet \
  --models openvla,pi0,smolvla,groot \
  --runtime remote \
  --remote-url http://gpu-box:8000
```

For separate model servers, use `--remote-map`:

```bash
vla-zoo compare pybullet \
  --models openvla,pi0,smolvla,groot \
  --runtime remote \
  --remote-map "openvla=http://gpu-box:8001,pi0=http://gpu-box:8002,smolvla=http://gpu-box:8003,groot=http://gpu-box:8004" \
  --markdown-out results/vla_runtime_comparison.md \
  --html-out results/vla_runtime_comparison.html
```

For repeatable comparisons, prefer a manifest:

```bash
vla-zoo compare pybullet --manifest examples/compare/pybullet_vla_remote.json
```

The manifest records the instruction, query cadence, render stride, each model endpoint, and output files. Keep real comparison outputs under `results/` or attach them to an issue; do not commit fabricated model numbers.

The HTML output is self-contained, so it can be attached to issues, release notes, or static docs without a server.

For a richer browser view with filters and charts, build a dashboard from result files:

```bash
vla-zoo compare dashboard \
  --results results/vla_runtime_comparison.json \
  --out results/vla_runtime_dashboard.html
```

Multiple JSON files can be passed as a comma-separated list.

The same dashboard can summarize ROS2 runtime status and diagnostics logs written as JSONL:

```bash
vla-zoo ros smoke-report --output-dir results
```

This runs the ROS2 smoke recording launch, writes status/diagnostics JSONL, and
creates `dashboard.html` plus `report_bundle.zip`.

The comparison output is intentionally runtime-centric, with extra scripted-scene telemetry for the PyBullet smoke task: frames, adapter query count, adapter errors, latency, action magnitude, cube lift, final cube distance to the goal, cube travel distance, grasp-attached frames, and phase completion. The smoke task uses a 15 cm placement zone for success. Treat these as deployment-path checks, not as model-quality claims.
