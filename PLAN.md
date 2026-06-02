# vla_zoo Plan and Claude Handoff

Updated: 2026-06-03 JST

This file is the handoff document for continuing `vla_zoo` development with another
agent. Read this before making changes.

## 0. Current Position

`vla_zoo` is a ROS2-native runtime, benchmark, and adapter hub for
Vision-Language-Action models. The project is intentionally runtime-first:

```text
camera + instruction + robot state + timestamp
  -> stable VLA adapter/runtime boundary
  -> typed VLAAction or VLAActionChunk
  -> Python API, ROS2 topic, HTTP response, benchmark step, or report artifact
```

The repository is no longer just an MVP skeleton. It now has:

- Stable Python API: `load_model()`, `list_models()`.
- Typed core runtime objects: `VLAObservation`, `VLAAction`, `VLAActionChunk`,
  `ActionSpec`.
- Built-in adapters: `dummy`, `scripted`, `random`, `openvla`, `pi0`,
  `smolvla`, `groot`.
- Local runtime and remote HTTP server/client.
- ROS2 packages, launch files, messages, runtime node, smoke/evidence tooling.
- PyBullet simulation demos and deterministic baseline comparison artifacts.
- Static report artifacts for README, GitHub Pages, action playground, GIF gallery,
  runtime dashboard, ROS2 remote smoke, and VLA evidence matrix.

Latest pushed development before this handoff file:

```text
af1d8ad improve runtime dashboard evidence view
40cb955 add gif gallery comparison matrix
cf1ce17 surface vla evidence links
434099c add html vla evidence matrix
bfdc016 add vla evidence matrix
bcc01b3 add ros2 remote smoke evidence
0789940 add remote action playground smoke
1b72d6f add action playground verification report
99c2a70 add action playground recorder
296e107 add action playground trace merge
d004701 enhance action playground comparison
0fbc05e add action playground demo
```

## 1. Product North Star

The project should feel closer to Nav2 / Autoware / robot_localization
infrastructure than to a notebook repo.

The target positioning:

> The missing ROS2-native runtime boundary for the post-OpenVLA VLA ecosystem.

What to optimize for:

- Boring stable runtime interface.
- Adapter contract clarity.
- ROS2-first deployment shape.
- Remote GPU inference path.
- Honest evidence and reproducible artifacts.
- Simulation and benchmark scaffolding without overclaiming model quality.

What not to optimize for yet:

- Training.
- RL.
- New manipulation algorithms.
- Direct robot actuation.
- Huge model downloads in CI.
- Claims that every VLA works on every robot.

## 2. Non-Negotiable Safety and Messaging Rules

Do not claim:

- "OpenVLA/pi0/SmolVLA/GR00T are all benchmarked successfully."
- "vla_zoo proves real robot task success."
- "zero-shot deployment is safe."
- "the VLA output should directly drive hardware."

Always say:

- The checked-in artifacts are runtime evidence unless explicitly marked otherwise.
- Dummy/scripted/random are baselines, not VLA model-quality measurements.
- OpenVLA has a lazy adapter and prompt path; the local CUDA prompt probe was blocked
  by free VRAM in the checked run.
- SmolVLA has a checked GPU inference-path probe with `lerobot/smolvla_base`, but this
  is not a robot task-success claim.
- pi0/openpi is remote-first and still needs a recorded real action probe.
- GR00T is experimental placeholder status unless a real serving adapter is added.
- ROS2 launch must remain dry-run safe by default.

For hardware:

- Publish typed action messages.
- Do not directly command motors in core.
- Hardware bridges belong outside core or behind explicit opt-in.
- Use watchdogs, stale action timeout, clipping, emergency stop integration, and a
  low-rate VLA outer loop plus high-rate deterministic controller.

## 3. Repository Map

Key source modules:

```text
src/vla_zoo/core/
  types.py              Typed observation/action/action spec dataclasses.
  model.py              BaseVLA predict wrapper.
  registry.py           Adapter registry, aliases, metadata, entry points.

src/vla_zoo/adapters/
  dummy.py              Always-working neutral baseline.
  baselines.py          scripted/random simulation baselines.
  openvla.py            Lazy Hugging Face OpenVLA adapter.
  pi0.py                Remote-first openpi/pi0 placeholder/scaffold.
  smolvla.py            Lazy LeRobot SmolVLA adapter.
  groot.py              Experimental GR00T placeholder.

src/vla_zoo/runtime/
  server.py             FastAPI serving app.
  remote.py             Remote client with BaseVLA-compatible predict().
  schemas.py            Request/response schema helpers.
  dashboard.py          Static runtime dashboard renderer.
  ros_smoke.py          ROS2 remote smoke evidence validation.
  ros_plan.py           ROS2 remote smoke plan generation.
  server_plan.py        Heavy VLA server plan generation.

src/vla_zoo/demo/
  pybullet.py           PyBullet simulation renderer/comparison.
  gif_suite.py          GIF suite generation, QA, gallery renderer.
  action_playground.py  Action trace/playground static artifacts.

src/vla_zoo/compare/
  evidence.py           VLA evidence matrix JSON/Markdown/HTML renderer.
  profiles.py           Adapter method profile comparison.
  cards.py              Adapter card generation.
  compatibility.py      Robot profile compatibility report.
  suite.py              Compare-suite artifact README.

src/vla_zoo/benchmark/
  base.py               BenchmarkEnv and BenchmarkRunner contracts.

ros2/
  vla_zoo/              ROS2 node, launch files, config.
  vla_zoo_msgs/         ROS2 message definitions.

tests/
  pytest suite for registry, adapters, CLI, schemas, demos, dashboard, ROS smoke.
```

Important generated/static artifact areas:

```text
docs/assets/vla_model_evidence_matrix.html
docs/assets/gif_suite/index.html
docs/assets/sample_compare_suite/runtime_dashboard.html
docs/assets/sample_compare_suite/pybullet_report.html
docs/assets/sample_ros2_remote_dummy/remote_smoke_check.md
docs/assets/sample_ros2_remote_dummy/dashboard.html
docs/assets/action_playground.html
docs/assets/action_playground_with_remote.html
docs/reports/model_comparison.md
docs/reports/remote_runtime_smoke.md
docs/adapters/*.md
```

## 4. Current Public Story

README front page currently emphasizes:

- Open First links:
  - VLA evidence matrix.
  - PyBullet GIF gallery.
  - PyBullet report.
  - ROS2 remote dummy evidence.
- Representative checked-in PyBullet GIFs.
- Runtime-centric verification status.
- Quickstart with dummy.
- SmolVLA GPU path.
- OpenVLA GPU/server path.
- ROS2 runtime and remote smoke path.
- Comparing VLA runtime paths.
- Known limitations and safety.

GitHub Pages currently has:

- Hero with evidence matrix and GIF gallery buttons.
- What Works Now section.
- Run commands.
- GPU VLA path.
- ROS2/remote deployment.
- Visible reports list.
- Links to evidence matrix, GIF gallery, dashboard, adapter cards, robot compatibility,
  ROS2 evidence, and PyBullet results.

## 5. Verified Artifact Status

Treat this as the current truth table.

| Area | Status | Main artifacts |
|---|---|---|
| Dummy Python API | verified | tests, CLI predict |
| Dummy remote HTTP | verified | `docs/reports/remote_runtime_smoke.md` |
| ROS2 local dummy | implemented | ROS2 launch/config/node/messages |
| ROS2 remote dummy | verified | `docs/assets/sample_ros2_remote_dummy/remote_smoke_check.md` |
| PyBullet baseline tasks | verified | `docs/assets/gif_suite/`, `sample_task_verification/baseline_tasks.*` |
| Runtime dashboard | verified as static report | `sample_compare_suite/runtime_dashboard.html` |
| VLA evidence matrix | generated and linked | `docs/assets/vla_model_evidence_matrix.html` |
| OpenVLA adapter | scaffolded/lazy | `docs/adapters/openvla.md`, prompt probe artifact |
| OpenVLA 7B local run | blocked | insufficient free VRAM in checked probe |
| pi0/openpi | remote-first placeholder | compatibility note, adapter card |
| SmolVLA | GPU inference-path verified | `smolvla_gpu_probe.md/json` |
| GR00T | experimental placeholder | adapter card / evidence matrix |

## 6. Commands That Should Keep Working

Use `rtk proxy` when running shell commands in this environment.

Install:

```bash
rtk proxy pip install -e ".[dev,cli,server,sim]"
```

Core checks:

```bash
rtk proxy env PYTHONPATH=src pytest -q
rtk proxy env PYTHONPATH=src ruff check src/vla_zoo tests ros2/vla_zoo/vla_zoo_ros
rtk proxy env PYTHONPATH=src mypy src/vla_zoo
rtk proxy git diff --check
```

CLI smoke:

```bash
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main list
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main predict --model dummy --instruction "hello"
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main doctor --no-ros
```

Regenerate VLA evidence matrix:

```bash
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main compare evidence \
  --models dummy,scripted,random,openvla,pi0,smolvla,groot \
  --out docs/assets/vla_model_evidence_matrix.json \
  --markdown-out docs/assets/vla_model_evidence_matrix.md \
  --html-out docs/assets/vla_model_evidence_matrix.html
```

Regenerate GIF gallery from existing GIF manifest:

```bash
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main demo gif-report \
  --manifest docs/assets/gif_suite/gif_manifest.json \
  --html-out docs/assets/gif_suite/index.html \
  --check-json-out docs/assets/gif_suite/gif_check.json
```

Validate GIF suite:

```bash
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main demo gif-check \
  docs/assets/gif_suite \
  --link-files README.md,docs/index.html,docs/assets/gif_suite/README.md \
  --strict
```

Regenerate runtime dashboard:

```bash
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main compare dashboard \
  --results docs/assets/sample_compare_suite/pybullet_results.json \
  --out docs/assets/sample_compare_suite/runtime_dashboard.html \
  --title "vla_zoo Comparison Suite"
```

Run compare suite:

```bash
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main compare suite \
  --out-dir results/vla_compare_suite
```

Run PyBullet task verification:

```bash
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main compare tasks \
  --models dummy,scripted,random \
  --tasks all \
  --out results/vla_task_verification/baseline_tasks.json \
  --markdown-out results/vla_task_verification/baseline_tasks.md \
  --html-out results/vla_task_verification/baseline_tasks.html
```

Remote dummy smoke:

```bash
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main demo action-playground-remote-smoke
```

ROS2 remote smoke evidence check:

```bash
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main ros remote-smoke-check \
  --status-log docs/assets/sample_ros2_remote_dummy/vla_status.jsonl \
  --diagnostics-log docs/assets/sample_ros2_remote_dummy/vla_diagnostics.jsonl \
  --action-log docs/assets/sample_ros2_remote_dummy/vla_actions.jsonl
```

## 7. Immediate Next Work

Recommended next sequence for Claude.

### 7.1 Add a Docs/Pages Link Verifier (DONE)

Status: implemented in `src/vla_zoo/docs/links.py` + CLI `vla-zoo report link-check`.
Parses Markdown/HTML links, skips external/anchors, resolves repo-relative and
`/`-rooted paths, emits a human table plus `--out` JSON, and supports `--strict`.
Verified over README/Pages: 79 local links OK, 0 broken. Tests in
`tests/test_docs_links.py` and a CLI help test in `tests/test_cli.py`.

Original spec for reference:

- New module, likely `src/vla_zoo/docs/links.py` or `src/vla_zoo/runtime/links.py`.
- CLI command, for example:

```bash
vla-zoo docs link-check README.md docs/index.html docs/assets/gif_suite/index.html
```

or, if avoiding a new Typer app:

```bash
vla-zoo report link-check --paths README.md,docs/index.html
```

Requirements:

- Parse Markdown links and HTML `href/src`.
- Ignore external `http(s)` by default.
- Resolve repo-relative local paths.
- Verify `.html`, `.md`, `.json`, `.gif`, `.png`, `.zip`, `.jsonl` links exist.
- Produce human table plus optional JSON output.
- Add tests with temporary markdown/html files.
- Use it in future before every README/Pages edit.

Acceptance:

```bash
rtk proxy env PYTHONPATH=src pytest -q tests/test_docs_links.py tests/test_cli.py
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main report link-check \
  --paths README.md,docs/index.html \
  --strict
```

### 7.2 Add a README/Pages Artifact Index (DONE)

Status: implemented in `src/vla_zoo/docs/artifact_index.py` + CLI `vla-zoo report index`.
Curated 16-entry catalog (title/path/category/status/kind/source command/caveat) across
all six categories, verifies on-disk existence vs `--root`, emits `--out` JSON and
`--html-out` HTML (hrefs relative to the output dir) plus a status table, and supports
`--strict`. Generated `docs/assets/artifact_index.json` + `.html`, linked from
`docs/index.html`, and link-checked clean. Tests in `tests/test_docs_index.py`.

Original spec for reference:

- `docs/assets/artifact_index.json`
- `docs/assets/artifact_index.html`
- CLI:

```bash
vla-zoo report index --out docs/assets/artifact_index.json --html-out docs/assets/artifact_index.html
```

Include:

- title
- path
- category
- status
- source command
- whether generated/checked/manual
- evidence caveat

Categories:

- "model evidence"
- "simulation"
- "runtime dashboard"
- "ROS2"
- "adapter docs"
- "GPU probes"

### 7.3 Make ROS2 Integration More Credible (metadata tests DONE)

Status: `tests/test_ros2_package_metadata.py` added (stdlib-only, no ROS2 install).
It parses and asserts: both `package.xml` files (name/format/build_type/deps/groups),
`setup.py` console_scripts + their module files + launch/config data_files,
`CMakeLists.txt` registers every `.msg`, `.msg` field contracts, every
`*.launch.py` parses and exposes `generate_launch_description`, the smoke launch
declares expected topic args and Node executables, and the safety invariant that any
launch `dry_run` arg defaults to `"true"`. A GitHub Actions ROS build job is still
optional/documented-only and not enabled.

Remaining/next useful tasks:

- Add a non-ROS syntax/lint test for launch files and package metadata.
- Add `tests/test_ros2_package_metadata.py`.
- Parse/check:
  - `ros2/vla_zoo/package.xml`
  - `ros2/vla_zoo/setup.py`
  - `ros2/vla_zoo/launch/*.launch.py`
  - `ros2/vla_zoo_msgs/msg/*.msg`
- Assert expected topics and launch args are present.
- Optionally add a GitHub Actions placeholder job that is commented/documented, not enabled.

Do not require a ROS2 installation in standard CI yet.

### 7.4 Strengthen SmolVLA Remote Path (plan + isolation docs DONE; local GPU run VERIFIED)

Local-runtime update (verified): `lerobot/smolvla_base` loads and predicts a 6-DoF action
through the public adapter on a local GPU. Measured: ~0.97 GB peak VRAM,
~60-133 ms steady latency (real-time capable; ~800 ms first-run warmup), ~23 s warm-cache
load. LeRobot deps are isolated in a `--system-site-packages` venv (`/tmp/lerobot_venv`).
Reproducible via `scripts/measure_lerobot_runtime.py --model smolvla` (artifact
`docs/assets/smolvla_local_runtime.json`, page `docs/smolvla_local_runtime.md`); the
`local_runtime`/`gpu_inference` cells carry measured numbers. Runtime-path claim on a
synthetic frame, not task success; `policy_quality` stays `not_verified`.

Remote serving (verified): a real `vla-zoo serve --model smolvla` FastAPI server passed a
health-first probe and returned a typed 6-DoF action over HTTP `/v1/predict`, recorded
end-to-end (`docs/assets/sample_task_verification/smolvla_remote_probe.{md,json}`). This is
the first real-model (non-dummy) remote `/v1/predict` recording in the repo, so the SmolVLA
`remote_server` cell is now `verified`.

ROS2 remote (verified): the real `VLARuntimeNode` was driven in `runtime=remote` mode
against the live SmolVLA server, recording 14 `RemoteVLAClient` actions + 106
status/diagnostics with 0 inference errors (`vla-zoo ros remote-smoke-check` passed,
`docs/assets/sample_ros2_remote_smolvla/`). The SmolVLA `ros2_remote` cell is now
`verified`. Note: this host's loopback has no `MULTICAST` flag, so cross-process DDS
discovery (the 3-process `smoke_record.launch.py`) does not work; `smoke_record.launch.py`
gained `model_name`/`runtime`/`remote_url` args for multicast-capable hosts, and
`scripts/record_ros2_remote_trace.py` runs the same real node/input/recorder in one process
to record the identical node -> RemoteVLAClient -> server path without that dependency.



Status: `vla-zoo smolvla-remote-plan` generates a reproducible isolated-env bring-up
plan (`src/vla_zoo/runtime/smolvla_plan.py`): venv create, `pip install -e ".[cli,server,smolvla]"`,
server command, `/health` check, robot-side `runtime=remote` probe
(`examples/python/smolvla_remote_client.py`), `compare pybullet --runtime remote`, and a
ROS2 remote smoke report command. `docs/smolvla_remote.md` documents why the `smolvla`
extra needs a dedicated venv (its `transformers`/`torch` pins clash with `openvla`).
Generated artifacts: `docs/assets/smolvla_remote_smoke_plan.{md,json}` and
`examples/serve/smolvla_remote_smoke_plan.{md,json}`. The evidence matrix `remote_server`
cell for SmolVLA stays `planned` (no recorded `/v1/predict` yet) but now links the plan
and isolation docs. Tests in `tests/test_smolvla_remote_plan.py`.

Reason: SmolVLA is the most feasible real model path already checked locally. The next
credible step is remote serving and robot-side client evidence.

Remaining/next useful tasks:

- Record an actual SmolVLA `/v1/predict` response from a real server and check it in.
- Promote the `remote_server` cell from `planned` to `partial`/`verified` only then.

Do not claim policy-quality success.

### 7.5 OpenVLA 7B Path (health-first remote probe DONE; local 4-bit GPU run VERIFIED)

Local-runtime update (verified): OpenVLA-7b now loads and predicts a 7-DoF action through
the public adapter on a local GPU. bf16 weights (~15 GB) do not fit a 16 GB
card, so the adapter gained `load_in_4bit` / `load_in_8bit` (bitsandbytes nf4 +
`device_map`, skipping the post-load `.to()`). Measured: ~4.6 GB peak VRAM, ~1.1-2.7 s
latency, ~20 s warm-cache load. OpenVLA's `trust_remote_code` needs `timm<1.0`, isolated in
a `--system-site-packages` venv. Evidence is reproducible via
`scripts/measure_openvla_runtime.py` (artifact `docs/assets/openvla_local_runtime.json`,
page `docs/openvla_local_runtime.md`); the `local_runtime` and `gpu_inference` cells are now
`verified`. This is a runtime-path claim on a synthetic frame, not task success;
`policy_quality` stays `not_verified`.



Status: `vla-zoo remote-probe` (`src/vla_zoo/runtime/remote_probe.py`) checks a server's
`/health` first and only then records one `/v1/predict` response, with three statuses
(`ok`/`unreachable`/`predict_failed`) and `--out`/`--markdown-out`/`--strict`. The Python
entry point is `examples/python/openvla_remote_probe.py`. `docs/openvla_remote.md`
documents the remote GPU bring-up and probe. The tool is verified end-to-end against the
in-repo `dummy` server (`docs/assets/sample_task_verification/remote_probe_dummy.{md,json}`)
without any model download. The OpenVLA `remote_server` evidence cell stays `planned`
(no real recorded OpenVLA `/v1/predict` yet) and now links the remote path docs and the
tooling sample. Tests in `tests/test_remote_probe.py` (injected fakes + a real
unreachable-server path; no downloads).

Reason: OpenVLA is important for credibility. Local 7B inference was previously blocked by
free VRAM; that is now resolved with 4-bit loading and verified with measured numbers. A
remote GPU server is still the recommended deployment path for the heavy stack.

Remote serving (verified): the `serve` command now exposes `--load-in-4bit` (threaded
through `_model_load_kwargs` -> `run_server` -> the adapter), so OpenVLA-7b fits a 16 GB
card on the server side too. A real `vla-zoo serve --model openvla --load-in-4bit` server
passed a health-first probe and returned a typed 7-DoF action over HTTP `/v1/predict`,
recorded end-to-end (`docs/assets/sample_task_verification/openvla_remote_probe.{md,json}`).
The OpenVLA `remote_server` cell is now `verified`.

ROS2 remote (verified): the real `VLARuntimeNode` was driven in `runtime=remote` mode
against the live OpenVLA-7b (4-bit) server, recording 7 `RemoteVLAClient` actions + 143
status/diagnostics with 0 inference errors (`vla-zoo ros remote-smoke-check` passed,
`docs/assets/sample_ros2_remote_openvla/`, via `scripts/record_ros2_remote_trace.py`). The
OpenVLA `ros2_remote` cell is now `verified` — OpenVLA is verified across contract / local /
gpu / remote / ros2.

Remaining/next useful tasks:

- Add a task-level probe (real scene frame) before any policy-quality claim.

### 7.6 pi0/openpi Path (remote-first docs + plan DONE)

Status: `examples/python/load_pi0_remote.py` (lightweight robot-side remote client),
a generated pi0 server plan artifact (`docs/assets/pi0_server_plan.{md,json}` and
`examples/serve/pi0_server_plan.{md,json}` via `vla-zoo serve-plan --models pi0`), and
`docs/pi0_remote.md` documenting LeRobot/openpi version compatibility (config-decode
mismatches, checkpoint-specific action shapes, dedicated `.venv-pi0`) plus the
health-first remote-probe usage. The pi0 `remote_server` evidence cell stays `planned`
(no recorded pi0 `/v1/predict` from a version-matched server) and now links the remote
docs and server plan. Tests in `tests/test_pi0_remote.py`.

Reason: pi0/openpi should not be forced into base env. Treat it as remote-first.

Local-load status (re-confirmed 2026-06-03 with LeRobot 0.5.1): `load_model("pi0",
enable_local=True, pretrained="lerobot/pi0")` fails on a concrete config-schema mismatch —
the cached checkpoint config carries `PI0Config` fields (`resize_imgs_with_padding`,
`adapt_to_pi_aloha`, `num_steps`, `use_cache`, `attention_implementation`, ...) that
LeRobot 0.5.1's `PI0Config` rejects (`draccus.DecodingError`). The `local_runtime` cell now
states this specific reason instead of a vague "compatibility varies". A version-matched
checkpoint or serving environment is the unblock path. (SmolVLA, which shares the adapter
base, loads cleanly — so this is checkpoint/config-specific, not an adapter bug.)

Remaining/next useful tasks:

- If a version-matched pi0 server/checkpoint becomes available, record a real action probe.
- Promote the `remote_server` cell off `planned` only then.

### 7.7 GR00T Path (DONE)

Reason: GR00T should stay experimental until a real serving adapter exists.

Status: DONE. GR00T is now explicitly blocked until the NVIDIA Isaac GR00T stack is
wired in, with one shared `GROOT_BLOCKED_NOTE` reused by the adapter, the registry
verification text, and the evidence matrix. The adapter raises (no fabricated
inference), the expected observation/action contract is documented in the adapter
docstring and `docs/groot_remote.md`, and the GR00T runtime evidence cells
(`local_runtime`/`gpu_inference`/`remote_server`) are `blocked` while `contract`
stays `partial`. Acceptance passed: 173 tests, ruff/mypy clean, link-check 230 ok / 0
broken, git diff --check clean.

Work items (all done):

- Keep `experimental=True`. (done)
- Add a proper "blocked until NVIDIA stack" note. (done — `GROOT_BLOCKED_NOTE`)
- Do not write fake local inference. (done — raises instead)
- Add an adapter card section for expected observation/action contract once known.
  (done — adapter docstring + `docs/groot_remote.md`)

Remaining/next useful tasks:

- When a real GR00T serving adapter and a recorded action probe exist, move the
  `remote_server`/`gpu_inference` cells off `blocked`.

## 8. Medium-Term Roadmap

### v0.1 release polish

- Ensure root README is concise but persuasive.
- Add `PLAN.md`.
- Add artifact index.
- Add link checker.
- Add release checklist.
- Add "Known limitations" stays visible.
- Make sure all report links on Pages work.

### v0.2 adapter expansion

- SmolVLA remote server evidence.
- pi0 remote adapter smoke.
- OpenVLA remote GPU smoke.
- More action chunk support in scheduler.
- Better adapter metadata schema.

### v0.3 benchmark credibility

- ROS bag replay benchmark stub -> working loader. (DONE for JSONL action logs;
  native rosbag2 .db3/.mcap decoding is still future work, gated on ROS2.)
- LIBERO smoke runner when deps are installed. (DONE — dependency-gated honest runner
  in `benchmark/libero.py`; emits the schema, raises MissingDependencyError without deps.)
- SimplerEnv runner. (DONE — dependency-gated honest runner in `benchmark/simpler.py`.)
- JSONL benchmark result schema. (DONE — `vla-zoo-benchmark/v1` in
  `benchmark/results.py`.)
- Latency and action-rate summary reports. (DONE — `BenchmarkSummary` +
  `vla-zoo bench --summary-md` / `vla-zoo bench-replay`.)
- Benchmark summaries on the report/Pages surface. (DONE — `vla-zoo bench-report`
  renders an HTML/Markdown comparison table; sample at
  `docs/assets/sample_benchmark/benchmark_report.html`.)

- LIBERO / SimplerEnv honest dependency-gated smoke runners. (DONE — `benchmark/libero.py`
  and `benchmark/simpler.py` declare their contract, gate on upstream deps, and emit the
  schema via an env-agnostic loop; `vla-zoo bench --benchmark libero|simpler`.)

Status: the versioned JSONL result schema, the latency/action-rate summary, the ROS bag
replay stub (over vla_zoo JSONL action logs), the benchmark comparison report on the Pages
surface, and the dependency-gated LIBERO / SimplerEnv smoke runners are all done.
Acceptance passed: 195 tests, ruff/mypy clean, link-check 240 ok / 0 broken, git diff
--check clean.

Remaining/next useful tasks:

- Native rosbag2 (.db3/.mcap) decoding behind the ROS2 stack.
- Wire the real LIBERO / SimplerEnv environment loops once their upstream stacks and a
  recorded run exist (the dependency-gated runners and the env-agnostic loop are ready).

### v0.4 robot readiness

- Lifecycle node design.
- Diagnostics improvements. (DONE — pure, versioned `RuntimeDiagnostics`
  (`vla-zoo-diagnostics/v1`) in `runtime/diagnostics.py` merges the `ActionClipGuard`
  counters, the `WatchdogStatus`, and latency into one schema with JSONL + Markdown +
  `to_key_values()` surfaces; the ROS2 node's `/diagnostics` payload is built from it.)
- Watchdog node/example. (DONE — pure `evaluate_watchdog` in `runtime/guard.py`; the ROS2
  node delegates to it.)
- Action clipping. (DONE — pure `clip_action_report` + `ActionClipGuard` with clip-rate
  reporting in `runtime/guard.py`; the ROS2 node delegates to it.)
- MoveIt Servo bridge example. (DONE — pure mapping in `runtime/servo_bridge.py`
  (`eef_delta`->TwistStamped, joint->JointJog) + dry-run-safe example
  `examples/ros2/moveit_servo_bridge.py` that runs the clip+watchdog guards.)
- ros2_control bridge example. (DONE — pure mapping in `runtime/control_bridge.py`
  (joint_position/velocity -> JointTrajectory point) + dry-run-safe example
  `examples/ros2/ros2_control_bridge.py` that runs the clip+watchdog guards.)
- Jetson + remote GPU deployment guide. (DONE — `docs/deployment.md` ties the remote
  serving path, ROS2 remote runtime, clip/watchdog guards, and bridge examples into one
  Jetson + remote-GPU topology.)

### v0.5 ecosystem hub

- External adapter plugin docs.
- Adapter registry page.
- Model cards.
- Third-party adapter CI template.
- Community benchmark contribution guide.

## 9. Suggested Issues to Create

Good first issues:

- Add docs link checker.
- Add artifact index page.
- Add ROS2 package metadata tests.
- Add MoveIt Servo action bridge example.
- Add ros2_control bridge example.
- Add Jetson remote GPU deployment guide.
- Add SO-101 / ALOHA launch example.
- Add ROS bag replay loader stub.
- Add LIBERO smoke benchmark stub.
- Add SimplerEnv smoke benchmark stub.

Adapter issues:

- Record SmolVLA remote server smoke.
- Record OpenVLA remote server smoke.
- Add pi0/openpi remote example.
- Add GR00T serving adapter design doc.

Report/UI issues:

- Add artifact index.
- Add dashboard screenshot preview.
- Add Pages status badges from generated JSON.
- Add action trace bundle viewer.

## 10. GitHub / Publishing Notes

Current repo remote:

```text
git@github.com:rsasaki0109/vla_zoo.git
```

The current workflow has been direct pushes to `main`. If continuing that style:

```bash
rtk proxy git status -sb
rtk proxy git add <explicit files>
rtk proxy git commit -m "<short message>"
rtk proxy git push origin main
```

Do not stage unrelated user changes. Always inspect `git status -sb` first.

If using PR workflow later:

- Branch: `codex/<description>`
- PR title: `[codex] <description>`
- Include validation commands in PR body.

## 11. Implementation Style Rules

Follow the existing project style:

- Keep base dependencies light.
- Heavy model deps go behind extras.
- Lazy import optional dependencies.
- Raise helpful missing-dependency errors.
- Do not download model weights in tests.
- Prefer structured dataclasses/schema helpers.
- Public API returns `VLAAction` or `VLAActionChunk`, not raw arrays.
- Report renderers should be deterministic and testable.
- Generated docs/assets should be reproducible from CLI commands.
- Tests should run on CPU.

Shell/tooling conventions in this environment:

- Prefix shell commands with `rtk proxy`.
- Use `rg` for search.
- Use `apply_patch` for file edits.
- Do not run destructive git commands.

## 12. Known Risks

| Risk | Impact | Mitigation |
|---|---|---|
| README overclaims model performance | Expert criticism | Keep evidence matrix and limitations explicit |
| Static links break | Bad first impression | Add link checker next |
| GIFs look like policy claims | Misleading | Keep "runtime artifact, not robot skill" text |
| OpenVLA local run blocked | Credibility issue | Move to remote GPU evidence path |
| ROS2 not in CI | Integration risk | Add metadata/syntax tests first, full ROS CI later |
| Heavy deps conflict with ROS2 env | Deployment pain | Keep remote server path and separate env guidance |
| Direct actuation temptation | Safety risk | Core publishes actions only |

## 13. Current Best Next Commit

Section 7 (adapter hardening), the v0.3 benchmark-credibility track, and the v0.4
robot-readiness track are done: the pure clip + watchdog guards (`runtime/guard.py`), both
hardware-bridge examples (`runtime/servo_bridge.py`, `runtime/control_bridge.py` + dry-run
examples), the Jetson + remote-GPU deployment guide (`docs/deployment.md`), and the pure,
versioned `RuntimeDiagnostics` schema (`runtime/diagnostics.py`, `vla-zoo-diagnostics/v1`)
that the ROS2 node's `/diagnostics` payload is built from are all in. The `diag-report`
CLI command that surfaces the diagnostics schema is now also done (DONE):

```text
add a vla-zoo diag-report CLI command for the runtime diagnostics schema (v0.4)  [DONE]
```

What landed: `vla-zoo diag-report` mirrors `bench-report`. It reads either a native
`vla-zoo-diagnostics/v1` JSONL (`--log`) or a recorded ROS2 `/diagnostics` DiagnosticArray
JSONL (`--from-ros-log`, selecting `--status-name`) and renders the latest record's Markdown
snapshot to stdout / `--markdown-out`, with `--jsonl-out` to persist the reconstructed native
log and `--json` for machine output. The `--from-ros-log` path uses a new pure inverse,
`diagnostics_from_key_values` (correctly parses string bools — `bool("False")` is `True`, so
`from_dict` cannot be used for KeyValue payloads). Real snapshots reconstructed from the
SmolVLA and OpenVLA-7b ROS2 runs are checked in at
`docs/assets/sample_ros2_remote_{smolvla,openvla}/runtime_diagnostics_snapshot.md` (+ native
JSONL), with artifact-index entries. Unit-tested with written fixtures (no ROS2/model deps);
runtime-path claims only.

The diag-report time-series summary is also done (DONE):

```text
render a diag-report time-series summary over a full diagnostics JSONL (v0.4)  [DONE]
```

What landed: `diag-report --summary` aggregates the whole log via the pure
`summarize_diagnostics` → `DiagnosticsSummary` (latency min/p50/max over each record's
`last_latency_ms`, max dropped_frames, final actions, peak action/element clip rate, and the
worst — highest-`level`, latest-on-ties — record + its status_text/index), rendered by
`format_diagnostics_summary_markdown`; the single-snapshot path stays the default. This
surfaces transients the latest-snapshot hides: both real runs end `ok` but the summary
reports the startup `warn`. Real summaries are checked in at
`docs/assets/sample_ros2_remote_{smolvla,openvla}/runtime_diagnostics_summary.md` with
artifact-index entries (SmolVLA p50 882 / max 5194 ms; OpenVLA p50 3563 / max 6273 ms).
Unit-tested with written fixtures; runtime-path claims only.

The diagnostics summary band in the HTML runtime dashboard is also done (DONE):

```text
show the diagnostics time-series summary in the runtime dashboard (v0.4)  [DONE]
```

What landed: `compare dashboard` / `report bundle` now render a server-side diagnostics
summary band above the toolbar. `load_diagnostics_summaries` reconstructs native
`vla-zoo-diagnostics/v1` records from a `--diagnostics-log` (handling both the flat native
JSONL and ROS2 DiagnosticArray KeyValue payloads), groups by model, and reduces each with
`summarize_diagnostics`; `_render_diagnostics_summary_band` emits one `scope-card` per model
whose severity follows the worst level (`ok-scope`/`warn`/`error`), surfacing the transient
the per-record rows bury. The band is omitted when no diagnostics log is present. A real
SmolVLA dashboard (106 records, worst level `warn`) is checked in at
`docs/assets/sample_ros2_remote_smolvla/dashboard.html` with an artifact-index entry.
Unit-tested with written fixtures; runtime-path claims only.

The diagnostics-surface loop is now closed — the recorder has a native emit path (DONE):

```text
have the ROS2 log recorder also write a native vla-zoo-diagnostics/v1 JSONL (v0.4)  [DONE]
```

What landed: `RuntimeLogRecorder` gained an optional native sink
(`record_native_diagnostics`, default on; `native_diagnostics_log_path`,
`native_diagnostics_status_name`). Its `/diagnostics` callback now also reconstructs native
records via the new pure `native_records_from_diagnostics_payload` (in `runtime/diagnostics.py`,
no ROS dep) and writes them to a `vla-zoo-diagnostics/v1` JSONL, so one recording emits both
the raw ROS dump and the native log — removing the `diag-report --from-ros-log` re-derivation
step. The pure transform is unit-tested with written fixtures; it is byte-identical to the
checked-in native logs (verified against both real runs). The recorder wiring was also
exercised live (same-process rclpy pub→record smoke: 3 published DiagnosticArrays → 3 native
records). Runtime-path claims only.

The real-scene task-level probe is now done — verified cells no longer run only on noise (DONE):

```text
record a real-scene action probe on PyBullet frames (no policy-quality claim) (v0.4)  [DONE]
```

What landed: `vla-zoo demo action-probe` drives an adapter through the existing PyBullet
pick-and-place rollout and records, for every fresh query, the *full* action vector the
adapter produced from a genuinely rendered camera frame. `run_simulation` gained an optional
`prediction_sink` (and `predict_adapter_action` now also returns the raw prediction) so the
full action is captured with no second model call; the new pure `demo/action_probe.py`
(`build_probe_record` → `summarize_probe_records` → `format_action_probe_summary_markdown`)
writes a canonical `vla_actions.jsonl` log that replays through `bench-replay` (`success=None`)
plus a runtime-evidence summary, all stamped `policy_quality=not_verified` / `real_scene=true`.
A real SmolVLA run (21 queries, action dim 6, latency p50 ~382 ms with `--return-action-chunk`
forcing a fresh encode per query) is checked in at
`docs/assets/sample_pybullet_smolvla/{smolvla_action_probe.jsonl,runtime_action_probe.md,json}`
with artifact-index entries and a `docs/smolvla_local_runtime.md` section. The pure pieces are
unit-tested (including a `load_action_log` round-trip); the heavy local adapter is gated behind
`--allow-local-heavy` so a plain `pytest` never downloads weights. Runtime-path claims only:
this upgrades the *input* from synthetic noise to a real render, nothing about skill.

The real-scene action probe is now surfaced where the honesty story is read — the VLA evidence
matrix (DONE):

```text
surface the real-scene action probe as a runtime-evidence cell in the VLA matrix (v0.4)  [DONE]
```

What landed: the SmolVLA `pybullet_tasks` cell in `compare/evidence.py` now distinguishes the
synthetic-frame runtime evidence (`local_runtime` / `gpu_inference`) from the recorded
real-scene probe (21 queries, 6-DoF, latency p50 ~382 ms), linking
`sample_pybullet_smolvla/runtime_action_probe.md`. The cell stays `partial` and
`policy_quality` stays `not_verified` with an explicit "upgrades the input from synthetic noise
to a real render, makes no task-success claim" note; `next_step` records that the remaining gap
is task-success / policy quality. The three checked-in matrix artifacts
(`docs/assets/vla_model_evidence_matrix.{json,md,html}`) were regenerated and the new link is
covered by `report link-check`. Existing evidence tests still pass (status assertions unchanged).

The real-scene runtime evidence now covers both heavy adapters — OpenVLA also has a probe (DONE):

```text
record an OpenVLA real-scene action probe on PyBullet frames (no policy-quality claim) (v0.4)  [DONE]
```

What landed: `vla-zoo demo action-probe` gained a generic `--adapter-kwarg key=value` option
(coerced to bool/int/float/str) so adapter-specific load args flow through without per-adapter
flags. A real OpenVLA-7b (4-bit) run — `--model openvla --allow-local-heavy --adapter-kwarg
load_in_4bit=true --adapter-kwarg unnorm_key=bridge_orig`, executed in the timm<1.0 openvla venv
(`/tmp/openvla_venv`, which has PyBullet) — recorded 21 queries (action dim 7, latency p50 ~2.0 s)
to `docs/assets/sample_pybullet_openvla/{openvla_action_probe.jsonl,runtime_action_probe.md,json}`
with artifact-index entries and a `docs/openvla_local_runtime.md` section. The OpenVLA
`pybullet_tasks` matrix cell flipped from `planned` to a `partial` recorded real-scene probe;
`policy_quality` stays `not_verified`. The three matrix artifacts were regenerated. Runtime-path
claims only.

The two real-scene probes are now directly comparable (DONE):

```text
render a real-scene runtime comparison report from the SmolVLA + OpenVLA action probes (v0.4)  [DONE]
```

What landed: `bench-replay` gained `--source` / `--note` overrides (default to the ROS
replay-stub values) so a non-ROS log is labeled honestly; `frames_to_records` takes a `source`
kwarg. Each probe log was replayed with `--source pybullet-action-probe` into a
`vla-zoo-benchmark/v1` summary (`success=None`), and `bench-report` rendered the side-by-side
comparison at `docs/assets/sample_pybullet_compare/runtime_probe_comparison.{html,md}`
(SmolVLA p50 ~382 ms vs OpenVLA-7b 4-bit p50 ~2.0 s, blank success rate). Artifact-index entries
were added and both runtime docs link the comparison. The `--source` override is unit-tested;
no task-success claim anywhere.

The real-scene action probes + comparison are now discoverable from the front page (DONE):

```text
surface the real-scene action probes + comparison on the GitHub Pages index (v0.4)  [DONE]
```

What landed: `docs/index.html` gained a "Real-scene action probes" tile in *What Works Now*
(pointing at the comparison) and three tiles in *Visible Reports* — the SmolVLA/OpenVLA probe
summaries and the side-by-side comparison — each keeping the "runtime path, not task success"
framing. All new links are covered by `report link-check` (39/39). Docs-only change; no code or
schema touched.

The pi0 local-load block was time-boxed and resolved into a precise, reproducible boundary (DONE):

```text
resolve the pi0 local-load block: version-matched checkpoint or a documented hard block (v0.4)  [DONE]
```

What landed: the block decomposed into two distinct findings. (1) The old `lerobot/pi0` config
schema is permanently rejected by LeRobot 0.5.1 (`draccus.DecodingError` on 8 named fields).
(2) `lerobot/pi0_base` is the **version-matched** checkpoint — it decodes cleanly (`PI0Config`,
32D action, `n_action_steps=50`), and its bf16 model fits a 16 GB GPU (~8.9 GB constructed),
verified by building the policy on the local 4070 Ti SUPER. The float32 config OOMs on 16 GB, so
the SmolVLA/pi0 adapter gained a `dtype` override (`--adapter-kwarg dtype=bfloat16`) that builds
the config with a pinned compute dtype. The pi0 adapter now defaults local loading to
`lerobot/pi0_base` (action spec `(32,)`), and `pi0` joined `HEAVY_LOCAL_MODELS` so the probe is
gated behind `--allow-local-heavy`. The *remaining* block is not a version/config/memory issue: pi0's
processor requires the **gated** `google/paligemma-3b-pt-224` tokenizer (`GatedRepoError 401`,
manual license acceptance + token). `pi0_compatibility_probe.md` was rewritten with the full
version matrix and reproduce commands; `pi0_remote.md`, `smolvla_local_runtime.md`, and the
regenerated evidence matrix (`local_runtime` cell + `next_step`) now carry this precise boundary.
The `local_runtime` cell stays `blocked` and `policy_quality` stays `not_verified`; no model
weights are downloaded in tests.

The pi0 license-gated boundary now fails loudly instead of silently (DONE):

```text
add a pi0 local-load preflight that fails loudly on missing weights / the gated PaliGemma tokenizer (v0.4)  [DONE]
```

What landed: `adapters/pi0.py` gained `run_pi0_local_preflight`, run before the heavy
`from_pretrained` on every local pi0 load. It probes two assets without ever downloading the
14 GB weights — `model.safetensors` via a metadata HEAD (cache hit short-circuits) and the
processor's tokenizer (read from `policy_preprocessor.json`'s `tokenizer_processor` step) via a
small-file resolve that trips `GatedRepoError` exactly as the real load would — and raises an
`AdapterError` with an actionable message: missing weights → "LeRobot silently returns a
randomly-initialized model, refusing"; gated tokenizer → "accept the license at
huggingface.co/<repo> and supply an HF token". The decision is a pure function
(`_pi0_local_load_error`) plus a pure extractor (`_pi0_tokenizer_repo`), both unit-tested without
heavy deps (5 new tests); the live path was verified in `.venv-smolvla` (raises the gated-tokenizer
`AdapterError` instead of loading random weights). `policy_quality` stays `not_verified`.

The 16 GB-fit knobs and the pi0 preflight are now documented for deployers (DONE):

```text
document the 16 GB-fit knobs (openvla 4-bit, smolvla/pi0 dtype=bfloat16) and the pi0 preflight in deployment.md (v0.4)  [DONE]
```

What landed: `deployment.md` gained a "Fitting heavy checkpoints on a 16 GB GPU" subsection under
the GPU-box serve section. A measured table records both knobs — OpenVLA-7b 4-bit (nf4) ~4.6 GB
via `serve --load-in-4bit` / `load_model(load_in_4bit=True)`, and the LeRobot `dtype="bfloat16"`
override (pi0_base ~8.9 GB, its float32 config OOMs) via `load_model(dtype=...)` /
`demo action-probe --adapter-kwarg dtype=bfloat16`, with the honest note that `serve` does not yet
expose a `dtype` flag. The pi0 preflight's loud-failure behavior (missing weights / gated
PaliGemma tokenizer) is summarized with a link to the compatibility probe. All flags were verified
against `--help` before documenting; the new internal links pass `report link-check --strict`.
Docs-only, runtime-centric, no policy-quality claim.

The "serve cannot pass dtype" follow-up turned out to be a doc error, not a code gap — corrected (DONE):

```text
correct the deployment.md dtype-serve note: vla-zoo serve --dtype already threads through (v0.4)  [DONE]
```

What landed: inspecting the code before writing the planned `serve --dtype` passthrough showed it
**already exists** — `serve` exposes `--dtype` (verified in `--help`) and threads it through
`_model_load_kwargs` → `run_server` → `create_app` → `load_model(runtime="local", dtype=...)` to the
LeRobot adapter (the live `load_model("pi0", dtype="bfloat16")` path already accepts it, raising the
gated-tokenizer `AdapterError` rather than a `TypeError`). The previous commit's deployment.md note
("serve does not expose a `dtype` flag yet") was therefore wrong; this commit corrects the table and
prose to show `vla-zoo serve --model pi0 --pretrained lerobot/pi0_base --dtype bfloat16` and
strengthens `test_model_load_kwargs_threads_quantization_flags` to lock the `dtype` threading. No
new feature; an honest correction backed by a test.

The dtype-serve path now has a recorded artifact, not just a unit test (DONE):

```text
record a remote probe against vla-zoo serve --model smolvla --dtype bfloat16 (dtype-serve evidence) (v0.4)  [DONE]
```

What landed: a real `vla-zoo serve --model smolvla --pretrained lerobot/smolvla_base --device cuda
--dtype bfloat16` server (run in `.venv-smolvla`; SmolVLA params confirmed `torch.bfloat16` in a
local load of the same config) passed a health-first `remote-probe --strict` and returned a typed
6-DoF action over HTTP. The recorded result is checked in at
`docs/assets/sample_task_verification/smolvla_dtype_serve_probe.{json,md}` with an artifact-index
entry (count 46 → 47), linked from `deployment.md` and `smolvla_local_runtime.md`. This upgrades the
`serve --dtype` claim from unit-tested wiring to a recorded runtime path. While setting it up,
verified that although `SmolVLAConfig` has no declared `dtype` field, the modeling reads the
attribute set by the override, so the params do load in bf16. `policy_quality` stays `not_verified`
(a served typed action is not a task-success claim).

With the OpenVLA/SmolVLA/pi0/dtype/serve evidence fully recorded, the remaining visible gap is
discoverability: the front-page `docs/index.html` surfaces the real-scene probes and comparisons but
not the new 16 GB-fit deployment story (the `deployment.md` knobs section + the bf16 dtype-serve
probe). The next best commit closes that:

```text
surface the 16 GB-fit deployment knobs and the bf16 dtype-serve probe on the GitHub Pages index (v0.4)
```

Reason: the established pattern is that recorded evidence is made discoverable from the front page
(as the real-scene probes were). A tile pointing at the `deployment.md` "Fitting heavy checkpoints
on a 16 GB GPU" section and/or the `smolvla_dtype_serve_probe.md` keeps the new, honest evidence one
click from the index. Docs-only, link-checked, runtime-centric; no code or schema touched.

Acceptance:

```bash
rtk proxy env PYTHONPATH=src pytest -q tests/test_evidence.py tests/test_cli.py
rtk proxy env PYTHONPATH=src ruff check src/vla_zoo tests
rtk proxy env PYTHONPATH=src mypy src/vla_zoo
rtk proxy env PYTHONPATH=src python3 -m vla_zoo.cli.main report link-check \
  --paths docs/index.html,docs/smolvla_local_runtime.md \
  --strict
```

## 14. One-Screen Claude Brief

Start here:

1. Run `rtk proxy git status -sb`.
2. Read `README.md`, `docs/index.html`, and this `PLAN.md`.
3. Do not redo finished evidence work.
4. Implement `docs/report link-check` next.
5. Keep all claims runtime-centric.
6. Run `pytest`, `ruff`, `mypy`, and `git diff --check`.
7. Commit only the files you changed.
8. Push to `origin main` only if the user asks to continue the direct-push style.

The repo is in a strong presentation state. The next quality jump is making the
visible report links and generated artifacts mechanically checked, then using that
foundation for more real-model remote evidence.
