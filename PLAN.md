# vla_ros2 — Plan and Handoff

Updated: 2026-06-08 JST

This document is the working handoff for `vla_ros2`. Read it before making
changes. It captures the mission, what the codebase looks like today, the
invariants to preserve, how to build/verify, and where to go next.

---

## 1. Mission and position

`vla_ros2` is a **ROS2-native, on-robot runtime for Vision-Language-Action (VLA)
models**. It does exactly one job: take a camera image, a natural-language
instruction, and robot state, run a VLA adapter **locally on the robot**, and
publish a typed action on a ROS2 topic.

```text
camera + instruction + robot state + timestamp
  -> stable VLA adapter boundary (local inference)
  -> typed VLAAction / VLAActionChunk
  -> ROS2 topics (/vla/action, /vla/action_chunk, /vla/status) + /diagnostics
```

The design is deliberately narrow. It is a runtime/integration layer, not a
benchmark suite, not a model zoo, not a research harness.

---

## 2. Pivot history (vla_zoo -> vla_ros2)

This repository began as `vla_zoo`, a broad "runtime + benchmark + adapter hub".
In June 2026 it was refocused into `vla_ros2` and the GitHub repository was
renamed `rsasaki0109/vla_zoo` -> `rsasaki0109/vla_ros2`.

Delivered across two merged PRs:

- **PR #7** — rename + prune. Renamed the package `vla_zoo` -> `vla_ros2`
  (ROS2 python module `vla_ros2_ros`, messages `vla_ros2_msgs`); dropped
  benchmark/compare/demo/docs and the remote path; replaced the CLI with a
  minimal one; updated CI.
- **PR #8** — README simulator hero GIF + metadata scrub. Added
  `scripts/record_sim_demo.py` (PyBullet pick-and-place driven by the runtime)
  and fixed leftover remote/PyBullet references in adapter metadata and the pi0
  error message.

### What was removed in the pivot
- `src/.../benchmark/`, `compare/`, `demo/`, `docs/` Python packages, and all
  generated report artifacts (GIF/HTML/leaderboard/roofline, `examples/`,
  `results/`, `log/`, `build/`, `install/`).
- The **remote-GPU HTTP path**: `runtime/remote.py`, `server.py`, `schemas.py`,
  `health.py`, the HTTP client, and `core/image.py` (only used by remote/server),
  plus the corresponding registry / node / config / launch plumbing. Inference
  is now **on-robot local only**.
- Unused core modules (`core/action.py`, `observation.py`, `config.py`).
- The large multi-command Typer CLI (~4.7k lines), replaced by a minimal
  `vla-ros2 list / predict`.
- Tests covering the removed surface; `scripts/` that depended on demo/remote.

### Intentional residual references
README / CHANGELOG / this file mention `vla_zoo` only to explain the migration.
That is expected; there should be no live code path or import named `vla_zoo`.

---

## 3. Repository layout

```text
src/vla_ros2/                    Python package (the adapter runtime)
  __init__.py                    public API: load_model, list_models, get_adapter_info, types
  core/
    types.py                     VLAObservation, VLAAction, VLAActionChunk, ActionSpec, ActionSpace
    model.py                     BaseVLA / VLAAdapter base classes (predict / predict_observation)
    registry.py                  built-in adapter table + entry-point discovery; load_model (local only)
    errors.py                    VLARos2Error hierarchy (UnknownModelError, MissingDependencyError, ...)
    __init__.py                  re-exports the public core surface
  adapters/
    dummy.py                     always-available zero-action adapter
    baselines.py                 RandomAdapter + ScriptedAdapter (phase-aware eef deltas)
    openvla.py                   OpenVLA HF adapter (needs [openvla])
    smolvla.py                   LeRobot SmolVLA adapter (needs [smolvla])
    pi0.py                       pi0/openpi adapter; local disabled unless enable_local=True
    groot.py                     experimental Isaac GR00T placeholder (no inference)
  runtime/
    local.py                     LocalVLARuntime (in-process inference)
    guard.py                     ActionClipGuard + watchdog (evaluate_watchdog, WatchdogConfig/Status)
    diagnostics.py               RuntimeDiagnostics record schema (vla-ros2-diagnostics/v1)
  cli/
    main.py                      minimal off-robot CLI: `vla-ros2 list` / `predict`

ros2/vla_ros2/                   ROS2 ament_python package
  vla_ros2_ros/
    node.py                      VLARuntimeNode (the runtime node)
    params.py                    RuntimeNodeParams dataclass
    converters.py                ROS <-> core type conversion (numpy, no cv_bridge)
    qos.py                       QoS profiles for image/instruction/action/status
    smoke_input.py               vla_smoke_input_node: publishes synthetic image/instruction/state
    log_recorder.py              vla_runtime_recorder: records action/status/diagnostics to JSONL
    action_replay.py             vla_action_replay_node: replays recorded actions
  launch/                        action_replay, dummy, log_recorder, openvla, smoke, smoke_record
  config/                        dummy.yaml, openvla.yaml (node parameter files)
  package.xml, setup.py, setup.cfg

ros2/vla_ros2_msgs/              ROS2 messages (ament_cmake)
  msg/VLAAction.msg, VLAActionChunk.msg, VLAInstruction.msg, VLAStatus.msg
  package.xml, CMakeLists.txt

scripts/
  record_sim_demo.py             PyBullet pick-and-place GIF, driven via the runtime (README hero)
  measure_openvla_runtime.py     local OpenVLA latency capture (needs GPU + [openvla])
  measure_lerobot_runtime.py     local LeRobot/SmolVLA latency capture

tests/                           10 stdlib/pytest tests (no GPU, no ROS2 install required)
docs/assets/sim_demo.gif         README hero GIF
README.md, PLAN.md, CHANGELOG.md, CITATION.cff, pyproject.toml
.github/workflows/ci.yml         install [dev], ruff (src tests ros2), mypy, pytest
```

---

## 4. Python API and core contracts

Public surface (`from vla_ros2 import ...`): `load_model`, `list_models`,
`get_adapter_info`, `BaseVLA`, and the typed objects.

- `VLAObservation(instruction, images={}, state={}, timestamp=None, metadata={})`
- `VLAAction(data: NDArray[float32], spec: ActionSpec, dt=None, confidence=None,
  chunk_index=None, metadata={})`
- `VLAActionChunk(actions: Sequence[VLAAction], metadata={})`
- `ActionSpec(action_space, shape, names=(), frame_id=None, control_hz=None,
  normalized=False, low=None, high=None, description="")`

`load_model(name, *, runtime="local", config=None, **kwargs) -> BaseVLA`.
**Only `runtime="local"` is supported**; anything else raises
`ConfigurationError`. Adapters are resolved from the built-in table in
`core/registry.py` and from the `vla_ros2.adapters` entry-point group declared in
`pyproject.toml` (keep both in sync — a mismatched group name silently loads no
adapters). `BaseVLA.predict(image=None, instruction=None, *, observation=None,
**kwargs)` is the convenience entry; adapters implement `predict_observation`.

### Adapters

| name     | loads without extras | extra            | notes |
|----------|----------------------|------------------|-------|
| dummy    | yes                  | —                | zero 7-DoF eef_delta; CI/dry-run baseline |
| random   | yes                  | —                | seeded random eef_delta |
| scripted | yes                  | —                | phase-aware eef_delta; drives the sim demo |
| openvla  | no                   | `[openvla]`      | OpenVLA-7b; needs torch+transformers+GPU |
| smolvla  | no                   | `[smolvla]`      | LeRobot SmolVLA; local CUDA path probed |
| pi0      | no                   | `[openpi]`       | local disabled by default; `enable_local=True` |
| groot    | no                   | —                | experimental placeholder; no inference ships |

---

## 5. Runtime and safety

- `runtime/local.py` — `LocalVLARuntime` wraps an adapter for in-process
  inference (loads the model, runs `predict_observation`).
- `runtime/guard.py` — `ActionClipGuard` bounds every action component to
  `[action_low, action_high]`; `evaluate_watchdog(...)` flags missing/stale
  image and instruction inputs and produces a `WatchdogStatus`.
- `runtime/diagnostics.py` — `RuntimeDiagnostics` is the single source of truth
  for the `/diagnostics` payload (latency, clip rate, watchdog, dropped frames).
  Schema version string: `vla-ros2-diagnostics/v1`.

**Safety invariants:**
- Launch files default `dry_run: true`. A metadata test asserts every launch file
  that declares `dry_run` defaults it to `"true"` — do not regress this.
- Actions are only published in dry-run when `publish_actions_in_dry_run: true`.
- The clip guard and watchdog are always in the loop in the node.

---

## 6. ROS2 integration

### Node — `vla_runtime_node` (`VLARuntimeNode`)
Loads an adapter (`load_model(model_name, runtime="local", ...)`), subscribes to
image/instruction/joint-state, runs inference on a single-worker thread pool at
`control_hz`, and publishes actions + status + diagnostics.

`RuntimeNodeParams` (declared with defaults in `node._declare_parameters`):
`model_name, runtime, dry_run, instruction_msg_type, image_topic,
instruction_topic, joint_state_topic, action_topic, action_chunk_topic,
status_topic, diagnostics_topic, publish_action_chunk, publish_diagnostics,
publish_actions_in_dry_run, control_hz, max_queue_size, require_image,
stale_image_timeout_sec, stale_instruction_timeout_sec, clip_actions,
action_low, action_high, device, pretrained, unnorm_key`.
(`runtime` is effectively always `local`; there is no `remote_url` anymore — if
you reintroduce a non-local runtime you must re-add its plumbing here AND in
`registry.load_model`.)

### Topics
| Dir | Topic (default)      | Type |
|-----|----------------------|------|
| in  | `/camera/image_raw`  | `sensor_msgs/Image` |
| in  | `/vla/instruction`   | `std_msgs/String` or `vla_ros2_msgs/VLAInstruction` |
| in  | `/joint_states`      | `sensor_msgs/JointState` |
| out | `/vla/action`        | `vla_ros2_msgs/VLAAction` |
| out | `/vla/action_chunk`  | `vla_ros2_msgs/VLAActionChunk` |
| out | `/vla/status`        | `vla_ros2_msgs/VLAStatus` |
| out | `/diagnostics`       | `diagnostic_msgs/DiagnosticArray` |

### Messages (`vla_ros2_msgs`)
- `VLAAction`: header, model_name, adapter_name, action_space, control_mode,
  frame_id, dt, data[], names[], confidence, chunk_index, metadata_json.
- `VLAActionChunk`: header, model_name, adapter_name, action_space, action_dt,
  actions[], metadata_json.
- `VLAInstruction`: header, text, task_id, metadata_json.
- `VLAStatus`: header, model_name, adapter_name, ready, dry_run, last_latency_ms,
  avg_latency_ms, action_rate_hz, dropped_frames, status_text, metadata_json.

### Node executables (`setup.py` console_scripts)
`vla_runtime_node`, `vla_smoke_input_node`, `vla_runtime_recorder`,
`vla_action_replay_node`.

### Launch files
`dummy.launch.py` (no GPU), `openvla.launch.py` (GPU + `[openvla]`),
`smoke.launch.py` (runtime + synthetic input, self-contained),
`smoke_record.launch.py` (smoke + JSONL recorder),
`log_recorder.launch.py`, `action_replay.launch.py`.

---

## 7. CLI

`vla-ros2` is a small off-robot sanity tool only (the real entry point is the
ROS2 node):
- `vla-ros2 list` — table of registered adapters.
- `vla-ros2 predict --model dummy [--instruction "..."]` — one local inference;
  a wiring check (no image supplied), not a task run.

---

## 8. Simulator demo (README hero GIF)

`scripts/record_sim_demo.py` renders `docs/assets/sim_demo.gif`: a Franka Panda
performing pick-and-place in a **real PyBullet physics sim**. Every control tick
builds a `VLAObservation` and calls `load_model("scripted")`; the returned
`VLAAction` end-effector delta + gripper channel command the arm via IK. A fixed
constraint models the grasp; the scene props are placed where the scripted
policy's integrated trajectory lands, so the runtime's own action stream produces
a coherent pick-and-place.

Honesty note: it is the **`scripted` baseline** (not a learned VLA), but the loop,
the physics, and the action stream are real. Needs `[sim]` (pybullet) + Pillow.
Reproduce: `.venv/bin/python scripts/record_sim_demo.py`.

---

## 9. Build, run, verify

### Python package
```bash
pip install -e ".[dev]"          # core + test/lint
ruff check src tests ros2
mypy src/vla_ros2
pytest                           # 10 files; no GPU, no ROS2 needed
```
Optional extras: `[openvla]`, `[smolvla]`, `[openpi]`, `[gpu]`, `[sim]`.

### ROS2 (this machine: ROS2 Jazzy at /opt/ros/jazzy)
```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vla_ros2_msgs vla_ros2
source install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"   # so the node imports our pip package
ros2 launch vla_ros2 dummy.launch.py        # or smoke.launch.py for a full graph
```

**Environment notes / gotchas**
- System `python3` (3.12) already has `rclpy` + `numpy`, `pydantic`, `pillow`,
  `typer`, `rich` — enough to run the node. Set `PYTHONPATH=$PWD/src` because the
  ament_python package installs only `vla_ros2_ros`, not our `vla_ros2` pip pkg.
- `converters.py` uses numpy directly; **`cv_bridge` is not required**.
- Do NOT `set -u` in shell scripts that `source` ROS2 setup files (they reference
  unbound vars and will abort).
- `build/`, `install/`, `log/`, `results/` are gitignored (colcon outputs).
- A 16 GB VRAM GPU + `torch` (CUDA) is available for openvla/smolvla;
  `.venv-smolvla` has LeRobot installed. (Never name the specific GPU model.)

### CI (`.github/workflows/ci.yml`)
Python 3.10/3.11/3.12 matrix: `pip install -e ".[dev]"`,
`ruff check src tests ros2`, `mypy src/vla_ros2`, `pytest`.
`tests/test_ros2_package_metadata.py` validates the `ros2/` package shape
(package.xml, setup.py entry points, messages, launch files) without a ROS2
install, so CI guards the integration on every push.

---

## 10. Conventions / invariants

- Package name `vla_ros2`, ROS2 module `vla_ros2_ros`, messages `vla_ros2_msgs`.
- `load_model` supports `runtime="local"` only.
- Every launch file that exposes `dry_run` defaults it to `"true"`.
- Keep the built-in adapter table and the `pyproject` `vla_ros2.adapters`
  entry-point group consistent.
- No `vla_zoo` live imports; the name appears only in migration prose.
- Git: commit under the author's own name (no Co-Authored-By); PR descriptions
  carry no AI-generated attribution.

---

## 11. Known limitations / debt
- The sim hero GIF is driven by the `scripted` baseline, not a learned VLA. A
  genuine closed-loop with smolvla/openvla is possible but the action space of
  the public checkpoints is not aligned to a PyBullet Panda, so a coherent task
  is not guaranteed without the policy's training env.
- `openvla` / `smolvla` / `pi0` are not exercised end-to-end in CI (no GPU);
  only their load/guard/metadata paths are tested.
- No live ROS2 node test (the metadata test parses files; it does not spin rclpy).
- Some adapter `metadata` strings are descriptive only and may drift.

## 12. Possible next steps
- A real-robot bring-up / hardware integration guide under `ros2/` (URDF, camera
  topic remaps, controller wiring, `action_low/high` calibration).
- A `launch_testing` smoke test that actually spins `vla_runtime_node` + the
  synthetic input node and asserts a `VLAAction` is published.
- Optional Gazebo (gz sim) integration: spawn an arm, bridge `/vla/action` to a
  joint controller, for a ROS2-native sim path alongside the PyBullet script.
- A closed-loop smolvla sim demo using a LeRobot-aligned environment (so the
  learned policy actually performs the task), if a "real VLA" GIF is wanted.
- Package the messages + node for a binary/colcon release and document a
  `rosdep`/workspace setup for users without this machine's environment.
