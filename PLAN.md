# vla_ros2 — Plan and Handoff

Updated: 2026-06-08 JST (integration wave: smoke test, bring-up, Gazebo, workspace, SmolVLA demo)

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
  sim/
    so100_kinematic.py           SO-100 kinematic stand-in for SmolVLA closed-loop demo

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
  config/                        dummy.yaml, openvla.yaml, robot.example.yaml
  tests/                         test_smoke_launch.py (launch_testing; needs ROS2 + colcon)
  package.xml, setup.py, setup.cfg

ros2/vla_ros2_gz/                Gazebo Sim integration (optional `gz` profile)
  vla_ros2_gz_ros/
    action_bridge.py             vla_action_bridge_node: /vla/action -> joint_trajectory_controller
  urdf/vla_arm.urdf.xacro         7-DoF arm + gz_ros2_control
  launch/                        gz_arm.launch.py, gz_smoke.launch.py
  config/                        vla_arm_controllers.yaml, gz_smoke.yaml

ros2/vla_ros2_msgs/              ROS2 messages (ament_cmake)
  msg/VLAAction.msg, VLAActionChunk.msg, VLAInstruction.msg, VLAStatus.msg
  package.xml, CMakeLists.txt

ros2/BRINGUP.md                  real-robot phased bring-up guide
ros2/SIM.md                      Gazebo Sim graph + action bridge
ros2/WORKSPACE.md                colcon / rosdep workspace bootstrap

scripts/
  bootstrap_ros2_workspace.sh    one-shot ROS2 workspace setup (core or gz profile)
  record_sim_demo.py             PyBullet pick-and-place GIF, driven via the runtime (README hero)
  record_smolvla_so100_demo.py   SmolVLA closed-loop on SO-100 kinematic stand-in
  measure_openvla_runtime.py     local OpenVLA latency capture (needs GPU + [openvla])
  measure_lerobot_runtime.py     local LeRobot/SmolVLA latency capture

tests/                           pytest (no GPU, no ROS2 install required for most tests)
docs/assets/sim_demo.gif         README hero GIF (scripted / PyBullet)
docs/assets/smolvla_so100_demo.gif  optional SmolVLA kinematic demo GIF
README.md, PLAN.md, CHANGELOG.md, CITATION.cff, pyproject.toml
.github/workflows/ci.yml         Python matrix + ros2-smoke (Jazzy launch test)
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

`vla_ros2_gz`: `gz_arm.launch.py` (Gazebo arm only), `gz_smoke.launch.py` (runtime +
smoke input + action bridge). See `ros2/SIM.md`.

---

## 7. CLI

`vla-ros2` is a small off-robot sanity tool only (the real entry point is the
ROS2 node):
- `vla-ros2 list` — table of registered adapters.
- `vla-ros2 predict --model dummy [--instruction "..."]` — one local inference;
  a wiring check (no image supplied), not a task run.

---

## 8. Simulator demos

### PyBullet hero GIF (`scripted` baseline)

`scripts/record_sim_demo.py` renders `docs/assets/sim_demo.gif`: a Franka Panda
performing pick-and-place in a **real PyBullet physics sim**. Every control tick
builds a `VLAObservation` and calls `load_model("scripted")`; the returned
`VLAAction` end-effector delta + gripper channel command the arm via IK.

Honesty note: it is the **`scripted` baseline** (not a learned VLA), but the loop,
the physics, and the action stream are real. Needs `[sim]` (pybullet) + Pillow.
Reproduce: `.venv/bin/python scripts/record_sim_demo.py`.

### SmolVLA closed-loop kinematic demo (`smolvla` adapter)

`scripts/record_smolvla_so100_demo.py` renders `docs/assets/smolvla_so100_demo.gif`:
a **real SmolVLA inference loop** (`lerobot/smolvla_base`) on a minimal SO-100-style
kinematic stand-in initialized from `lerobot/svla_so100_stacking`. Observations use
LeRobot-aligned keys (6D state, three 256×256 camera slots).

Honesty note: this is **not** the official SO-100 physics sim; `smolvla_base` is a
**base** checkpoint and task success is not guaranteed without fine-tuning. Needs
`[smolvla]`, a GPU, and a LeRobot dataset download on first run.
Reproduce: `.venv-smolvla/bin/python scripts/record_smolvla_so100_demo.py`.

---

## 9. Build, run, verify

### Python package
```bash
pip install -e ".[dev]"          # core + test/lint
ruff check src tests ros2
mypy src/vla_ros2
pytest                           # unit + metadata tests; no GPU for most
```
Optional extras: `[openvla]`, `[smolvla]`, `[openpi]`, `[gpu]`, `[sim]`.

### ROS2 workspace (Jazzy)

Quick path:

```bash
./scripts/bootstrap_ros2_workspace.sh              # core: msgs + vla_ros2
VLA_ROS2_PROFILE=gz ./scripts/bootstrap_ros2_workspace.sh   # + vla_ros2_gz
source install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ros2 launch vla_ros2 smoke.launch.py
```

Manual path and `rosdep` notes: `ros2/WORKSPACE.md`. Real-robot wiring:
`ros2/BRINGUP.md`. Gazebo graph: `ros2/SIM.md`.

Launch smoke test (local or CI):

```bash
colcon test --packages-select vla_ros2 --python-testing pytest --event-handlers console_direct+
colcon test-result --verbose
```

**Environment notes / gotchas**
- Use `colcon build --base-paths ros2` (not `--paths`).
- `rosdep install` must pass `--skip-keys "ament_python"` (provided by the ROS underlay).
- System `python3` (3.12) already has `rclpy` + common deps on this machine. Set
  `PYTHONPATH=$PWD/src` because the ament_python package installs only `vla_ros2_ros`,
  not our `vla_ros2` pip pkg (bootstrap runs `pip install -e .` as well).
- `converters.py` uses numpy directly; **`cv_bridge` is not required**.
- Do NOT `set -u` in shell scripts that `source` ROS2 setup files (they reference
  unbound vars and will abort).
- `build/`, `install/`, `log/`, `results/` are gitignored (colcon outputs).
- A CUDA GPU is available for openvla/smolvla; `.venv-smolvla` has LeRobot installed.

### CI (`.github/workflows/ci.yml`)
- **test** — Python 3.10/3.11/3.12: `pip install -e ".[dev]"`, ruff, mypy, pytest.
- **ros2-smoke** — Jazzy: colcon build + `launch_testing` smoke test on `vla_ros2`.
- `tests/test_ros2_package_metadata.py` guards ROS2 package shape without a ROS2 install.

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
- The README hero GIF is still the **`scripted` PyBullet** demo. The SmolVLA GIF uses
  a kinematic stand-in; task success is not guaranteed with `smolvla_base` alone.
- `openvla` / `smolvla` / `pi0` are not exercised end-to-end in CI (no GPU);
  only metadata, guard, and kinematic-unit paths are tested locally.
- Gazebo (`vla_ros2_gz`) is not in CI; validate with `./scripts/gz_smoke_validate.sh`
  (see `notes/2026-06-08_gazebo_closed_loop.md`).
- The Gazebo action bridge maps `eef_delta` to joint increments for sim convenience;
  real robots need their own Cartesian IK / controller bridge (`ros2/BRINGUP.md`).
- Some adapter `metadata` strings are descriptive only and may drift.

## 12. Possible next steps
- **Real-robot validation**: follow `ros2/BRINGUP.md` on hardware. Reference bridge:
  `vla_controller_bridge_node` + Phase C gate `./scripts/bringup_validate.sh c`
  (see `notes/2026-06-08_real_robot_bringup.md`).
- ~~**README polish**: surface `smolvla_so100_demo.gif` alongside the PyBullet hero.~~ Done.
- **SmolVLA fine-tune**: train on `lerobot/svla_so100_stacking` so the kinematic
  or real SO-100 demo actually completes the stacking task.
- **Bloom / rosdistro**: release `vla_ros2_msgs` and `vla_ros2` (see `WORKSPACE.md` §7).
- **Gazebo CI** (optional): nightly or self-hosted `./scripts/gz_smoke_validate.sh` (local gates done).
- **GPU adapter smoke** on a self-hosted runner (load + one `predict`, not task metrics).
