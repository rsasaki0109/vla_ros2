# vla_ros2

ROS2-native on-robot runtime for Vision-Language-Action (VLA) models.

[![CI](https://github.com/rsasaki0109/vla_ros2/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/vla_ros2/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-22314E)](ros2)

> VLA models move fast; robots need a stable runtime interface.
> `vla_ros2` wires **camera + instruction + robot state** to **typed actions**
> and publishes them on ROS2 topics, running inference locally on the robot.

```text
camera + instruction + robot state + timestamp
  -> stable VLA adapter boundary (local inference)
  -> typed VLAAction / VLAActionChunk
  -> ROS2 topic (/vla/action, /vla/action_chunk) + /diagnostics
```

This project was previously `vla_zoo`, a broad runtime/benchmark/adapter hub.
It has been refocused into a single job: **run a VLA policy on a robot through
ROS2**. Benchmarking, comparison reports, PyBullet demos, and the remote-GPU
HTTP path have been removed.

## Demos

### Scripted PyBullet (baseline runtime loop)

![vla_ros2 PyBullet pick-and-place driven by the runtime](docs/assets/sim_demo.gif)

A **Franka Panda in a PyBullet physics sim** performing pick-and-place, driven
through the real `vla_ros2` runtime: every control tick builds a
`VLAObservation`, calls `load_model("scripted")`, and the returned
`VLAAction` end-effector delta + gripper channel command the arm. It is the
`scripted` baseline adapter (not a learned VLA), but the loop, the physics, and
the action stream are real.

Reproduce: [`scripts/record_sim_demo.py`](scripts/record_sim_demo.py)

### SmolVLA closed-loop (learned policy)

![SmolVLA inference on a LeRobot-aligned SO-100 kinematic stand-in](docs/assets/smolvla_so100_demo.gif)

**Real SmolVLA inference** (`lerobot/smolvla_base`) in a closed loop on a
LeRobot-aligned SO-100 kinematic stand-in initialized from
`lerobot/svla_so100_stacking`. The base checkpoint is not fine-tuned for your
setup; task success is not guaranteed. Needs `pip install -e ".[smolvla]"` and a GPU.

Reproduce: [`scripts/record_smolvla_so100_demo.py`](scripts/record_smolvla_so100_demo.py)

Fine-tune on the same dataset (optional; improves task fit vs `smolvla_base` alone):

```bash
./scripts/finetune_smolvla_so100.sh
CKPT="$(./scripts/finetune_smolvla_so100.sh --print-checkpoint)"
.venv-smolvla/bin/python scripts/record_smolvla_so100_demo.py --pretrained "$CKPT"
```

### SmolVLA × Gazebo actuation (ROS2 closed loop)

![SmolVLA inference driving Gazebo arm actuation](docs/assets/gz_smolvla_demo.gif)

**Real SmolVLA inference** through the ROS2 runtime, SO-100-style camera rendered from
live `joint_states`, and 6D joint commands into `joint_trajectory_controller`.
Camera views are synthetic (not Gazebo RGB); joint motion is from the real sim graph.

Reproduce: [`scripts/record_gz_smolvla_demo.sh`](scripts/record_gz_smolvla_demo.sh)

Launch manually: [ros2/SIM.md](ros2/SIM.md) (`gz_smolvla.launch.py`; GPU required).

### VLA Playground (browser)

Try adapters locally before wiring ROS2:

```bash
pip install -e ".[playground,smolvla]"
python scripts/vla_playground.py
# open http://127.0.0.1:7860
```

Upload an image, enter an instruction, pick an adapter, and inspect the predicted action.

## Layout

```text
src/vla_ros2/         Python package (adapter runtime)
  core/               typed contracts: VLAObservation/Action/ActionChunk, registry
  adapters/           dummy, random, scripted, openvla, smolvla, pi0, groot
  runtime/            local runtime, action-clip guard + watchdog, diagnostics
  cli/                minimal off-robot sanity CLI (vla-ros2 list / predict)
ros2/vla_ros2/        ROS2 ament_python package: runtime node, launch, config
ros2/vla_ros2_msgs/   ROS2 messages: VLAAction, VLAActionChunk, VLAStatus, VLAInstruction
```

## Install (Python package)

```bash
pip install -e .                 # core runtime + minimal CLI
pip install -e ".[openvla]"      # + OpenVLA dependencies
pip install -e ".[smolvla]"      # + SmolVLA (LeRobot) dependencies
pip install -e ".[playground]"   # + Gradio browser playground
pip install -e ".[dev]"          # + test/lint tooling
```

Sanity-check adapters off-robot, without ROS2:

```bash
vla-ros2 list                              # list registered adapters
vla-ros2 predict --model dummy             # run one local inference (wiring check)
```

## Run (ROS2 node)

Build the workspace first — see [ros2/WORKSPACE.md](ros2/WORKSPACE.md) or run
`./scripts/bootstrap_ros2_workspace.sh`.

The runtime node loads an adapter, subscribes to image/instruction/state, runs
local inference at `control_hz`, and publishes typed actions plus a diagnostics
record. It defaults to `dry_run: true` for safe bring-up.

```bash
source install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# dummy adapter, no GPU / weights required
ros2 launch vla_ros2 dummy.launch.py

# OpenVLA local inference (needs a GPU and openvla extras)
ros2 launch vla_ros2 openvla.launch.py dry_run:=false
```

`ros2 launch vla_ros2 smoke.launch.py` brings up the runtime node plus a
synthetic-input node, and a real typed `VLAAction` flows on `/vla/action`.

For real-robot wiring (topic remaps, clip calibration, phased `dry_run`
bring-up), see [ros2/BRINGUP.md](ros2/BRINGUP.md).

For Gazebo Sim (spawn arm, bridge `/vla/action` to `joint_trajectory_controller`),
see [ros2/SIM.md](ros2/SIM.md).

### Topics

| Direction | Topic (default) | Type |
|---|---|---|
| in  | `/camera/image_raw` | `sensor_msgs/Image` |
| in  | `/vla/instruction` | `std_msgs/String` or `vla_ros2_msgs/VLAInstruction` |
| in  | `/joint_states` | `sensor_msgs/JointState` |
| out | `/vla/action` | `vla_ros2_msgs/VLAAction` |
| out | `/vla/action_chunk` | `vla_ros2_msgs/VLAActionChunk` |
| out | `/vla/status` | `vla_ros2_msgs/VLAStatus` |
| out | `/diagnostics` | `diagnostic_msgs/DiagnosticArray` |

### Safety

- `dry_run` defaults to `true`; actions are not published until you opt in.
- An action-clip guard bounds every action to `[action_low, action_high]`.
- A watchdog flags stale/missing images and instructions and downgrades status.

## Adapters

`dummy`, `random`, `scripted` always load (no ML dependencies) and are used for
bring-up and tests. `openvla`, `smolvla`, `pi0`, `groot` require their optional
dependency extras and a suitable GPU. Inference runs **on the robot** (local
runtime); there is no remote-server path.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests ros2
mypy src/vla_ros2
pytest
```

`tests/test_ros2_package_metadata.py` validates the `ros2/` package shape
(package.xml, setup.py entry points, messages, launch files) without requiring
a ROS2/colcon install, so CI guards the ROS2 integration on every push.

## License

Apache-2.0. See [LICENSE](LICENSE).
