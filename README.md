# vla_ros2

ROS2-native on-robot runtime for Vision-Language-Action (VLA) models.

[![CI](https://github.com/rsasaki0109/vla_ros2/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/vla_ros2/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

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
pip install -e ".[dev]"          # + test/lint tooling
```

Sanity-check adapters off-robot, without ROS2:

```bash
vla-ros2 list                              # list registered adapters
vla-ros2 predict --model dummy             # run one local inference (wiring check)
```

## Run (ROS2 node)

The runtime node loads an adapter, subscribes to image/instruction/state, runs
local inference at `control_hz`, and publishes typed actions plus a diagnostics
record. It defaults to `dry_run: true` for safe bring-up.

```bash
# build the colcon workspace (vla_ros2_msgs first, then vla_ros2)
colcon build --packages-select vla_ros2_msgs vla_ros2
source install/setup.bash

# dummy adapter, no GPU / weights required
ros2 launch vla_ros2 dummy.launch.py

# OpenVLA local inference (needs a GPU and openvla extras)
ros2 launch vla_ros2 openvla.launch.py dry_run:=false
```

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
