# ROS2 workspace setup

This guide bootstraps a **source workspace** for the `vla_ros2` ROS2 packages on a
clean machine. The packages are **not** released into the ROS 2 distro yet; build
from this repository.

## Packages

| Package | Role | Profile |
|---------|------|---------|
| `vla_ros2_msgs` | `VLAAction`, `VLAStatus`, … message definitions | `core` |
| `vla_ros2` | Runtime node, launch files, smoke/recorder tools | `core` |
| `vla_ros2_gz` | Gazebo Sim arm + `/vla/action` bridge | `gz` |

**Profiles**

- `core` — runtime node only (robot or off-robot smoke tests)
- `gz` — `core` + Gazebo integration (`vla_ros2_gz`)

---

## 1. Prerequisites

- Ubuntu 24.04 (Noble) + [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- Python 3.10+
- `colcon`, `rosdep`, `vcstool` (via `ros-dev-tools` or individual packages)

```bash
sudo apt update
sudo apt install -y \
  python3-pip python3-venv \
  ros-jazzy-ros-base \
  ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge \
  python3-colcon-common-extensions python3-rosdep
```

For the **`gz`** profile, also install:

```bash
sudo apt install -y \
  ros-jazzy-gz-ros2-control ros-jazzy-ros2-control \
  ros-jazzy-controller-manager \
  ros-jazzy-joint-state-broadcaster ros-jazzy-joint-trajectory-controller \
  ros-jazzy-robot-state-publisher ros-jazzy-xacro
```

---

## 2. Quick bootstrap (recommended)

From a clone of this repository:

```bash
# core runtime (default)
./scripts/bootstrap_ros2_workspace.sh

# or include Gazebo packages
VLA_ROS2_PROFILE=gz ./scripts/bootstrap_ros2_workspace.sh
```

The script:

1. Sources `/opt/ros/$ROS_DISTRO/setup.bash` (default `jazzy`)
2. Runs `rosdep install` over `ros2/` (skips `ament_python`, already provided by ROS)
3. `pip install -e ".[dev]"` for the Python adapter runtime
4. `colcon build --base-paths ros2`
5. Prints `source install/setup.bash` and `PYTHONPATH` reminders

---

## 3. Manual setup

```bash
export ROS_DISTRO=jazzy
source /opt/ros/${ROS_DISTRO}/setup.bash

git clone https://github.com/rsasaki0109/vla_ros2.git
cd vla_ros2

sudo rosdep init 2>/dev/null || true
rosdep update
rosdep install --from-paths ros2 --ignore-src -y --skip-keys "ament_python"

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

colcon build --base-paths ros2 \
  --packages-up-to vla_ros2          # core
# colcon build --base-paths ros2 \
#   --packages-up-to vla_ros2_gz      # gz (includes msgs + runtime)

source install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

`PYTHONPATH` is required because the colcon packages install `vla_ros2_ros` nodes,
not the pip `vla_ros2` adapter package. Alternatively, `pip install -e .` into the
same Python that runs the nodes (the bootstrap script does this).

---

## 4. Verify

### Core

```bash
ros2 launch vla_ros2 smoke.launch.py
# other terminal:
ros2 topic echo /vla/action --once
```

### Launch test (optional)

```bash
colcon test --packages-select vla_ros2 --python-testing pytest \
  --event-handlers console_direct+
colcon test-result --verbose
```

### Gazebo (`gz` profile)

```bash
ros2 launch vla_ros2_gz gz_smoke.launch.py
```

See [SIM.md](SIM.md) for phased sim bring-up.

---

## 5. `rosdep` notes

- `ament_python` is a **buildtool** provided by the sourced ROS underlay, not an
  apt package. Bootstrap passes `--skip-keys "ament_python"`.
- All other keys in `package.xml` files resolve to `ros-jazzy-*` packages on
  Noble when the optional apt groups above are installed.
- The pip package `vla_ros2` (adapters, guards, CLI) is **not** in rosdep;
  install with `pip install -e .` as shown.

Check dependencies without installing:

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths ros2 --ignore-src --simulate --skip-keys "ament_python"
```

---

## 6. Build options

| Goal | Command |
|------|---------|
| All ROS packages | `colcon build --base-paths ros2` |
| Runtime only | `colcon build --base-paths ros2 --packages-up-to vla_ros2` |
| Messages only | `colcon build --base-paths ros2 --packages-select vla_ros2_msgs` |
| Rebuild after edit | `colcon build --base-paths ros2 --packages-select vla_ros2` |

Artifacts land in `build/`, `install/`, `log/` (gitignored).

---

## 7. Future binary release (Bloom)

These packages are intended for a future `rosdep`/`apt` release via Bloom. Until
then, use the source workflow above. A typical release sequence will be:

1. Tag `vla_ros2_msgs` → build and bloom-release to `ros-jazzy-vla-ros2-msgs`
2. Tag `vla_ros2` (depends on released msgs)
3. Tag `vla_ros2_gz` (optional sim extra)

The pip `vla_ros2` library will remain on PyPI separately from the ROS packages.

---

## Related docs

- [BRINGUP.md](BRINGUP.md) — real-robot wiring
- [SIM.md](SIM.md) — Gazebo Sim graph
- `/PLAN.md` — architecture handoff
- `/README.md` — project overview
