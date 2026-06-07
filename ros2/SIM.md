# Gazebo Sim integration

ROS2-native simulation path for `vla_ros2`: spawn a 7-DoF arm in Gazebo Sim,
run the runtime against synthetic or bridged inputs, and route `/vla/action` to
`joint_trajectory_controller` via `vla_action_bridge_node`.

PyBullet (`scripts/record_sim_demo.py`) remains the README physics demo; this
stack is for ROS2 graph testing without hardware.

```text
vla_smoke_input_node ──► vla_runtime_node ──► /vla/action
        │                      ▲
        └── /joint_states ◄────┘ (from gz + ros2_control)
                                      │
                             vla_action_bridge_node
                                      │
                             joint_trajectory_controller ──► Gazebo arm
```

---

## Dependencies (ROS2 Jazzy)

```bash
sudo apt install \
  ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control ros-jazzy-controller-manager \
  ros-jazzy-joint-state-broadcaster ros-jazzy-joint-trajectory-controller \
  ros-jazzy-robot-state-publisher ros-jazzy-xacro
```

Also install the Python runtime:

```bash
pip install -e ".[dev]"
```

---

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vla_ros2_msgs vla_ros2 vla_ros2_gz
source install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

---

## Launch phases

### Phase 1 — Sim + runtime only (no arm motion)

Default: `dry_run:=true`, `enable_actuation:=false`. The runtime publishes
`/vla/action`; the bridge ignores commands.

```bash
ros2 launch vla_ros2_gz gz_smoke.launch.py
```

Verify:

```bash
ros2 topic echo /vla/action --once
ros2 topic echo /joint_states --once
```

### Phase 2 — Bridge actuation in sim

Enable the bridge (runtime may stay in dry-run or not):

```bash
ros2 launch vla_ros2_gz gz_smoke.launch.py \
  enable_actuation:=true \
  dry_run:=false \
  publish_actions_in_dry_run:=true
```

Try `model_name:=random` to see the arm move from non-zero actions.

### Phase 3 — Gazebo arm only (no VLA graph)

```bash
ros2 launch vla_ros2_gz gz_arm.launch.py
```

---

## Action bridge semantics

`vla_action_bridge_node` maps each element of a 7-DoF `eef_delta` action to a
joint position increment:

```text
target[joint_i] += data[i] * joint_scales[i]
```

This is a **sim convenience**, not Cartesian IK. Real robots should implement
their own bridge (see [BRINGUP.md](BRINGUP.md)).

Tune scales in `vla_ros2_gz/config/gz_smoke.yaml` or via launch overrides.

---

## Files

| Path | Role |
|------|------|
| `vla_ros2_gz/urdf/vla_arm.urdf.xacro` | 7-DoF arm + `gz_ros2_control` |
| `vla_ros2_gz/config/vla_arm_controllers.yaml` | `joint_trajectory_controller` |
| `vla_ros2_gz/config/gz_smoke.yaml` | Runtime + bridge defaults |
| `vla_ros2_gz/launch/gz_arm.launch.py` | Gazebo + controllers |
| `vla_ros2_gz/launch/gz_smoke.launch.py` | Full VLA + sim graph |
| `vla_ros2_gz/vla_ros2_gz_ros/action_bridge.py` | `/vla/action` → trajectory |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'vla_ros2'` | `pip install -e .` and/or `export PYTHONPATH=$PWD/src:$PYTHONPATH` |
| Controllers fail to spawn | Ensure `gz_ros2_control` is installed; rebuild after URDF edits |
| Arm static despite actions | Set `enable_actuation:=true` and `dry_run:=false` |
| `dummy` produces no motion | Expected — zeros hold position; try `model_name:=random` |

---

## Related

- Workspace bootstrap: [WORKSPACE.md](WORKSPACE.md)
- Real-robot bring-up: [BRINGUP.md](BRINGUP.md)
- SmolVLA closed-loop kinematic demo: [`scripts/record_smolvla_so100_demo.py`](../scripts/record_smolvla_so100_demo.py)
- Launch smoke test (no Gazebo): `vla_ros2/tests/test_smoke_launch.py`
- Architecture handoff: `/PLAN.md`
