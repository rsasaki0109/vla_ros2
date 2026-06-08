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

Headless (recommended for CI/local gates):

```bash
ros2 launch vla_ros2_gz gz_smoke.launch.py gz_args:="-s"
```

Automated gate:

```bash
./scripts/gz_smoke_validate.sh 1    # or `all` for Phase 1 + 2
```

Uses an isolated `ROS_DOMAIN_ID` by default to avoid collisions with other
ROS stacks on the same machine.

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

```bash
ros2 launch vla_ros2_gz gz_smoke.launch.py gz_args:="-s" \
  enable_actuation:=true dry_run:=false model_name:=random
```

Automated actuation gate (checks bridge trajectories):

```bash
./scripts/gz_smoke_validate.sh 2
```

### Phase 3 — SmolVLA closed-loop (learned policy → Gazebo)

**Real SmolVLA inference** through `vla_runtime_node`, SO-100-style rendered camera
from live `joint_states`, and a 6D joint bridge into `joint_trajectory_controller`.

Requires GPU + SmolVLA extras:

```bash
pip install -e ".[smolvla]"    # same Python env that runs the ROS nodes
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ros2 launch vla_ros2_gz gz_smolvla.launch.py \
  dry_run:=false enable_actuation:=true control_hz:=2.0
```

Inference-only gate (no arm motion):

```bash
./scripts/gz_smolvla_validate.sh infer
```

Honesty note: the Gazebo arm is a stand-in, not the SO-100 URDF; task success is not
guaranteed with `smolvla_base` alone. The loop, inference, and ROS graph are real.

Demo GIF (recorded from the live graph; synthetic camera, real joint actuation):

![SmolVLA × Gazebo actuation demo](../../docs/assets/gz_smolvla_demo.gif)

```bash
./scripts/record_gz_smolvla_demo.sh
# kinematic fallback (no Gazebo): python scripts/record_gz_smolvla_demo.py --offline
```

Fine-tune on `lerobot/svla_so100_stacking` and point the launch at the checkpoint:

```bash
./scripts/finetune_smolvla_so100.sh
CKPT="$(./scripts/finetune_smolvla_so100.sh --print-checkpoint)"
ros2 launch vla_ros2_gz gz_smolvla.launch.py pretrained:=$CKPT enable_actuation:=true
```

### Phase 4 — Gazebo arm only (no VLA graph)

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

`vla_smolvla_joint_bridge_node` maps 6D SmolVLA joint targets (degrees + gripper) to
absolute Gazebo joint positions with optional blending (`action_blend`).

---

## Files

| Path | Role |
|------|------|
| `vla_ros2_gz/urdf/vla_arm.urdf.xacro` | 7-DoF arm + `gz_ros2_control` |
| `vla_ros2_gz/config/vla_arm_controllers.yaml` | `joint_trajectory_controller` |
| `vla_ros2_gz/config/gz_smoke.yaml` | Runtime + bridge defaults |
| `vla_ros2_gz/launch/gz_arm.launch.py` | Gazebo + controllers |
| `vla_ros2_gz/launch/gz_smoke.launch.py` | Full VLA + sim graph |
| `vla_ros2_gz/launch/gz_smolvla.launch.py` | SmolVLA + SO-100 render + joint bridge |
| `vla_ros2_gz/config/gz_smolvla.yaml` | SmolVLA runtime + bridge defaults |
| `vla_ros2_gz/vla_ros2_gz_ros/action_bridge.py` | `eef_delta` → trajectory |
| `vla_ros2_gz/vla_ros2_gz_ros/smolvla_joint_bridge.py` | 6D SmolVLA → trajectory |
| `vla_ros2/vla_ros2_ros/smolvla_input.py` | Render 256² camera from `joint_states` |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'vla_ros2'` | `pip install -e .` and/or `export PYTHONPATH=$PWD/src:$PYTHONPATH` |
| Controllers fail to spawn | Ensure `gz_ros2_control` is installed; rebuild after URDF edits |
| Arm static despite actions | Set `enable_actuation:=true` and `dry_run:=false` |
| `dummy` produces no motion | Expected — zeros hold position; try `model_name:=random` |
| Spawner lock / controller timeout | Remove `~/.ros/locks/ros2-control-controller-spawner.lock`; use isolated `ROS_DOMAIN_ID`; see `scripts/gz_smoke_validate.sh` |
| Controller spawner races Gazebo init | `gz_arm.launch.py` delays spawner 5s and loads both controllers in one `--activate-as-group` call |
| SmolVLA inference fails / OOM | Lower `control_hz`; confirm `pip install -e ".[smolvla]"` and CUDA; try `device:=cuda:0` |

---

## Related

- Workspace bootstrap: [WORKSPACE.md](WORKSPACE.md)
- Real-robot bring-up: [BRINGUP.md](BRINGUP.md)
- SmolVLA closed-loop kinematic demo: [`scripts/record_smolvla_so100_demo.py`](../scripts/record_smolvla_so100_demo.py)
- SmolVLA × Gazebo GIF: [`scripts/record_gz_smolvla_demo.sh`](../scripts/record_gz_smolvla_demo.sh)
- SmolVLA fine-tune: [`scripts/finetune_smolvla_so100.sh`](../scripts/finetune_smolvla_so100.sh)
- Browser playground: [`scripts/vla_playground.py`](../scripts/vla_playground.py)
- Launch smoke test (no Gazebo): `vla_ros2/tests/test_smoke_launch.py`
- Architecture handoff: `/PLAN.md`
