# Real-robot bring-up guide

This guide walks through wiring `vla_runtime_node` to a physical robot: camera,
instruction, joint state in; typed `VLAAction` out. It assumes ROS2 Jazzy and a
colcon workspace that already builds `vla_ros2_msgs` + `vla_ros2`.

The runtime **does not drive motors directly**. You provide a downstream bridge
(controller node, MoveIt servo, custom IK, etc.) that subscribes to `/vla/action`
and sends commands to your arm.

```text
[camera] --Image--> vla_runtime_node --VLAAction--> [your controller bridge] --> arm
[instruction] ----^              |
[joint_states] ---^              +-- VLAStatus, /diagnostics
```

---

## 1. Prerequisites

Also install the Python runtime:

```bash
pip install -e ".[dev]"          # core + test/lint
```

`cv_bridge` is **not** required; image conversion uses numpy directly.

Build the colcon workspace first — see [WORKSPACE.md](WORKSPACE.md).

### Robot-side publishers

Before starting the runtime, confirm these topics exist (names may differ on your
robot — remap in config):

| Input | Default topic | Message type | Notes |
|-------|---------------|--------------|-------|
| Camera | `/camera/image_raw` | `sensor_msgs/Image` | `rgb8` or `bgr8`; primary view used for inference |
| Instruction | `/vla/instruction` | `std_msgs/String` or `vla_ros2_msgs/VLAInstruction` | Set `instruction_msg_type` accordingly |
| Joint state | `/joint_states` | `sensor_msgs/JointState` | Passed to the adapter as observation state |

Quick checks:

```bash
ros2 topic hz /camera/image_raw
ros2 topic echo /joint_states --once
ros2 topic pub /vla/instruction std_msgs/msg/String "{data: 'pick up the cup'}" --once
```

---

## 2. Bring-up phases

Work through these phases in order. Do not skip ahead to `dry_run:=false` until
each gate passes.

### Phase A — Workspace smoke (no robot)

Verify the runtime graph end-to-end with synthetic inputs:

```bash
ros2 launch vla_ros2 smoke.launch.py
# another terminal:
ros2 topic echo /vla/action --once
ros2 topic echo /vla/status --once
```

Or run the automated launch test:

```bash
colcon test --packages-select vla_ros2 --python-testing pytest --event-handlers console_direct+
colcon test-result --verbose
```

### Phase B — Dry-run on robot I/O (`dummy` adapter)

Use the `dummy` adapter (zero actions, no GPU) to validate topic wiring and
watchdog behaviour on real sensors.

1. Copy and edit the example config:

   ```bash
   cp install/vla_ros2/share/vla_ros2/config/robot.example.yaml ~/my_robot.yaml
   # edit image_topic, joint_state_topic, instruction_msg_type, clip bounds
   ```

2. Launch with `dry_run` left at `true` (default):

   ```bash
   ros2 launch vla_ros2 dummy.launch.py \
     params_file:=/path/to/my_robot.yaml \
     instruction_msg_type:=vla_instruction
   ```

3. Gate checks:

   - `ros2 topic echo /vla/status` shows `ready: true` (after camera +
     instruction arrive).
   - `status_text` is not a watchdog message (`stale image`, `waiting for
     instruction`, etc.).
   - `/diagnostics` reports sane latency and zero/minimal clip rate with dummy
     zeros.

Optional: record a short trace for offline review:

```bash
ros2 launch vla_ros2 log_recorder.launch.py output_dir:=results/bringup
```

### Phase C — Observe actions without actuating

Keep `dry_run: true` but allow action messages on the wire so your controller
bridge can be tested in isolation:

```bash
ros2 launch vla_ros2 dummy.launch.py \
  params_file:=/path/to/my_robot.yaml \
  publish_actions_in_dry_run:=true
```

Your bridge should subscribe to `/vla/action` but **must not** forward commands
to the arm until Phase D. Confirm parsing of `data[]`, `names[]`, `frame_id`,
and `action_space`.

### Phase D — Live actuation (`dry_run:=false`)

Only after clip bounds and the controller bridge are verified:

```bash
ros2 launch vla_ros2 dummy.launch.py \
  params_file:=/path/to/my_robot.yaml \
  dry_run:=false
```

Start with `control_hz` low (2–5 Hz) and an e-stop reachable. Increase rate only
after stable tracking.

### Phase E — Learned policy

When I/O and safety layers are solid, swap the adapter:

```bash
pip install -e ".[openvla]"   # GPU + weights required
ros2 launch vla_ros2 openvla.launch.py \
  params_file:=/path/to/my_robot.yaml \
  dry_run:=true \
  image_topic:=/camera/color/image_raw \
  require_image:=true
```

Move to `dry_run:=false` using the same gates as Phase D. **Align `action_low` /
`action_high` and your controller bridge with the adapter's action space** — public
checkpoints may use spaces that do not match your arm without additional
kinematic wrapping.

---

## 3. Parameter reference

All parameters are declared on `vla_runtime_node`. Key fields for bring-up:

| Parameter | Default | Bring-up notes |
|-----------|---------|----------------|
| `dry_run` | `true` | Must stay `true` until you explicitly opt in |
| `publish_actions_in_dry_run` | `false` | Set `true` in Phase C to test the action topic |
| `image_topic` | `/camera/image_raw` | Remap to your camera driver |
| `instruction_topic` | `/vla/instruction` | Your task/UI node publishes here |
| `instruction_msg_type` | `string` | Use `vla_instruction` for `VLAInstruction` (task_id + metadata) |
| `joint_state_topic` | `/joint_states` | Remap if your robot uses a namespaced topic |
| `control_hz` | `5.0` | Inference + publish rate; start low on hardware |
| `require_image` | `false` | Set `true` on real robots / learned policies |
| `stale_image_timeout_sec` | `1.0` | Watchdog: no image within this window → not ready |
| `stale_instruction_timeout_sec` | `5.0` | Watchdog for instruction freshness |
| `clip_actions` | `true` | Keep enabled |
| `action_low` / `action_high` | empty | Comma-separated floats; see §4 |

Full list lives in `vla_ros2_ros/node.py` (`_declare_parameters`).

---

## 4. Action clip calibration

The clip guard clamps every element of `VLAAction.data` to `[action_low,
action_high]`. Empty bounds fall back to the adapter's declared `ActionSpec`
limits (if any).

For the default 7-DoF `eef_delta` layout (`x, y, z, roll, pitch, yaw,
gripper`), start conservative:

```yaml
action_low: "-0.05,-0.05,-0.05,-0.2,-0.2,-0.2,0.0"
action_high: "0.05,0.05,0.05,0.2,0.2,0.2,1.0"
```

Tune procedure:

1. Log actions in dry-run with your target adapter (`log_recorder` or
   `smoke_record.launch.py`).
2. Inspect max absolute values per dimension in JSONL.
3. Set bounds slightly above observed peaks, never below what the controller can
   safely execute.
4. Watch `/diagnostics` clip rates — sustained clipping means bounds are too
   tight or the policy mismatch is large.

Broadcast rule: a **single-value** bound applies to all dimensions; otherwise
provide one value per flattened action element.

---

## 5. Controller bridge (your code)

A **reference** parse-only bridge ships in this repo for Phase C:

```bash
# Synthetic graph (no robot) — automated gate:
./scripts/bringup_validate.sh c

# Runtime + bridge on your robot params:
ros2 launch vla_ros2 bringup_phase_c.launch.py params_file:=/path/to/my_robot.yaml
```

`vla_controller_bridge_node` subscribes to `/vla/action`, logs/publishes parsed fields on
`/vla/bridge/parsed`, and stays non-actuating until `enable_actuation:=true`. Optional
`publish_cmd_vel:=true` maps `eef_delta` to `geometry_msgs/Twist` on `/cmd_vel` — only use
when your stack expects that and E-stop is verified.

Config: `vla_ros2/config/controller_bridge.example.yaml`

`vla_ros2` publishes `vla_ros2_msgs/VLAAction`:

| Field | Meaning |
|-------|---------|
| `data[]` | Action vector (adapter-specific units) |
| `names[]` | Label per element (e.g. `x`, `y`, `z`, …) |
| `action_space` | e.g. `eef_delta` |
| `frame_id` | Target frame for Cartesian deltas |
| `dt` | Nominal control period (seconds) |
| `metadata_json` | Adapter/run metadata |

Minimal bridge sketch (pseudocode — implement in your robot stack):

```python
# Subscribe: /vla/action (vla_ros2_msgs/VLAAction)
# 1. Parse data[] + names[] + frame_id
# 2. Transform delta to your controller's command type (Twist, JointTrajectory, Servo, …)
# 3. Respect dry_run on the robot side as a second interlock until bring-up is complete
```

Keep actuation **out of** `vla_ros2` — one bridge per robot lets the runtime stay
adapter-focused.

---

## 6. Instruction publishing

### `std_msgs/String`

Prefer the helper (matches `instruction_qos()` — `TRANSIENT_LOCAL` + `RELIABLE`):

```bash
python3 scripts/publish_instruction.py --text "pick up the red block"
```

Or with `ros2 topic pub` (must match durability):

```bash
ros2 topic pub /vla/instruction std_msgs/msg/String \
  "{data: 'pick up the red block'}" --once \
  --qos-durability transient_local --qos-reliability reliable
```

Set `instruction_msg_type:=string`.

### `vla_ros2_msgs/VLAInstruction`

```bash
ros2 topic pub /vla/instruction vla_ros2_msgs/msg/VLAInstruction \
  "{text: 'pick up the red block', task_id: 'pick_red_001', metadata_json: '{}'}" --once
```

Set `instruction_msg_type:=vla_instruction`. Prefer this when your task manager
already tracks `task_id` and metadata.

---

## 7. Debugging tools

| Tool | Purpose |
|------|---------|
| `ros2 topic echo /vla/status` | `ready`, `status_text`, latency, dropped frames |
| `ros2 topic echo /diagnostics` | Clip rate, watchdog, schema `vla-ros2-diagnostics/v1` |
| `vla-ros2 list` / `vla-ros2 predict --model dummy` | Off-robot adapter sanity (no ROS2) |
| `ros2 launch vla_ros2 log_recorder.launch.py` | JSONL trace of action/status/diagnostics |
| `ros2 launch vla_ros2 action_replay.launch.py` | Replay recorded actions (controller testing) |

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Node exits: `No module named 'vla_ros2'` | Pip package not visible to node process | `pip install -e .` and/or `export PYTHONPATH=$PWD/src:$PYTHONPATH` |
| `ready: false`, stale image | Camera topic mismatch or slow driver | Fix `image_topic`; check `ros2 topic hz`; increase `stale_image_timeout_sec` temporarily |
| `ready: false`, waiting for instruction | No publisher on instruction topic | Publish instruction; check `instruction_msg_type` matches message type |
| No `/vla/action` in dry-run | Expected — set `publish_actions_in_dry_run:=true` or `dry_run:=false` | See Phase C/D |
| QoS warnings on `/vla/instruction` | `ros2 topic pub` defaults to volatile durability | Use `scripts/publish_instruction.py` or `--qos-durability transient_local` |
| QoS warnings on `/vla/status` | Status uses **best effort** | Match `status_qos()` in subscribers (see `tests/test_smoke_launch.py`) |
| High clip rate in diagnostics | Bounds too tight or wrong action space | Recalibrate §4; verify adapter/controller alignment |
| Inference slow / GPU OOM | Model too large for onboard GPU | Use a smaller adapter (`smolvla`), lower `control_hz`, or reduce image size upstream |

---

## 9. Safety checklist

- [ ] `dry_run` defaults to `true` in launch files (do not regress)
- [ ] `action_low` / `action_high` set before `dry_run:=false`
- [ ] Controller bridge tested with `publish_actions_in_dry_run:=true` first
- [ ] Watchdog timeouts appropriate for camera rate
- [ ] E-stop verified independent of ROS
- [ ] Clip + diagnostics monitored during first motion trials

---

## 10. Related files

- Example robot config: `vla_ros2/config/robot.example.yaml`
- Dashcam-only example (no joint_states publisher): `vla_ros2/config/bringup.dashcam.example.yaml`
- Automated Phase A/B gates: `scripts/bringup_validate.sh`
- Phase C bridge launch: `vla_ros2/launch/bringup_phase_c.launch.py`
- Reference bridge node: `vla_ros2_ros/controller_bridge.py`
- Instruction publisher helper: `scripts/publish_instruction.py`
- Launch entry points: `vla_ros2/launch/dummy.launch.py`, `openvla.launch.py`, `smoke.launch.py`
- Launch smoke test: `vla_ros2/tests/test_smoke_launch.py`
- Architecture handoff: `/PLAN.md`
- Workspace bootstrap: [WORKSPACE.md](WORKSPACE.md)
- Gazebo Sim integration: [SIM.md](SIM.md)
