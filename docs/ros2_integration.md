# ROS2 Integration

The ROS2 workspace contains two packages:

- `ros2/vla_zoo`: runtime node, launch files, and YAML configs
- `ros2/vla_zoo_msgs`: message definitions

## Build

```bash
pip install -e .
colcon build --base-paths ros2 --symlink-install
source install/setup.bash
vla-zoo doctor
ros2 launch vla_zoo dummy.launch.py
```

If ROS2 is running under a different Python interpreter than the editable install, export the source package before launching:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

To observe action messages during dummy dry-run demos, opt in explicitly:

```bash
ros2 launch vla_zoo dummy.launch.py publish_actions_in_dry_run:=true
```

For a self-contained runtime smoke test, launch the runtime together with synthetic
camera, joint state, and typed instruction inputs:

```bash
ros2 launch vla_zoo smoke.launch.py
```

`smoke.launch.py` keeps `dry_run:=true`, uses the `dummy` adapter, sets
`instruction_msg_type:=vla_instruction`, requires a fresh image, and publishes
dummy actions for logging/demo visibility.

To record the same smoke run for dashboards and issue reports:

```bash
vla-zoo ros smoke-report --output-dir results/ros2_smoke
```

That command runs `smoke_record.launch.py`, writes JSONL logs, and builds
`dashboard.html`, `action_trace.html`, `action_analysis.json`,
`action_analysis.md`, and `report_bundle.zip`.

The manual sequence is:

```bash
ros2 launch vla_zoo smoke_record.launch.py output_dir:=results/ros2_smoke
```

Stop the launch after the JSONL files are non-empty, then build a dashboard from the logs:

```bash
vla-zoo compare dashboard \
  --status-log results/ros2_smoke/vla_status.jsonl \
  --diagnostics-log results/ros2_smoke/vla_diagnostics.jsonl \
  --out results/ros2_smoke/dashboard.html
vla-zoo ros action-trace \
  --action-log results/ros2_smoke/vla_actions.jsonl \
  --out results/ros2_smoke/action_trace.html
vla-zoo ros action-analyze \
  --action-log results/ros2_smoke/vla_actions.jsonl \
  --out results/ros2_smoke/action_analysis.json \
  --markdown-out results/ros2_smoke/action_analysis.md
```

## Topics

Inputs:

- `/camera/image_raw`: `sensor_msgs/msg/Image`
- `/vla/instruction`: `std_msgs/msg/String` by default, or `vla_zoo_msgs/msg/VLAInstruction`
- `/joint_states`: optional `sensor_msgs/msg/JointState`

Outputs:

- `/vla/action`: `vla_zoo_msgs/msg/VLAAction`
- `/vla/action_chunk`: `vla_zoo_msgs/msg/VLAActionChunk`
- `/vla/status`: `vla_zoo_msgs/msg/VLAStatus`
- `/diagnostics`: `diagnostic_msgs/msg/DiagnosticArray`

## Parameters

Core runtime parameters:

- `model_name`: adapter name such as `dummy`, `openvla`, or `smolvla`
- `runtime`: `local` or `remote`
- `dry_run`: keep the node in non-commanding mode
- `instruction_msg_type`: `string`, `vla_instruction`, or `vla_zoo_msgs/VLAInstruction`
- `publish_actions_in_dry_run`: publish actions anyway for demos and logging
- `control_hz`: outer-loop VLA inference rate
- `max_queue_size`: ROS publisher/subscriber queue depth
- `device`, `pretrained`, `unnorm_key`, `remote_url`: adapter/runtime configuration

Topic parameters include `image_topic`, `instruction_topic`, `joint_state_topic`, `action_topic`, `action_chunk_topic`, `status_topic`, and `diagnostics_topic`.

Safety and observability parameters:

- `require_image`: wait for a camera frame before inference
- `stale_image_timeout_sec`: stop inference when the latest image is too old
- `stale_instruction_timeout_sec`: stop inference when the instruction is too old
- `clip_actions`: clip outgoing actions using configured or adapter-declared bounds
- `action_low`, `action_high`: scalar or flattened action bounds
- `publish_diagnostics`: publish a ROS diagnostics status

## QoS

- Image subscription uses sensor data QoS.
- Instruction uses reliable transient local QoS.
- Actions use reliable QoS.
- Status uses best effort QoS.
- Diagnostics uses best effort QoS.

## Runtime Safety Behavior

`dry_run` defaults to true. In dry-run mode the node computes predictions and updates status/diagnostics, but suppresses `/vla/action` and `/vla/action_chunk` unless `publish_actions_in_dry_run:=true` is set. This keeps launch files useful for validation while avoiding accidental downstream command publication.

The watchdog checks image and instruction freshness before starting each inference. If a timeout is hit, `/vla/status` and `/diagnostics` report `stale image` or `stale instruction` and the node skips inference until fresh inputs arrive.

## Typed Instructions

The default instruction topic type is `std_msgs/msg/String` for easy demos:

```bash
ros2 launch vla_zoo dummy.launch.py
python examples/ros2/publish_instruction.py --instruction "pick up the red block"
```

For benchmark runs, issue reports, and task-level logging, use `VLAInstruction`:

```bash
ros2 launch vla_zoo dummy.launch.py \
  instruction_msg_type:=vla_instruction \
  publish_actions_in_dry_run:=true
python examples/ros2/publish_instruction.py \
  --typed \
  --task-id pick_red_block_001 \
  --instruction "pick up the red block"
```

`VLAInstruction.task_id` and `metadata_json` are copied into `VLAObservation.metadata`,
`VLAStatus.metadata_json`, and diagnostics key-values. This keeps ROS bag replay,
dashboards, and future benchmark runners aligned on the same task identity without
changing the model adapter API.

## Smoke Input Node

`vla_smoke_input_node` publishes a deterministic RGB image with a moving red block,
a `vla_zoo_msgs/msg/VLAInstruction`, and optional `sensor_msgs/msg/JointState`.
It is for launch validation and ROS bag/report demos, not for measuring model skill.

Useful launch arguments:

- `image_topic`: camera topic to publish and subscribe
- `instruction_topic`: typed instruction topic
- `joint_state_topic`: optional joint state topic
- `publish_hz`: synthetic input rate
- `instruction`: instruction text
- `task_id`: typed task identifier copied into runtime metadata

## Runtime Dashboard From Logs

The static dashboard can ingest JSONL logs shaped like `VLAStatus` and `DiagnosticArray` messages:

```bash
vla-zoo ros smoke-report --output-dir results
vla-zoo compare dashboard \
  --status-log results/vla_status.jsonl \
  --diagnostics-log results/vla_diagnostics.jsonl \
  --out results/vla_ros_runtime_dashboard.html
vla-zoo report bundle \
  --status-log results/vla_status.jsonl \
  --diagnostics-log results/vla_diagnostics.jsonl \
  --out results/vla_runtime_report_bundle.zip
vla-zoo ros action-trace \
  --action-log results/vla_actions.jsonl \
  --out results/vla_action_trace.html
vla-zoo ros action-analyze \
  --action-log results/vla_actions.jsonl \
  --markdown-out results/vla_action_analysis.md
```

This is intended for issue reports and field debugging: attach the JSONL plus generated HTML instead of screenshots alone.
The bundle command packages the logs, normalized records, generated dashboard, and adapter inventory into one zip.

The recorder node subscribes to `/vla/status` and `/diagnostics` by default. It does not command hardware and can run beside `dummy.launch.py`, `openvla.launch.py`, or `remote.launch.py`.
It can also record `/vla/action` to `vla_actions.jsonl`; the action trace HTML
visualizes action magnitude, timing, action space, and per-dimension values.
The action analysis report flags low action rate, large action gaps, repeated
actions, near-zero actions, and per-dimension ranges.

## Hardware Bridges

The MVP does not command hardware. Downstream bridge packages may translate:

- `VLAAction` to `trajectory_msgs/JointTrajectory`
- `VLAAction` to `geometry_msgs/Twist`
- `VLAAction` to MoveIt Servo commands
- `VLAAction` to ros2_control controller commands
