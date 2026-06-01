# ROS2 Integration

The ROS2 workspace contains two packages:

- `ros2/vla_zoo`: runtime node, launch files, and YAML configs
- `ros2/vla_zoo_msgs`: message definitions

## Build

```bash
pip install -e .
colcon build --base-paths ros2 --symlink-install
source install/setup.bash
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

## Topics

Inputs:

- `/camera/image_raw`: `sensor_msgs/msg/Image`
- `/vla/instruction`: `std_msgs/msg/String`
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

## Runtime Dashboard From Logs

The static dashboard can ingest JSONL logs shaped like `VLAStatus` and `DiagnosticArray` messages:

```bash
ros2 launch vla_zoo log_recorder.launch.py output_dir:=results
vla-zoo compare dashboard \
  --status-log results/vla_status.jsonl \
  --diagnostics-log results/vla_diagnostics.jsonl \
  --out results/vla_ros_runtime_dashboard.html
```

This is intended for issue reports and field debugging: attach the JSONL plus generated HTML instead of screenshots alone.

The recorder node subscribes to `/vla/status` and `/diagnostics` by default. It does not command hardware and can run beside `dummy.launch.py`, `openvla.launch.py`, or `remote.launch.py`.

## Hardware Bridges

The MVP does not command hardware. Downstream bridge packages may translate:

- `VLAAction` to `trajectory_msgs/JointTrajectory`
- `VLAAction` to `geometry_msgs/Twist`
- `VLAAction` to MoveIt Servo commands
- `VLAAction` to ros2_control controller commands
