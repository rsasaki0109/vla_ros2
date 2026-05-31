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

## Topics

Inputs:

- `/camera/image_raw`: `sensor_msgs/msg/Image`
- `/vla/instruction`: `std_msgs/msg/String`
- `/joint_states`: optional `sensor_msgs/msg/JointState`

Outputs:

- `/vla/action`: `vla_zoo_msgs/msg/VLAAction`
- `/vla/action_chunk`: `vla_zoo_msgs/msg/VLAActionChunk`
- `/vla/status`: `vla_zoo_msgs/msg/VLAStatus`

## Parameters

The node exposes `model_name`, `runtime`, `dry_run`, topic names, `control_hz`, `device`, `pretrained`, `unnorm_key`, and `remote_url`.

## QoS

- Image subscription uses sensor data QoS.
- Instruction uses reliable transient local QoS.
- Actions use reliable QoS.
- Status uses best effort QoS.

## Hardware Bridges

The MVP does not command hardware. Downstream bridge packages may translate:

- `VLAAction` to `trajectory_msgs/JointTrajectory`
- `VLAAction` to `geometry_msgs/Twist`
- `VLAAction` to MoveIt Servo commands
- `VLAAction` to ros2_control controller commands
