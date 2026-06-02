from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    image_topic = LaunchConfiguration("image_topic")
    instruction_topic = LaunchConfiguration("instruction_topic")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    action_topic = LaunchConfiguration("action_topic")
    status_topic = LaunchConfiguration("status_topic")
    diagnostics_topic = LaunchConfiguration("diagnostics_topic")
    publish_hz = LaunchConfiguration("publish_hz")
    control_hz = LaunchConfiguration("control_hz")
    dry_run = LaunchConfiguration("dry_run")
    publish_actions_in_dry_run = LaunchConfiguration("publish_actions_in_dry_run")
    instruction = LaunchConfiguration("instruction")
    task_id = LaunchConfiguration("task_id")
    output_dir = LaunchConfiguration("output_dir")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("vla_zoo"), "config", "dummy.yaml"]
                ),
            ),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("instruction_topic", default_value="/vla/instruction"),
            DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
            DeclareLaunchArgument("action_topic", default_value="/vla/action"),
            DeclareLaunchArgument("status_topic", default_value="/vla/status"),
            DeclareLaunchArgument("diagnostics_topic", default_value="/diagnostics"),
            DeclareLaunchArgument("publish_hz", default_value="5.0"),
            DeclareLaunchArgument("control_hz", default_value="5.0"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("publish_actions_in_dry_run", default_value="true"),
            DeclareLaunchArgument("instruction", default_value="pick up the red block"),
            DeclareLaunchArgument("task_id", default_value="ros2_smoke_pick_red_block"),
            DeclareLaunchArgument("output_dir", default_value="results/ros2_smoke"),
            DeclareLaunchArgument("status_log_name", default_value="vla_status.jsonl"),
            DeclareLaunchArgument("diagnostics_log_name", default_value="vla_diagnostics.jsonl"),
            DeclareLaunchArgument("max_records", default_value="0"),
            DeclareLaunchArgument("flush_every", default_value="1"),
            Node(
                package="vla_zoo",
                executable="vla_runtime_node",
                name="vla_runtime_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "model_name": "dummy",
                        "runtime": "local",
                        "dry_run": dry_run,
                        "instruction_msg_type": "vla_instruction",
                        "image_topic": image_topic,
                        "instruction_topic": instruction_topic,
                        "joint_state_topic": joint_state_topic,
                        "action_topic": action_topic,
                        "status_topic": status_topic,
                        "diagnostics_topic": diagnostics_topic,
                        "publish_actions_in_dry_run": publish_actions_in_dry_run,
                        "require_image": True,
                        "control_hz": control_hz,
                    },
                ],
            ),
            Node(
                package="vla_zoo",
                executable="vla_smoke_input_node",
                name="vla_smoke_input_node",
                output="screen",
                parameters=[
                    {
                        "image_topic": image_topic,
                        "instruction_topic": instruction_topic,
                        "joint_state_topic": joint_state_topic,
                        "publish_hz": publish_hz,
                        "instruction": instruction,
                        "task_id": task_id,
                    },
                ],
            ),
            Node(
                package="vla_zoo",
                executable="vla_runtime_recorder",
                name="vla_runtime_log_recorder",
                output="screen",
                parameters=[
                    {
                        "status_topic": status_topic,
                        "diagnostics_topic": diagnostics_topic,
                        "status_log_path": PathJoinSubstitution(
                            [output_dir, LaunchConfiguration("status_log_name")]
                        ),
                        "diagnostics_log_path": PathJoinSubstitution(
                            [output_dir, LaunchConfiguration("diagnostics_log_name")]
                        ),
                        "record_status": True,
                        "record_diagnostics": True,
                        "max_records": LaunchConfiguration("max_records"),
                        "flush_every": LaunchConfiguration("flush_every"),
                    },
                ],
            ),
        ]
    )
