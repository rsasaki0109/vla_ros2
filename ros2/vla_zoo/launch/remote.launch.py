from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("vla_zoo"), "config", "remote.yaml"]
                ),
            ),
            DeclareLaunchArgument("model_name", default_value="openvla"),
            DeclareLaunchArgument("remote_url", default_value="http://localhost:8000"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("instruction_msg_type", default_value="string"),
            DeclareLaunchArgument("diagnostics_topic", default_value="/diagnostics"),
            DeclareLaunchArgument("publish_diagnostics", default_value="true"),
            DeclareLaunchArgument("publish_actions_in_dry_run", default_value="false"),
            DeclareLaunchArgument("require_image", default_value="true"),
            DeclareLaunchArgument("stale_image_timeout_sec", default_value="1.0"),
            DeclareLaunchArgument("stale_instruction_timeout_sec", default_value="5.0"),
            DeclareLaunchArgument("clip_actions", default_value="true"),
            Node(
                package="vla_zoo",
                executable="vla_runtime_node",
                name="vla_runtime_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "model_name": LaunchConfiguration("model_name"),
                        "runtime": "remote",
                        "remote_url": LaunchConfiguration("remote_url"),
                        "dry_run": LaunchConfiguration("dry_run"),
                        "instruction_msg_type": LaunchConfiguration("instruction_msg_type"),
                        "diagnostics_topic": LaunchConfiguration("diagnostics_topic"),
                        "publish_diagnostics": LaunchConfiguration("publish_diagnostics"),
                        "publish_actions_in_dry_run": LaunchConfiguration(
                            "publish_actions_in_dry_run"
                        ),
                        "require_image": LaunchConfiguration("require_image"),
                        "stale_image_timeout_sec": LaunchConfiguration(
                            "stale_image_timeout_sec"
                        ),
                        "stale_instruction_timeout_sec": LaunchConfiguration(
                            "stale_instruction_timeout_sec"
                        ),
                        "clip_actions": LaunchConfiguration("clip_actions"),
                    },
                ],
            ),
        ]
    )
