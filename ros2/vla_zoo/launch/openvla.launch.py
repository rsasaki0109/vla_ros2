from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("vla_zoo"), "config", "openvla.yaml"]
            ),
        ),
        DeclareLaunchArgument("model_name", default_value="openvla"),
        DeclareLaunchArgument("runtime", default_value="local"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("instruction_topic", default_value="/vla/instruction"),
        DeclareLaunchArgument("instruction_msg_type", default_value="string"),
        DeclareLaunchArgument("action_topic", default_value="/vla/action"),
        DeclareLaunchArgument("diagnostics_topic", default_value="/diagnostics"),
        DeclareLaunchArgument("publish_diagnostics", default_value="true"),
        DeclareLaunchArgument("publish_actions_in_dry_run", default_value="false"),
        DeclareLaunchArgument("require_image", default_value="true"),
        DeclareLaunchArgument("stale_image_timeout_sec", default_value="1.0"),
        DeclareLaunchArgument("stale_instruction_timeout_sec", default_value="5.0"),
        DeclareLaunchArgument("clip_actions", default_value="true"),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("pretrained", default_value="openvla/openvla-7b"),
        DeclareLaunchArgument("unnorm_key", default_value="bridge_orig"),
        DeclareLaunchArgument("remote_url", default_value="http://localhost:8000"),
    ]
    node = Node(
        package="vla_zoo",
        executable="vla_runtime_node",
        name="vla_runtime_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "model_name": LaunchConfiguration("model_name"),
                "runtime": LaunchConfiguration("runtime"),
                "dry_run": LaunchConfiguration("dry_run"),
                "image_topic": LaunchConfiguration("image_topic"),
                "instruction_topic": LaunchConfiguration("instruction_topic"),
                "instruction_msg_type": LaunchConfiguration("instruction_msg_type"),
                "action_topic": LaunchConfiguration("action_topic"),
                "diagnostics_topic": LaunchConfiguration("diagnostics_topic"),
                "publish_diagnostics": LaunchConfiguration("publish_diagnostics"),
                "publish_actions_in_dry_run": LaunchConfiguration(
                    "publish_actions_in_dry_run"
                ),
                "require_image": LaunchConfiguration("require_image"),
                "stale_image_timeout_sec": LaunchConfiguration("stale_image_timeout_sec"),
                "stale_instruction_timeout_sec": LaunchConfiguration(
                    "stale_instruction_timeout_sec"
                ),
                "clip_actions": LaunchConfiguration("clip_actions"),
                "device": LaunchConfiguration("device"),
                "pretrained": LaunchConfiguration("pretrained"),
                "unnorm_key": LaunchConfiguration("unnorm_key"),
                "remote_url": LaunchConfiguration("remote_url"),
            },
        ],
    )
    return LaunchDescription([*declared_arguments, node])
