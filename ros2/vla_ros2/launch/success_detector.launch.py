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
                    [FindPackageShare("vla_ros2"), "config", "success_detector.yaml"]
                ),
            ),
            DeclareLaunchArgument("vlm_name", default_value="smolvlm"),
            DeclareLaunchArgument(
                "pretrained", default_value="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            ),
            DeclareLaunchArgument("device", default_value="auto"),
            DeclareLaunchArgument("check_hz", default_value="0.5"),
            DeclareLaunchArgument("instruction_msg_type", default_value="string"),
            Node(
                package="vla_ros2",
                executable="vla_success_detector_node",
                name="vla_success_detector_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "vlm_name": LaunchConfiguration("vlm_name"),
                        "pretrained": LaunchConfiguration("pretrained"),
                        "device": LaunchConfiguration("device"),
                        "check_hz": LaunchConfiguration("check_hz"),
                        "instruction_msg_type": LaunchConfiguration("instruction_msg_type"),
                    },
                ],
            ),
        ]
    )
