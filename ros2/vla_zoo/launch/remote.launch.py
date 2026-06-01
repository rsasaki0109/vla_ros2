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
                    },
                ],
            ),
        ]
    )
