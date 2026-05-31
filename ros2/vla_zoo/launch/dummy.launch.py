from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("vla_zoo"), "config", "dummy.yaml"]
                ),
            ),
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
                        "dry_run": True,
                    },
                ],
            )
        ]
    )
