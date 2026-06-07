from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    dry_run = LaunchConfiguration("dry_run")
    publish_actions_in_dry_run = LaunchConfiguration("publish_actions_in_dry_run")
    instruction_msg_type = LaunchConfiguration("instruction_msg_type")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("vla_ros2"), "config", "dummy.yaml"]
                ),
            ),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("publish_actions_in_dry_run", default_value="false"),
            DeclareLaunchArgument("instruction_msg_type", default_value="string"),
            Node(
                package="vla_ros2",
                executable="vla_runtime_node",
                name="vla_runtime_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "model_name": "dummy",
                        "runtime": "local",
                        "dry_run": dry_run,
                        "instruction_msg_type": instruction_msg_type,
                        "publish_actions_in_dry_run": publish_actions_in_dry_run,
                    },
                ],
            )
        ]
    )
