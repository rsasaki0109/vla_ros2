from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    enable_actuation = LaunchConfiguration("enable_actuation")
    publish_cmd_vel = LaunchConfiguration("publish_cmd_vel")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("vla_ros2"), "config", "controller_bridge.example.yaml"]
                ),
            ),
            DeclareLaunchArgument("enable_actuation", default_value="false"),
            DeclareLaunchArgument("publish_cmd_vel", default_value="false"),
            Node(
                package="vla_ros2",
                executable="vla_controller_bridge_node",
                name="vla_controller_bridge_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "enable_actuation": enable_actuation,
                        "publish_cmd_vel": publish_cmd_vel,
                    },
                ],
            ),
        ]
    )
