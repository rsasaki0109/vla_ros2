from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    output_dir = LaunchConfiguration("output_dir")
    return LaunchDescription(
        [
            DeclareLaunchArgument("action_topic", default_value="/vla/action"),
            DeclareLaunchArgument("status_topic", default_value="/vla/status"),
            DeclareLaunchArgument("diagnostics_topic", default_value="/diagnostics"),
            DeclareLaunchArgument("output_dir", default_value="results"),
            DeclareLaunchArgument("action_log_name", default_value="vla_actions.jsonl"),
            DeclareLaunchArgument("status_log_name", default_value="vla_status.jsonl"),
            DeclareLaunchArgument("diagnostics_log_name", default_value="vla_diagnostics.jsonl"),
            DeclareLaunchArgument("record_actions", default_value="true"),
            DeclareLaunchArgument("record_status", default_value="true"),
            DeclareLaunchArgument("record_diagnostics", default_value="true"),
            DeclareLaunchArgument("max_records", default_value="0"),
            DeclareLaunchArgument("flush_every", default_value="1"),
            Node(
                package="vla_ros2",
                executable="vla_runtime_recorder",
                name="vla_runtime_log_recorder",
                output="screen",
                parameters=[
                    {
                        "action_topic": LaunchConfiguration("action_topic"),
                        "status_topic": LaunchConfiguration("status_topic"),
                        "diagnostics_topic": LaunchConfiguration("diagnostics_topic"),
                        "action_log_path": PathJoinSubstitution(
                            [output_dir, LaunchConfiguration("action_log_name")]
                        ),
                        "status_log_path": PathJoinSubstitution(
                            [output_dir, LaunchConfiguration("status_log_name")]
                        ),
                        "diagnostics_log_path": PathJoinSubstitution(
                            [output_dir, LaunchConfiguration("diagnostics_log_name")]
                        ),
                        "record_actions": LaunchConfiguration("record_actions"),
                        "record_status": LaunchConfiguration("record_status"),
                        "record_diagnostics": LaunchConfiguration("record_diagnostics"),
                        "max_records": LaunchConfiguration("max_records"),
                        "flush_every": LaunchConfiguration("flush_every"),
                    },
                ],
            ),
        ]
    )
