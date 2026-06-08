from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "action_log_path",
                default_value="results/ros2_smoke/vla_actions.jsonl",
            ),
            DeclareLaunchArgument("action_topic", default_value="/vla/action_replay"),
            DeclareLaunchArgument("status_topic", default_value="/vla/replay_status"),
            DeclareLaunchArgument("use_recorded_timing", default_value="true"),
            DeclareLaunchArgument("replay_hz", default_value="5.0"),
            DeclareLaunchArgument("speed", default_value="1.0"),
            DeclareLaunchArgument("start_delay_sec", default_value="1.0"),
            DeclareLaunchArgument("loop", default_value="false"),
            DeclareLaunchArgument("stamp_now", default_value="true"),
            DeclareLaunchArgument("frame_id_override", default_value=""),
            DeclareLaunchArgument("max_actions", default_value="0"),
            DeclareLaunchArgument("max_queue_size", default_value="10"),
            Node(
                package="vla_ros2",
                executable="vla_action_replay_node",
                name="vla_action_replay_node",
                output="screen",
                parameters=[
                    {
                        "action_log_path": LaunchConfiguration("action_log_path"),
                        "action_topic": LaunchConfiguration("action_topic"),
                        "status_topic": LaunchConfiguration("status_topic"),
                        "use_recorded_timing": LaunchConfiguration("use_recorded_timing"),
                        "replay_hz": LaunchConfiguration("replay_hz"),
                        "speed": LaunchConfiguration("speed"),
                        "start_delay_sec": LaunchConfiguration("start_delay_sec"),
                        "loop": LaunchConfiguration("loop"),
                        "stamp_now": LaunchConfiguration("stamp_now"),
                        "frame_id_override": LaunchConfiguration("frame_id_override"),
                        "max_actions": LaunchConfiguration("max_actions"),
                        "max_queue_size": LaunchConfiguration("max_queue_size"),
                    },
                ],
            ),
        ]
    )
