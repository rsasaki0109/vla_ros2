import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _pythonpath_for_vla_ros2() -> str:
    repo_src = Path(__file__).resolve().parents[3] / "src"
    src = str(repo_src)
    existing = os.environ.get("PYTHONPATH", "")
    if not existing:
        return src
    if src in existing.split(os.pathsep):
        return existing
    return os.pathsep.join((src, existing))


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    dry_run = LaunchConfiguration("dry_run")
    publish_actions_in_dry_run = LaunchConfiguration("publish_actions_in_dry_run")
    enable_actuation = LaunchConfiguration("enable_actuation")
    model_name = LaunchConfiguration("model_name")
    control_hz = LaunchConfiguration("control_hz")
    publish_hz = LaunchConfiguration("publish_hz")
    gz_args = LaunchConfiguration("gz_args")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("vla_ros2_gz"), "config", "gz_smoke.yaml"]
                ),
            ),
            DeclareLaunchArgument("model_name", default_value="dummy"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("publish_actions_in_dry_run", default_value="true"),
            DeclareLaunchArgument("enable_actuation", default_value="false"),
            DeclareLaunchArgument("control_hz", default_value="5.0"),
            DeclareLaunchArgument("publish_hz", default_value="5.0"),
            DeclareLaunchArgument("gz_args", default_value="-s"),
            SetEnvironmentVariable("PYTHONPATH", _pythonpath_for_vla_ros2()),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [FindPackageShare("vla_ros2_gz"), "launch", "gz_arm.launch.py"]
                        )
                    ]
                ),
                launch_arguments={"gz_args": gz_args}.items(),
            ),
            Node(
                package="vla_ros2",
                executable="vla_runtime_node",
                name="vla_runtime_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "model_name": model_name,
                        "runtime": "local",
                        "dry_run": dry_run,
                        "publish_actions_in_dry_run": publish_actions_in_dry_run,
                        "control_hz": control_hz,
                        "require_image": True,
                    },
                ],
            ),
            Node(
                package="vla_ros2",
                executable="vla_smoke_input_node",
                name="vla_smoke_input_node",
                output="screen",
                parameters=[
                    {
                        "publish_hz": publish_hz,
                        "instruction": "pick up the red block",
                        "task_id": "gz_smoke_pick_red_block",
                    },
                ],
            ),
            Node(
                package="vla_ros2_gz",
                executable="vla_action_bridge_node",
                name="vla_action_bridge_node",
                output="screen",
                parameters=[
                    params_file,
                    {"enable_actuation": enable_actuation},
                ],
            ),
        ]
    )
