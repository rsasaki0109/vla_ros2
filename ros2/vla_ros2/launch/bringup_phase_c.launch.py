"""Phase C bring-up: runtime publishes actions; reference bridge parses only."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
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
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("vla_ros2"), "config", "bringup.dashcam.example.yaml"]
                ),
            ),
            SetEnvironmentVariable("PYTHONPATH", _pythonpath_for_vla_ros2()),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("vla_ros2"), "launch", "dummy.launch.py"]
                    )
                ),
                launch_arguments={
                    "params_file": params_file,
                    "dry_run": "true",
                    "publish_actions_in_dry_run": "true",
                    "instruction_msg_type": "string",
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("vla_ros2"), "launch", "controller_bridge.launch.py"]
                    )
                ),
                launch_arguments={
                    "enable_actuation": "false",
                    "publish_cmd_vel": "false",
                }.items(),
            ),
        ]
    )
