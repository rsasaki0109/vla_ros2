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
    control_hz = LaunchConfiguration("control_hz")
    device = LaunchConfiguration("device")
    pretrained = LaunchConfiguration("pretrained")
    gz_args = LaunchConfiguration("gz_args")
    action_blend = LaunchConfiguration("action_blend")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("vla_ros2_gz"), "config", "gz_smolvla.yaml"]
                ),
            ),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("publish_actions_in_dry_run", default_value="true"),
            DeclareLaunchArgument("enable_actuation", default_value="false"),
            DeclareLaunchArgument("control_hz", default_value="2.0"),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("pretrained", default_value="lerobot/smolvla_base"),
            DeclareLaunchArgument("gz_args", default_value="-s"),
            DeclareLaunchArgument("action_blend", default_value="0.35"),
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
                        "model_name": "smolvla",
                        "runtime": "local",
                        "dry_run": dry_run,
                        "publish_actions_in_dry_run": publish_actions_in_dry_run,
                        "control_hz": control_hz,
                        "require_image": True,
                        "device": device,
                        "pretrained": pretrained,
                    },
                ],
            ),
            Node(
                package="vla_ros2",
                executable="vla_smolvla_input_node",
                name="vla_smolvla_input_node",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="vla_ros2_gz",
                executable="vla_smolvla_joint_bridge_node",
                name="vla_smolvla_joint_bridge_node",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "enable_actuation": enable_actuation,
                        "action_blend": action_blend,
                    },
                ],
            ),
        ]
    )
