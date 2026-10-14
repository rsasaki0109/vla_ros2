"""Runtime + proprioceptive success detector + task sequencer, wired together.

    ros2 launch vla_ros2 task_pipeline.launch.py \\
        steps_json:='["Put the red cube on the blue cube.", "Put the green cube on the red cube."]'
    ros2 launch vla_ros2 task_pipeline.launch.py planner:=hf \\
        goal:="Build a tower with blue at the bottom, then green, then red on top."

The sequencer publishes each step as VLAInstruction (with task_id), the runtime
acts on it, vla_state_success_node judges it from joint_states, and the
sequencer advances. dry_run defaults to true like every other launch file.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    def arg(name: str) -> LaunchConfiguration:
        return LaunchConfiguration(name)

    def text(name: str) -> ParameterValue:
        # launch_ros YAML-parses string arguments: '["a"]' would become a list and ""
        # nothing, breaking the nodes' string parameters. Keep these as strings.
        return ParameterValue(LaunchConfiguration(name), value_type=str)

    return LaunchDescription(
        [
            # Either an explicit step list, or a goal (used as-is by planner:=static,
            # broken into steps by a local LLM with planner:=hf_tower or planner:=hf).
            DeclareLaunchArgument("steps_json", default_value=""),
            DeclareLaunchArgument("goal", default_value="Put the red cube on the blue cube."),
            DeclareLaunchArgument("planner", default_value="static"),
            DeclareLaunchArgument("planner_model", default_value="Qwen/Qwen2.5-1.5B-Instruct"),
            DeclareLaunchArgument("planner_device", default_value="auto"),
            DeclareLaunchArgument("model_name", default_value="dummy"),
            DeclareLaunchArgument("pretrained", default_value="lerobot/smolvla_base"),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("model_kwargs_json", default_value=""),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("publish_actions_in_dry_run", default_value="false"),
            DeclareLaunchArgument("control_hz", default_value="5.0"),
            DeclareLaunchArgument("async_chunks", default_value="false"),
            DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("state_config_json", default_value=""),
            DeclareLaunchArgument("step_timeout_sec", default_value="0.0"),
            DeclareLaunchArgument("max_retries", default_value="0"),
            Node(
                package="vla_ros2",
                executable="vla_runtime_node",
                name="vla_runtime_node",
                output="screen",
                parameters=[
                    {
                        "model_name": arg("model_name"),
                        "runtime": "local",
                        "pretrained": text("pretrained"),
                        "device": text("device"),
                        "model_kwargs_json": text("model_kwargs_json"),
                        "dry_run": arg("dry_run"),
                        "publish_actions_in_dry_run": arg("publish_actions_in_dry_run"),
                        "control_hz": arg("control_hz"),
                        "async_chunks": arg("async_chunks"),
                        "instruction_msg_type": "vla_instruction",
                        "joint_state_topic": arg("joint_state_topic"),
                        "image_topic": arg("image_topic"),
                    }
                ],
            ),
            Node(
                package="vla_ros2",
                executable="vla_state_success_node",
                name="vla_state_success_node",
                output="screen",
                parameters=[
                    {
                        "instruction_msg_type": "vla_instruction",
                        "joint_state_topic": arg("joint_state_topic"),
                        "config_json": text("state_config_json"),
                    }
                ],
            ),
            Node(
                package="vla_ros2",
                executable="vla_task_sequencer_node",
                name="vla_task_sequencer_node",
                output="screen",
                parameters=[
                    {
                        "steps_json": text("steps_json"),
                        "goal": text("goal"),
                        "planner": text("planner"),
                        "planner_model": text("planner_model"),
                        "planner_device": text("planner_device"),
                        "step_timeout_sec": arg("step_timeout_sec"),
                        "max_retries": arg("max_retries"),
                    }
                ],
            ),
        ]
    )
