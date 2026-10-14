"""Publish a list of instructions one after another, advancing on task success.

Publishes the current step as ``VLAInstruction`` (text + task_id) and re-sends it
at ``republish_hz`` so the runtime's stale-instruction watchdog stays satisfied.
Advances when ``/vla/task_status`` reports success for the current task_id
(``vla_state_success_node`` / ``vla_success_detector_node`` with
``instruction_msg_type: vla_instruction`` echo it). Progress is published as JSON
on ``sequence_status_topic``. Never publishes actions.
"""

from __future__ import annotations

import json
from time import monotonic
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vla_ros2_msgs.msg import VLAInstruction as VLAInstructionMsg
from vla_ros2_msgs.msg import VLATaskStatus as VLATaskStatusMsg

from vla_ros2.planner import load_planner
from vla_ros2.sequencer import SequenceStep, TaskSequencer, resolve_steps
from vla_ros2_ros.qos import instruction_qos, status_qos


class TaskSequencerNode(Node):
    def __init__(self) -> None:
        super().__init__("vla_task_sequencer_node")
        self.declare_parameter("steps", [""])
        # Same as `steps` but as a JSON array string; easier to pass through launch args.
        self.declare_parameter("steps_json", "")
        # With no steps, plan them from `goal` at startup (vla_ros2.planner). The default
        # "static" planner uses the goal itself as the only step; "hf_tower" asks a local
        # LLM for a tower's colors bottom to top and writes the steps; "hf" asks for steps.
        self.declare_parameter("goal", "")
        self.declare_parameter("planner", "static")
        self.declare_parameter("planner_model", "Qwen/Qwen2.5-1.5B-Instruct")
        self.declare_parameter("planner_device", "auto")
        self.declare_parameter("task_id_prefix", "seq")
        self.declare_parameter("instruction_topic", "/vla/instruction")
        self.declare_parameter("task_status_topic", "/vla/task_status")
        self.declare_parameter("sequence_status_topic", "/vla/sequence_status")
        self.declare_parameter("republish_hz", 1.0)
        self.declare_parameter("step_timeout_sec", 0.0)
        self.declare_parameter("max_retries", 0)

        def value(name: str) -> Any:
            return self.get_parameter(name).value

        def plan(goal: str) -> list[str]:
            planner = load_planner(
                str(value("planner")),
                pretrained=str(value("planner_model")),
                device=str(value("planner_device")),
            )
            planned = planner.plan(goal)
            self.get_logger().info(f"planned {len(planned)} steps for {goal!r}: {planned}")
            return planned

        steps, self._steps_source = resolve_steps(
            steps=[str(step) for step in value("steps")],
            steps_json=str(value("steps_json")),
            goal=str(value("goal")),
            plan=plan,
        )
        self._goal = str(value("goal"))
        self._sequencer = TaskSequencer(
            steps,
            task_id_prefix=str(value("task_id_prefix")),
            step_timeout_sec=float(value("step_timeout_sec")),
            max_retries=int(value("max_retries")),
        )
        self._instruction_pub = self.create_publisher(
            VLAInstructionMsg, str(value("instruction_topic")), instruction_qos(10)
        )
        self._sequence_pub = self.create_publisher(
            String, str(value("sequence_status_topic")), status_qos(10)
        )
        self.create_subscription(
            VLATaskStatusMsg, str(value("task_status_topic")), self._status_cb, status_qos(10)
        )
        self._publish_step(self._sequencer.start(monotonic()))
        self.create_timer(1.0 / max(1e-3, float(value("republish_hz"))), self._tick)
        self.get_logger().info(f"vla_task_sequencer_node: {len(self._sequencer.steps)} steps")

    def _publish_step(self, step: SequenceStep | None) -> None:
        if step is None:
            return
        msg = VLAInstructionMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.text = step.instruction
        msg.task_id = step.task_id
        msg.metadata_json = json.dumps(
            {"source": "vla_task_sequencer_node", "step": step.index, "attempt": step.attempt}
        )
        self._instruction_pub.publish(msg)

    def _publish_progress(self) -> None:
        progress = {
            **self._sequencer.to_dict(),
            "goal": self._goal,
            "plan_source": self._steps_source,
            "plan": list(self._sequencer.steps),
        }
        self._sequence_pub.publish(String(data=json.dumps(progress)))

    def _status_cb(self, msg: VLATaskStatusMsg) -> None:
        if not msg.success:
            return
        state_before = self._sequencer.state
        next_step = self._sequencer.on_success(msg.task_id, monotonic())
        if next_step is not None:
            self.get_logger().info(f"step done; next: {next_step.instruction!r}")
            self._publish_step(next_step)
            self._publish_progress()
        elif state_before == "running" and self._sequencer.state == "done":
            self.get_logger().info("sequence done")
            self._publish_progress()

    def _tick(self) -> None:
        retried = self._sequencer.tick(monotonic())
        if retried is not None:
            self.get_logger().warning(f"step timed out; retry {retried.attempt}")
        if self._sequencer.state == "failed":
            self.get_logger().error("sequence failed: step timed out", throttle_duration_sec=10.0)
        self._publish_step(self._sequencer.current)
        self._publish_progress()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TaskSequencerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
