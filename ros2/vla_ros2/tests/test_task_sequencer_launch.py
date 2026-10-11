"""Launch test: the sequencer walks two steps, advancing on grasp -> release success.

vla_task_sequencer_node publishes step instructions; vla_state_success_node judges
each step from joint_states. The test plays a synthetic pick-and-place per step
and checks: step 0 -> step 1 only after the first release, then "done".
"""

from __future__ import annotations

import json
import os
import time
import unittest
from collections.abc import Callable
from pathlib import Path

import launch
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest
import rclpy
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from vla_ros2_msgs.msg import VLAInstruction
from vla_ros2_ros.qos import action_qos, instruction_qos, status_qos

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
_STEPS = ["Put the red cube on the blue cube.", "Put the green cube on the red cube."]
_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_PICK_AND_PLACE = [
    (0.0, 25.0, 1.0),
    (25.0, 13.0, 0.5),
    (13.0, 13.0, 1.5),
    (13.0, 20.0, 0.5),
    (20.0, 0.0, 0.5),
    (0.0, 0.0, 1.0),
]


def _pythonpath_for_vla_ros2() -> str:
    src = str(_REPO_SRC)
    existing = os.environ.get("PYTHONPATH", "")
    if not existing:
        return src
    if src in existing.split(os.pathsep):
        return existing
    return os.pathsep.join((src, existing))


@pytest.mark.launch_test
def generate_test_description() -> launch.LaunchDescription:
    return launch.LaunchDescription(
        [
            SetEnvironmentVariable("PYTHONPATH", _pythonpath_for_vla_ros2()),
            Node(
                package="vla_ros2",
                executable="vla_task_sequencer_node",
                name="vla_task_sequencer_node",
                parameters=[{"steps": _STEPS, "task_id_prefix": "t", "republish_hz": 2.0}],
            ),
            Node(
                package="vla_ros2",
                executable="vla_state_success_node",
                name="vla_state_success_node",
                parameters=[{"instruction_msg_type": "vla_instruction", "publish_hz": 10.0}],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestTaskSequencer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()
        cls._node = rclpy.create_node("vla_task_sequencer_test")
        cls._instructions: list[VLAInstruction] = []
        cls._progress: list[dict] = []
        cls._node.create_subscription(
            VLAInstruction, "/vla/instruction", cls._instructions.append, instruction_qos(10)
        )
        cls._node.create_subscription(
            String,
            "/vla/sequence_status",
            lambda msg: cls._progress.append(json.loads(msg.data)),
            status_qos(10),
        )
        cls._joint_pub = cls._node.create_publisher(JointState, "/joint_states", action_qos(50))
        cls._arm = 0.0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._node.destroy_node()

    def _spin_until(self, condition: Callable[[], bool], timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if condition():
                return True
            rclpy.spin_once(self._node, timeout_sec=0.02)
        return condition()

    def _play_pick_and_place(self) -> None:
        for start, end, seconds in _PICK_AND_PLACE:
            steps = int(seconds * 30)
            for k in range(steps):
                gripper = start + (end - start) * k / steps
                if start != end:
                    type(self)._arm += 0.5
                msg = JointState()
                msg.header.stamp = self._node.get_clock().now().to_msg()
                msg.name = list(_JOINTS)
                msg.position = [self._arm] * 5 + [gripper]
                self._joint_pub.publish(msg)
                deadline = time.monotonic() + 1.0 / 30
                while time.monotonic() < deadline:
                    rclpy.spin_once(self._node, timeout_sec=0.005)

    def _latest_task_id(self) -> str:
        return self._instructions[-1].task_id if self._instructions else ""

    def test_walks_both_steps_then_done(self) -> None:
        self.assertTrue(self._spin_until(lambda: self._latest_task_id() == "t-0-0", 20.0))
        self.assertEqual(self._instructions[-1].text, _STEPS[0])
        self._spin_until(lambda: False, 1.5)  # let the success node receive step 0
        self.assertNotIn("t-1-0", {msg.task_id for msg in self._instructions})

        self._play_pick_and_place()
        self.assertTrue(
            self._spin_until(lambda: self._latest_task_id() == "t-1-0", 5.0),
            f"did not advance; progress={self._progress[-1:]}",
        )
        self.assertEqual(self._instructions[-1].text, _STEPS[1])
        self._spin_until(lambda: False, 1.5)

        self._play_pick_and_place()
        self.assertTrue(
            self._spin_until(lambda: any(p["state"] == "done" for p in self._progress), 5.0),
            f"did not finish; progress={self._progress[-1:]}",
        )
        done = next(p for p in self._progress if p["state"] == "done")
        self.assertEqual(done["completed"], ["t-0-0", "t-1-0"])


@launch_testing.post_shutdown_test()
class TestTaskSequencerShutdown(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def test_exit_codes(self, proc_info: launch_testing.ProcInfo) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
