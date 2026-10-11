"""Launch test: vla_state_success_node reports success only after grasp -> release.

The test itself publishes an instruction and a synthetic SO-100 pick-and-place
joint trajectory at 30 Hz, and checks the published VLATaskStatus: not done while
holding, done after the release.
"""

from __future__ import annotations

import os
import time
import unittest
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
from vla_ros2_msgs.msg import VLATaskStatus
from vla_ros2_ros.qos import action_qos, instruction_qos, status_qos

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# (gripper start, gripper end, seconds); the arm moves whenever the gripper does.
_SEGMENTS = [
    (0.0, 0.0, 0.5),
    (0.0, 25.0, 1.0),
    (25.0, 13.0, 0.5),
    (13.0, 13.0, 2.0),
    (13.0, 20.0, 0.5),
    (20.0, 0.0, 0.5),
    (0.0, 0.0, 1.0),
]
_HOLD_ENDS_SEC = 4.0


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
                executable="vla_state_success_node",
                name="vla_state_success_node",
                parameters=[{"publish_hz": 10.0}],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestStateSuccess(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()
        cls._node = rclpy.create_node("vla_state_success_test")
        cls._statuses: list[tuple[float, VLATaskStatus]] = []
        cls._node.create_subscription(
            VLATaskStatus,
            "/vla/task_status",
            lambda msg: cls._statuses.append((time.monotonic(), msg)),
            status_qos(50),
        )
        cls._instruction_pub = cls._node.create_publisher(
            String, "/vla/instruction", instruction_qos(10)
        )
        cls._joint_pub = cls._node.create_publisher(JointState, "/joint_states", action_qos(50))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._node.destroy_node()

    def _spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self._node, timeout_sec=0.01)

    def test_success_only_after_release(self) -> None:
        self._instruction_pub.publish(String(data="Put the red cube on top of the blue cube."))
        self._spin_for(2.0)  # discovery + instruction delivery
        start = time.monotonic()
        arm = 0.0
        for gripper_start, gripper_end, seconds in _SEGMENTS:
            steps = int(seconds * 30)
            for k in range(steps):
                gripper = gripper_start + (gripper_end - gripper_start) * k / steps
                if gripper_start != gripper_end:
                    arm += 0.5
                msg = JointState()
                msg.header.stamp = self._node.get_clock().now().to_msg()
                msg.name = list(_JOINTS)
                msg.position = [arm, arm, arm, arm, arm, gripper]
                self._joint_pub.publish(msg)
                self._spin_for(1.0 / 30)
        self._spin_for(1.0)

        during_hold = [
            msg for t, msg in self._statuses if t - start < _HOLD_ENDS_SEC and msg.instruction
        ]
        self.assertTrue(during_hold, "expected status messages during the trajectory")
        self.assertFalse(any(msg.success for msg in during_hold), "fired before the release")
        self.assertIn("holding", {msg.answer_text for msg in during_hold})
        final = self._statuses[-1][1]
        self.assertTrue(final.success, f"expected success, last phase={final.answer_text}")
        self.assertEqual(final.answer_text, "done")


@launch_testing.post_shutdown_test()
class TestStateSuccessShutdown(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def test_exit_codes(self, proc_info: launch_testing.ProcInfo) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
