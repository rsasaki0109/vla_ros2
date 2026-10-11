"""Launch test: vla_success_detector_node publishes VLATaskStatus.

Uses the always-available dummy VLM, so no weights are needed. Two detectors run
against the smoke input node: one through success_detector.launch.py (dummy
answers "no"), one started directly with dummy_answer "yes" and
compare_with_start, which must latch success after required_consecutive checks.
"""

from __future__ import annotations

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
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from vla_ros2_msgs.msg import VLATaskStatus
from vla_ros2_ros.qos import status_qos

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"


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
    detector_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("vla_ros2"), "launch", "success_detector.launch.py"]
            )
        ),
        launch_arguments={
            "vlm_name": "dummy",
            "check_hz": "4.0",
            "instruction_msg_type": "vla_instruction",
        }.items(),
    )
    return launch.LaunchDescription(
        [
            SetEnvironmentVariable("PYTHONPATH", _pythonpath_for_vla_ros2()),
            Node(
                package="vla_ros2",
                executable="vla_smoke_input_node",
                name="vla_smoke_input_node",
                parameters=[{"publish_hz": 5.0}],
            ),
            detector_launch,
            Node(
                package="vla_ros2",
                executable="vla_success_detector_node",
                name="vla_success_detector_yes",
                parameters=[
                    {
                        "vlm_name": "dummy",
                        "dummy_answer": "yes",
                        "check_hz": 4.0,
                        "required_consecutive": 2,
                        "compare_with_start": True,
                        "instruction_msg_type": "vla_instruction",
                        "task_status_topic": "/vla/task_status_yes",
                    }
                ],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestSuccessDetectorLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()
        cls._node = rclpy.create_node("vla_success_detector_test_listener")
        cls._no: list[VLATaskStatus] = []
        cls._yes: list[VLATaskStatus] = []
        cls._node.create_subscription(
            VLATaskStatus, "/vla/task_status", lambda msg: cls._no.append(msg), status_qos(10)
        )
        cls._node.create_subscription(
            VLATaskStatus, "/vla/task_status_yes", lambda msg: cls._yes.append(msg), status_qos(10)
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._node.destroy_node()

    def _spin_until(self, condition: Callable[[], bool], timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline and not condition():
            rclpy.spin_once(self._node, timeout_sec=0.2)

    def test_launch_file_detector_reports_not_done(self) -> None:
        self._spin_until(lambda: len(self._no) >= 2, 30.0)
        self.assertGreaterEqual(len(self._no), 2, "expected /vla/task_status messages")
        last = self._no[-1]
        self.assertEqual(last.vlm_name, "dummy")
        self.assertFalse(last.success)
        self.assertFalse(last.last_check_success)
        self.assertEqual(last.instruction, "pick up the red block")

    def test_yes_detector_latches_success_with_start_reference(self) -> None:
        self._spin_until(lambda: any(msg.success for msg in self._yes), 30.0)
        self.assertTrue(any(msg.success for msg in self._yes), "expected latched success")
        first_success = next(msg for msg in self._yes if msg.success)
        self.assertGreaterEqual(first_success.consecutive_success, 2)
        self.assertIn('"compare": true', first_success.metadata_json)


@launch_testing.post_shutdown_test()
class TestSuccessDetectorShutdown(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def test_exit_codes(self, proc_info: launch_testing.ProcInfo) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
