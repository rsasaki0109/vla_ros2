"""Launch test: smoke.launch.py publishes VLAAction and VLAStatus.

Requires a sourced ROS2 workspace (vla_ros2_msgs + vla_ros2 built). The pip
package or repo src/ must be on PYTHONPATH so vla_runtime_node can import adapters.
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
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from vla_ros2_msgs.msg import VLAAction, VLAStatus
from vla_ros2_ros.qos import action_qos, status_qos

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
    smoke_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("vla_ros2"), "launch", "smoke.launch.py"])
        ),
        launch_arguments={
            "control_hz": "5.0",
            "publish_hz": "5.0",
            "dry_run": "true",
            "publish_actions_in_dry_run": "true",
        }.items(),
    )
    return launch.LaunchDescription(
        [
            SetEnvironmentVariable("PYTHONPATH", _pythonpath_for_vla_ros2()),
            smoke_launch,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestSmokeLaunch(unittest.TestCase):
    """Spin smoke.launch.py and assert the runtime graph publishes outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()
        cls._node = rclpy.create_node("vla_ros2_smoke_test_listener")
        cls._actions: list[VLAAction] = []
        cls._statuses: list[VLAStatus] = []
        cls._node.create_subscription(
            VLAAction,
            "/vla/action",
            lambda msg: cls._actions.append(msg),
            action_qos(10),
        )
        cls._node.create_subscription(
            VLAStatus,
            "/vla/status",
            lambda msg: cls._statuses.append(msg),
            status_qos(10),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._node.destroy_node()

    def test_action_and_status_published(self) -> None:
        deadline = time.time() + 20.0
        while time.time() < deadline and not self._actions:
            rclpy.spin_once(self._node, timeout_sec=0.2)

        self.assertGreater(len(self._actions), 0, "expected at least one /vla/action message")
        self.assertEqual(self._actions[0].model_name, "dummy")
        self.assertEqual(self._actions[0].adapter_name, "DummyAdapter")
        self.assertGreater(len(self._actions[0].data), 0)

        status_deadline = time.time() + 5.0
        while time.time() < status_deadline and not any(status.ready for status in self._statuses):
            rclpy.spin_once(self._node, timeout_sec=0.2)

        self.assertTrue(
            any(status.ready for status in self._statuses),
            "expected at least one ready /vla/status message",
        )


@launch_testing.post_shutdown_test()
class TestSmokeLaunchShutdown(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def test_exit_codes(self, proc_info: launch_testing.ProcInfo) -> None:
        launch_testing.asserts.assertExitCodes(
            proc_info,
            allowable_exit_codes=[0, -2, -15],
        )
