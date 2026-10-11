"""Launch test: task_pipeline.launch.py runs a two-step task end to end.

Runtime (dummy adapter) + vla_state_success_node + vla_task_sequencer_node from
the launch file. The test plays a synthetic pick-and-place per step and checks
that the runtime's actions carry step 0's instruction until the first release,
step 1's after it, and that the sequence finishes.
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
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from vla_ros2_msgs.msg import VLAAction
from vla_ros2_ros.qos import action_qos, status_qos

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
    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("vla_ros2"), "launch", "task_pipeline.launch.py"]
            )
        ),
        launch_arguments={
            "steps_json": json.dumps(_STEPS),
            "publish_actions_in_dry_run": "true",
            "control_hz": "10.0",
        }.items(),
    )
    return launch.LaunchDescription(
        [
            SetEnvironmentVariable("PYTHONPATH", _pythonpath_for_vla_ros2()),
            pipeline,
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestTaskPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()
        cls._node = rclpy.create_node("vla_task_pipeline_test")
        cls._acted_on: list[str] = []
        cls._progress: list[dict] = []
        cls._node.create_subscription(
            VLAAction,
            "/vla/action",
            lambda msg: cls._acted_on.append(json.loads(msg.metadata_json).get("instruction", "")),
            action_qos(50),
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

    def test_runtime_follows_the_sequence(self) -> None:
        self.assertTrue(
            self._spin_until(lambda: _STEPS[0] in self._acted_on, 30.0),
            f"runtime never acted on step 0; saw {set(self._acted_on)}",
        )
        self._spin_until(lambda: False, 1.0)
        self.assertNotIn(_STEPS[1], self._acted_on)

        self._play_pick_and_place()
        self.assertTrue(
            self._spin_until(lambda: self._acted_on[-1:] == [_STEPS[1]], 5.0),
            f"runtime did not switch to step 1; progress={self._progress[-1:]}",
        )

        self._play_pick_and_place()
        self.assertTrue(
            self._spin_until(lambda: any(p["state"] == "done" for p in self._progress), 5.0),
            f"sequence did not finish; progress={self._progress[-1:]}",
        )


@launch_testing.post_shutdown_test()
class TestTaskPipelineShutdown(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def test_exit_codes(self, proc_info: launch_testing.ProcInfo) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
