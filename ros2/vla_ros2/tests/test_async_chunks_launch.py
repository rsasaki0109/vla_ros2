"""Launch test: async chunk execution keeps /vla/action flowing through inference.

Two runtime nodes see the same smoke input. Both run the dummy adapter with
20-step chunks and a simulated 0.5 s inference (latency_sec), at 10 Hz:

- sync  (async_chunks=false): one inference per published action
- async (async_chunks=true):  one buffered action per tick, next chunk inferred ahead

The async node must publish close to control_hz with no inference-sized gaps.
"""

from __future__ import annotations

import json
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
from vla_ros2_msgs.msg import VLAAction
from vla_ros2_ros.qos import action_qos

_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
_CONTROL_HZ = 10.0
# 0.5 s inference is ~6 ticks at 10 Hz; chunks must be >= 2x that to never starve.
_MODEL_KWARGS = json.dumps({"chunk_size": 20, "latency_sec": 0.5})


def _pythonpath_for_vla_ros2() -> str:
    src = str(_REPO_SRC)
    existing = os.environ.get("PYTHONPATH", "")
    if not existing:
        return src
    if src in existing.split(os.pathsep):
        return existing
    return os.pathsep.join((src, existing))


def _runtime(name: str, *, async_chunks: bool) -> Node:
    return Node(
        package="vla_ros2",
        executable="vla_runtime_node",
        name=name,
        parameters=[
            {
                "model_name": "dummy",
                "dry_run": True,
                "publish_actions_in_dry_run": True,
                "instruction_msg_type": "vla_instruction",
                "control_hz": _CONTROL_HZ,
                "stale_image_timeout_sec": 2.0,
                "async_chunks": async_chunks,
                "model_kwargs_json": _MODEL_KWARGS,
                "action_topic": f"/{name}/action",
                "action_chunk_topic": f"/{name}/action_chunk",
                "status_topic": f"/{name}/status",
                "diagnostics_topic": f"/{name}/diagnostics",
            }
        ],
    )


@pytest.mark.launch_test
def generate_test_description() -> launch.LaunchDescription:
    return launch.LaunchDescription(
        [
            SetEnvironmentVariable("PYTHONPATH", _pythonpath_for_vla_ros2()),
            Node(
                package="vla_ros2",
                executable="vla_smoke_input_node",
                name="vla_smoke_input_node",
                parameters=[{"publish_hz": 10.0}],
            ),
            _runtime("runtime_sync", async_chunks=False),
            _runtime("runtime_async", async_chunks=True),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def _rate_and_max_gap(stamps: list[float]) -> tuple[float, float]:
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    duration = stamps[-1] - stamps[0]
    return (len(gaps) / duration if duration > 0 else 0.0), max(gaps, default=0.0)


class TestAsyncChunks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()
        cls._node = rclpy.create_node("vla_async_chunk_test_listener")
        cls._stamps: dict[str, list[float]] = {"runtime_sync": [], "runtime_async": []}
        for name, stamps in cls._stamps.items():
            cls._node.create_subscription(
                VLAAction,
                f"/{name}/action",
                lambda _msg, stamps=stamps: stamps.append(time.monotonic()),
                action_qos(50),
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._node.destroy_node()

    def test_async_publishes_at_control_rate_through_inference(self) -> None:
        # Warm up until both nodes publish, then measure a fixed window; the async
        # node's first chunk (and its latency estimate) is excluded by the warm-up.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not all(
            len(stamps) >= 3 for stamps in self._stamps.values()
        ):
            rclpy.spin_once(self._node, timeout_sec=0.1)
        for stamps in self._stamps.values():
            stamps.clear()
        window_end = time.monotonic() + 6.0
        while time.monotonic() < window_end:
            rclpy.spin_once(self._node, timeout_sec=0.05)

        sync_rate, sync_gap = _rate_and_max_gap(self._stamps["runtime_sync"])
        async_rate, async_gap = _rate_and_max_gap(self._stamps["runtime_async"])
        report = (
            f"sync {sync_rate:.2f} Hz (max gap {sync_gap:.2f} s) vs "
            f"async {async_rate:.2f} Hz (max gap {async_gap:.2f} s)"
        )
        print(report)
        self.assertGreater(len(self._stamps["runtime_async"]), 10, report)
        self.assertGreater(async_rate, 0.7 * _CONTROL_HZ, report)
        self.assertLess(async_gap, 0.35, report)  # no 0.5 s inference stall
        self.assertGreater(async_rate, 2.0 * sync_rate, report)


@launch_testing.post_shutdown_test()
class TestAsyncChunksShutdown(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def test_exit_codes(self, proc_info: launch_testing.ProcInfo) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
