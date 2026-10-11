from __future__ import annotations

import numpy as np

from vla_ros2.task_success import GraspReleaseConfig, GraspReleaseDetector, detect_success_time


def _trajectory(hz: float, *, segments: list[tuple[float, float, float]]) -> tuple[list, list]:
    """Piecewise-linear gripper (start, end, seconds); arm moves while the gripper moves."""

    states, stamps, t = [], [], 0.0
    arm = np.zeros(5)
    for start, end, seconds in segments:
        steps = max(1, int(seconds * hz))
        for k in range(steps):
            gripper = start + (end - start) * k / steps
            arm = arm + (0.5 if start != end else 0.0)
            states.append([*arm.tolist(), gripper])
            stamps.append(t)
            t += 1.0 / hz
    return states, stamps


PICK_AND_PLACE = [
    (0.0, 0.0, 1.0),  # rest, closed
    (0.0, 25.0, 1.0),  # open to approach
    (25.0, 13.0, 0.5),  # close onto the cube
    (13.0, 13.0, 2.0),  # holding
    (13.0, 20.0, 0.5),  # release
    (20.0, 0.0, 0.5),  # close at rest
    (0.0, 0.0, 1.0),
]


def test_pick_and_place_fires_after_release_at_any_rate() -> None:
    for hz in (10.0, 30.0, 100.0):
        states, stamps = _trajectory(hz, segments=PICK_AND_PLACE)
        fired = detect_success_time(states, stamps)
        assert fired is not None, hz
        assert fired > 5.0  # after hold (ends 5.0 s) and release began


def test_never_grasping_does_not_fire() -> None:
    states, stamps = _trajectory(30.0, segments=[(0.0, 0.0, 10.0)])
    assert detect_success_time(states, stamps) is None


def test_slowly_opening_through_hold_range_is_not_a_hold() -> None:
    # Opening slowly to 25 passes through 6-19 for seconds but never closed onto anything.
    states, stamps = _trajectory(
        30.0, segments=[(0.0, 25.0, 6.0), (25.0, 25.0, 2.0), (25.0, 0.0, 1.0), (0.0, 0.0, 2.0)]
    )
    assert detect_success_time(states, stamps) is None


def test_holding_without_release_is_not_success() -> None:
    states, stamps = _trajectory(30.0, segments=PICK_AND_PLACE[:4])
    detector = GraspReleaseDetector()
    for state, stamp in zip(states, stamps, strict=True):
        detector.update(state, stamp)
    assert detector.phase == "holding"
    assert detector.success is False
    assert detector.hold_value is not None and abs(detector.hold_value - 13.0) < 0.5


def test_reset_and_out_of_order_samples() -> None:
    states, stamps = _trajectory(30.0, segments=PICK_AND_PLACE)
    detector = GraspReleaseDetector(GraspReleaseConfig())
    for state, stamp in zip(states, stamps, strict=True):
        detector.update(state, stamp)
    assert detector.success
    assert detector.update(states[0], stamps[0]) is True  # stale sample ignored
    detector.reset()
    assert detector.phase == "searching" and not detector.success
