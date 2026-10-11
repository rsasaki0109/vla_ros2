"""Task-success detection from proprioception: grasp, release, settle.

On ``svla_so100_stacking`` the gripper goes rest(closed) -> open (approach) ->
a steady partly-closed value (the cube stops it closing: holding) -> open
(release) -> closed / arm still. A single frame cannot tell start from end (the
arm rests in both), but this sequence can: success is declared once a hold has
been followed by a release and the arm has settled.

This checks that a pick-and-place happened, not that the cube landed where the
instruction asked; pair it with a perception check when that matters.
Thresholds are in the robot's units (SO-100 LeRobot: degrees, gripper 0-30) and
times in seconds, so the detector works at any sample rate.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

PHASES = ("searching", "holding", "released", "done")


@dataclass(frozen=True)
class GraspReleaseConfig:
    gripper_index: int = 5
    arm_indices: tuple[int, ...] = (0, 1, 2, 3, 4)
    hold_low: float = 6.0
    """Gripper value range that counts as closed on an object."""
    hold_high: float = 19.0
    hold_sec: float = 1.0
    """The gripper must stay within ``hold_range`` for this long."""
    hold_range: float = 1.5
    approach_delta: float = 2.5
    """Before the hold, the gripper must have been this much more open (it closed onto
    something rather than drifting through the range while opening)."""
    release_delta: float = 3.0
    """Opening this much past the hold value counts as a release."""
    closed: float = 3.0
    """Gripper at or below this after a release counts as back at rest."""
    settle_speed: float = 15.0
    """Arm joint speed (units/s) below which the arm counts as still."""
    settle_sec: float = 0.5


class GraspReleaseDetector:
    """Streaming detector: feed (state, stamp) in order, read ``success``."""

    def __init__(self, config: GraspReleaseConfig | None = None) -> None:
        self.config = config or GraspReleaseConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = "searching"
        self.success = False
        self.hold_value: float | None = None
        self._window: deque[tuple[float, float]] = deque()
        self._peak_before_window = float("-inf")
        self._last_arm: tuple[float, ...] | None = None
        self._last_stamp: float | None = None
        self._settled_sec = 0.0

    def update(self, state: Sequence[float], stamp: float) -> bool:
        cfg = self.config
        gripper = float(state[cfg.gripper_index])
        arm = tuple(float(state[i]) for i in cfg.arm_indices)
        dt = None if self._last_stamp is None else stamp - self._last_stamp
        if dt is not None and dt <= 0:
            return self.success  # out-of-order or duplicate sample

        if self.phase == "searching":
            self._update_hold(gripper, stamp)
        elif self.phase == "holding":
            assert self.hold_value is not None
            if gripper >= self.hold_value + cfg.release_delta:
                self.phase = "released"
        elif self.phase == "released" and dt is not None and self._last_arm is not None:
            speed = max(abs(a - b) for a, b in zip(arm, self._last_arm, strict=True)) / dt
            self._settled_sec = self._settled_sec + dt if speed <= cfg.settle_speed else 0.0
            if gripper <= cfg.closed or self._settled_sec >= cfg.settle_sec:
                self.phase = "done"
                self.success = True

        self._last_arm = arm
        self._last_stamp = stamp
        return self.success

    def _update_hold(self, gripper: float, stamp: float) -> None:
        cfg = self.config
        self._window.append((stamp, gripper))
        # Keep the window at hold_sec; values leaving it feed the "was more open" peak.
        while len(self._window) > 1 and self._window[1][0] <= stamp - cfg.hold_sec:
            _, old = self._window.popleft()
            self._peak_before_window = max(self._peak_before_window, old)
        if stamp - self._window[0][0] < cfg.hold_sec:
            return
        values = [value for _, value in self._window]
        mean = sum(values) / len(values)
        if (
            cfg.hold_low <= mean <= cfg.hold_high
            and max(values) - min(values) <= cfg.hold_range
            and self._peak_before_window >= mean + cfg.approach_delta
        ):
            self.phase = "holding"
            self.hold_value = mean


def detect_success_time(
    states: Sequence[Sequence[float]],
    stamps: Sequence[float],
    config: GraspReleaseConfig | None = None,
) -> float | None:
    """Run a detector over a recorded trajectory; return the stamp it fired at, if any."""

    detector = GraspReleaseDetector(config)
    for state, stamp in zip(states, stamps, strict=True):
        if detector.update(state, float(stamp)):
            return float(stamp)
    return None
