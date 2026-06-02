from __future__ import annotations

import numpy as np

from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk
from vla_zoo.runtime.guard import (
    ActionClipGuard,
    WatchdogConfig,
    clip_action_report,
    evaluate_watchdog,
)


def _spec(low: tuple[float, ...] | None, high: tuple[float, ...] | None) -> ActionSpec:
    return ActionSpec(action_space="eef_delta", shape=(3,), low=low, high=high)


def _action(values: list[float], *, low=(-1.0, -1.0, -1.0), high=(1.0, 1.0, 1.0)) -> VLAAction:
    return VLAAction(data=np.asarray(values, dtype=np.float32), spec=_spec(low, high))


def test_clip_report_counts_clamped_elements() -> None:
    report = clip_action_report(_action([2.0, 0.5, -3.0]))

    assert report.clipped is True
    assert report.clipped_elements == 2  # 2.0 and -3.0 are out of [-1, 1]
    assert report.total_elements == 3
    np.testing.assert_allclose(report.action.to_numpy(), [1.0, 0.5, -1.0])
    assert report.action.metadata["was_clipped"] is True


def test_clip_report_no_bounds_is_noop() -> None:
    report = clip_action_report(_action([5.0, 5.0, 5.0], low=None, high=None))

    assert report.clipped is False
    assert report.clipped_elements == 0
    np.testing.assert_allclose(report.action.to_numpy(), [5.0, 5.0, 5.0])


def test_clip_override_broadcasts_single_value() -> None:
    report = clip_action_report(
        _action([0.5, 0.9, 0.2], low=None, high=None),
        action_low=(0.0,),
        action_high=(0.3,),
    )

    np.testing.assert_allclose(report.action.to_numpy(), [0.3, 0.3, 0.2])
    assert report.clipped_elements == 2


def test_clip_guard_accumulates_clip_rate() -> None:
    guard = ActionClipGuard()
    guard.clip(_action([2.0, 0.0, 0.0]))  # clipped
    guard.clip(_action([0.0, 0.0, 0.0]))  # not clipped

    assert guard.total_actions == 2
    assert guard.clipped_actions == 1
    assert guard.action_clip_rate == 0.5
    assert guard.clipped_elements == 1
    assert guard.total_elements == 6
    payload = guard.to_dict()
    assert payload["clipped_actions"] == 1


def test_clip_guard_handles_chunks() -> None:
    guard = ActionClipGuard()
    chunk = VLAActionChunk(actions=[_action([2.0, 0.0, 0.0]), _action([0.0, 5.0, 0.0])])
    result = guard.clip(chunk)

    assert isinstance(result, VLAActionChunk)
    assert guard.total_actions == 2
    assert guard.clipped_actions == 2
    assert result.metadata["clip_actions"] is True


def test_watchdog_waits_for_image_when_required() -> None:
    status = evaluate_watchdog(image_age_sec=None, instruction_age_sec=0.1)

    assert status.ok is False
    assert status.reason == "waiting for image"


def test_watchdog_flags_stale_image() -> None:
    status = evaluate_watchdog(
        image_age_sec=2.5,
        instruction_age_sec=0.1,
        config=WatchdogConfig(stale_image_timeout_sec=1.0),
    )

    assert status.ok is False
    assert status.reason == "stale image: 2.50s"


def test_watchdog_flags_stale_instruction() -> None:
    status = evaluate_watchdog(
        image_age_sec=0.1,
        instruction_age_sec=9.0,
        config=WatchdogConfig(stale_instruction_timeout_sec=5.0),
    )

    assert status.ok is False
    assert status.reason == "stale instruction: 9.00s"


def test_watchdog_ok_when_fresh() -> None:
    status = evaluate_watchdog(image_age_sec=0.1, instruction_age_sec=0.2)

    assert status.ok is True
    assert status.reason is None


def test_watchdog_timeout_zero_disables_check() -> None:
    status = evaluate_watchdog(
        image_age_sec=100.0,
        instruction_age_sec=100.0,
        config=WatchdogConfig(stale_image_timeout_sec=0.0, stale_instruction_timeout_sec=0.0),
    )

    assert status.ok is True
