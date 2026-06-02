"""Pure, framework-agnostic runtime safety guards: action clipping + staleness watchdog.

These guards are the single source of truth for two robot-readiness safety layers. They
are deliberately free of ROS2/numpy-runtime side effects beyond array math so they can be
unit-tested directly and reused by the ROS2 node. The core never actuates motors; these
guards only shape/flag the action stream and report counters for diagnostics/JSONL.

- Action clipping clamps each action to the adapter's declared ``low``/``high`` (or a
  configured override) and reports a per-element and per-action clip rate.
- The staleness watchdog flags stale image/instruction inputs from their ages, returning
  the same status text the ROS2 runtime publishes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk


@dataclass(frozen=True)
class ClipReport:
    """Result of clipping a single action: the clipped action plus clip counts."""

    action: VLAAction
    clipped: bool
    clipped_elements: int
    total_elements: int


def _resolve_bound(
    spec: ActionSpec,
    *,
    bound: str,
    override: tuple[float, ...] | None,
    flat_size: int,
) -> np.ndarray | None:
    """Resolve a low/high bound array, preferring a configured override over the spec.

    A length-1 override broadcasts across the action. An override whose length matches
    neither 1 nor the flattened action size is ignored (returns the spec bound, or None).
    """

    spec_bound = spec.low if bound == "low" else spec.high
    if override:
        if len(override) == 1:
            return np.full(spec.shape, override[0], dtype=np.float32)
        if len(override) == flat_size:
            return np.asarray(override, dtype=np.float32).reshape(spec.shape)
        # invalid override length: fall back to the declared spec bound
    if spec_bound is None:
        return None
    return np.asarray(spec_bound, dtype=np.float32).reshape(spec.shape)


def clip_action_report(
    action: VLAAction,
    *,
    action_low: tuple[float, ...] | None = None,
    action_high: tuple[float, ...] | None = None,
) -> ClipReport:
    """Clip an action to its bounds and report how many elements were clamped."""

    flat_size = int(action.data.size)
    low = _resolve_bound(action.spec, bound="low", override=action_low, flat_size=flat_size)
    high = _resolve_bound(action.spec, bound="high", override=action_high, flat_size=flat_size)
    if low is None and high is None:
        return ClipReport(
            action=action, clipped=False, clipped_elements=0, total_elements=flat_size
        )

    data = action.to_numpy()
    clipped_data = np.clip(data, low, high).astype(np.float32)
    clipped_elements = int(np.count_nonzero(clipped_data != data))
    was_clipped = clipped_elements > 0
    new_action = VLAAction(
        data=clipped_data,
        spec=action.spec,
        dt=action.dt,
        confidence=action.confidence,
        chunk_index=action.chunk_index,
        metadata={**action.metadata, "clip_actions": True, "was_clipped": was_clipped},
    )
    return ClipReport(
        action=new_action,
        clipped=was_clipped,
        clipped_elements=clipped_elements,
        total_elements=flat_size,
    )


@dataclass
class ActionClipGuard:
    """Stateful action-clipping guard that accumulates clip-rate counters."""

    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None
    total_actions: int = 0
    clipped_actions: int = 0
    clipped_elements: int = 0
    total_elements: int = 0

    def _clip_one(self, action: VLAAction) -> VLAAction:
        report = clip_action_report(
            action, action_low=self.action_low, action_high=self.action_high
        )
        self.total_actions += 1
        self.total_elements += report.total_elements
        self.clipped_elements += report.clipped_elements
        if report.clipped:
            self.clipped_actions += 1
        return report.action

    def clip(self, prediction: VLAAction | VLAActionChunk) -> VLAAction | VLAActionChunk:
        """Clip an action or every action in a chunk, updating the counters."""

        if isinstance(prediction, VLAActionChunk):
            clipped = [self._clip_one(action) for action in prediction.actions]
            return VLAActionChunk(
                actions=clipped,
                metadata={**prediction.metadata, "clip_actions": True},
            )
        return self._clip_one(prediction)

    @property
    def action_clip_rate(self) -> float:
        return self.clipped_actions / self.total_actions if self.total_actions else 0.0

    @property
    def element_clip_rate(self) -> float:
        return self.clipped_elements / self.total_elements if self.total_elements else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_actions": self.total_actions,
            "clipped_actions": self.clipped_actions,
            "clipped_elements": self.clipped_elements,
            "total_elements": self.total_elements,
            "action_clip_rate": self.action_clip_rate,
            "element_clip_rate": self.element_clip_rate,
        }


@dataclass(frozen=True)
class WatchdogConfig:
    """Staleness watchdog thresholds. A timeout of 0 disables that check."""

    require_image: bool = True
    stale_image_timeout_sec: float = 1.0
    stale_instruction_timeout_sec: float = 5.0


#: Shared default watchdog config (frozen, so safe to reuse as a function default).
DEFAULT_WATCHDOG_CONFIG = WatchdogConfig()


@dataclass(frozen=True)
class WatchdogStatus:
    """Outcome of a staleness check: whether inputs are fresh enough to run."""

    ok: bool
    reason: str | None
    image_age_sec: float | None = None
    instruction_age_sec: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "image_age_sec": self.image_age_sec,
            "instruction_age_sec": self.instruction_age_sec,
        }


def evaluate_watchdog(
    *,
    image_age_sec: float | None,
    instruction_age_sec: float | None,
    config: WatchdogConfig = DEFAULT_WATCHDOG_CONFIG,
) -> WatchdogStatus:
    """Flag stale inputs from their ages. ``None`` age means the input was never received.

    The reason strings match what the ROS2 runtime publishes so dashboards and JSONL stay
    consistent: ``"waiting for image"``, ``"stale image: <age>s"``,
    ``"stale instruction: <age>s"``.
    """

    def _status(reason: str | None) -> WatchdogStatus:
        return WatchdogStatus(
            ok=reason is None,
            reason=reason,
            image_age_sec=image_age_sec,
            instruction_age_sec=instruction_age_sec,
        )

    if config.require_image and image_age_sec is None:
        return _status("waiting for image")
    if (
        image_age_sec is not None
        and config.stale_image_timeout_sec > 0
        and image_age_sec > config.stale_image_timeout_sec
    ):
        return _status(f"stale image: {image_age_sec:.2f}s")
    if (
        instruction_age_sec is not None
        and config.stale_instruction_timeout_sec > 0
        and instruction_age_sec > config.stale_instruction_timeout_sec
    ):
        return _status(f"stale instruction: {instruction_age_sec:.2f}s")
    return _status(None)
