"""Run a list of instructions in order, advancing on task success.

Each step is published with its own ``task_id`` (``<prefix>-<step>-<attempt>``), so
success detectors reset even when two steps share the same text, and a success
is only accepted when its ``task_id`` matches the current step. A stale "done"
from the previous step therefore cannot skip the next one. With a step timeout,
a step is retried ``max_retries`` times and the sequence then stops as failed
rather than moving on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceStep:
    index: int
    attempt: int
    instruction: str
    task_id: str


class TaskSequencer:
    """State machine: ``idle`` -> ``running`` -> ``done`` | ``failed``."""

    def __init__(
        self,
        steps: Sequence[str],
        *,
        task_id_prefix: str = "seq",
        step_timeout_sec: float = 0.0,
        max_retries: int = 0,
    ) -> None:
        cleaned = [step.strip() for step in steps if step.strip()]
        if not cleaned:
            msg = "TaskSequencer needs at least one non-empty step"
            raise ValueError(msg)
        if max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        self.steps = cleaned
        self.task_id_prefix = task_id_prefix
        self.step_timeout_sec = step_timeout_sec
        self.max_retries = max_retries
        self.state = "idle"
        self.index = 0
        self.attempt = 0
        self.step_started_at: float | None = None
        self.completed: list[str] = []

    @property
    def current(self) -> SequenceStep | None:
        if self.state != "running":
            return None
        return SequenceStep(
            index=self.index,
            attempt=self.attempt,
            instruction=self.steps[self.index],
            task_id=f"{self.task_id_prefix}-{self.index}-{self.attempt}",
        )

    def start(self, now: float) -> SequenceStep:
        self.state = "running"
        self.index = 0
        self.attempt = 0
        self.completed = []
        self.step_started_at = now
        step = self.current
        assert step is not None
        return step

    def on_success(self, task_id: str, now: float) -> SequenceStep | None:
        """Advance if ``task_id`` is the current step; return the new step, if any."""

        step = self.current
        if step is None or task_id != step.task_id:
            return None
        self.completed.append(step.task_id)
        if self.index + 1 >= len(self.steps):
            self.state = "done"
            self.step_started_at = None
            return None
        self.index += 1
        self.attempt = 0
        self.step_started_at = now
        return self.current

    def tick(self, now: float) -> SequenceStep | None:
        """Handle the step timeout; return a retried step when one starts."""

        if (
            self.state != "running"
            or self.step_timeout_sec <= 0
            or self.step_started_at is None
            or now - self.step_started_at < self.step_timeout_sec
        ):
            return None
        if self.attempt >= self.max_retries:
            self.state = "failed"
            return None
        self.attempt += 1
        self.step_started_at = now
        return self.current

    def to_dict(self) -> dict[str, object]:
        step = self.current
        return {
            "state": self.state,
            "step": self.index,
            "steps": len(self.steps),
            "attempt": self.attempt,
            "instruction": step.instruction if step else None,
            "task_id": step.task_id if step else None,
            "completed": list(self.completed),
        }


def resolve_steps(
    *,
    steps: Sequence[str] = (),
    steps_json: str = "",
    goal: str = "",
    plan: Callable[[str], list[str]] | None = None,
) -> tuple[list[str], str]:
    """Pick the step list: explicit ``steps_json``, then ``steps``, then ``plan(goal)``.

    Returns ``(steps, source)`` with source ``"steps_json"``, ``"steps"`` or ``"planner"``.
    """

    if steps_json.strip():
        parsed = json.loads(steps_json)
        if not isinstance(parsed, list):
            msg = "steps_json must be a JSON array of strings"
            raise ValueError(msg)
        cleaned = [str(step).strip() for step in parsed if str(step).strip()]
        if cleaned:
            return cleaned, "steps_json"
    cleaned = [step.strip() for step in steps if step.strip()]
    if cleaned:
        return cleaned, "steps"
    if goal.strip() and plan is not None:
        return plan(goal.strip()), "planner"
    msg = "no steps: set steps, steps_json, or goal with a planner"
    raise ValueError(msg)
