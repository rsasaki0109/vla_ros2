"""Always-available VLM stand-in for tests and ROS2 wiring checks."""

from __future__ import annotations

from collections.abc import Sequence

from vla_ros2.vlm.base import BaseVLM, ImageArray


class DummyVLM(BaseVLM):
    """Returns a fixed answer; ``answer="yes"`` simulates a succeeded task."""

    name = "dummy"
    model_id = "none"

    def __init__(self, *, answer: str = "no", **_: object) -> None:
        self.answer = answer
        self.calls = 0

    def generate(
        self,
        prompt: str,
        images: Sequence[ImageArray] = (),
        *,
        max_new_tokens: int = 16,
    ) -> str:
        self.calls += 1
        return self.answer

    def yes_probability(self, prompt: str, images: Sequence[ImageArray] = ()) -> float | None:
        self.calls += 1
        return 1.0 if self.answer.strip().lower().startswith("yes") else 0.0
