"""Task-success detection: ask a VLM whether the instruction has been completed."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from vla_ros2.vlm.base import BaseVLM, ImageArray

DEFAULT_PROMPT_TEMPLATE = (
    'You are watching a robot arm. The robot was asked to: "{instruction}". '
    "Look at the image. Has the robot fully completed this task? Answer yes or no."
)

_YES_RE = re.compile(r"^\W*(yes|yeah|yep|true|completed|done|success)\b", re.IGNORECASE)
_NO_RE = re.compile(r"^\W*(no|nope|false|not)\b", re.IGNORECASE)


def parse_yes_no(text: str) -> bool | None:
    """Parse a VLM answer into True/False, or None when it is neither."""

    if _YES_RE.match(text):
        return True
    if _NO_RE.match(text):
        return False
    return None


@dataclass(frozen=True)
class SuccessJudgement:
    success: bool | None
    confidence: float | None
    answer: str
    latency_ms: float
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SuccessDetector:
    """Builds the success prompt and turns the VLM's reply into a judgement.

    With ``use_logits`` (default) the adapter's P(yes) decides against
    ``threshold``; adapters without logits fall back to parsing generated text.
    """

    def __init__(
        self,
        vlm: BaseVLM,
        *,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        threshold: float = 0.5,
        use_logits: bool = True,
        max_new_tokens: int = 8,
    ) -> None:
        if "{instruction}" not in prompt_template:
            msg = "prompt_template must contain '{instruction}'"
            raise ValueError(msg)
        self.vlm = vlm
        self.prompt_template = prompt_template
        self.threshold = threshold
        self.use_logits = use_logits
        self.max_new_tokens = max_new_tokens

    def prompt(self, instruction: str) -> str:
        return self.prompt_template.format(instruction=instruction.strip())

    def judge(self, instruction: str, images: Sequence[ImageArray]) -> SuccessJudgement:
        prompt = self.prompt(instruction)
        start = perf_counter()
        confidence = self.vlm.yes_probability(prompt, images) if self.use_logits else None
        if confidence is not None:
            success: bool | None = confidence >= self.threshold
            answer = "yes" if success else "no"
        else:
            answer = self.vlm.generate(prompt, images, max_new_tokens=self.max_new_tokens)
            success = parse_yes_no(answer)
        latency_ms = (perf_counter() - start) * 1000.0
        return SuccessJudgement(
            success=success,
            confidence=confidence,
            answer=answer,
            latency_ms=latency_ms,
            prompt=prompt,
            metadata={"vlm": self.vlm.name, "model_id": self.vlm.model_id},
        )


class SuccessLatch:
    """Debounces noisy per-frame judgements into a task-level success flag.

    ``success`` latches once ``required_consecutive`` checks in a row say yes,
    and stays latched until :meth:`reset` (e.g. a new instruction arrives).
    """

    def __init__(self, required_consecutive: int = 2) -> None:
        if required_consecutive < 1:
            msg = "required_consecutive must be >= 1"
            raise ValueError(msg)
        self.required_consecutive = required_consecutive
        self.consecutive = 0
        self.success = False

    def update(self, judgement: SuccessJudgement) -> bool:
        if judgement.success:
            self.consecutive += 1
        else:
            self.consecutive = 0
        if self.consecutive >= self.required_consecutive:
            self.success = True
        return self.success

    def reset(self) -> None:
        self.consecutive = 0
        self.success = False
