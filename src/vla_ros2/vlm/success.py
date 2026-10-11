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

# Two-image variant: the start-of-task frame first, the current frame second.
# Asking what changed avoids judging a single top-down frame in isolation.
DEFAULT_COMPARE_PROMPT_TEMPLATE = (
    'The first image shows the scene when a robot arm was asked to: "{instruction}". '
    "The second image shows the scene now. Has the task been fully completed in the "
    "second image? Answer yes or no."
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
    Passing ``reference`` (the frame from when the instruction started) switches
    to ``compare_prompt_template`` and sends ``[reference, *images]``.
    """

    def __init__(
        self,
        vlm: BaseVLM,
        *,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        compare_prompt_template: str = DEFAULT_COMPARE_PROMPT_TEMPLATE,
        threshold: float = 0.5,
        use_logits: bool = True,
        max_new_tokens: int = 8,
    ) -> None:
        for template in (prompt_template, compare_prompt_template):
            if "{instruction}" not in template:
                msg = "prompt templates must contain '{instruction}'"
                raise ValueError(msg)
        self.vlm = vlm
        self.prompt_template = prompt_template
        self.compare_prompt_template = compare_prompt_template
        self.threshold = threshold
        self.use_logits = use_logits
        self.max_new_tokens = max_new_tokens

    def prompt(self, instruction: str, *, compare: bool = False) -> str:
        template = self.compare_prompt_template if compare else self.prompt_template
        return template.format(instruction=instruction.strip())

    def judge(
        self,
        instruction: str,
        images: Sequence[ImageArray],
        *,
        reference: ImageArray | None = None,
    ) -> SuccessJudgement:
        compare = reference is not None
        prompt = self.prompt(instruction, compare=compare)
        if reference is not None:
            images = [reference, *images]
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
            metadata={"vlm": self.vlm.name, "model_id": self.vlm.model_id, "compare": compare},
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


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    balanced_accuracy: float
    auroc: float
    num_positive: int
    num_negative: int


def calibrate_threshold(scores: Sequence[float], labels: Sequence[bool]) -> ThresholdCalibration:
    """Pick the P(yes) threshold that maximizes balanced accuracy on labeled frames.

    ``auroc`` is threshold-free: 1.0 means every done frame scored above every
    not-done frame, 0.5 means the VLM cannot tell them apart.
    """

    if len(scores) != len(labels):
        msg = "scores and labels must have the same length"
        raise ValueError(msg)
    positives = [float(s) for s, label in zip(scores, labels, strict=True) if label]
    negatives = [float(s) for s, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        msg = "need at least one done and one not-done frame"
        raise ValueError(msg)

    pairs = [(p > n) + 0.5 * (p == n) for p in positives for n in negatives]
    auroc = sum(pairs) / len(pairs)

    ordered = sorted(set(positives + negatives))
    candidates = [(a + b) / 2.0 for a, b in zip(ordered, ordered[1:], strict=False)]
    candidates = candidates or [ordered[0]]
    best_threshold, best_accuracy = candidates[0], -1.0
    for threshold in candidates:
        tpr = sum(p >= threshold for p in positives) / len(positives)
        tnr = sum(n < threshold for n in negatives) / len(negatives)
        accuracy = (tpr + tnr) / 2.0
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = threshold, accuracy
    return ThresholdCalibration(
        threshold=best_threshold,
        balanced_accuracy=best_accuracy,
        auroc=auroc,
        num_positive=len(positives),
        num_negative=len(negatives),
    )
