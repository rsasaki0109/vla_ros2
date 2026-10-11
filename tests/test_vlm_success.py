from __future__ import annotations

import importlib.util
from collections.abc import Sequence

import numpy as np
import pytest

from vla_ros2.core.errors import MissingDependencyError, UnknownModelError
from vla_ros2.vlm import (
    BaseVLM,
    SuccessDetector,
    SuccessJudgement,
    SuccessLatch,
    calibrate_threshold,
    get_vlm_info,
    list_vlms,
    load_vlm,
    parse_yes_no,
)
from vla_ros2.vlm.base import ImageArray

FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


class TextOnlyVLM(BaseVLM):
    name = "text-only"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.image_counts: list[int] = []

    def generate(
        self, prompt: str, images: Sequence[ImageArray] = (), *, max_new_tokens: int = 16
    ) -> str:
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        return self.answer


class FixedProbVLM(TextOnlyVLM):
    def __init__(self, p_yes: float) -> None:
        super().__init__("unused")
        self.p_yes = p_yes

    def yes_probability(self, prompt: str, images: Sequence[ImageArray] = ()) -> float | None:
        return self.p_yes


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes.", True),
        ("yes, the cube is stacked", True),
        (" Done", True),
        ("No", False),
        ("not yet", False),
        ("The red cube is on the table.", None),
        ("", None),
    ],
)
def test_parse_yes_no(text: str, expected: bool | None) -> None:
    assert parse_yes_no(text) is expected


def test_registry_lists_and_loads_dummy() -> None:
    assert {"dummy", "smolvlm"} <= {info.name for info in list_vlms()}
    vlm = load_vlm("dummy", answer="yes")
    assert vlm.name == "dummy"
    assert vlm.yes_probability("q", [FRAME]) == 1.0


def test_registry_resolves_alias_and_rejects_unknown() -> None:
    assert get_vlm_info("SmolVLM2").name == "smolvlm"
    with pytest.raises(UnknownModelError):
        load_vlm("not-a-vlm")


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is not None,
    reason="transformers installed; missing-dependency path not reachable",
)
def test_smolvlm_without_extras_raises_missing_dependency() -> None:
    with pytest.raises(MissingDependencyError):
        load_vlm("smolvlm2")


def test_detector_uses_logits_against_threshold() -> None:
    assert SuccessDetector(FixedProbVLM(0.7), threshold=0.6).judge("stack", [FRAME]).success
    judgement = SuccessDetector(FixedProbVLM(0.55), threshold=0.6).judge("stack", [FRAME])
    assert judgement.success is False
    assert judgement.confidence == pytest.approx(0.55)


def test_detector_falls_back_to_generated_text() -> None:
    vlm = TextOnlyVLM("Yes, it is on top.")
    judgement = SuccessDetector(vlm).judge("put the red cube on the blue cube", [FRAME])
    assert judgement.success is True
    assert judgement.confidence is None
    assert "put the red cube on the blue cube" in vlm.prompts[0]


def test_detector_compares_with_reference_frame() -> None:
    vlm = TextOnlyVLM("yes")
    detector = SuccessDetector(vlm)
    reference = np.ones((32, 32, 3), dtype=np.uint8)
    judgement = detector.judge("stack the cubes", [FRAME], reference=reference)
    assert judgement.success is True
    assert judgement.metadata["compare"] is True
    assert vlm.image_counts == [2]
    assert "first image" in vlm.prompts[0]
    detector.judge("stack the cubes", [FRAME])
    assert vlm.image_counts == [2, 1]
    assert "first image" not in vlm.prompts[1]


def test_detector_rejects_template_without_instruction() -> None:
    with pytest.raises(ValueError):
        SuccessDetector(TextOnlyVLM("yes"), prompt_template="Is it done?")
    with pytest.raises(ValueError):
        SuccessDetector(TextOnlyVLM("yes"), compare_prompt_template="Changed?")


def _judgement(success: bool | None) -> SuccessJudgement:
    return SuccessJudgement(success=success, confidence=None, answer="", latency_ms=0.0, prompt="")


def test_latch_needs_consecutive_yes_and_resets() -> None:
    latch = SuccessLatch(required_consecutive=2)
    assert latch.update(_judgement(True)) is False
    assert latch.update(_judgement(None)) is False  # unparseable answer breaks the streak
    assert latch.update(_judgement(True)) is False
    assert latch.update(_judgement(True)) is True
    assert latch.update(_judgement(False)) is True  # stays latched for the task
    latch.reset()
    assert latch.success is False
    assert latch.consecutive == 0


def test_calibrate_threshold_separable_scores() -> None:
    calibration = calibrate_threshold([0.2, 0.3, 0.6, 0.8], [False, False, True, True])
    assert calibration.auroc == 1.0
    assert calibration.balanced_accuracy == 1.0
    assert calibration.threshold == pytest.approx(0.45)
    assert (calibration.num_positive, calibration.num_negative) == (2, 2)


def test_calibrate_threshold_inseparable_scores() -> None:
    calibration = calibrate_threshold([0.4, 0.4, 0.4, 0.4], [False, True, False, True])
    assert calibration.auroc == 0.5
    assert calibration.balanced_accuracy == 0.5


def test_calibrate_threshold_needs_both_classes() -> None:
    with pytest.raises(ValueError):
        calibrate_threshold([0.1, 0.2], [True, True])
