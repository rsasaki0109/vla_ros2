from __future__ import annotations

import numpy as np
import pytest

from vla_ros2 import load_model
from vla_ros2.core.model import BaseVLA
from vla_ros2.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation
from vla_ros2.eval_offline import (
    OfflineFrame,
    action_errors,
    evaluate_offline,
    prediction_to_array,
)

SPEC = ActionSpec(action_space="custom", shape=(2,))


class ChunkModel(BaseVLA):
    """Predicts a fixed 3-step chunk."""

    def __init__(self, rows: list[list[float]]) -> None:
        super().__init__(name="chunk", model_id="chunk-model", action_spec=SPEC)
        self.rows = rows

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        return VLAActionChunk(
            actions=[
                VLAAction(data=np.asarray(row, dtype=np.float32), spec=SPEC) for row in self.rows
            ]
        )


def _frame(target: list[list[float]], state: list[float] | None = None) -> OfflineFrame:
    return OfflineFrame(
        observation=VLAObservation(instruction="stack"),
        target=np.asarray(target, dtype=np.float32),
        state=None if state is None else np.asarray(state, dtype=np.float32),
    )


def test_prediction_to_array_handles_action_and_chunk() -> None:
    action = VLAAction(data=np.asarray([1.0, 2.0], dtype=np.float32), spec=SPEC)
    assert prediction_to_array(action).shape == (1, 2)
    chunk = ChunkModel([[0, 0], [1, 1], [2, 2]]).predict_observation(VLAObservation("x"))
    assert prediction_to_array(chunk).shape == (3, 2)


def test_action_errors_truncates_to_shorter_horizon() -> None:
    errors = action_errors(
        [np.asarray([[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]])],
        [np.asarray([[0.0, 0.0], [1.0, 1.0]])],  # recorded episode ends after 2 steps
    )
    assert errors.horizon == 2
    assert errors.first_step_mae == pytest.approx(1.0)
    assert errors.mae == pytest.approx(1.5)
    assert errors.rmse == pytest.approx(np.sqrt((1 + 4) / 2))
    assert errors.mae_per_dim == pytest.approx((1.5, 1.5))


def test_action_errors_rejects_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="action dim mismatch"):
        action_errors([np.zeros((1, 7))], [np.zeros((1, 6))])


def test_evaluate_offline_scores_model_and_hold_state_baseline() -> None:
    model = ChunkModel([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    frames = [
        _frame([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], state=[0.0, 0.0]),
        _frame([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]], state=[1.0, 1.0]),
    ]
    result = evaluate_offline(model, frames)
    assert result.num_frames == 2
    assert result.model == "chunk-model"
    assert result.model_errors.mae == pytest.approx(1 / 6)
    assert result.baseline_errors is not None
    # hold-state errors: frame 1 -> (1,2,3), frame 2 -> (0,1,3)
    assert result.baseline_errors.mae == pytest.approx(10 / 6)
    assert result.model_errors.mae < result.baseline_errors.mae
    payload = result.to_dict()
    assert payload["hold_state_baseline_errors"]["horizon"] == 3
    assert len(payload["per_frame"]) == 2


def test_evaluate_offline_runs_built_in_dummy_adapter() -> None:
    model = load_model("dummy")
    frames = [_frame([[0.0] * 7, [1.0] * 7])]
    result = evaluate_offline(model, frames)
    assert result.model_errors.horizon == 1  # dummy predicts a single step
    assert result.model_errors.mae == pytest.approx(0.0)
    assert result.baseline_errors is None  # no state supplied
