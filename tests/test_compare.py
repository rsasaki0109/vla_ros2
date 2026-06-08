from __future__ import annotations

import json

from vla_ros2.compare import (
    compare_models,
    compare_results_to_json,
    compare_results_to_rows,
)
from vla_ros2.core.types import VLAObservation


def test_compare_baselines_on_same_observation() -> None:
    observation = VLAObservation(
        instruction="pick up the cube",
        metadata={"phase": "approach"},
    )
    results = compare_models(["dummy", "random", "scripted"], observation)
    assert len(results) == 3
    assert all(item.ok for item in results)
    assert {item.name for item in results} == {"dummy", "random", "scripted"}
    assert results[0].action_shape == (7,)

    rows = compare_results_to_rows(results)
    assert rows[0]["model"] == "dummy"
    assert rows[0]["ok"] is True

    payload = json.loads(compare_results_to_json(results))
    assert payload[1]["name"] == "random"


def test_compare_unknown_model_is_not_called() -> None:
    observation = VLAObservation(instruction="test")
    results = compare_models(["dummy", "not-a-model"], observation)
    assert results[0].ok
    assert not results[1].ok
    assert results[1].error is not None
