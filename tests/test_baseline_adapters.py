from __future__ import annotations

import numpy as np

from vla_zoo import load_model
from vla_zoo.core.types import VLAAction, VLAObservation


def test_random_adapter_is_seeded() -> None:
    first = load_model("random", seed=7, scale=0.5)
    second = load_model("random", seed=7, scale=0.5)

    action_a = first.predict(image=None, instruction="test")
    action_b = second.predict(image=None, instruction="test")

    assert isinstance(action_a, VLAAction)
    assert isinstance(action_b, VLAAction)
    assert np.allclose(action_a.to_numpy(), action_b.to_numpy())
    assert action_a.metadata["baseline"] == "random"


def test_scripted_adapter_uses_phase_metadata() -> None:
    model = load_model("scripted")
    action = model.predict(
        observation=VLAObservation(
            instruction="pick up the red block",
            metadata={"phase": "lift"},
        )
    )

    assert isinstance(action, VLAAction)
    data = action.to_numpy()
    assert data[2] > 0.0
    assert data[6] < 0.0
    assert action.metadata["baseline"] == "scripted"


def test_scripted_alias_loads() -> None:
    model = load_model("heuristic")
    action = model.predict(image=None, instruction="test")

    assert isinstance(action, VLAAction)
    assert action.metadata["baseline"] == "scripted"
