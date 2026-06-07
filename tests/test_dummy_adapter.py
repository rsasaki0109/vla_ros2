from __future__ import annotations

from vla_ros2 import load_model
from vla_ros2.core.types import VLAAction


def test_dummy_predict_returns_vla_action() -> None:
    model = load_model("dummy")
    action = model.predict(image=None, instruction="move forward")
    assert isinstance(action, VLAAction)
    assert action.spec.action_space == "eef_delta"
    assert action.to_numpy().shape == (7,)
    assert action.to_numpy().sum() == 0.0
