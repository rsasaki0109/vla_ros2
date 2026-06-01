from __future__ import annotations

import numpy as np

from vla_zoo.core.types import ActionSpec, VLAAction
from vla_zoo.runtime.remote import jsonable
from vla_zoo.runtime.schemas import action_to_response, response_to_action


def test_remote_schema_roundtrip() -> None:
    spec = ActionSpec(
        action_space="eef_delta",
        shape=(7,),
        names=("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
        frame_id="base_link",
        control_hz=5.0,
    )
    action = VLAAction(data=np.zeros((7,), dtype=np.float32), spec=spec, dt=0.2)
    response = action_to_response(action)
    recovered = response_to_action(response)
    assert recovered.spec == spec
    assert recovered.tolist() == action.tolist()


def test_remote_jsonable_converts_numpy_state() -> None:
    payload = jsonable(
        {
            "state": np.asarray([1.0, 2.0], dtype=np.float32),
            "nested": {"target": (0.1, 0.2, 0.3)},
        }
    )

    assert payload == {
        "state": [1.0, 2.0],
        "nested": {"target": [0.1, 0.2, 0.3]},
    }
