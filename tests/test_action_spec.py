from __future__ import annotations

import numpy as np
import pytest

from vla_ros2.core.types import ActionSpec, VLAAction


def test_action_spec_validation() -> None:
    with pytest.raises(ValueError):
        ActionSpec(action_space="eef_delta", shape=(0,))
    with pytest.raises(ValueError):
        ActionSpec(action_space="eef_delta", shape=(2,), names=("x",))
    spec = ActionSpec(action_space="eef_delta", shape=(2,), low=(-1.0, -1.0), high=(1.0, 1.0))
    action = VLAAction(data=np.array([0.0, 0.5]), spec=spec)
    assert action.tolist() == [0.0, 0.5]
