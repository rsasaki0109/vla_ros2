from __future__ import annotations

import numpy as np
import pytest

from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk
from vla_zoo.demo.pybullet import PyBulletDemoConfig, prediction_to_demo_action


def test_pybullet_demo_config_defaults_to_dummy() -> None:
    config = PyBulletDemoConfig()
    assert config.model_name == "dummy"
    assert config.runtime == "local"
    assert config.out.name == "simulation_pick_place.gif"


def test_prediction_to_demo_action_uses_7d_gripper_slot() -> None:
    spec = ActionSpec(action_space="eef_delta", shape=(7,))
    action = VLAAction(
        data=np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.7], dtype=np.float32),
        spec=spec,
    )

    assert prediction_to_demo_action(action) == pytest.approx((0.1, -0.2, 0.3, 0.7))


def test_prediction_to_demo_action_accepts_chunks() -> None:
    spec = ActionSpec(action_space="eef_delta", shape=(4,))
    action = VLAAction(data=np.array([2.0, 0.0, -2.0, 0.5], dtype=np.float32), spec=spec)
    chunk = VLAActionChunk(actions=[action])

    assert prediction_to_demo_action(chunk) == pytest.approx((1.0, 0.0, -1.0, 0.5))
