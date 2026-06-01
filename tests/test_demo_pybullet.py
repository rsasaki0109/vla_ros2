from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vla_zoo.core.model import BaseVLA
from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation
from vla_zoo.demo.pybullet import (
    PyBulletDemoConfig,
    predict_adapter_action,
    prediction_to_demo_action,
    simulation_state_vector,
)


class _CaptureModel(BaseVLA):
    def __init__(self) -> None:
        self.observation: VLAObservation | None = None
        super().__init__(
            name="capture",
            model_id="capture",
            action_spec=ActionSpec(action_space="custom", shape=(6,)),
        )

    def predict_observation(self, observation: VLAObservation) -> VLAAction:
        self.observation = observation
        return VLAAction(
            data=np.asarray([0.1, 0.2, 0.3, 0.4, 0.0, 0.0], dtype=np.float32),
            spec=self.action_spec,
        )


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


def test_simulation_state_vector_is_fixed_6d() -> None:
    state = simulation_state_vector(
        target=(0.5, 0.1, 0.2),
        cube_position=(0.6, -0.1, 0.03),
        gripper=0.75,
    )

    assert state.dtype == np.float32
    assert state.shape == (6,)
    assert state.tolist() == pytest.approx([0.5, 0.1, 0.2, 0.6, -0.1, 0.75])


def test_predict_adapter_action_builds_multicamera_observation() -> None:
    model = _CaptureModel()
    image = Image.new("RGB", (2, 2))

    action, error, latency = predict_adapter_action(
        model,
        {
            "primary": image,
            "observation.images.camera1": image,
            "observation.images.camera2": image,
            "observation.images.camera3": image,
        },
        PyBulletDemoConfig(),
        phase="observe",
        target=(0.5, 0.1, 0.2),
        cube_position=(0.6, -0.1, 0.03),
        cube_goal_position=(0.6, 0.2, 0.03),
        gripper=1.0,
        attached=False,
        sim_time=1.25,
    )

    assert error is None
    assert latency is not None
    assert action == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert model.observation is not None
    assert set(model.observation.images) == {
        "primary",
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    }
    assert model.observation.state["state"].shape == (6,)
    assert model.observation.metadata["state_names"] == (
        "eef_target_x",
        "eef_target_y",
        "eef_target_z",
        "cube_x",
        "cube_y",
        "gripper_open",
    )
