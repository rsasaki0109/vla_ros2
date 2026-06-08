from __future__ import annotations

import numpy as np

from vla_ros2.sim.so100_kinematic import (
    apply_joint_action,
    observation_images,
    scene_from_dataset_state,
)


def test_apply_joint_action_blends_toward_target() -> None:
    state = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 0.2], dtype=np.float32)
    target = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 0.8], dtype=np.float32)
    updated = apply_joint_action(state, target, blend=0.5)
    assert np.allclose(updated, (state + target) / 2.0)


def test_observation_images_match_smolvla_camera_keys() -> None:
    scene = scene_from_dataset_state(np.zeros(6, dtype=np.float32))
    images = observation_images(scene, size=128)
    assert set(images) == {"camera1", "camera2", "camera3"}
    for array in images.values():
        assert array.shape == (128, 128, 3)
        assert array.dtype == np.uint8
