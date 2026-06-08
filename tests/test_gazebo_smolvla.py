from __future__ import annotations

import numpy as np

from vla_ros2.sim.gazebo_smolvla import (
    gazebo_joints_to_smolvla_state,
    smolvla_action_to_gazebo_positions,
)


def test_roundtrip_gripper_channel() -> None:
    names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
    positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.04]
    state = gazebo_joints_to_smolvla_state(names, positions)
    commanded = smolvla_action_to_gazebo_positions(state)
    assert commanded["gripper"] == 0.04


def test_smolvla_arm_angles_use_degrees() -> None:
    commanded = smolvla_action_to_gazebo_positions([90.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert abs(commanded["joint_1"] - np.pi / 2) < 1e-5
