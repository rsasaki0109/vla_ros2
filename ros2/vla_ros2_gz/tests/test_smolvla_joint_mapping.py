from __future__ import annotations

from vla_ros2.sim.gazebo_smolvla import (
    blend_joint_targets,
    gazebo_joints_to_smolvla_state,
    smolvla_action_to_gazebo_positions,
)


def test_gazebo_joints_to_smolvla_state_maps_degrees_and_gripper() -> None:
    names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
    positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02]
    state = gazebo_joints_to_smolvla_state(names, positions)
    assert state == [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]


def test_smolvla_action_to_gazebo_positions_maps_back() -> None:
    commanded = smolvla_action_to_gazebo_positions([90.0, 0.0, 45.0, 0.0, 10.0, 1.0])
    assert commanded["gripper"] == 0.04
    assert abs(commanded["joint_1"] - 1.5707963267948966) < 1e-5


def test_blend_joint_targets_interpolates() -> None:
    updated = blend_joint_targets(
        current={"joint_1": 0.0, "gripper": 0.0},
        commanded={"joint_1": 1.0, "gripper": 0.04},
        blend=0.5,
        joint_limits={"joint_1": (-2.5, 2.5), "gripper": (0.0, 0.04)},
    )
    assert updated["joint_1"] == 0.5
    assert updated["gripper"] == 0.02
