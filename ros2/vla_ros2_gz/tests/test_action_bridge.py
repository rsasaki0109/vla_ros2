from __future__ import annotations

from vla_ros2_gz_ros.action_bridge import integrate_joint_targets


def test_integrate_joint_targets_applies_scales_and_clips() -> None:
    targets = {"joint_1": 0.0, "gripper": 0.02}
    updated = integrate_joint_targets(
        data=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        joint_scales=[0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.01],
        targets=targets,
        joint_limits={
            "joint_1": (-2.5, 2.5),
            "gripper": (0.0, 0.04),
        },
    )
    assert updated["joint_1"] == 0.15
    assert updated["gripper"] == 0.01
