from __future__ import annotations

from vla_ros2_ros.action_parse import (
    eef_delta_named_values,
    parse_action_fields,
    parsed_twist_from_eef_delta,
)


def test_parse_action_fields_builds_named_values() -> None:
    parsed = parse_action_fields(
        model_name="dummy",
        adapter_name="DummyAdapter",
        action_space="eef_delta",
        control_mode="eef_delta",
        frame_id="base_link",
        dt=0.2,
        data=[0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 1.0],
        names=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        metadata_json='{"dry_run": true}',
    )
    assert parsed.named_values["x"] == 0.1
    assert parsed.named_values["gripper"] == 1.0
    assert parsed.metadata["dry_run"] is True


def test_eef_delta_named_values_filters_keys() -> None:
    values = eef_delta_named_values({"x": 1.0, "foo": 9.0, "gripper": 0.5})
    assert values == {"x": 1.0, "gripper": 0.5}


def test_parsed_twist_from_eef_delta_maps_linear_and_angular() -> None:
    parsed = parse_action_fields(
        model_name="dummy",
        adapter_name="DummyAdapter",
        action_space="eef_delta",
        control_mode="eef_delta",
        frame_id="base_link",
        dt=0.2,
        data=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
        names=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    )
    twist = parsed_twist_from_eef_delta(parsed)
    assert twist is not None
    assert twist.linear_x == 0.1
    assert twist.linear_z == 0.3
    assert twist.angular_z == 0.6
