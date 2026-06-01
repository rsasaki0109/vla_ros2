from __future__ import annotations

from PIL import Image

from vla_zoo.demo.pybullet import (
    CUBE_GOAL_POSITION,
    PHASE_ORDER,
    PyBulletComparisonResult,
    RenderSample,
    compare_pybullet_models,
    format_pybullet_comparison_html,
    format_pybullet_comparison_markdown,
    summarize_pybullet_samples,
)


def _sample(
    phase: str,
    *,
    frame_index: int,
    cube_position: tuple[float, float, float],
    attached: bool = False,
) -> RenderSample:
    return RenderSample(
        image=Image.new("RGB", (2, 2)),
        phase=phase,
        position=(0.58, 0.0, 0.2),
        cube_position=cube_position,
        cube_goal_position=CUBE_GOAL_POSITION,
        scripted_action=(0.0, 0.0, 0.0, 0.0),
        adapter_action=(0.0, 0.0, 0.0, 0.0),
        adapter_error=None,
        adapter_latency_ms=1.0,
        adapter_query_count=frame_index + 1,
        adapter_query_fresh=True,
        attached=attached,
        sim_time=float(frame_index),
        model_name="dummy",
        runtime="local",
        frame_index=frame_index,
    )


def test_pybullet_comparison_markdown_includes_remote_url() -> None:
    markdown = format_pybullet_comparison_markdown(
        [
            PyBulletComparisonResult(
                model_name="openvla",
                runtime="remote",
                ok=True,
                remote_url="http://gpu-box:8001",
                frames=10,
                adapter_queries=2,
                mean_latency_ms=12.34,
                mean_abs_action=0.25,
                task_success=True,
                cube_lifted=True,
                final_cube_distance_to_goal=0.02,
                cube_moved_distance=0.38,
                phase_completion=1.0,
            )
        ]
    )

    assert "`openvla`" in markdown
    assert "http://gpu-box:8001" in markdown
    assert "12.34" in markdown
    assert "`pick_red_block`" in markdown
    assert "Goal dist m" in markdown
    assert "true" in markdown


def test_pybullet_comparison_skips_smolvla_local_heavy_by_default() -> None:
    result = compare_pybullet_models(["smolvla"])[0]

    assert result.ok is False
    assert result.last_error is not None
    assert "local heavy adapter skipped" in result.last_error


def test_pybullet_comparison_html_includes_summary_and_json() -> None:
    html = format_pybullet_comparison_html(
        [
            PyBulletComparisonResult(
                model_name="dummy",
                runtime="local",
                ok=True,
                frames=14,
                adapter_queries=2,
                mean_latency_ms=0.03,
                mean_abs_action=0.0,
                task_success=True,
                cube_lifted=True,
                final_cube_distance_to_goal=0.0,
            )
        ],
        title="Dummy Report",
    )

    assert "<!doctype html>" in html
    assert "Dummy Report" in html
    assert "<code>pick_red_block</code>" in html
    assert "<code>dummy</code>" in html
    assert "task success" in html
    assert "Goal dist m" in html
    assert "Raw JSON" in html


def test_pybullet_summary_computes_scene_task_telemetry() -> None:
    cube_positions = [
        (0.58, -0.16, 0.035),
        (0.58, -0.16, 0.035),
        (0.58, -0.16, 0.035),
        (0.58, -0.16, 0.035),
        (0.58, -0.16, 0.24),
        (0.58, 0.04, 0.24),
        (0.58, 0.22, 0.10),
        CUBE_GOAL_POSITION,
        CUBE_GOAL_POSITION,
    ]
    samples = [
        _sample(
            phase,
            frame_index=index,
            cube_position=cube_positions[index],
            attached=phase in {"close gripper", "lift", "transport", "place"},
        )
        for index, phase in enumerate(PHASE_ORDER)
    ]

    result = summarize_pybullet_samples("dummy", "local", samples)

    assert result.ok is True
    assert result.task_success is True
    assert result.cube_lifted is True
    assert result.grasp_attached_frames == 4
    assert result.phase_completion == 1.0
    assert result.final_cube_distance_to_goal == 0.0
    assert result.cube_moved_distance is not None
    assert result.cube_moved_distance > 0.35


def test_pybullet_tasks_have_distinct_instructions() -> None:
    from vla_zoo.demo.pybullet import default_pybullet_tasks, pybullet_task_by_id

    tasks = default_pybullet_tasks()
    assert len(tasks) >= 3
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert pybullet_task_by_id("move_red_block_left").instruction.startswith("move")
