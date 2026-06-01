from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vla_zoo.demo.action_playground import (
    ActionPlaygroundRecord,
    build_action_playground_records,
    build_action_playground_task_records,
    check_action_playground_records,
    format_action_playground_check_markdown,
    format_action_playground_html,
    load_action_playground_trace,
    load_action_playground_traces,
    load_playground_specs,
    merge_action_playground_records,
    samples_to_playground_frames,
    write_action_playground_html,
    write_action_playground_trace,
)
from vla_zoo.demo.gif_suite import PyBulletGifResult, PyBulletGifSpec, write_gif_manifest
from vla_zoo.demo.pybullet import RenderSample, pybullet_task_by_id


def _sample(
    *,
    frame_index: int,
    phase: str = "observe",
    adapter_action: tuple[float, float, float, float] | None = (0.1, -0.2, 0.0, 0.5),
    adapter_query_fresh: bool = True,
) -> RenderSample:
    return RenderSample(
        image=Image.new("RGB", (8, 6), color=(20, 40, 60)),
        phase=phase,
        position=(0.1, 0.2, 0.3),
        cube_position=(0.58, 0.22, 0.035),
        cube_goal_position=(0.58, 0.22, 0.035),
        scripted_action=(0.0, 0.1, 0.2, 0.0),
        adapter_action=adapter_action,
        adapter_error=None,
        adapter_latency_ms=4.2 if adapter_query_fresh else None,
        adapter_query_count=1,
        adapter_query_fresh=adapter_query_fresh,
        attached=phase == "lift",
        sim_time=0.25 * frame_index,
        model_name="dummy",
        runtime="local",
        frame_index=frame_index,
    )


def _manifest(tmp_path: Path) -> Path:
    gif_path = tmp_path / "simulation_pick_red_block_dummy.gif"
    gif_path.write_bytes(b"GIF89a")
    manifest = tmp_path / "gif_manifest.json"
    write_gif_manifest(
        manifest,
        [
            PyBulletGifResult(
                spec=PyBulletGifSpec(
                    model_name="dummy",
                    task_id="pick_red_block",
                    instruction="pick up the red block",
                    out=gif_path,
                ),
                ok=True,
                frames=2,
                bytes=gif_path.stat().st_size,
            )
        ],
    )
    return manifest


def test_load_playground_specs_from_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    specs = load_playground_specs(manifest)

    assert len(specs) == 1
    assert specs[0].model_name == "dummy"
    assert specs[0].task_id == "pick_red_block"


def test_samples_to_playground_frames_prefers_adapter_action() -> None:
    frames = samples_to_playground_frames(
        [
            _sample(frame_index=0, adapter_action=(0.1, -0.2, 0.0, 0.5)),
            _sample(frame_index=1, adapter_action=None, adapter_query_fresh=False),
        ]
    )

    assert frames[0].displayed_action == (0.1, -0.2, 0.0, 0.5)
    assert frames[0].action_magnitude > 0.5
    assert frames[1].displayed_action == (0.0, 0.1, 0.2, 0.0)
    assert not frames[1].adapter_query_fresh


def test_build_action_playground_records_with_fake_simulator(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    records = build_action_playground_records(
        manifest,
        simulator=lambda _spec: [
            _sample(frame_index=0, phase="observe"),
            _sample(frame_index=1, phase="lift", adapter_query_fresh=False),
        ],
    )

    assert len(records) == 1
    assert records[0].ok
    assert records[0].frames[0].phase == "observe"
    assert records[0].summary["adapter_queries"] == 1


def test_action_playground_html_and_trace_outputs(tmp_path: Path) -> None:
    records = build_action_playground_records(
        _manifest(tmp_path),
        simulator=lambda _spec: [_sample(frame_index=0)],
    )
    trace = tmp_path / "action_playground.json"
    html_path = tmp_path / "action_playground.html"

    write_action_playground_trace(trace, records)
    write_action_playground_html(html_path, records, title="Test Playground")
    html = format_action_playground_html(records, title="Inline Test", path_relative_to=tmp_path)

    payload = json.loads(trace.read_text(encoding="utf-8"))
    assert payload["schema"] == "vla_zoo.action_playground.v1"
    assert payload["records"][0]["model_name"] == "dummy"
    assert "Test Playground" in html_path.read_text(encoding="utf-8")
    assert "Trace frame" in html
    assert "Task Comparison" in html
    assert "comparison-chart" in html
    assert "What This Does Not Show" in html

    script_json = html.split('<script id="payload" type="application/json">', 1)[1].split(
        "</script>",
        1,
    )[0]
    assert json.loads(script_json)["records"][0]["model_name"] == "dummy"


def test_action_playground_trace_load_and_merge(tmp_path: Path) -> None:
    records = build_action_playground_records(
        _manifest(tmp_path),
        simulator=lambda _spec: [_sample(frame_index=0)],
    )
    replacement = ActionPlaygroundRecord(
        model_name=records[0].model_name,
        task_id=records[0].task_id,
        instruction=records[0].instruction,
        gif_path=records[0].gif_path,
        runtime=records[0].runtime,
        ok=False,
        frames=records[0].frames,
        summary={"adapter_queries": 99},
        error="external trace replacement",
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_action_playground_trace(first, records)
    write_action_playground_trace(second, [replacement])

    loaded = load_action_playground_trace(first)
    merged = merge_action_playground_records([*loaded, replacement])
    loaded_merged = load_action_playground_traces([first, second])

    assert len(loaded) == 1
    assert len(merged) == 1
    assert merged[0].error == "external trace replacement"
    assert loaded_merged[0].summary["adapter_queries"] == 99


def test_action_playground_check_report_marks_verified_records(tmp_path: Path) -> None:
    records = build_action_playground_records(
        _manifest(tmp_path),
        simulator=lambda _spec: [_sample(frame_index=0), _sample(frame_index=1)],
    )

    report = check_action_playground_records(
        records,
        expected_models=("dummy",),
        expected_tasks=("pick_red_block",),
        min_frames=2,
    )
    markdown = format_action_playground_check_markdown(report)

    assert report.ok
    assert report.ok_count == 1
    assert report.records[0].gif_exists
    assert "| pick_red_block | ok (local) |" in markdown
    assert "runtime-path report" in markdown


def test_action_playground_check_report_surfaces_missing_expected_model(
    tmp_path: Path,
) -> None:
    records = build_action_playground_records(
        _manifest(tmp_path),
        simulator=lambda _spec: [_sample(frame_index=0), _sample(frame_index=1)],
    )

    report = check_action_playground_records(
        records,
        expected_models=("dummy", "openvla"),
        expected_tasks=("pick_red_block",),
        min_frames=2,
    )
    markdown = format_action_playground_check_markdown(report)

    assert not report.ok
    assert "missing expected model trace: openvla" in report.issues
    assert "| pick_red_block | ok (local) | missing |" in markdown


def test_build_action_playground_task_records_uses_remote_and_reference_gif(
    tmp_path: Path,
) -> None:
    seen_specs: list[PyBulletGifSpec] = []

    def fake_simulator(spec: PyBulletGifSpec) -> list[RenderSample]:
        seen_specs.append(spec)
        return [_sample(frame_index=0)]

    records = build_action_playground_task_records(
        models=("openvla",),
        tasks=(pybullet_task_by_id("pick_red_block"),),
        out_dir=tmp_path,
        runtime="remote",
        remote_url="http://default:8000",
        remote_urls={"openvla": "http://gpu-box:8001"},
        reference_gif_model="scripted",
        simulator=fake_simulator,
    )

    assert len(records) == 1
    assert records[0].model_name == "openvla"
    assert records[0].runtime == "remote"
    assert records[0].gif_path.endswith("simulation_pick_red_block_scripted.gif")
    assert records[0].summary["remote_url"] == "http://gpu-box:8001"
    assert seen_specs[0].remote_url == "http://gpu-box:8001"
