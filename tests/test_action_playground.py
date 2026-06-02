from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vla_zoo.demo.action_playground import (
    build_action_playground_records,
    format_action_playground_html,
    load_playground_specs,
    samples_to_playground_frames,
    write_action_playground_html,
    write_action_playground_trace,
)
from vla_zoo.demo.gif_suite import PyBulletGifResult, PyBulletGifSpec, write_gif_manifest
from vla_zoo.demo.pybullet import RenderSample


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
    assert "What This Does Not Show" in html

    script_json = html.split('<script id="payload" type="application/json">', 1)[1].split(
        "</script>",
        1,
    )[0]
    assert json.loads(script_json)["records"][0]["model_name"] == "dummy"
