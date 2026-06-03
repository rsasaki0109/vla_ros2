from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from typer.testing import CliRunner

from vla_zoo.cli.main import (
    _load_json_manifest,
    _manifest_int,
    _manifest_targets,
    _model_load_kwargs,
    app,
)


def test_model_load_kwargs_threads_quantization_flags() -> None:
    base = _model_load_kwargs(pretrained=None, device="cuda:0", dtype=None, unnorm_key=None)
    assert "load_in_4bit" not in base  # only added when requested

    quant = _model_load_kwargs(
        pretrained="openvla/openvla-7b",
        device="cuda:0",
        dtype=None,
        unnorm_key="bridge_orig",
        load_in_4bit=True,
    )
    assert quant["load_in_4bit"] is True
    assert quant["pretrained"] == "openvla/openvla-7b"
    assert "load_in_8bit" not in quant
    assert "dtype" not in quant  # only added when requested

    # serve --dtype must thread through to the LeRobot adapter load_model(..., dtype=...)
    lerobot = _model_load_kwargs(
        pretrained="lerobot/pi0_base",
        device="cuda",
        dtype="bfloat16",
        unnorm_key=None,
    )
    assert lerobot["dtype"] == "bfloat16"
    assert lerobot["pretrained"] == "lerobot/pi0_base"


def test_cli_list() -> None:
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "dummy" in result.output
    assert "openvla" in result.output


def test_cli_demo_pybullet_help() -> None:
    result = CliRunner().invoke(app, ["demo", "pybullet", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--runtime" in result.output
    assert "--task" in result.output
    assert "--render-stride" in result.output


def test_cli_demo_action_probe_help() -> None:
    result = CliRunner().invoke(app, ["demo", "action-probe", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--allow-local-heavy" in result.output
    assert "--summary-md" in result.output


def test_cli_demo_action_probe_blocks_local_heavy_without_flag() -> None:
    # must not attempt to download/load smolvla weights from a plain `pytest` run
    result = CliRunner().invoke(app, ["demo", "action-probe", "--model", "smolvla"])
    assert result.exit_code == 1
    assert "allow-local-heavy" in result.output


def test_cli_demo_action_probe_blocks_pi0_local_heavy_without_flag() -> None:
    # pi0_base is a ~3B local checkpoint; gate it behind --allow-local-heavy too
    result = CliRunner().invoke(app, ["demo", "action-probe", "--model", "pi0"])
    assert result.exit_code == 1
    assert "allow-local-heavy" in result.output


def test_parse_adapter_kwargs_coerces_types() -> None:
    from vla_zoo.cli.main import _parse_adapter_kwargs

    parsed = _parse_adapter_kwargs(
        ["load_in_4bit=true", "unnorm_key=bridge_orig", "max_new_tokens=8", "temp=0.5"]
    )

    assert parsed == {
        "load_in_4bit": True,
        "unnorm_key": "bridge_orig",
        "max_new_tokens": 8,
        "temp": 0.5,
    }


def test_parse_adapter_kwargs_rejects_bare_token() -> None:
    import pytest

    from vla_zoo.cli.main import _parse_adapter_kwargs

    with pytest.raises(ValueError, match="key=value"):
        _parse_adapter_kwargs(["load_in_4bit"])


def test_cli_demo_gif_suite_help() -> None:
    result = CliRunner().invoke(app, ["demo", "gif-suite", "--help"])
    assert result.exit_code == 0
    assert "--models" in result.output
    assert "--tasks" in result.output
    assert "--out-dir" in result.output
    assert "--markdown-out" in result.output


def test_cli_demo_gif_check_help() -> None:
    result = CliRunner().invoke(app, ["demo", "gif-check", "--help"])
    assert result.exit_code == 0
    assert "--expected-width" in result.output
    assert "--link-files" in result.output
    assert "--markdown-out" in result.output


def test_cli_demo_gif_report_help() -> None:
    result = CliRunner().invoke(app, ["demo", "gif-report", "--help"])
    assert result.exit_code == 0
    assert "--manifest" in result.output
    assert "--html-out" in result.output
    assert "--check-json-out" in result.output


def test_cli_demo_action_playground_help() -> None:
    result = CliRunner().invoke(app, ["demo", "action-playground", "--help"])
    assert result.exit_code == 0
    assert "--manifest" in result.output
    assert "--trace-out" in result.output
    assert "--allow-local-heavy" in result.output


def test_cli_demo_action_playground_record_help() -> None:
    result = CliRunner().invoke(app, ["demo", "action-playground-record", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--runtime" in result.output
    assert "--remote-map" in result.output
    assert "--reference-gif-model" in result.output


def test_cli_demo_action_playground_view_help() -> None:
    result = CliRunner().invoke(app, ["demo", "action-playground-view", "--help"])
    assert result.exit_code == 0
    assert "--trace" in result.output
    assert "--merged-out" in result.output
    assert "--out" in result.output


def test_cli_demo_action_playground_check_help() -> None:
    result = CliRunner().invoke(app, ["demo", "action-playground-check", "--help"])
    assert result.exit_code == 0
    assert "--trace" in result.output
    assert "--expected-models" in result.output
    assert "--markdown-out" in result.output


def test_cli_demo_action_playground_remote_smoke_help() -> None:
    result = CliRunner().invoke(app, ["demo", "action-playground-remote-smoke", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--base-trace" in result.output
    assert "--merged-out" in result.output
    assert "--startup-timeout-sec" in result.output


def test_cli_report_link_check_help() -> None:
    result = CliRunner().invoke(app, ["report", "link-check", "--help"])
    assert result.exit_code == 0
    assert "--paths" in result.output
    assert "--root" in result.output
    assert "--strict" in result.output


def test_cli_report_index_help() -> None:
    result = CliRunner().invoke(app, ["report", "index", "--help"])
    assert result.exit_code == 0
    assert "--out" in result.output
    assert "--html-out" in result.output
    assert "--strict" in result.output


def test_cli_doctor_json() -> None:
    result = CliRunner().invoke(app, ["doctor", "--json", "--no-ros"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["summary"]["ok"] >= 1
    assert any(check["name"] == "gpu.nvidia_smi" for check in payload["checks"])
    assert any(check["name"] == "gpu.torch_cuda" for check in payload["checks"])
    assert any(check["name"] == "adapter.dummy.predict" for check in payload["checks"])


def test_cli_serve_help_exposes_gpu_options() -> None:
    result = CliRunner().invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--device" in result.output
    assert "--dtype" in result.output
    assert "--pretrained" in result.output
    assert "--unnorm-key" in result.output


def test_cli_serve_plan_writes_artifacts(tmp_path: Path) -> None:
    json_out = tmp_path / "servers.json"
    markdown_out = tmp_path / "servers.md"

    result = CliRunner().invoke(
        app,
        [
            "serve-plan",
            "--models",
            "openvla,pi0",
            "--public-host",
            "gpu-box",
            "--base-port",
            "8101",
            "--out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["remote_map"] == "openvla=http://gpu-box:8101,pi0=http://gpu-box:8102"
    assert "lerobot/pi0_base" in result.output
    assert "VLA GPU Server Plan" in markdown_out.read_text(encoding="utf-8")


def test_cli_gpu_smoke_help() -> None:
    result = CliRunner().invoke(app, ["gpu", "smoke", "--help"])

    assert result.exit_code == 0
    assert "--device" in result.output
    assert "--matrix-size" in result.output
    assert "--iterations" in result.output


def test_cli_ros_remote_smoke_plan_writes_artifacts(tmp_path: Path) -> None:
    json_out = tmp_path / "ros_remote.json"
    markdown_out = tmp_path / "ros_remote.md"

    result = CliRunner().invoke(
        app,
        [
            "ros",
            "remote-smoke-plan",
            "--model",
            "smolvla",
            "--remote-url",
            "http://gpu-box:8003",
            "--output-dir",
            "results/ros2_smolvla",
            "--out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["model_name"] == "smolvla"
    assert payload["remote_url"] == "http://gpu-box:8003"
    assert "remote_smoke_record.launch.py" in result.output
    assert "ROS2 Remote Smoke Plan" in markdown_out.read_text(encoding="utf-8")


def test_cli_compare_adapters() -> None:
    result = CliRunner().invoke(app, ["compare", "adapters"])
    assert result.exit_code == 0
    assert "dummy" in result.output
    assert "openvla" in result.output


def test_cli_compare_methods_json() -> None:
    result = CliRunner().invoke(app, ["compare", "methods", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    names = {item["name"] for item in payload}
    assert {"dummy", "openvla", "pi0", "smolvla", "groot"}.issubset(names)
    openvla = next(item for item in payload if item["name"] == "openvla")
    assert openvla["action_space"] == "eef_delta"
    assert "single RGB image" in openvla["input_requirements"]


def test_cli_compare_methods_writes_markdown(tmp_path: Path) -> None:
    out = tmp_path / "methods.md"

    result = CliRunner().invoke(app, ["compare", "methods", "--markdown-out", str(out)])

    assert result.exit_code == 0
    text = out.read_text(encoding="utf-8")
    assert "VLA Method Profiles" in text
    assert "`dummy`" in text
    assert "`openvla`" in text


def test_cli_compare_evidence_writes_artifacts(tmp_path: Path) -> None:
    json_out = tmp_path / "evidence.json"
    markdown_out = tmp_path / "evidence.md"
    html_out = tmp_path / "evidence.html"

    result = CliRunner().invoke(
        app,
        [
            "compare",
            "evidence",
            "--models",
            "openvla,smolvla",
            "--out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--html-out",
            str(html_out),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    records = {record["model"]: record for record in payload["records"]}
    assert records["openvla"]["evidence"]["gpu_inference"]["status"] == "verified"
    assert records["smolvla"]["evidence"]["gpu_inference"]["status"] == "verified"
    text = markdown_out.read_text(encoding="utf-8")
    assert "VLA Model Evidence Matrix" in text
    assert "not a model-quality leaderboard" in text
    html = html_out.read_text(encoding="utf-8")
    assert "VLA Model Evidence Matrix" in html
    assert "status-verified" in html
    assert "This is not a model-quality leaderboard" in html


def test_cli_compare_suite_no_pybullet(tmp_path: Path) -> None:
    out_dir = tmp_path / "suite"

    result = CliRunner().invoke(
        app,
        ["compare", "suite", "--out-dir", str(out_dir), "--no-pybullet"],
    )

    assert result.exit_code == 0
    assert (out_dir / "README.md").exists()
    assert (out_dir / "method_profiles.json").exists()
    assert (out_dir / "method_profiles.md").exists()
    assert not (out_dir / "pybullet_results.json").exists()
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "vla_zoo Comparison Suite" in readme
    assert "Method profiles do not load model weights" in readme


def test_cli_compare_dashboard_help() -> None:
    result = CliRunner().invoke(app, ["compare", "dashboard", "--help"])
    assert result.exit_code == 0
    assert "--results" in result.output
    assert "--status-log" in result.output
    assert "--diagnostics-log" in result.output
    assert "--out" in result.output


def test_cli_compare_dashboard_accepts_status_log(tmp_path: Path) -> None:
    log = tmp_path / "status.jsonl"
    log.write_text(
        json.dumps(
            {
                "model_name": "dummy",
                "ready": True,
                "last_latency_ms": 1.0,
                "status_text": "ready",
                "metadata": {"runtime": "local"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "dashboard.html"

    result = CliRunner().invoke(
        app,
        ["compare", "dashboard", "--status-log", str(log), "--out", str(out)],
    )

    assert result.exit_code == 0
    assert out.exists()
    assert "Fleet Health" in out.read_text(encoding="utf-8")


def test_cli_report_bundle_help() -> None:
    result = CliRunner().invoke(app, ["report", "bundle", "--help"])
    assert result.exit_code == 0
    assert "--status-log" in result.output
    assert "--diagnostics-log" in result.output
    assert "--out" in result.output


def test_cli_report_bundle_creates_zip(tmp_path: Path) -> None:
    log = tmp_path / "status.jsonl"
    log.write_text(
        json.dumps(
            {
                "model_name": "dummy",
                "ready": True,
                "last_latency_ms": 1.0,
                "status_text": "ready",
                "metadata": {"runtime": "local"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "bundle.zip"

    result = CliRunner().invoke(
        app,
        ["report", "bundle", "--status-log", str(log), "--out", str(out)],
    )

    assert result.exit_code == 0
    with ZipFile(out) as bundle:
        names = set(bundle.namelist())
        assert "README.txt" in names
        assert "dashboard.html" in names
        assert "records.json" in names
        assert "metadata.json" in names
        assert "inputs/status/00_status.jsonl" in names
        metadata = json.loads(bundle.read("metadata.json"))
    assert metadata["record_count"] == 1
    assert metadata["inputs"]["status_logs"] == [str(log)]


def test_cli_ros_smoke_report_help() -> None:
    result = CliRunner().invoke(app, ["ros", "smoke-report", "--help"])

    assert result.exit_code == 0
    assert "--duration-sec" in result.output
    assert "--skip-launch" in result.output
    assert "--fastdds-udp" in result.output
    assert "--action-log-name" in result.output


def test_cli_ros_remote_smoke_report_help() -> None:
    result = CliRunner().invoke(app, ["ros", "remote-smoke-report", "--help"])

    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--remote-url" in result.output
    assert "--skip-launch" in result.output
    assert "--remote-check-name" in result.output
    assert "Publish typed action" in result.output


def test_cli_ros_remote_smoke_check_help() -> None:
    result = CliRunner().invoke(app, ["ros", "remote-smoke-check", "--help"])

    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--remote-url" in result.output
    assert "--require-actions" in result.output
    assert "--markdown-out" in result.output


def test_cli_ros_action_trace(tmp_path: Path) -> None:
    action_log = tmp_path / "actions.jsonl"
    out = tmp_path / "trace.html"
    action_log.write_text(
        json.dumps(
            {
                "header": {"stamp": {"sec": 1, "nanosec": 0}},
                "model_name": "dummy",
                "adapter_name": "DummyAdapter",
                "action_space": "eef_delta",
                "data": [0.0, 0.0, 0.0],
                "names": ["x", "y", "z"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["ros", "action-trace", "--action-log", str(action_log), "--out", str(out)],
    )

    assert result.exit_code == 0
    assert out.exists()
    assert "Action Timeline" in out.read_text(encoding="utf-8")


def test_cli_ros_action_analyze(tmp_path: Path) -> None:
    action_log = tmp_path / "actions.jsonl"
    json_out = tmp_path / "analysis.json"
    markdown_out = tmp_path / "analysis.md"
    action_log.write_text(
        json.dumps(
            {
                "header": {"stamp": {"sec": 1, "nanosec": 0}},
                "model_name": "dummy",
                "adapter_name": "DummyAdapter",
                "action_space": "eef_delta",
                "data": [0.0, 0.1, 0.0],
                "names": ["x", "y", "z"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "header": {"stamp": {"sec": 2, "nanosec": 0}},
                "model_name": "dummy",
                "adapter_name": "DummyAdapter",
                "action_space": "eef_delta",
                "data": [0.0, 0.2, 0.0],
                "names": ["x", "y", "z"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ros",
            "action-analyze",
            "--action-log",
            str(action_log),
            "--out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["action_count"] == 2
    assert payload["action_dim"] == 3
    assert "action_rate_hz" in payload
    assert "Dimensions" in markdown_out.read_text(encoding="utf-8")


def test_cli_ros_smoke_report_skip_launch(tmp_path: Path) -> None:
    action_log = tmp_path / "vla_actions.jsonl"
    status_log = tmp_path / "vla_status.jsonl"
    diagnostics_log = tmp_path / "vla_diagnostics.jsonl"
    action_log.write_text(
        json.dumps(
            {
                "header": {"stamp": {"sec": 1, "nanosec": 0}},
                "model_name": "dummy",
                "adapter_name": "DummyAdapter",
                "action_space": "eef_delta",
                "data": [0.0, 0.0, 0.0],
                "names": ["x", "y", "z"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    status_log.write_text(
        json.dumps(
            {
                "model_name": "dummy",
                "ready": True,
                "last_latency_ms": 1.0,
                "status_text": "ready",
                "metadata": {"runtime": "local"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics_log.write_text(
        json.dumps(
            {
                "status": [
                    {
                        "name": "vla_zoo/vla_runtime_node",
                        "hardware_id": "dummy",
                        "level": 0,
                        "message": "ready",
                        "values": [{"key": "runtime", "value": "local"}],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["ros", "smoke-report", "--skip-launch", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert (tmp_path / "dashboard.html").exists()
    assert (tmp_path / "report_bundle.zip").exists()
    assert (tmp_path / "action_trace.html").exists()
    assert (tmp_path / "action_analysis.json").exists()
    assert (tmp_path / "action_analysis.md").exists()
    with ZipFile(tmp_path / "report_bundle.zip") as bundle:
        names = set(bundle.namelist())
        assert "action_trace.html" in names
        assert "action_analysis.json" in names
        assert "action_analysis.md" in names
        assert "inputs/actions/00_vla_actions.jsonl" in names
    assert "ROS2 smoke report written" in result.output


def test_cli_ros_remote_smoke_report_skip_launch(tmp_path: Path) -> None:
    action_log = tmp_path / "vla_actions.jsonl"
    status_log = tmp_path / "vla_status.jsonl"
    diagnostics_log = tmp_path / "vla_diagnostics.jsonl"
    action_log.write_text(
        json.dumps(
            {
                "header": {"stamp": {"sec": 1, "nanosec": 0}},
                "model_name": "openvla",
                "adapter_name": "RemoteVLAClient",
                "action_space": "eef_delta",
                "data": [0.0] * 7,
                "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                "metadata": {"model": "openvla", "latency_ms": 42.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    status_log.write_text(
        json.dumps(
            {
                "model_name": "openvla",
                "adapter_name": "RemoteVLAClient",
                "ready": True,
                "dry_run": True,
                "last_latency_ms": 42.0,
                "status_text": "dry_run: action suppressed",
                "metadata": {
                    "runtime": "remote",
                    "remote_url": "http://gpu-box:8001",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics_log.write_text(
        json.dumps(
            {
                "status": [
                    {
                        "name": "vla_zoo/vla_runtime_node",
                        "hardware_id": "openvla",
                        "level": 0,
                        "message": "ready",
                        "values": [
                            {"key": "runtime", "value": "remote"},
                            {"key": "remote_url", "value": "http://gpu-box:8001"},
                        ],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "ros",
            "remote-smoke-report",
            "--skip-launch",
            "--model",
            "openvla",
            "--remote-url",
            "http://gpu-box:8001",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "dashboard.html").exists()
    assert (tmp_path / "report_bundle.zip").exists()
    assert (tmp_path / "remote_smoke_check.json").exists()
    assert (tmp_path / "remote_smoke_check.md").exists()
    payload = json.loads((tmp_path / "remote_smoke_check.json").read_text(encoding="utf-8"))
    assert payload["ok"]
    assert payload["remote_action_count"] == 1
    assert "ROS2 remote smoke report written" in result.output


def test_cli_ros_remote_smoke_check_writes_artifacts(tmp_path: Path) -> None:
    (tmp_path / "vla_actions.jsonl").write_text(
        json.dumps(
            {
                "model_name": "dummy",
                "adapter_name": "RemoteVLAClient",
                "action_space": "eef_delta",
                "data": [0.0] * 7,
                "names": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "vla_status.jsonl").write_text(
        json.dumps(
            {
                "model_name": "dummy",
                "adapter_name": "RemoteVLAClient",
                "ready": True,
                "dry_run": True,
                "last_latency_ms": 12.0,
                "status_text": "ready",
                "metadata": {
                    "runtime": "remote",
                    "remote_url": "http://127.0.0.1:8766",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "vla_diagnostics.jsonl").write_text(
        json.dumps(
            {
                "status": [
                    {
                        "name": "vla_zoo/vla_runtime_node",
                        "hardware_id": "dummy",
                        "level": 0,
                        "message": "ready",
                        "values": [
                            {"key": "runtime", "value": "remote"},
                            {"key": "remote_url", "value": "http://127.0.0.1:8766"},
                        ],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "check.json"
    markdown_out = tmp_path / "check.md"

    result = CliRunner().invoke(
        app,
        [
            "ros",
            "remote-smoke-check",
            "--output-dir",
            str(tmp_path),
            "--model",
            "dummy",
            "--remote-url",
            "http://127.0.0.1:8766",
            "--out",
            str(out),
            "--markdown-out",
            str(markdown_out),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"]
    assert payload["remote_status_count"] == 1
    assert "ROS2 Remote Runtime Smoke Check" in markdown_out.read_text(encoding="utf-8")


def test_cli_compare_pybullet_help() -> None:
    result = CliRunner().invoke(app, ["compare", "pybullet", "--help"])
    assert result.exit_code == 0
    assert "--models" in result.output
    assert "--manifest" in result.output
    assert "--remote-map" in result.output
    assert "--allow-local-heavy" in result.output
    assert "--markdown-out" in result.output
    assert "--html-out" in result.output


def test_cli_compare_tasks_help() -> None:
    result = CliRunner().invoke(app, ["compare", "tasks", "--help"])
    assert result.exit_code == 0
    assert "--models" in result.output
    assert "--tasks" in result.output
    assert "--remote-map" in result.output
    assert "--allow-local-heavy" in result.output


def test_cli_compare_manifest_loads_targets(tmp_path: Path) -> None:
    manifest = tmp_path / "comparison.json"
    manifest.write_text(
        json.dumps(
            {
                "render_stride": 48,
                "models": [
                    {
                        "name": "dummy",
                        "runtime": "remote",
                        "remote_url": "http://127.0.0.1:8010",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = _load_json_manifest(manifest)
    targets = _manifest_targets(payload)

    assert _manifest_int(payload, "render_stride", 12) == 48
    assert targets[0].model_name == "dummy"
    assert targets[0].runtime == "remote"
    assert targets[0].remote_url == "http://127.0.0.1:8010"


def _native_diag_log(path: Path) -> Path:
    from vla_zoo.runtime.diagnostics import RuntimeDiagnostics
    from vla_zoo.runtime.guard import ActionClipGuard, evaluate_watchdog

    record = RuntimeDiagnostics.from_parts(
        model="dummy",
        status_text="ready",
        level="ok",
        clip_guard=ActionClipGuard(),
        watchdog=evaluate_watchdog(image_age_sec=0.1, instruction_age_sec=0.2),
        last_latency_ms=12.5,
        avg_latency_ms=11.0,
    )
    path.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")
    return path


def test_diag_report_renders_native_log(tmp_path: Path) -> None:
    log = _native_diag_log(tmp_path / "diag.jsonl")

    result = CliRunner().invoke(app, ["diag-report", "--log", str(log)])

    assert result.exit_code == 0, result.output
    assert "Runtime Diagnostics Snapshot" in result.output
    assert "vla-zoo-diagnostics/v1" in result.output


def test_diag_report_reconstructs_from_ros_log(tmp_path: Path) -> None:
    ros_log = tmp_path / "vla_diagnostics.jsonl"
    ros_log.write_text(
        json.dumps(
            {
                "status": [
                    {
                        "name": "vla_zoo/vla_runtime_node",
                        "values": [
                            {"key": "schema_version", "value": "vla-zoo-diagnostics/v1"},
                            {"key": "model", "value": "smolvla"},
                            {"key": "status_text", "value": "inference pending"},
                            {"key": "level", "value": "ok"},
                            {"key": "last_latency_ms", "value": "120.500"},
                            {"key": "pending_inference", "value": "False"},
                            {"key": "watchdog_ok", "value": "True"},
                        ],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "snapshot.md"
    jsonl_out = tmp_path / "native.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "diag-report",
            "--from-ros-log",
            str(ros_log),
            "--markdown-out",
            str(out),
            "--jsonl-out",
            str(jsonl_out),
        ],
    )

    assert result.exit_code == 0, result.output
    rendered = out.read_text(encoding="utf-8")
    assert "smolvla" in rendered
    assert "120.50 ms" in rendered
    # native JSONL must round-trip back through the schema reader
    from vla_zoo.runtime.diagnostics import read_diagnostics_jsonl

    records = read_diagnostics_jsonl(jsonl_out)
    assert records[-1].pending_inference is False
    assert records[-1].watchdog_ok is True


def _write_action_log(path: Path, *, model: str, latencies: list[float]) -> None:
    lines = []
    for index, latency in enumerate(latencies):
        record = {
            "header": {"stamp": {"sec": index, "nanosec": 0}, "frame_id": "world"},
            "model_name": model,
            "action_space": "eef_delta",
            "data": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            "names": [],
            "metadata": {"latency_ms": latency},
        }
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_bench_aggregate_from_log_ranks_replayed_logs(tmp_path: Path) -> None:
    fast = tmp_path / "fast.jsonl"
    slow = tmp_path / "slow.jsonl"
    _write_action_log(fast, model="smolvla", latencies=[10.0, 12.0, 11.0])
    _write_action_log(slow, model="openvla", latencies=[100.0, 120.0, 110.0])

    result = CliRunner().invoke(
        app,
        ["bench-aggregate", "--from-log", f"{slow},{fast}", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # the faster log ranks first even though it was passed second
    assert payload["ranked"][0]["summary"]["model"] == "smolvla"
    assert payload["ranked"][0]["rank"] == 1
    # a replayed log never invents a task-success claim
    assert all(e["summary"]["success_rate"] is None for e in payload["ranked"])


def test_bench_aggregate_combines_summaries_and_logs(tmp_path: Path) -> None:
    log = tmp_path / "probe.jsonl"
    _write_action_log(log, model="smolvla", latencies=[10.0, 12.0])

    summary_json = tmp_path / "remote.json"
    summary_json.write_text(
        json.dumps(
            {
                "schema_version": "vla-zoo-benchmark/v1",
                "model": "openvla",
                "source": "remote-server",
                "sample_count": 5,
                "success_count": 0,
                "success_rate": None,
                "latency_ms_p50": 2000.0,
                "latency_ms_p95": 2500.0,
                "latency_ms_mean": 2100.0,
                "action_rate_hz": 0.5,
                "exception_count": 0,
                "note": None,
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["bench-aggregate", "--summaries", str(summary_json), "--from-log", str(log), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 2
    assert payload["ranked"][0]["summary"]["model"] == "smolvla"


def test_bench_aggregate_requires_an_input() -> None:
    result = CliRunner().invoke(app, ["bench-aggregate"])
    assert result.exit_code == 1
    assert "no inputs provided" in result.output


def test_bench_aggregate_from_log_missing_file_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["bench-aggregate", "--from-log", str(tmp_path / "nope.jsonl")]
    )
    assert result.exit_code == 1
    assert "action log not found" in result.output


def test_quickstart_writes_report_and_proves_boundary(tmp_path: Path) -> None:
    out_dir = tmp_path / "qs"
    result = CliRunner().invoke(
        app, ["quickstart", "--out-dir", str(out_dir), "--episodes", "2"]
    )

    assert result.exit_code == 0, result.output
    assert "runtime boundary works" in result.output
    assert (out_dir / "report.html").is_file()
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert {r["model"] for r in payload["rows"]} == {"dummy", "scripted", "random"}


def test_quickstart_json_mode_does_not_write_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "qs"
    result = CliRunner().invoke(
        app, ["quickstart", "--out-dir", str(out_dir), "--episodes", "1", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "vla-zoo-quickstart/v1"
    assert not out_dir.exists()  # --json prints only, writes nothing


def test_compare_leaderboard_from_log_ranks_and_lists_blocked(tmp_path: Path) -> None:
    fast = tmp_path / "fast.jsonl"
    _write_action_log(fast, model="smolvla", latencies=[10.0, 12.0, 11.0])

    result = CliRunner().invoke(
        app, ["compare", "leaderboard", "--from-log", str(fast), "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    by_model = {e["model"]: e for e in payload["entries"]}
    assert by_model["smolvla"]["rank"] == 1
    # blocked adapters are surfaced honestly with no fabricated latency
    assert by_model["pi0"]["rank"] is None
    assert by_model["pi0"]["latency_ms_p50"] is None
    assert by_model["pi0"]["status"] == "blocked"


def test_compare_leaderboard_no_blocked_drops_unmeasured(tmp_path: Path) -> None:
    fast = tmp_path / "fast.jsonl"
    _write_action_log(fast, model="smolvla", latencies=[10.0, 12.0])

    result = CliRunner().invoke(
        app,
        ["compare", "leaderboard", "--from-log", str(fast), "--no-blocked", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {e["model"] for e in payload["entries"]} == {"smolvla"}


def test_compare_leaderboard_html_out(tmp_path: Path) -> None:
    fast = tmp_path / "fast.jsonl"
    _write_action_log(fast, model="smolvla", latencies=[10.0])
    out = tmp_path / "board.html"

    result = CliRunner().invoke(
        app, ["compare", "leaderboard", "--from-log", str(fast), "--html-out", str(out)]
    )

    assert result.exit_code == 0, result.output
    rendered = out.read_text(encoding="utf-8")
    assert "VLA Runtime Leaderboard" in rendered
    assert "badge" in rendered


def test_diag_report_requires_exactly_one_input() -> None:
    assert CliRunner().invoke(app, ["diag-report"]).exit_code == 1


def test_diag_report_summary_aggregates_log(tmp_path: Path) -> None:
    from vla_zoo.runtime.diagnostics import RuntimeDiagnostics
    from vla_zoo.runtime.guard import ActionClipGuard, evaluate_watchdog

    def _rec(level: str, latency: float) -> dict[str, object]:
        return RuntimeDiagnostics.from_parts(
            model="dummy",
            status_text="waiting for instruction" if level == "warn" else "ok",
            level=level,
            clip_guard=ActionClipGuard(),
            watchdog=evaluate_watchdog(image_age_sec=0.1, instruction_age_sec=0.1),
            last_latency_ms=latency,
            avg_latency_ms=latency,
        ).to_dict()

    log = tmp_path / "diag.jsonl"
    log.write_text(
        "\n".join(json.dumps(r) for r in [_rec("warn", 10.0), _rec("ok", 30.0)]) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["diag-report", "--log", str(log), "--summary", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["record_count"] == 2
    assert payload["latency_ms_max"] == 30.0
    assert payload["worst_level"] == "warn"


def test_cli_rtc_sim_json_reports_boundary_reduction() -> None:
    result = CliRunner().invoke(
        app,
        ["rtc-sim", "--chunks", "10", "--horizon", "16", "--execute", "8", "--delay", "4", "--json"],  # noqa: E501
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "vla-zoo-rtc-sim/v1"
    assert payload["rtc"]["mean_boundary_jump"] < payload["naive"]["mean_boundary_jump"]


def test_cli_rtc_sim_rejects_infeasible_timing() -> None:
    result = CliRunner().invoke(
        app,
        ["rtc-sim", "--horizon", "8", "--execute", "6", "--delay", "4"],
    )

    assert result.exit_code == 1
    assert "must not exceed horizon" in result.output


def test_cli_rtc_sim_help_lists_options() -> None:
    result = CliRunner().invoke(app, ["rtc-sim", "--help"])

    assert result.exit_code == 0
    assert "--horizon" in result.output
    assert "--delay" in result.output
    assert "--action-log" in result.output


def test_cli_demo_rtc_gif_renders(tmp_path) -> None:
    out = tmp_path / "rtc.gif"
    result = CliRunner().invoke(
        app,
        ["demo", "rtc-gif", "--out", str(out), "--chunks", "5", "--width", "560"],
    )

    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "reduction" in result.output


def test_cli_demo_rtc_gif_help_lists_options() -> None:
    result = CliRunner().invoke(app, ["demo", "rtc-gif", "--help"])

    assert result.exit_code == 0
    assert "--horizon" in result.output
    assert "--delay" in result.output


def test_cli_compare_roofline_help_lists_options() -> None:
    result = CliRunner().invoke(app, ["compare", "roofline", "--help"])

    assert result.exit_code == 0
    assert "--from-log" in result.output
    assert "--hardware" in result.output


def test_cli_compare_roofline_list_hardware() -> None:
    result = CliRunner().invoke(app, ["compare", "roofline", "--list-hardware"])

    assert result.exit_code == 0
    assert "local_16gb" in result.output
    assert "GB/s" in result.output


def test_cli_compare_roofline_rejects_unknown_hardware() -> None:
    result = CliRunner().invoke(app, ["compare", "roofline", "--hardware", "nope"])

    assert result.exit_code == 1
    assert "unknown hardware profile" in result.output


def test_cli_compare_roofline_from_log_reports_bands() -> None:
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "roofline",
            "--from-log",
            "docs/assets/sample_pybullet_smolvla/smolvla_action_probe.jsonl",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "vla-zoo-roofline/v1"
    models = {c["model"]: c for c in payload["comparisons"]}
    assert models["smolvla"]["measured_p50_ms"] is not None


def test_cli_rtc_record_help_lists_options() -> None:
    result = CliRunner().invoke(app, ["rtc-record", "--help"])

    assert result.exit_code == 0
    assert "--control-hz" in result.output
    assert "--allow-local-heavy" in result.output


def test_cli_rtc_record_blocks_heavy_without_flag() -> None:
    result = CliRunner().invoke(app, ["rtc-record", "--model", "smolvla", "--out", "x.json"])

    assert result.exit_code == 1
    assert "allow-local-heavy" in result.output


def test_cli_rtc_sim_trace_replays_recorded_chunks(tmp_path) -> None:
    import numpy as np

    from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation
    from vla_zoo.runtime.rtc_executor import record_rtc_trace

    spec = ActionSpec(action_space="eef_delta", shape=(3,))

    class _Fake:
        name = "fake"

        def __init__(self) -> None:
            self._rng = np.random.default_rng(0)

        def predict(self, *, observation: VLAObservation) -> VLAActionChunk:
            base = np.sin(0.2 * np.arange(16)[:, None] + np.array([0.0, 1.0, 2.0])[None, :])
            off = self._rng.normal(0, 0.7, size=3)
            chunk = (base + off[None, :]).astype(np.float32)
            return VLAActionChunk(actions=[VLAAction(data=r, spec=spec) for r in chunk])

    obs = [VLAObservation(instruction="pick") for _ in range(8)]
    ticks = iter([v for i in range(8) for v in (i * 1.0, i * 1.0 + 0.13)])
    trace = record_rtc_trace(
        _Fake(), obs, control_hz=30.0, execute_horizon=8, clock=lambda: next(ticks)
    )
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace.to_dict()), encoding="utf-8")

    result = CliRunner().invoke(app, ["rtc-sim", "--trace", str(trace_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["rtc"]["mean_boundary_jump"] < payload["naive"]["mean_boundary_jump"]
