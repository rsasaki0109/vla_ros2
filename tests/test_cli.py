from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from typer.testing import CliRunner

from vla_zoo.cli.main import _load_json_manifest, _manifest_int, _manifest_targets, app


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


def test_cli_compare_pybullet_help() -> None:
    result = CliRunner().invoke(app, ["compare", "pybullet", "--help"])
    assert result.exit_code == 0
    assert "--models" in result.output
    assert "--manifest" in result.output
    assert "--remote-map" in result.output
    assert "--allow-local-heavy" in result.output
    assert "--markdown-out" in result.output
    assert "--html-out" in result.output


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
