from __future__ import annotations

import importlib
from pathlib import Path

from vla_zoo.compare.evidence import build_evidence_matrix
from vla_zoo.runtime.server_plan import build_server_plan, format_server_plan_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pi0_server_plan_carries_checkpoint_and_install_hint() -> None:
    plan = build_server_plan(["pi0"], public_host="gpu-box", base_port=8001)
    entry = plan.entries[0]

    assert entry.model == "pi0"
    command = entry.command
    assert command[command.index("--pretrained") + 1] == "lerobot/pi0_base"
    assert "--device" in command
    assert "openpi" in entry.install_hint


def test_pi0_server_plan_markdown_flags_version_sensitivity() -> None:
    markdown = format_server_plan_markdown(build_server_plan(["pi0"]))

    assert "`pi0`" in markdown
    assert "version-sensitive" in markdown


def test_pi0_evidence_remote_server_links_remote_docs() -> None:
    record = build_evidence_matrix(["pi0"])[0]
    cell = record.evidence["remote_server"]

    assert cell.status == "planned"  # honest: no recorded pi0 /v1/predict yet
    hrefs = {link.href for link in cell.links}
    assert "../pi0_remote.md" in hrefs
    assert "pi0_server_plan.md" in hrefs


def test_pi0_remote_artifacts_exist() -> None:
    assert (REPO_ROOT / "docs" / "pi0_remote.md").is_file()
    assert (REPO_ROOT / "docs" / "assets" / "pi0_server_plan.md").is_file()
    assert (REPO_ROOT / "examples" / "python" / "load_pi0_remote.py").is_file()


def test_load_pi0_remote_example_imports() -> None:
    spec = importlib.util.spec_from_file_location(
        "load_pi0_remote",
        REPO_ROOT / "examples" / "python" / "load_pi0_remote.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
