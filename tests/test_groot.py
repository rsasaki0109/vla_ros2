from __future__ import annotations

from pathlib import Path

import pytest

from vla_zoo.adapters.groot import GROOT_BLOCKED_NOTE, GR00TAdapter
from vla_zoo.compare.evidence import build_evidence_matrix
from vla_zoo.core.errors import MissingDependencyError
from vla_zoo.core.registry import get_adapter_info
from vla_zoo.core.types import VLAObservation

REPO_ROOT = Path(__file__).resolve().parents[1]


def _observation() -> VLAObservation:
    return VLAObservation(instruction="pick up the red block")


def test_groot_blocked_note_is_explicit_about_nvidia_stack() -> None:
    assert "NVIDIA Isaac GR00T" in GROOT_BLOCKED_NOTE
    assert "no task-success claim" in GROOT_BLOCKED_NOTE.lower()


def test_groot_adapter_does_not_fabricate_actions() -> None:
    adapter = GR00TAdapter()

    # gr00t is not installed in the test environment, so the dependency guard fires
    # first with the shared blocked note. Either way, no action is returned.
    with pytest.raises((MissingDependencyError, NotImplementedError)) as excinfo:
        adapter.predict_observation(_observation())
    assert "NVIDIA Isaac GR00T" in str(excinfo.value)


def test_groot_registry_marks_blocked_and_experimental() -> None:
    info = get_adapter_info("groot")

    assert info.experimental is True
    assert "NVIDIA Isaac GR00T" in str(info.metadata["blocked_reason"])
    assert "blocked" in str(info.metadata["verification"]).lower()


def test_groot_evidence_runtime_cells_are_blocked() -> None:
    record = build_evidence_matrix(["groot"])[0]
    evidence = record.evidence

    for column in ("local_runtime", "gpu_inference", "remote_server"):
        assert evidence[column].status == "blocked", column

    hrefs = {link.href for link in evidence["local_runtime"].links}
    assert "../groot_remote.md" in hrefs
    # The contract is declared even while blocked, so it stays partial, not verified.
    assert record.evidence["contract"].status == "partial"
    assert record.evidence["policy_quality"].status == "not_verified"


def test_groot_blocked_doc_documents_expected_contract() -> None:
    doc = (REPO_ROOT / "docs" / "groot_remote.md").read_text(encoding="utf-8")

    assert "Blocked" in doc
    assert "NVIDIA Isaac GR00T" in doc
    assert "observation/action contract" in doc
    assert "VLAActionChunk" in doc
