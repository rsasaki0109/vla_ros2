from __future__ import annotations

from vla_zoo.compare.evidence import (
    build_evidence_matrix,
    evidence_matrix_payload,
    format_evidence_matrix_html,
    format_evidence_matrix_markdown,
)


def test_evidence_matrix_marks_smolvla_gpu_verified() -> None:
    records = build_evidence_matrix(["smolvla"])

    record = records[0]

    assert record.model == "smolvla"
    assert record.evidence["gpu_inference"].status == "verified"
    assert record.evidence["pybullet_tasks"].status == "partial"
    assert "SmolVLA" in format_evidence_matrix_markdown(records)


def test_evidence_matrix_keeps_openvla_policy_quality_unverified() -> None:
    records = build_evidence_matrix(["openvla"])
    payload = evidence_matrix_payload(records)
    record = records[0]

    assert payload["schema"] == "vla_zoo.vla_model_evidence_matrix.v1"
    # Local 4-bit GPU inference is now measured and verified, but policy quality stays
    # explicitly unverified: a runtime-path claim is not a task-success claim.
    assert record.evidence["gpu_inference"].status == "verified"
    assert record.evidence["policy_quality"].status == "not_verified"
    assert "No task-success" in record.evidence["policy_quality"].summary


def test_evidence_matrix_deduplicates_aliases() -> None:
    records = build_evidence_matrix(["pi0", "openpi", "pi05"])

    assert [record.model for record in records] == ["pi0"]


def test_evidence_matrix_html_renders_status_and_links() -> None:
    records = build_evidence_matrix(["openvla", "smolvla"])

    html = format_evidence_matrix_html(records)

    assert "<!doctype html>" in html
    assert "This is not a model-quality leaderboard" in html
    assert "status-verified" in html
    # Both OpenVLA and SmolVLA now link their measured local-runtime evidence pages.
    assert "local runtime evidence" in html
