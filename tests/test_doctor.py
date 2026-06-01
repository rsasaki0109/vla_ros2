from __future__ import annotations

from vla_zoo.runtime.doctor import format_doctor_table, run_doctor, summarize_checks


def test_run_doctor_includes_dummy_predict() -> None:
    checks = run_doctor(include_ros=False)
    names = {check.name for check in checks}

    assert "python" in names
    assert "gpu.nvidia_smi" in names
    assert "gpu.torch_cuda" in names
    assert "adapter.dummy.predict" in names
    assert any(check.name == "adapter.dummy.predict" and check.ok for check in checks)


def test_doctor_summary_and_table() -> None:
    checks = run_doctor(include_ros=False)
    summary = summarize_checks(checks)
    table = format_doctor_table(checks)

    assert summary["ok"] >= 1
    assert "summary:" in table
    assert "adapter.dummy.predict" in table
