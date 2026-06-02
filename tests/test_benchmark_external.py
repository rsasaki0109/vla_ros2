from __future__ import annotations

import pytest

from vla_zoo.benchmark.libero import (
    LIBERO_SOURCE,
    LIBERO_SPEC,
    SUPPORTED_SUITES,
    run_libero_smoke,
)
from vla_zoo.benchmark.results import RESULT_SCHEMA_VERSION
from vla_zoo.benchmark.simpler import (
    SIMPLER_SOURCE,
    SIMPLER_SPEC,
    SUPPORTED_TASKS,
    run_simpler_smoke,
)
from vla_zoo.core.errors import MissingDependencyError
from vla_zoo.core.registry import load_model
from vla_zoo.core.types import VLAObservation


class _FakeEnv:
    """Minimal benchmark env following the reset/step protocol, no heavy deps."""

    name = "fake"

    def __init__(self, task: str) -> None:
        self.task = task
        self.steps = 0

    def reset(self, task_id: str | None = None) -> VLAObservation:
        self.steps = 0
        return VLAObservation(instruction=f"{self.task} {task_id or ''}")

    def step(self, action: object) -> tuple[VLAObservation, dict[str, object]]:
        self.steps += 1
        return VLAObservation(instruction="continue"), {"success": True, "done": True}


def test_specs_declare_contract_without_fabricating_success() -> None:
    for spec in (LIBERO_SPEC, SIMPLER_SPEC):
        assert spec.action_space == "eef_delta"
        assert spec.observation  # observation contract is declared
        assert "No task-success numbers are fabricated" in spec.note
        assert spec.suites


def test_libero_smoke_is_dependency_gated() -> None:
    model = load_model("dummy")
    with pytest.raises(MissingDependencyError) as excinfo:
        run_libero_smoke(model, model_name="dummy", episodes=1)
    assert "LIBERO" in str(excinfo.value)


def test_simpler_smoke_is_dependency_gated() -> None:
    model = load_model("dummy")
    with pytest.raises(MissingDependencyError) as excinfo:
        run_simpler_smoke(model, model_name="dummy", episodes=1)
    assert "SimplerEnv" in str(excinfo.value)


def test_libero_rejects_unknown_suite() -> None:
    model = load_model("dummy")
    with pytest.raises(ValueError, match="Unknown LIBERO suite"):
        run_libero_smoke(model, model_name="dummy", suite="not_a_suite")


def test_libero_smoke_emits_schema_with_injected_env() -> None:
    model = load_model("dummy")
    records, action_rate_hz = run_libero_smoke(
        model,
        model_name="dummy",
        suite=SUPPORTED_SUITES[0],
        episodes=2,
        env_factory=_FakeEnv,
    )

    assert len(records) == 2
    assert all(r.source == LIBERO_SOURCE for r in records)
    assert all(r.schema_version == RESULT_SCHEMA_VERSION for r in records)
    assert all(r.success is True for r in records)
    assert action_rate_hz is None or action_rate_hz > 0


def test_simpler_smoke_emits_schema_with_injected_env() -> None:
    model = load_model("dummy")
    records, _ = run_simpler_smoke(
        model,
        model_name="dummy",
        task=SUPPORTED_TASKS[0],
        episodes=3,
        env_factory=_FakeEnv,
    )

    assert len(records) == 3
    assert all(r.source == SIMPLER_SOURCE for r in records)
