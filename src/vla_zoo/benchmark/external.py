"""Shared contract type for external (dependency-gated) benchmark smoke runners.

LIBERO and SimplerEnv are real benchmarks whose environments live in heavy upstream
packages. vla_zoo does not vendor them, so their smoke runners are dependency-gated: the
declared contract is always queryable, but actually running requires the upstream stack.
No task-success numbers are fabricated — a runner emits the ``vla-zoo-benchmark/v1``
schema only for episodes it really executed (against a real or injected env).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalBenchmarkSpec:
    """Declared, queryable contract for an external benchmark smoke runner."""

    name: str
    source: str
    upstream: str
    suites: tuple[str, ...]
    action_space: str
    observation: tuple[str, ...]
    install_hint: str
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "upstream": self.upstream,
            "suites": list(self.suites),
            "action_space": self.action_space,
            "observation": list(self.observation),
            "install_hint": self.install_hint,
            "note": self.note,
        }
