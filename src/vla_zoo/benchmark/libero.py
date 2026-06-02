"""Dependency-gated LIBERO benchmark smoke runner.

LIBERO environments live in the upstream ``libero`` package (plus robosuite/MuJoCo),
which vla_zoo does not vendor. This runner declares the LIBERO contract and emits the
``vla-zoo-benchmark/v1`` schema, but actually building the environment requires the
upstream stack. It never fabricates task-success numbers: with no env it raises a clear
:class:`~vla_zoo.core.errors.MissingDependencyError`, and the schema-emitting loop runs
only against a real or injected env.
"""

from __future__ import annotations

from collections.abc import Callable

from vla_zoo.benchmark.base import BenchmarkEnv
from vla_zoo.benchmark.external import ExternalBenchmarkSpec
from vla_zoo.benchmark.results import EpisodeRecord
from vla_zoo.benchmark.runner import run_env_episode_records
from vla_zoo.core.errors import MissingDependencyError
from vla_zoo.core.model import BaseVLA

SUPPORTED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

LIBERO_SOURCE = "libero-smoke"

LIBERO_SPEC = ExternalBenchmarkSpec(
    name="libero",
    source=LIBERO_SOURCE,
    upstream="LIBERO (robosuite / MuJoCo)",
    suites=SUPPORTED_SUITES,
    action_space="eef_delta",
    observation=(
        "agentview + wrist RGB frames",
        "robot proprioceptive state",
        "language task instruction",
    ),
    install_hint='Install the upstream LIBERO stack (pip install "vla_zoo[libero]").',
    note=(
        "LIBERO smoke runner is dependency-gated: it emits the vla-zoo-benchmark/v1 schema "
        "but requires the upstream LIBERO environment to run. No task-success numbers are "
        "fabricated."
    ),
)


def _import_libero_dependencies() -> None:
    try:
        import libero  # noqa: F401
    except ImportError as exc:
        msg = (
            "LIBERO smoke runner requires the upstream LIBERO environment. "
            f"{LIBERO_SPEC.install_hint}"
        )
        raise MissingDependencyError(msg) from exc


def run_libero_smoke(
    model: BaseVLA,
    *,
    model_name: str,
    suite: str = SUPPORTED_SUITES[0],
    episodes: int = 3,
    env_factory: Callable[[str], BenchmarkEnv] | None = None,
) -> tuple[list[EpisodeRecord], float | None]:
    """Run a LIBERO smoke episode set and return schema records + action rate.

    ``env_factory`` lets callers (and tests) inject a benchmark env without the heavy
    LIBERO install. When it is ``None``, the upstream dependency guard runs first; a real
    env builder is not yet wired, so the dependency-present path raises
    ``NotImplementedError`` rather than fabricating a run.
    """

    if suite not in SUPPORTED_SUITES:
        msg = f"Unknown LIBERO suite {suite!r}; expected one of {SUPPORTED_SUITES}"
        raise ValueError(msg)

    if env_factory is None:
        _import_libero_dependencies()
        msg = (
            "LIBERO dependencies are present, but the real environment loop is not wired "
            "yet. Provide env_factory to run, or record a real LIBERO run before claiming "
            "results."
        )
        raise NotImplementedError(msg)

    env = env_factory(suite)
    return run_env_episode_records(
        model,
        env,
        source=LIBERO_SOURCE,
        model_name=model_name,
        episodes=episodes,
    )
