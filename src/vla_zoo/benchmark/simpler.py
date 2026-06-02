"""Dependency-gated SimplerEnv benchmark smoke runner.

SimplerEnv exposes Gymnasium-style manipulation environments in an upstream package that
vla_zoo does not vendor. This runner declares the SimplerEnv contract and emits the
``vla-zoo-benchmark/v1`` schema, but building the environment requires the upstream stack.
It never fabricates task-success numbers: with no env it raises a clear
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

API_STYLE = "gymnasium"

SUPPORTED_TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "widowx_put_eggplant_in_basket",
)

SIMPLER_SOURCE = "simpler-smoke"

SIMPLER_SPEC = ExternalBenchmarkSpec(
    name="simpler",
    source=SIMPLER_SOURCE,
    upstream="SimplerEnv (Gymnasium-style)",
    suites=SUPPORTED_TASKS,
    action_space="eef_delta",
    observation=(
        "RGB camera frame(s)",
        "robot proprioceptive state",
        "language task instruction",
    ),
    install_hint='Install the upstream SimplerEnv stack (pip install "vla_zoo[simpler]").',
    note=(
        "SimplerEnv smoke runner is dependency-gated: it emits the vla-zoo-benchmark/v1 "
        "schema but requires the upstream SimplerEnv environment to run. No task-success "
        "numbers are fabricated."
    ),
)


def _import_simpler_dependencies() -> None:
    try:
        import gymnasium  # noqa: F401
        import simpler_env  # noqa: F401
    except ImportError as exc:
        msg = (
            "SimplerEnv smoke runner requires the upstream SimplerEnv environment. "
            f"{SIMPLER_SPEC.install_hint}"
        )
        raise MissingDependencyError(msg) from exc


def run_simpler_smoke(
    model: BaseVLA,
    *,
    model_name: str,
    task: str = SUPPORTED_TASKS[0],
    episodes: int = 3,
    env_factory: Callable[[str], BenchmarkEnv] | None = None,
) -> tuple[list[EpisodeRecord], float | None]:
    """Run a SimplerEnv smoke episode set and return schema records + action rate.

    ``env_factory`` lets callers (and tests) inject a benchmark env without the heavy
    SimplerEnv install. When it is ``None``, the upstream dependency guard runs first; a
    real env builder is not yet wired, so the dependency-present path raises
    ``NotImplementedError`` rather than fabricating a run.
    """

    if task not in SUPPORTED_TASKS:
        msg = f"Unknown SimplerEnv task {task!r}; expected one of {SUPPORTED_TASKS}"
        raise ValueError(msg)

    if env_factory is None:
        _import_simpler_dependencies()
        msg = (
            "SimplerEnv dependencies are present, but the real environment loop is not wired "
            "yet. Provide env_factory to run, or record a real SimplerEnv run before claiming "
            "results."
        )
        raise NotImplementedError(msg)

    env = env_factory(task)
    return run_env_episode_records(
        model,
        env,
        source=SIMPLER_SOURCE,
        model_name=model_name,
        episodes=episodes,
    )
