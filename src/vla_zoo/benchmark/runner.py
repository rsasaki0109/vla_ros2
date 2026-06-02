from __future__ import annotations

from time import perf_counter

from vla_zoo.benchmark.base import BenchmarkEnv, BenchmarkRunner
from vla_zoo.benchmark.metrics import MetricsAccumulator
from vla_zoo.benchmark.results import EpisodeRecord
from vla_zoo.core.model import BaseVLA
from vla_zoo.core.types import VLAAction, VLAActionChunk, VLAObservation

SMOKE_SOURCE = "smoke-benchmark"


def _action_count(action: VLAAction | VLAActionChunk) -> int:
    if isinstance(action, VLAActionChunk):
        return len(action.actions)
    return 1


class SmokeBenchmarkEnv:
    name = "smoke"

    def __init__(self) -> None:
        self.steps = 0

    def reset(self, task_id: str | None = None) -> VLAObservation:
        self.steps = 0
        return VLAObservation(
            instruction=f"smoke task {task_id or 'default'}",
            images={},
            state={"step": self.steps},
        )

    def step(self, action: VLAAction | VLAActionChunk) -> tuple[VLAObservation, dict[str, object]]:
        self.steps += 1
        obs = VLAObservation(
            instruction="continue smoke task",
            images={},
            state={"step": self.steps},
        )
        return obs, {"success": True, "episode_return": 1.0, "done": True}


class SimpleBenchmarkRunner(BenchmarkRunner):
    def run(
        self,
        model: BaseVLA,
        env: BenchmarkEnv,
        *,
        episodes: int,
        seed: int,
    ) -> dict[str, object]:
        del seed
        metrics = MetricsAccumulator(episodes=episodes)
        for episode in range(episodes):
            observation = env.reset(task_id=str(episode))
            try:
                start = perf_counter()
                action = model.predict(observation=observation)
                metrics.latencies_ms.append((perf_counter() - start) * 1000.0)
                _, info = env.step(action)
                if bool(info.get("success", False)):
                    metrics.successes += 1
            except Exception:
                metrics.exceptions += 1
        summary: dict[str, object] = dict(metrics.summary())
        elapsed = sum(metrics.latencies_ms) / 1000.0
        summary.update(
            {
                "benchmark": env.name,
                "episodes": episodes,
                "action_rate_hz": episodes / elapsed if elapsed > 0 else 0.0,
                "sim_steps_per_sec": 0.0,
                "real_time_factor": 0.0,
                "action_smoothness": 0.0,
            }
        )
        return summary


def run_smoke_benchmark(model: BaseVLA, *, episodes: int = 3, seed: int = 0) -> dict[str, object]:
    return SimpleBenchmarkRunner().run(model, SmokeBenchmarkEnv(), episodes=episodes, seed=seed)


def run_smoke_episode_records(
    model: BaseVLA,
    *,
    model_name: str,
    episodes: int = 3,
    seed: int = 0,
) -> tuple[list[EpisodeRecord], float | None]:
    """Run the smoke benchmark and return per-episode schema records + action rate.

    This mirrors :class:`SimpleBenchmarkRunner` but emits the versioned
    :class:`~vla_zoo.benchmark.results.EpisodeRecord` schema so results can be written
    as JSONL and summarized. The returned action rate is wall-clock based.
    """

    del seed
    env = SmokeBenchmarkEnv()
    records: list[EpisodeRecord] = []
    total_latency_s = 0.0
    for episode in range(episodes):
        observation = env.reset(task_id=str(episode))
        latency_ms: float | None = None
        success: bool | None = None
        num_actions = 0
        error: str | None = None
        try:
            start = perf_counter()
            action = model.predict(observation=observation)
            elapsed = perf_counter() - start
            latency_ms = elapsed * 1000.0
            total_latency_s += elapsed
            num_actions = _action_count(action)
            _, info = env.step(action)
            success = bool(info.get("success", False))
        except Exception as exc:  # noqa: BLE001 - recorded as an error row, not raised
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            EpisodeRecord(
                model=model_name,
                source=SMOKE_SOURCE,
                index=episode,
                task_id=str(episode),
                success=success,
                latency_ms=latency_ms,
                num_actions=num_actions,
                error=error,
            )
        )
    action_rate_hz = (episodes / total_latency_s) if total_latency_s > 0 else None
    return records, action_rate_hz
