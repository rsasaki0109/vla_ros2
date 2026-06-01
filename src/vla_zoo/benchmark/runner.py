from __future__ import annotations

from time import perf_counter

from vla_zoo.benchmark.base import BenchmarkEnv, BenchmarkRunner
from vla_zoo.benchmark.metrics import MetricsAccumulator
from vla_zoo.core.model import BaseVLA
from vla_zoo.core.types import VLAAction, VLAActionChunk, VLAObservation


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
