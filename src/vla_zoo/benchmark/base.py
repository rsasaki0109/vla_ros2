from __future__ import annotations

from typing import Protocol

from vla_zoo.core.model import BaseVLA
from vla_zoo.core.types import VLAAction, VLAActionChunk, VLAObservation


class BenchmarkEnv(Protocol):
    name: str

    def reset(self, task_id: str | None = None) -> VLAObservation:
        ...

    def step(self, action: VLAAction | VLAActionChunk) -> tuple[VLAObservation, dict[str, object]]:
        ...


class BenchmarkRunner:
    def run(
        self,
        model: BaseVLA,
        env: BenchmarkEnv,
        *,
        episodes: int,
        seed: int,
    ) -> dict[str, object]:
        raise NotImplementedError
