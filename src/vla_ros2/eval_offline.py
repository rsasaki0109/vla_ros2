"""Open-loop (offline) evaluation of a VLA adapter against recorded demonstrations.

For each sampled frame of a recorded episode the adapter predicts an action
chunk from the recorded observation, and the chunk is compared with the actions
the demonstrator actually took from that frame on. No simulator or robot is
involved, so the number is reproducible, but it measures imitation error, not
task success.

A ``hold-state`` baseline (repeat the current joint state) is reported beside the
model: for absolute joint-target actions such as SO-100's, a policy that does not
beat it has not learned the motion.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vla_ros2.core.errors import MissingDependencyError
from vla_ros2.core.model import BaseVLA
from vla_ros2.core.types import VLAAction, VLAActionChunk, VLAObservation


@dataclass(frozen=True)
class OfflineFrame:
    observation: VLAObservation
    target: NDArray[np.float32]
    """Recorded actions from this frame on, shape ``(horizon, action_dim)``."""
    state: NDArray[np.float32] | None = None
    episode_index: int = 0
    frame_index: int = 0


@dataclass(frozen=True)
class ActionErrors:
    horizon: int
    mae: float
    rmse: float
    first_step_mae: float
    mae_per_dim: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "mae": self.mae,
            "rmse": self.rmse,
            "first_step_mae": self.first_step_mae,
            "mae_per_dim": list(self.mae_per_dim),
        }


@dataclass(frozen=True)
class OfflineEvalResult:
    model: str
    num_frames: int
    model_errors: ActionErrors
    baseline_errors: ActionErrors | None
    infer_ms_mean: float
    per_frame: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "num_frames": self.num_frames,
            "infer_ms_mean": self.infer_ms_mean,
            "model_errors": self.model_errors.to_dict(),
            "hold_state_baseline_errors": (
                self.baseline_errors.to_dict() if self.baseline_errors else None
            ),
            "per_frame": self.per_frame,
        }


def prediction_to_array(prediction: VLAAction | VLAActionChunk) -> NDArray[np.float32]:
    """Return a ``(steps, action_dim)`` array from an action or action chunk."""

    if isinstance(prediction, VLAActionChunk):
        rows = [np.asarray(a.data, dtype=np.float32).reshape(-1) for a in prediction.actions]
        return np.stack(rows)
    return np.asarray(prediction.data, dtype=np.float32).reshape(1, -1)


def _aligned(
    pred: NDArray[np.float32], target: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    pred = np.atleast_2d(np.asarray(pred, dtype=np.float32))
    target = np.atleast_2d(np.asarray(target, dtype=np.float32))
    if pred.shape[1] != target.shape[1]:
        msg = f"action dim mismatch: predicted {pred.shape[1]}, recorded {target.shape[1]}"
        raise ValueError(msg)
    steps = min(len(pred), len(target))
    if steps == 0:
        msg = "empty prediction or target"
        raise ValueError(msg)
    return pred[:steps], target[:steps]


def action_errors(
    preds: Iterable[NDArray[np.float32]], targets: Iterable[NDArray[np.float32]]
) -> ActionErrors:
    """Aggregate errors over frames; each pair is truncated to the shorter horizon."""

    diffs: list[NDArray[np.float32]] = []
    first: list[NDArray[np.float32]] = []
    for pred, target in zip(preds, targets, strict=True):
        aligned_pred, aligned_target = _aligned(pred, target)
        diff = aligned_pred - aligned_target
        diffs.append(diff)
        first.append(diff[0])
    if not diffs:
        msg = "no frames to evaluate"
        raise ValueError(msg)
    stacked = np.concatenate(diffs, axis=0)
    return ActionErrors(
        horizon=max(len(diff) for diff in diffs),
        mae=float(np.mean(np.abs(stacked))),
        rmse=float(np.sqrt(np.mean(stacked**2))),
        first_step_mae=float(np.mean(np.abs(np.stack(first)))),
        mae_per_dim=tuple(float(value) for value in np.mean(np.abs(stacked), axis=0)),
    )


def evaluate_offline(model: BaseVLA, frames: Iterable[OfflineFrame]) -> OfflineEvalResult:
    """Run ``model`` on each recorded frame and score it against the recorded actions."""

    preds: list[NDArray[np.float32]] = []
    targets: list[NDArray[np.float32]] = []
    baseline_preds: list[NDArray[np.float32]] = []
    baseline_targets: list[NDArray[np.float32]] = []
    latencies: list[float] = []
    per_frame: list[dict[str, Any]] = []
    for frame in frames:
        start = perf_counter()
        prediction = model.predict_observation(frame.observation)
        latencies.append((perf_counter() - start) * 1000.0)
        pred = prediction_to_array(prediction)
        frame_errors = action_errors([pred], [frame.target])
        preds.append(pred)
        targets.append(frame.target)
        row: dict[str, Any] = {
            "episode": frame.episode_index,
            "frame": frame.frame_index,
            "mae": frame_errors.mae,
            "first_step_mae": frame_errors.first_step_mae,
        }
        if frame.state is not None and frame.state.size == frame.target.shape[-1]:
            hold = np.repeat(frame.state.reshape(1, -1), len(frame.target), axis=0)
            baseline_preds.append(hold)
            baseline_targets.append(frame.target)
            row["hold_state_mae"] = action_errors([hold], [frame.target]).mae
        per_frame.append(row)
    if not preds:
        msg = "no frames to evaluate"
        raise ValueError(msg)
    return OfflineEvalResult(
        model=model.model_id,
        num_frames=len(preds),
        model_errors=action_errors(preds, targets),
        baseline_errors=(
            action_errors(baseline_preds, baseline_targets) if baseline_preds else None
        ),
        infer_ms_mean=float(np.mean(latencies)),
        per_frame=per_frame,
    )


def iter_lerobot_frames(
    dataset_id: str,
    episodes: Iterable[int],
    *,
    stride: int = 30,
    horizon: int = 50,
    max_frames_per_episode: int | None = None,
) -> Iterator[OfflineFrame]:
    """Yield frames from a LeRobot dataset with the recorded action chunk as target.

    Images are passed through under their dataset keys (``observation.images.top``
    etc.), so a checkpoint fine-tuned on the dataset sees its own camera names.
    Requires the ``[smolvla]`` extra (LeRobot).
    """

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        msg = 'LeRobot datasets require: pip install "vla_ros2[smolvla]"'
        raise MissingDependencyError(msg) from exc

    for episode in episodes:
        dataset = LeRobotDataset(dataset_id, episodes=[int(episode)])
        rows = dataset.hf_dataset["action"]
        actions = np.stack([np.asarray(row, dtype=np.float32) for row in rows])
        camera_keys = list(dataset.meta.camera_keys)
        indices = range(0, len(dataset), max(1, stride))
        if max_frames_per_episode is not None:
            indices = indices[:max_frames_per_episode]
        for index in indices:
            item = dataset[index]
            state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
            yield OfflineFrame(
                observation=VLAObservation(
                    instruction=str(item["task"]),
                    images={key: item[key] for key in camera_keys},
                    state={"observation.state": state},
                ),
                target=actions[index : index + horizon],
                state=state,
                episode_index=int(episode),
                frame_index=int(index),
            )
