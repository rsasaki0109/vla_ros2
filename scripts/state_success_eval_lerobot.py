#!/usr/bin/env python3
"""Evaluate the grasp-release success detector on recorded LeRobot episodes.

Uses only the recorded joint states and timestamps (no video decoding). For each
episode it reports whether and where (fraction of the episode) the detector
fires, and checks the 30% and 50% prefixes, before the place, where it must not.

    pip install -e ".[smolvla]"
    python scripts/state_success_eval_lerobot.py --train-episodes 0-9 --out results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vla_ros2.task_success import GraspReleaseConfig, detect_success_time

DEFAULT_DATASET = "lerobot/svla_so100_stacking"


def _parse_range(text: str) -> set[int]:
    start, _, end = text.partition("-")
    return set(range(int(start), int(end or start) + 1))


def evaluate_episode(
    states: np.ndarray, stamps: np.ndarray, config: GraspReleaseConfig
) -> dict[str, Any]:
    length = len(states)
    fired = detect_success_time(states, stamps, config)
    row: dict[str, Any] = {
        "frames": length,
        "fired": fired is not None,
        "fire_fraction": None,
    }
    if fired is not None and length > 1:
        row["fire_fraction"] = float((fired - stamps[0]) / (stamps[-1] - stamps[0]))
    for cut in (0.3, 0.5):
        stop = int(cut * length)
        early = detect_success_time(states[:stop], stamps[:stop], config) if stop else None
        row[f"fired_by_{int(cut * 100)}pct"] = early is not None
    return row


def summarize(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    fractions = [row["fire_fraction"] for row in rows.values() if row["fired"]]
    return {
        "episodes": len(rows),
        "fired": sum(row["fired"] for row in rows.values()),
        "missed": sorted(e for e, row in rows.items() if not row["fired"]),
        "false_alarm_by_30pct": sum(row["fired_by_30pct"] for row in rows.values()),
        "false_alarm_by_50pct": sum(row["fired_by_50pct"] for row in rows.values()),
        "fire_fraction_median": float(np.median(fractions)) if fractions else None,
        "fire_fraction_min": float(min(fractions)) if fractions else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--train-episodes", default="0-9", help="range used to tune thresholds")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    data = LeRobotDataset(args.dataset).hf_dataset
    episodes = np.asarray(data["episode_index"])
    states = np.stack([np.asarray(row, dtype=np.float64) for row in data["observation.state"]])
    stamps = np.asarray(data["timestamp"], dtype=np.float64).reshape(-1)
    config = GraspReleaseConfig()
    train = _parse_range(args.train_episodes)

    rows: dict[int, dict[str, Any]] = {}
    for episode in sorted(set(episodes.tolist())):
        mask = episodes == episode
        rows[int(episode)] = evaluate_episode(states[mask], stamps[mask], config)

    report = {
        "train": summarize({e: r for e, r in rows.items() if e in train}),
        "held_out": summarize({e: r for e, r in rows.items() if e not in train}),
    }
    for split, summary in report.items():
        print(
            f"{split:8} fired {summary['fired']}/{summary['episodes']} "
            f"missed={summary['missed']} false alarms @30%={summary['false_alarm_by_30pct']} "
            f"@50%={summary['false_alarm_by_50pct']} "
            f"fire at median {summary['fire_fraction_median']:.2f} "
            f"(earliest {summary['fire_fraction_min']:.2f}) of the episode"
        )
    if args.out:
        payload = {"dataset": args.dataset, "config": vars(config), **report, "episodes": rows}
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
