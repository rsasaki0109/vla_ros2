#!/usr/bin/env python3
"""Score a planner on tower-building goals with one correct plan each.

pip install -e ".[vlm]"
python scripts/planner_eval.py --pretrained Qwen/Qwen2.5-1.5B-Instruct --every 3
python scripts/planner_eval.py --planner hf_tower --pretrained Qwen/Qwen2.5-1.5B-Instruct --every 3
python scripts/planner_eval.py --planner hf_tower --goals multi

--examples tower gives the step planner (hf) the same few-shot goals as hf_tower,
written as steps, to separate the effect of the examples from the format.

"wrong" counts plans that were returned but are not correct (what a robot would run);
"refused" counts PlanError, including hf_tower's cross-check disagreeing. For
hf_tower, "unchecked" scores the first answer alone, as with --no-cross-check.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from vla_ros2.planner import (
    TOWER_EXAMPLES,
    PlanError,
    load_planner,
    parse_towers,
    towers_steps,
)
from vla_ros2.planner_eval import multi_tower_tasks, score_multi_plan, score_plan, tower_tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", default="hf")
    parser.add_argument("--pretrained", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--every", type=int, default=1, help="evaluate every Nth tower (all its phrasings)"
    )
    parser.add_argument("--examples", choices=("default", "tower"), default="default")
    parser.add_argument(
        "--goals",
        choices=("tower", "multi"),
        default="tower",
        help="one tower per goal (tower_tasks) or two towers (multi_tower_tasks)",
    )
    parser.add_argument("--no-cross-check", action="store_true", help="hf_tower only")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    kwargs: dict[str, Any] = {"pretrained": args.pretrained, "device": args.device}
    if args.planner == "hf_tower":
        kwargs["cross_check"] = not args.no_cross_check
    if args.examples == "tower" and args.planner == "hf":
        kwargs["examples"] = [
            (goal, towers_steps(parse_towers(json.dumps(out)))) for goal, out in TOWER_EXAMPLES
        ]
    planner = load_planner(args.planner, **kwargs)
    tasks: list[Any]
    if args.goals == "multi":
        tasks = list(multi_tower_tasks())[:: max(1, args.every)]
    else:
        towers = sorted({task.tower for task in tower_tasks()}, key=lambda t: (len(t), t))
        keep = set(towers[:: max(1, args.every)])
        tasks = [task for task in tower_tasks() if task.tower in keep]
    rows: list[dict[str, Any]] = []
    by_group: dict[str, list[bool]] = defaultdict(list)
    start = time.perf_counter()
    score = score_multi_plan if args.goals == "multi" else score_plan
    for task in tasks:
        try:
            steps = planner.plan(task.goal)
            error = None
        except PlanError as exc:
            steps, error = [], str(exc)
        correct = error is None and score(task, steps)
        unchecked = getattr(planner, "last_unchecked", None)
        row = {"goal": task.goal, "steps": steps, "correct": correct, "error": error}
        for key in ("last_reply", "last_check_reply"):
            if getattr(planner, key, None) is not None:
                row[key.removeprefix("last_")] = getattr(planner, key)
        if unchecked is not None:
            row["unchecked_steps"] = unchecked
            row["unchecked_correct"] = bool(unchecked) and score(task, unchecked)
        rows.append(row)
        if args.goals != "multi":
            by_group[f"height {len(task.tower)}"].append(correct)
        by_group[task.phrasing].append(correct)
    elapsed = time.perf_counter() - start

    total = sum(row["correct"] for row in rows)
    refused = sum(row["error"] is not None for row in rows)
    wrong = len(rows) - total - refused
    unchecked_total = sum(row.get("unchecked_correct", False) for row in rows)
    print(
        f"{args.planner}/{args.examples} {args.goals} {args.pretrained}: "
        f"{total}/{len(rows)} correct, {wrong} wrong, {refused} refused"
        + (f", unchecked {unchecked_total} correct" if "unchecked_correct" in rows[0] else "")
        + f", {elapsed / len(rows):.1f} s/goal"
    )
    for group, results in sorted(by_group.items()):
        print(f"  {group:10} {sum(results)}/{len(results)}")
    if args.out:
        payload = {
            "planner": args.planner,
            "examples": args.examples,
            "goals": args.goals,
            "pretrained": args.pretrained,
            "device": args.device,
            "correct": total,
            "tasks": len(rows),
            "wrong": wrong,
            "refused": refused,
            "unchecked_correct": unchecked_total,
            "sec_per_goal": elapsed / len(rows),
            "by_group": {k: [sum(v), len(v)] for k, v in sorted(by_group.items())},
            "rows": rows,
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
