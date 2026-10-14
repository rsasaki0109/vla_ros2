#!/usr/bin/env python3
"""Score a planner on tower-building goals with one correct plan each.

pip install -e ".[vlm]"
python scripts/planner_eval.py --pretrained Qwen/Qwen2.5-1.5B-Instruct --every 3
python scripts/planner_eval.py --planner hf_tower --pretrained Qwen/Qwen2.5-3B-Instruct --every 3

--examples tower gives the step planner (hf) the same three few-shot goals as
hf_tower, written as steps, to separate the effect of the examples from the format.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from vla_ros2.planner import TOWER_EXAMPLES, PlanError, load_planner, tower_steps
from vla_ros2.planner_eval import score_plan, tower_tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", default="hf")
    parser.add_argument("--pretrained", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--every", type=int, default=1, help="evaluate every Nth tower (all its phrasings)"
    )
    parser.add_argument("--examples", choices=("default", "tower"), default="default")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    kwargs: dict[str, Any] = {"pretrained": args.pretrained, "device": args.device}
    if args.examples == "tower" and args.planner == "hf":
        kwargs["examples"] = [(goal, tower_steps(tower)) for goal, tower in TOWER_EXAMPLES]
    planner = load_planner(args.planner, **kwargs)
    towers = sorted({task.tower for task in tower_tasks()}, key=lambda t: (len(t), t))
    keep = set(towers[:: max(1, args.every)])
    tasks = [task for task in tower_tasks() if task.tower in keep]
    rows: list[dict[str, Any]] = []
    by_group: dict[str, list[bool]] = defaultdict(list)
    start = time.perf_counter()
    for task in tasks:
        try:
            steps = planner.plan(task.goal)
            error = None
        except PlanError as exc:
            steps, error = [], str(exc)
        correct = error is None and score_plan(task, steps)
        rows.append({"goal": task.goal, "steps": steps, "correct": correct, "error": error})
        by_group[f"height {len(task.tower)}"].append(correct)
        by_group[task.phrasing].append(correct)
    elapsed = time.perf_counter() - start

    total = sum(row["correct"] for row in rows)
    parse_failures = sum(row["error"] is not None for row in rows)
    print(
        f"{args.planner}/{args.examples} {args.pretrained}: {total}/{len(rows)} correct "
        f"({100 * total / len(rows):.0f}%), parse failures {parse_failures}, "
        f"{elapsed / len(rows):.1f} s/goal"
    )
    for group, results in sorted(by_group.items()):
        print(f"  {group:10} {sum(results)}/{len(results)}")
    if args.out:
        payload = {
            "planner": args.planner,
            "examples": args.examples,
            "pretrained": args.pretrained,
            "device": args.device,
            "correct": total,
            "tasks": len(rows),
            "parse_failures": parse_failures,
            "sec_per_goal": elapsed / len(rows),
            "by_group": {k: [sum(v), len(v)] for k, v in sorted(by_group.items())},
            "rows": rows,
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
