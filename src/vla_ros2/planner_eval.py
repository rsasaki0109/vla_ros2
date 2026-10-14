"""Tower-building goals with a single correct plan, to score planners objectively.

A tower bottom -> top (A, B, C) has exactly one valid instruction list:
"Put the B cube on the A cube.", "Put the C cube on the B cube.". Goals use
colors that do not appear in the planners' few-shot examples. The few-shot examples
of ``HFTowerPlanner`` cover the ``bottom_up``, ``top_down`` and ``chain`` phrasings;
``HELD_OUT_PHRASINGS`` are in no example.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import permutations

EVAL_COLORS = ("red", "green", "blue", "gray")
SEEN_PHRASINGS = ("bottom_up", "top_down", "chain")
HELD_OUT_PHRASINGS = ("top_list", "bottom_list", "on_top_of", "top_sentences")
_STEP_RE = re.compile(
    r"^\s*(?:put|place|stack)\s+the\s+(\w+)\s+(?:cube|block)\s+on\s+(?:top\s+of\s+)?"
    r"the\s+(\w+)\s+(?:cube|block)\.?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TowerTask:
    goal: str
    tower: tuple[str, ...]
    """Colors from bottom to top."""
    phrasing: str

    @property
    def expected(self) -> list[tuple[str, str]]:
        return [(top, below) for below, top in zip(self.tower, self.tower[1:], strict=False)]


def _phrase(tower: Sequence[str], phrasing: str) -> str:
    if phrasing == "bottom_up":
        middle = "".join(f", then {color}" for color in tower[1:-1])
        return f"Build a tower with {tower[0]} at the bottom{middle}, then {tower[-1]} on top."
    if phrasing == "top_down":
        rest = ", ".join(f"{color} under it" for color in reversed(tower[:-1]))
        return f"Make a stack with {tower[-1]} on top, {rest}."
    if phrasing == "chain":
        chain = " on ".join(f"the {color} cube" for color in reversed(tower))
        return f"Stack {chain}."
    if phrasing == "top_list":
        return f"From top to bottom, the tower is {', '.join(reversed(tower))}."
    if phrasing == "bottom_list":
        return f"From bottom to top, the tower is {', '.join(tower)}."
    if phrasing == "on_top_of":
        pairs = reversed(list(zip(tower[1:], tower, strict=False)))
        return f"Put {', '.join(f'{upper} on top of {lower}' for upper, lower in pairs)}."
    if phrasing == "top_sentences":
        below = "".join(f" Below it is {color}." for color in reversed(tower[1:-1]))
        top, bottom = tower[-1].capitalize(), tower[0].capitalize()
        return f"{top} goes on top.{below} {bottom} is at the base."
    msg = f"unknown phrasing {phrasing!r}"
    raise ValueError(msg)


def tower_tasks(
    colors: Sequence[str] = EVAL_COLORS,
    heights: Sequence[int] = (2, 3, 4),
    phrasings: Sequence[str] = SEEN_PHRASINGS + HELD_OUT_PHRASINGS,
) -> Iterator[TowerTask]:
    for height in heights:
        for tower in permutations(colors, height):
            for phrasing in phrasings:
                yield TowerTask(goal=_phrase(tower, phrasing), tower=tower, phrasing=phrasing)


def plan_pairs(steps: Sequence[str]) -> list[tuple[str, str]] | None:
    """(moved, target) per step, or None if any step is not a put-X-on-Y instruction."""

    pairs = []
    for step in steps:
        match = _STEP_RE.match(step)
        if match is None:
            return None
        pairs.append((match.group(1).lower(), match.group(2).lower()))
    return pairs


def score_plan(task: TowerTask, steps: Sequence[str]) -> bool:
    return plan_pairs(steps) == task.expected


MULTI_TOWER_COLORS = EVAL_COLORS + ("brown", "black")


@dataclass(frozen=True)
class MultiTowerTask:
    """Several towers from cubes that all start on the table; any valid order is fine."""

    goal: str
    towers: tuple[tuple[str, ...], ...]
    """Each tower's colors from bottom to top."""
    phrasing: str

    @property
    def expected_on(self) -> dict[str, str]:
        return {
            top: below
            for tower in self.towers
            for below, top in zip(tower, tower[1:], strict=False)
        }


def simulate(steps: Sequence[str]) -> dict[str, str] | None:
    """Final {cube: cube it sits on} after ``steps``, starting with every cube on the table.

    Returns None if a step is not a put-X-on-Y instruction or cannot be done: X or Y
    has a cube on it, or X is Y.
    """

    pairs = plan_pairs(steps)
    if pairs is None:
        return None
    on: dict[str, str] = {}
    for moved, target in pairs:
        covered = set(on.values())
        if moved == target or moved in covered or target in covered:
            return None
        on[moved] = target
    return on


def score_multi_plan(task: MultiTowerTask, steps: Sequence[str]) -> bool:
    return simulate(steps) == task.expected_on


def multi_tower_tasks(
    colors: Sequence[str] = MULTI_TOWER_COLORS,
    heights: Sequence[tuple[int, int]] = ((2, 2), (2, 3), (3, 3)),
    per_phrasing: int = 10,
    phrasings: Sequence[str] = SEEN_PHRASINGS + HELD_OUT_PHRASINGS,
) -> Iterator[MultiTowerTask]:
    """Two-tower goals: both towers in one phrasing, then mixed-phrasing pairs."""

    candidates = [
        (perm[:first], perm[first:])
        for first, second in heights
        for perm in permutations(colors, first + second)
    ]
    count = per_phrasing * (len(phrasings) + 1)
    picked = candidates[:: max(1, len(candidates) // count)][:count]
    combos = [(p, p) for p in phrasings] + [
        (phrasings[i], phrasings[(i + 3) % len(phrasings)]) for i in range(len(phrasings))
    ]
    for index, (first, second) in enumerate(picked):
        if index < per_phrasing * len(phrasings):
            p1, p2 = combos[index // per_phrasing]
        else:
            p1, p2 = combos[len(phrasings) + index % len(phrasings)]
        goal = (
            f"Build two separate towers. First: {_phrase(first, p1)} Second: {_phrase(second, p2)}"
        )
        label = p1 if p1 == p2 else "mixed"
        yield MultiTowerTask(goal=goal, towers=(first, second), phrasing=label)
