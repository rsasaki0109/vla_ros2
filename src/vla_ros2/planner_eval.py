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
