"""Turn one goal into a list of short instructions for the task sequencer.

The VLA executes single instructions in its training style ("Put the red cube on
the blue cube."); a planner writes that list from a goal ("build a tower: blue at
the bottom, then green, then red"). ``HFPlanner`` asks a local instruction-tuned
LLM for a JSON array; ``parse_plan`` extracts and validates it. ``HFTowerPlanner``
asks, per tower, only for the colors in the order the goal names them plus whether
that order runs top-down or bottom-up; code puts them bottom to top (``parse_towers``)
and writes the instructions (``towers_steps``), so the model never reorders anything.
With ``cross_check`` it also asks for plain bottom-to-top lists
(``parse_tower_lists``) and refuses the plan when the two answers disagree. Plans
are only suggestions for the sequencer: they never command the robot directly.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from vla_ros2.core.errors import MissingDependencyError, UnknownModelError

DEFAULT_SYSTEM_PROMPT = (
    "You plan tasks for a robot arm that can only follow one short instruction at a "
    'time, of the form "Put the <color> cube on the <color> cube." Reply with a JSON '
    "array of instruction strings, in execution order, and nothing else."
)

DEFAULT_EXAMPLES: tuple[tuple[str, list[str]], ...] = (
    (
        "Build a tower with yellow at the bottom, then purple, then orange on top.",
        ["Put the purple cube on the yellow cube.", "Put the orange cube on the purple cube."],
    ),
    (
        "Stack the pink cube on the white cube.",
        ["Put the pink cube on the white cube."],
    ),
)


TOWER_SYSTEM_PROMPT = (
    "You read goals for a robot arm that stacks colored cubes into towers. For each "
    "tower, list the cube colors in the order the goal mentions them, and say whether "
    'that order runs from the top of the tower down ("top_down") or from the bottom up '
    '("bottom_up"). Reply with a JSON array holding one object {"order": "top_down" or '
    '"bottom_up", "colors": [...]} per tower, one lowercase word per color, and nothing '
    "else."
)

_EXAMPLE_GOALS = (
    "Build a tower with yellow at the bottom, then purple, then orange on top.",
    "Make a stack with white on top, pink under it, yellow under it.",
    "Stack the orange cube on the white cube on the purple cube.",
    "Make two stacks: one with white at the bottom and pink on top, and one with orange "
    "on top of yellow.",
)

TOWER_EXAMPLES: tuple[tuple[str, list[dict[str, Any]]], ...] = tuple(
    zip(
        _EXAMPLE_GOALS,
        (
            [{"order": "bottom_up", "colors": ["yellow", "purple", "orange"]}],
            [{"order": "top_down", "colors": ["white", "pink", "yellow"]}],
            [{"order": "top_down", "colors": ["orange", "white", "purple"]}],
            [
                {"order": "bottom_up", "colors": ["white", "pink"]},
                {"order": "top_down", "colors": ["orange", "yellow"]},
            ],
        ),
        strict=True,
    )
)

# Cross-check prompts: plain bottom-to-top lists. A one-tower answer is checked with a
# one-tower prompt, since a multi-tower example makes small models split single towers.
CHECK_SYSTEM_PROMPT = (
    "You read goals for a robot arm that stacks colored cubes into one tower. Reply "
    "with a JSON array of the cube colors from the BOTTOM of the tower to the TOP, "
    "one lowercase word per color, and nothing else."
)

CHECK_EXAMPLES: tuple[tuple[str, list[str]], ...] = tuple(
    zip(
        _EXAMPLE_GOALS[:3],
        (
            ["yellow", "purple", "orange"],
            ["yellow", "pink", "white"],
            ["purple", "white", "orange"],
        ),
        strict=True,
    )
)

MULTI_CHECK_SYSTEM_PROMPT = (
    "You read goals for a robot arm that stacks colored cubes into towers. Reply with a "
    "JSON array holding one array per tower, each listing that tower's cube colors from "
    "the BOTTOM of the tower to the TOP, one lowercase word per color, and nothing else."
)

MULTI_CHECK_EXAMPLES: tuple[tuple[str, list[list[str]]], ...] = (
    *((goal, [colors]) for goal, colors in CHECK_EXAMPLES),
    (_EXAMPLE_GOALS[3], [["white", "pink"], ["yellow", "orange"]]),
)


class PlanError(ValueError):
    """Raised when a planner reply cannot be turned into a usable plan."""


def _first_string_array(text: str) -> list[str]:
    match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if match is None:
        msg = f"no JSON array in planner reply: {text[:200]!r}"
        raise PlanError(msg)
    try:
        parsed: Any = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        msg = f"planner reply is not valid JSON: {match.group(0)[:200]!r}"
        raise PlanError(msg) from exc
    if not isinstance(parsed, list) or not all(isinstance(step, str) for step in parsed):
        msg = "planner reply must be a JSON array of strings"
        raise PlanError(msg)
    return parsed


def parse_plan(text: str, *, max_steps: int = 10, max_words: int = 20) -> list[str]:
    """Extract the first JSON array of strings from ``text`` and validate it."""

    parsed = _first_string_array(text)
    steps = [" ".join(step.split()) for step in parsed if step.strip()]
    if not steps:
        msg = "planner returned an empty plan"
        raise PlanError(msg)
    if len(steps) > max_steps:
        msg = f"planner returned {len(steps)} steps (max {max_steps})"
        raise PlanError(msg)
    for step in steps:
        if len(step.split()) > max_words:
            msg = f"planner step too long ({len(step.split())} words): {step!r}"
            raise PlanError(msg)
    return steps


def _first_json(text: str) -> Any:
    """Decode the first JSON object or array in ``text``.

    Small models often stop right before the closing ``]`` of an array (``[{...}``),
    so a reply that ends inside an array gets up to two brackets appended.
    """

    match = re.search(r"[\[{]", text)
    if match is None:
        msg = f"no JSON in planner reply: {text[:200]!r}"
        raise PlanError(msg)
    body = text[match.start() :].rstrip()
    for suffix in ("", "]", "]]"):
        try:
            parsed, _ = json.JSONDecoder().raw_decode(body + suffix)
        except json.JSONDecodeError:
            continue
        return parsed
    msg = f"planner reply is not valid JSON: {body[:200]!r}"
    raise PlanError(msg)


def _tower_colors(items: Any, *, max_height: int) -> list[str]:
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        msg = f"tower colors must be a list of strings: {items!r}"
        raise PlanError(msg)
    colors = []
    for item in items:
        words = item.lower().replace("cube", " ").replace("block", " ").split()
        if len(words) != 1 or not words[0].isalpha():
            msg = f"tower entry is not one color word: {item!r}"
            raise PlanError(msg)
        colors.append(words[0])
    if len(colors) < 2:
        msg = f"a tower needs at least 2 cubes, got {colors}"
        raise PlanError(msg)
    if len(colors) > max_height:
        msg = f"tower of {len(colors)} cubes (max {max_height})"
        raise PlanError(msg)
    return colors


def _check_towers(towers: list[list[str]], *, max_towers: int) -> list[list[str]]:
    if not towers:
        msg = "planner returned no towers"
        raise PlanError(msg)
    if len(towers) > max_towers:
        msg = f"planner returned {len(towers)} towers (max {max_towers})"
        raise PlanError(msg)
    colors = [color for tower in towers for color in tower]
    if len(set(colors)) != len(colors):
        msg = f"towers repeat a color: {towers}"
        raise PlanError(msg)
    return towers


def parse_towers(text: str, *, max_height: int = 8, max_towers: int = 4) -> list[list[str]]:
    """Bottom-to-top towers from ``[{"order": ..., "colors": [...]}, ...]`` in ``text``.

    A single object (no array) is one tower. ``"top_down"`` lists are reversed here.
    """

    parsed = _first_json(text)
    objects = [parsed] if isinstance(parsed, dict) else parsed
    if not isinstance(objects, list) or not all(isinstance(obj, dict) for obj in objects):
        msg = f"planner reply must be an array of tower objects: {parsed!r}"
        raise PlanError(msg)
    towers = []
    for obj in objects:
        if obj.get("order") not in ("top_down", "bottom_up"):
            msg = f'tower needs "order": "top_down" or "bottom_up": {obj!r}'
            raise PlanError(msg)
        colors = _tower_colors(obj.get("colors"), max_height=max_height)
        towers.append(colors[::-1] if obj["order"] == "top_down" else colors)
    return _check_towers(towers, max_towers=max_towers)


def parse_tower_lists(text: str, *, max_height: int = 8, max_towers: int = 4) -> list[list[str]]:
    """Bottom-to-top towers from ``[[...], ...]`` in ``text``; a flat list is one tower."""

    parsed = _first_json(text)
    if isinstance(parsed, list) and parsed and all(isinstance(item, str) for item in parsed):
        parsed = [parsed]
    if not isinstance(parsed, list):
        msg = f"planner reply must be an array of color arrays: {parsed!r}"
        raise PlanError(msg)
    towers = [_tower_colors(items, max_height=max_height) for items in parsed]
    return _check_towers(towers, max_towers=max_towers)


def tower_steps(tower: Sequence[str]) -> list[str]:
    """Instructions that build ``tower`` (colors bottom to top), one cube per step."""

    return [
        f"Put the {top} cube on the {below} cube."
        for below, top in zip(tower, tower[1:], strict=False)
    ]


def towers_steps(towers: Sequence[Sequence[str]]) -> list[str]:
    """Instructions that build each tower in turn, starting from cubes on the table."""

    return [step for tower in towers for step in tower_steps(tower)]


class BasePlanner(ABC):
    name = "base"
    model_id = "none"

    @abstractmethod
    def plan(self, goal: str) -> list[str]:
        """Return instructions for ``goal`` in execution order."""


class StaticPlanner(BasePlanner):
    """Returns preset steps (or the goal itself); for tests and wiring checks."""

    name = "static"

    def __init__(self, steps: Sequence[str] | None = None, **_: object) -> None:
        self.steps = list(steps or [])

    def plan(self, goal: str) -> list[str]:
        return list(self.steps) if self.steps else [goal.strip()]


class HFPlanner(BasePlanner):
    """Local instruction-tuned LLM via ``transformers`` (greedy decoding)."""

    name = "hf"

    def __init__(
        self,
        *,
        pretrained: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        examples: Sequence[tuple[str, Any]] = DEFAULT_EXAMPLES,
        max_new_tokens: int = 160,
        retries: int = 1,
        **_: object,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            msg = 'HFPlanner requires transformers and torch: pip install "vla_ros2[vlm]"'
            raise MissingDependencyError(msg) from exc
        self._torch = torch
        self.model_id = pretrained
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained, dtype=dtype).to(self.device)
        self.model.eval()
        self.system_prompt = system_prompt
        self.examples = list(examples)
        self.max_new_tokens = max_new_tokens
        self.retries = retries
        self.last_reply = ""

    def parse(self, reply: str) -> list[str]:
        return parse_plan(reply)

    def _messages(
        self,
        goal: str,
        system_prompt: str | None = None,
        examples: Sequence[tuple[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt or self.system_prompt}]
        for example_goal, example_steps in self.examples if examples is None else examples:
            messages.append({"role": "user", "content": example_goal})
            messages.append({"role": "assistant", "content": json.dumps(example_steps)})
        messages.append({"role": "user", "content": goal.strip()})
        return messages

    def generate(self, goal: str, *, messages: list[dict[str, str]] | None = None) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages if messages is not None else self._messages(goal),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

    def plan(self, goal: str) -> list[str]:
        error: PlanError | None = None
        for _ in range(self.retries + 1):
            self.last_reply = self.generate(goal)
            try:
                return self.parse(self.last_reply)
            except PlanError as exc:
                error = exc
        assert error is not None
        raise error


class HFTowerPlanner(HFPlanner):
    """``HFPlanner`` that asks for each tower's colors and direction, not for steps.

    With ``cross_check`` (default) it asks a second time for plain bottom-to-top lists
    (``CHECK_*`` for one tower, ``MULTI_CHECK_*`` for more) and raises ``PlanError``
    unless both answers give the same towers.
    """

    name = "hf_tower"

    def __init__(
        self,
        *,
        system_prompt: str = TOWER_SYSTEM_PROMPT,
        examples: Sequence[tuple[str, Any]] = TOWER_EXAMPLES,
        cross_check: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(system_prompt=system_prompt, examples=examples, **kwargs)
        self.cross_check = cross_check
        self.last_towers: list[list[str]] = []
        self.last_check_towers: list[list[str]] = []
        self.last_check_reply = ""
        self.last_unchecked: list[str] = []

    def parse(self, reply: str) -> list[str]:
        self.last_towers = parse_towers(reply)
        return towers_steps(self.last_towers)

    def plan(self, goal: str) -> list[str]:
        self.last_towers, self.last_check_towers, self.last_unchecked = [], [], []
        self.last_check_reply = ""
        steps = super().plan(goal)
        self.last_unchecked = steps
        if not self.cross_check:
            return steps
        if len(self.last_towers) == 1:
            messages = self._messages(goal, CHECK_SYSTEM_PROMPT, CHECK_EXAMPLES)
        else:
            messages = self._messages(goal, MULTI_CHECK_SYSTEM_PROMPT, MULTI_CHECK_EXAMPLES)
        self.last_check_reply = self.generate(goal, messages=messages)
        self.last_check_towers = parse_tower_lists(self.last_check_reply)
        if sorted(self.last_check_towers) != sorted(self.last_towers):
            msg = (
                f"cross-check disagrees: {self.last_towers} vs {self.last_check_towers} "
                "(bottom to top); not planning"
            )
            raise PlanError(msg)
        return steps


_PLANNERS: dict[str, type[BasePlanner]] = {
    "static": StaticPlanner,
    "hf": HFPlanner,
    "hf_tower": HFTowerPlanner,
}


def load_planner(name: str, **kwargs: Any) -> BasePlanner:
    try:
        factory = _PLANNERS[name]
    except KeyError as exc:
        msg = f"Unknown planner {name!r}. Known planners: {', '.join(_PLANNERS)}"
        raise UnknownModelError(msg) from exc
    return factory(**kwargs)
