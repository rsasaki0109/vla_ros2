"""Turn one goal into a list of short instructions for the task sequencer.

The VLA executes single instructions in its training style ("Put the red cube on
the blue cube."); a planner writes that list from a goal ("build a tower: blue at
the bottom, then green, then red"). ``HFPlanner`` asks a local instruction-tuned
LLM for a JSON array; ``parse_plan`` extracts and validates it. Plans are only
suggestions for the sequencer: they never command the robot directly.
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


class PlanError(ValueError):
    """Raised when a planner reply cannot be turned into a usable plan."""


def parse_plan(text: str, *, max_steps: int = 10, max_words: int = 20) -> list[str]:
    """Extract the first JSON array of strings from ``text`` and validate it."""

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
        examples: Sequence[tuple[str, list[str]]] = DEFAULT_EXAMPLES,
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

    def _messages(self, goal: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for example_goal, example_steps in self.examples:
            messages.append({"role": "user", "content": example_goal})
            messages.append({"role": "assistant", "content": json.dumps(example_steps)})
        messages.append({"role": "user", "content": goal.strip()})
        return messages

    def generate(self, goal: str) -> str:
        inputs = self.tokenizer.apply_chat_template(
            self._messages(goal),
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
                return parse_plan(self.last_reply)
            except PlanError as exc:
                error = exc
        assert error is not None
        raise error


_PLANNERS: dict[str, type[BasePlanner]] = {"static": StaticPlanner, "hf": HFPlanner}


def load_planner(name: str, **kwargs: Any) -> BasePlanner:
    try:
        factory = _PLANNERS[name]
    except KeyError as exc:
        msg = f"Unknown planner {name!r}. Known planners: {', '.join(_PLANNERS)}"
        raise UnknownModelError(msg) from exc
    return factory(**kwargs)
