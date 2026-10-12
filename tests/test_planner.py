from __future__ import annotations

import pytest

from vla_ros2.core.errors import UnknownModelError
from vla_ros2.planner import PlanError, StaticPlanner, load_planner, parse_plan
from vla_ros2.planner_eval import TowerTask, plan_pairs, score_plan, tower_tasks


def test_parse_plan_extracts_first_json_array() -> None:
    reply = (
        'Sure!\n["Put the green cube on the blue cube.",  "Put the red  cube on the green cube."]'
    )
    assert parse_plan(reply) == [
        "Put the green cube on the blue cube.",
        "Put the red cube on the green cube.",
    ]


@pytest.mark.parametrize(
    "reply",
    ["no list here", "[1, 2]", "[]", '["a", ', '["' + "word " * 30 + '"]', str(["x"] * 11)],
)
def test_parse_plan_rejects_bad_replies(reply: str) -> None:
    with pytest.raises(PlanError):
        parse_plan(reply.replace("'", '"'))


def test_static_planner_and_registry() -> None:
    assert StaticPlanner().plan(" do it ") == ["do it"]
    assert load_planner("static", steps=["a", "b"]).plan("goal") == ["a", "b"]
    with pytest.raises(UnknownModelError):
        load_planner("nope")


def test_tower_goals_and_expected_plans() -> None:
    task = TowerTask(goal="", tower=("blue", "green", "red"), phrasing="bottom_up")
    assert task.expected == [("green", "blue"), ("red", "green")]
    tasks = list(tower_tasks())
    # 4P2 + 4P3 + 4P4 towers, 3 phrasings each
    assert len(tasks) == (12 + 24 + 24) * 3
    goals = {t.phrasing: t.goal for t in tasks if t.tower == ("blue", "green", "red")}
    assert goals["bottom_up"] == (
        "Build a tower with blue at the bottom, then green, then red on top."
    )
    assert goals["top_down"] == "Make a stack with red on top, green under it, blue under it."
    assert goals["chain"] == "Stack the red cube on the green cube on the blue cube."


def test_score_plan_requires_exact_order() -> None:
    task = TowerTask(goal="", tower=("blue", "green", "red"), phrasing="chain")
    good = ["Put the green cube on the blue cube.", "Place the red block on top of the green cube"]
    assert score_plan(task, good)
    assert not score_plan(task, list(reversed(good)))
    assert not score_plan(task, ["Put the green cube on the blue cube."])
    assert plan_pairs(["Pick up the red cube."]) is None
