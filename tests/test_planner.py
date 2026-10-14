from __future__ import annotations

import pytest

from vla_ros2.core.errors import UnknownModelError
from vla_ros2.planner import (
    TOWER_EXAMPLES,
    HFTowerPlanner,
    PlanError,
    StaticPlanner,
    load_planner,
    parse_plan,
    parse_tower,
    tower_steps,
)
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
    # 4P2 + 4P3 + 4P4 towers, 4 phrasings each
    assert len(tasks) == (12 + 24 + 24) * 4
    goals = {t.phrasing: t.goal for t in tasks if t.tower == ("blue", "green", "red")}
    assert goals["bottom_up"] == (
        "Build a tower with blue at the bottom, then green, then red on top."
    )
    assert goals["top_down"] == "Make a stack with red on top, green under it, blue under it."
    assert goals["chain"] == "Stack the red cube on the green cube on the blue cube."
    assert goals["top_list"] == "From top to bottom, the tower is red, green, blue."


def test_score_plan_requires_exact_order() -> None:
    task = TowerTask(goal="", tower=("blue", "green", "red"), phrasing="chain")
    good = ["Put the green cube on the blue cube.", "Place the red block on top of the green cube"]
    assert score_plan(task, good)
    assert not score_plan(task, list(reversed(good)))
    assert not score_plan(task, ["Put the green cube on the blue cube."])
    assert plan_pairs(["Pick up the red cube."]) is None


def test_parse_tower_and_compile_steps() -> None:
    assert parse_tower('Bottom to top: ["Blue", "green cube", " red "]') == [
        "blue",
        "green",
        "red",
    ]
    assert tower_steps(["blue", "green", "red"]) == [
        "Put the green cube on the blue cube.",
        "Put the red cube on the green cube.",
    ]


@pytest.mark.parametrize(
    "reply", ['["red"]', '["red", "red"]', '["light blue", "red"]', '["r2", "d2"]', "none"]
)
def test_parse_tower_rejects_bad_replies(reply: str) -> None:
    with pytest.raises(PlanError):
        parse_tower(reply)


def test_tower_examples_score_as_correct_plans() -> None:
    for goal, tower in TOWER_EXAMPLES:
        task = TowerTask(goal=goal, tower=tuple(tower), phrasing="example")
        assert score_plan(task, tower_steps(tower))


def test_tower_planner_parses_reply_without_a_model() -> None:
    planner = HFTowerPlanner.__new__(HFTowerPlanner)
    assert planner.parse('["gray", "red"]') == ["Put the red cube on the gray cube."]
    assert planner.last_tower == ["gray", "red"]
