from __future__ import annotations

import json

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
from vla_ros2.planner_eval import (
    MultiTowerTask,
    TowerTask,
    multi_tower_tasks,
    plan_pairs,
    score_multi_plan,
    score_plan,
    simulate,
    tower_tasks,
)


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
    # 4P2 + 4P3 + 4P4 towers, 7 phrasings each
    assert len(tasks) == (12 + 24 + 24) * 7
    goals = {t.phrasing: t.goal for t in tasks if t.tower == ("blue", "green", "red")}
    assert goals["bottom_up"] == (
        "Build a tower with blue at the bottom, then green, then red on top."
    )
    assert goals["top_down"] == "Make a stack with red on top, green under it, blue under it."
    assert goals["chain"] == "Stack the red cube on the green cube on the blue cube."
    assert goals["top_list"] == "From top to bottom, the tower is red, green, blue."
    assert goals["bottom_list"] == "From bottom to top, the tower is blue, green, red."
    assert goals["on_top_of"] == "Put red on top of green, green on top of blue."
    assert goals["top_sentences"] == "Red goes on top. Below it is green. Blue is at the base."


def test_score_plan_requires_exact_order() -> None:
    task = TowerTask(goal="", tower=("blue", "green", "red"), phrasing="chain")
    good = ["Put the green cube on the blue cube.", "Place the red block on top of the green cube"]
    assert score_plan(task, good)
    assert not score_plan(task, list(reversed(good)))
    assert not score_plan(task, ["Put the green cube on the blue cube."])
    assert plan_pairs(["Pick up the red cube."]) is None


def test_parse_tower_and_compile_steps() -> None:
    reply = 'Sure: {"order": "top_down", "colors": ["Red", "green cube", " blue "]}'
    assert parse_tower(reply) == ["blue", "green", "red"]
    assert parse_tower('{"order": "bottom_up", "colors": ["blue", "red"]}') == ["blue", "red"]
    assert tower_steps(["blue", "green", "red"]) == [
        "Put the green cube on the blue cube.",
        "Put the red cube on the green cube.",
    ]


@pytest.mark.parametrize(
    "reply",
    [
        '{"order": "up", "colors": ["red", "blue"]}',
        '{"order": "top_down", "colors": "red, blue"}',
        '{"order": "top_down", "colors": ["red"]}',
        '{"order": "top_down", "colors": ["red", "red"]}',
        '{"order": "top_down", "colors": ["light blue", "red"]}',
        '["red", "blue"]',
        "{not json}",
        "none",
    ],
)
def test_parse_tower_rejects_bad_replies(reply: str) -> None:
    with pytest.raises(PlanError):
        parse_tower(reply)


def test_tower_examples_score_as_correct_plans() -> None:
    for goal, out in TOWER_EXAMPLES:
        tower = parse_tower(json.dumps(out))
        task = TowerTask(goal=goal, tower=tuple(tower), phrasing="example")
        assert score_plan(task, tower_steps(tower))


def test_tower_planner_parses_reply_without_a_model() -> None:
    planner = HFTowerPlanner.__new__(HFTowerPlanner)
    reply = '{"order": "top_down", "colors": ["red", "gray"]}'
    assert planner.parse(reply) == ["Put the red cube on the gray cube."]
    assert planner.last_tower == ["gray", "red"]


def test_simulate_checks_each_step_and_returns_final_state() -> None:
    assert simulate(
        ["Put the green cube on the blue cube.", "Put the red cube on the green cube."]
    ) == {
        "green": "blue",
        "red": "green",
    }
    # red onto green, then green (now covered) cannot move
    assert (
        simulate(["Put the red cube on the green cube.", "Put the green cube on the blue cube."])
        is None
    )
    # blue is covered by green, so red cannot go on it
    assert (
        simulate(["Put the green cube on the blue cube.", "Put the red cube on the blue cube."])
        is None
    )
    assert simulate(["Put the red cube on the red cube."]) is None
    assert simulate(["Pick up the red cube."]) is None
    # moving a clear cube again is allowed
    assert simulate(
        ["Put the red cube on the blue cube.", "Put the red cube on the gray cube."]
    ) == {"red": "gray"}


def test_multi_tower_plans_accept_any_valid_order() -> None:
    task = MultiTowerTask(
        goal="", towers=(("blue", "red"), ("gray", "green", "brown")), phrasing="x"
    )
    steps = [
        "Put the green cube on the gray cube.",
        "Put the red cube on the blue cube.",
        "Put the brown cube on the green cube.",
    ]
    assert score_multi_plan(task, steps)
    assert not score_multi_plan(task, steps[:2])
    assert not score_multi_plan(task, list(reversed(steps)))


def test_multi_tower_tasks() -> None:
    tasks = list(multi_tower_tasks())
    assert len(tasks) == 80
    assert {t.phrasing for t in tasks} >= {"bottom_up", "top_list", "mixed"}
    for task in tasks:
        colors = [c for tower in task.towers for c in tower]
        assert len(colors) == len(set(colors))
        assert task.goal.startswith("Build two separate towers. First: ")
