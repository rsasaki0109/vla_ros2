from __future__ import annotations

import pytest

from vla_ros2.sequencer import TaskSequencer


def test_advances_only_on_current_task_id() -> None:
    seq = TaskSequencer(["stack red on blue", "stack green on red"], task_id_prefix="s")
    first = seq.start(now=0.0)
    assert (first.instruction, first.task_id) == ("stack red on blue", "s-0-0")
    assert seq.on_success("other-task", now=1.0) is None
    second = seq.on_success("s-0-0", now=2.0)
    assert second is not None and second.task_id == "s-1-0"
    # A late "done" for step 0 must not complete step 1.
    assert seq.on_success("s-0-0", now=3.0) is None
    assert seq.state == "running"
    assert seq.on_success("s-1-0", now=4.0) is None
    assert seq.state == "done"
    assert seq.completed == ["s-0-0", "s-1-0"]
    assert seq.current is None


def test_identical_step_texts_get_distinct_task_ids() -> None:
    seq = TaskSequencer(["wipe", "wipe"])
    first = seq.start(now=0.0)
    second = seq.on_success(first.task_id, now=1.0)
    assert second is not None
    assert second.instruction == first.instruction
    assert second.task_id != first.task_id


def test_timeout_retries_then_fails_without_skipping() -> None:
    seq = TaskSequencer(["a", "b"], step_timeout_sec=10.0, max_retries=1)
    seq.start(now=0.0)
    assert seq.tick(now=5.0) is None
    retry = seq.tick(now=10.0)
    assert retry is not None and retry.task_id.endswith("-0-1")
    assert seq.on_success("seq-0-0", now=11.0) is None  # old attempt no longer counts
    assert seq.tick(now=20.0) is None
    assert seq.state == "failed"
    assert seq.to_dict()["step"] == 0  # never moved on to "b"


def test_no_timeout_by_default_and_input_validation() -> None:
    seq = TaskSequencer([" a ", ""])
    assert seq.steps == ["a"]
    seq.start(now=0.0)
    assert seq.tick(now=1e9) is None
    assert seq.state == "running"
    with pytest.raises(ValueError):
        TaskSequencer(["", "  "])
    with pytest.raises(ValueError):
        TaskSequencer(["a"], max_retries=-1)
