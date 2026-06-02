from __future__ import annotations

from vla_zoo.runtime.server_plan import build_server_plan, format_server_plan_markdown


def test_server_plan_assigns_ports_and_remote_map() -> None:
    plan = build_server_plan(["openvla", "pi0"], public_host="gpu-box", base_port=9001)

    assert [entry.port for entry in plan.entries] == [9001, 9002]
    assert plan.remote_map == "openvla=http://gpu-box:9001,pi0=http://gpu-box:9002"
    assert "--runtime remote" in plan.compare_command
    assert "--unnorm-key" in plan.entries[0].command
    assert "bridge_orig" in plan.entries[0].command


def test_server_plan_pi0_uses_explicit_checkpoint() -> None:
    plan = build_server_plan(["pi0"])
    command = plan.entries[0].command

    assert command[:4] == ("vla-zoo", "serve", "--model", "pi0")
    assert "--pretrained" in command
    assert "lerobot/pi0_base" in command
    assert "--device" in command


def test_server_plan_markdown_includes_no_verification_claim() -> None:
    plan = build_server_plan(["pi0"])
    markdown = format_server_plan_markdown(plan)

    assert "not a claim" in markdown
    assert "vla-zoo compare pybullet" in markdown
