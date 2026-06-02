from __future__ import annotations

from vla_zoo.compare.profiles import format_method_profiles_markdown, method_profiles


def test_method_profiles_include_runtime_shape() -> None:
    profiles = method_profiles(status_provider=lambda name: f"status:{name}")
    by_name = {profile.name: profile for profile in profiles}

    assert by_name["dummy"].local_runtime == "supported"
    assert by_name["openvla"].action_space == "eef_delta"
    assert by_name["pi0"].remote_runtime == "recommended"
    assert by_name["smolvla"].action_chunks == "internal queue; chunk output optional"
    assert by_name["groot"].action_space == "custom"


def test_method_profiles_markdown_is_readme_ready() -> None:
    profiles = method_profiles(status_provider=lambda _: "available")
    markdown = format_method_profiles_markdown(profiles)

    assert "## VLA Method Profiles" in markdown
    assert "| Method | Family | Role | Inputs | Action | Chunks | Local | Remote |" in markdown
    assert "`openvla`" in markdown
    assert "single RGB image" in markdown
