from __future__ import annotations

from vla_zoo.compare.suite import SuiteArtifact, format_suite_readme


def test_format_suite_readme_embeds_artifacts_and_scope() -> None:
    readme = format_suite_readme(
        title="Suite",
        created_at="2026-06-02T00:00:00+00:00",
        command="vla-zoo compare suite --no-pybullet",
        artifacts=[
            SuiteArtifact(
                label="methods",
                path="method_profiles.md",
                description="method table",
            )
        ],
        method_profiles_markdown="## Methods\n\n| A |\n|---|\n| x |\n",
    )

    assert "# Suite" in readme
    assert "`method_profiles.md`: method table" in readme
    assert "vla-zoo compare suite --no-pybullet" in readme
    assert "Method profiles do not load model weights" in readme
