from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteArtifact:
    label: str
    path: str
    description: str


def format_suite_readme(
    *,
    title: str,
    created_at: str,
    command: str,
    artifacts: list[SuiteArtifact],
    method_profiles_markdown: str,
    pybullet_markdown: str | None = None,
) -> str:
    artifact_lines = [
        f"- `{artifact.path}`: {artifact.description}" for artifact in artifacts
    ]
    sections = [
        f"# {title}",
        "",
        f"Generated at: `{created_at}`",
        "",
        "This artifact directory is intended for README snippets, issue reports, and release "
        "notes. It compares VLA runtime integration shape first, then optional deterministic "
        "PyBullet smoke-scene telemetry.",
        "",
        "## Artifacts",
        "",
        *artifact_lines,
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        method_profiles_markdown.strip(),
    ]
    if pybullet_markdown is not None:
        sections.extend(["", pybullet_markdown.strip()])
    sections.extend(
        [
            "",
            "## Scope",
            "",
            "- Method profiles do not load model weights.",
            "- PyBullet reports are deterministic runtime smoke checks, not model-quality claims.",
            "- External model projects and checkpoints are not redistributed by vla_zoo.",
            "- Real robot deployment still requires robot-specific action bridges "
            "and safety checks.",
        ]
    )
    return "\n".join(sections) + "\n"
