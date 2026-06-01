from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# Markdown image/link and HTML href/src targets. Matches the pattern used by the
# GIF suite checker so both tools recognize the same link shapes.
LOCAL_LINK_RE = re.compile(
    r"""(?:!\[[^\]]*\]\(([^)]+)\)|\[[^\]]+\]\(([^)]+)\)|(?:href|src)=["']([^"']+)["'])"""
)

# Schemes/prefixes that are not local repository files and are skipped by default.
EXTERNAL_PREFIXES = ("http://", "https://", "//", "mailto:", "tel:", "data:")

# Local link statuses.
STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_EXTERNAL = "external"
STATUS_ANCHOR = "anchor"
STATUS_SOURCE_MISSING = "source_missing"


@dataclass(frozen=True)
class LinkResult:
    """One link occurrence discovered in a scanned file."""

    source: str
    link: str
    status: str
    resolved: str | None = None

    @property
    def is_broken(self) -> bool:
        return self.status in (STATUS_MISSING, STATUS_SOURCE_MISSING)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LinkCheckReport:
    """Aggregated link-check results across one or more files."""

    results: tuple[LinkResult, ...]

    @property
    def checked(self) -> int:
        return sum(1 for result in self.results if result.status in (STATUS_OK, STATUS_MISSING))

    @property
    def ok(self) -> int:
        return sum(1 for result in self.results if result.status == STATUS_OK)

    @property
    def broken(self) -> tuple[LinkResult, ...]:
        return tuple(result for result in self.results if result.is_broken)

    @property
    def skipped_external(self) -> int:
        return sum(1 for result in self.results if result.status == STATUS_EXTERNAL)

    @property
    def ok_overall(self) -> bool:
        return not self.broken

    def to_dict(self) -> dict[str, object]:
        return {
            "checked": self.checked,
            "ok": self.ok,
            "broken": len(self.broken),
            "skipped_external": self.skipped_external,
            "ok_overall": self.ok_overall,
            "results": [result.to_dict() for result in self.results],
        }


def _clean_link(raw: str) -> str:
    """Strip a Markdown/HTML link target down to its local path component."""

    link = raw.strip()
    # Markdown links may carry an optional title: [text](path "Title").
    if " " in link:
        link = link.split(" ", 1)[0].strip()
    # Drop anchor fragments and query strings.
    link = link.split("#", 1)[0]
    link = link.split("?", 1)[0]
    return link.strip()


def extract_links(text: str) -> list[str]:
    """Return raw link targets found in Markdown or HTML text, in order."""

    links: list[str] = []
    for match in LOCAL_LINK_RE.finditer(text):
        raw = next((group for group in match.groups() if group), None)
        if raw is not None:
            links.append(raw)
    return links


def _resolve(link: str, *, source: Path, root: Path) -> Path:
    """Resolve a cleaned local link against the source file or repo root."""

    if link.startswith("/"):
        return root / link.lstrip("/")
    return source.parent / link


def check_file(source: Path, *, root: Path) -> list[LinkResult]:
    """Scan one Markdown/HTML file and classify every link it contains."""

    source_key = str(source)
    if not source.is_file():
        return [LinkResult(source=source_key, link=str(source), status=STATUS_SOURCE_MISSING)]

    text = source.read_text(encoding="utf-8")
    results: list[LinkResult] = []
    for raw in extract_links(text):
        if raw.strip().startswith(EXTERNAL_PREFIXES):
            results.append(LinkResult(source=source_key, link=raw.strip(), status=STATUS_EXTERNAL))
            continue
        cleaned = _clean_link(raw)
        if not cleaned:
            # Pure anchor or empty target; nothing to verify on disk.
            results.append(LinkResult(source=source_key, link=raw.strip(), status=STATUS_ANCHOR))
            continue
        target = _resolve(cleaned, source=source, root=root)
        status = STATUS_OK if target.exists() else STATUS_MISSING
        results.append(
            LinkResult(
                source=source_key,
                link=cleaned,
                status=status,
                resolved=str(target),
            )
        )
    return results


def check_paths(paths: Sequence[Path], *, root: Path = Path(".")) -> LinkCheckReport:
    """Check local links across several Markdown/HTML files."""

    results: list[LinkResult] = []
    for path in paths:
        results.extend(check_file(path, root=root))
    return LinkCheckReport(results=tuple(results))


def link_report_payload(report: LinkCheckReport) -> dict[str, object]:
    """Return a machine-readable payload for the report."""

    return report.to_dict()


def format_link_report_table(report: LinkCheckReport) -> str:
    """Render a human-readable status table for the report."""

    lines = [
        f"{'status':<14} {'source':<40} link",
        f"{'-' * 14} {'-' * 40} {'-' * 40}",
    ]
    for result in report.results:
        source = result.source
        if len(source) > 40:
            source = f"...{source[-37:]}"
        lines.append(f"{result.status:<14} {source:<40} {result.link}")
    lines.append("")
    lines.append(
        f"checked={report.checked} ok={report.ok} broken={len(report.broken)} "
        f"skipped_external={report.skipped_external}"
    )
    return "\n".join(lines)
