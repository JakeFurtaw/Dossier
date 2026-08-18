"""Deterministic citation verification. No model calls, no GPU.

Two stages, each checking against evidence local to that agent's own
message list:

1. Researcher level — the ``report_findings`` text is checked against that
   researcher's own ``web_search`` / ``browse_page`` observations. A verdict
   line is appended to the report so the planner and the evaluator can see it.
2. Final answer level — the planner's final answer is checked against the
   researcher reports it received (``spawn_*`` observations). The audit table
   is written into the run report.

Checks performed:
- Provenance: every cited URL must appear in a tool observation.
- Number grounding (only where per-URL page text exists, i.e. the researcher
  level): the numbers cited on the same line as a URL should appear in the
  observed content for that URL.

URLs are compared in a canonical form (lowercase host, no www., no
old.reddit.com, no fragment/query, no trailing slash) so trivial formatting
differences do not read as fabrications.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from langchain_core.messages import BaseMessage, ToolMessage

from workflow.util import message_text

_URL_RE = re.compile(r"https?://[^\s\)\]<>\"'`]+")
_SEARCH_URL_RE = re.compile(r"^\s*URL:\s*(\S+)", re.MULTILINE)
_BROWSE_HEADER_RE = re.compile(r"###\s+Content from:\s*(\S+)")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_TRAILING_PUNCT = ".,;:!?)]}'\""

def _is_spawn_tool(name: str) -> bool:
    return (name or "").startswith("spawn_")


@dataclass
class Evidence:
    """What tool output says about one URL."""

    url: str
    sources: list[str] = field(default_factory=list)
    body: str = ""  # observed text for this URL (browse_page extracts)


@dataclass
class CitationRow:
    url: str
    verified: bool
    sources: list[str] = field(default_factory=list)
    numbers_found: int = 0
    numbers_total: int = 0
    grounding_checked: bool = False


@dataclass
class CitationAudit:
    stage: str
    rows: list[CitationRow] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def verified_count(self) -> int:
        return sum(1 for row in self.rows if row.verified)

    @property
    def all_verified(self) -> bool:
        return all(row.verified for row in self.rows)

    @property
    def unverified(self) -> list[CitationRow]:
        return [row for row in self.rows if not row.verified]


def normalize_url(url: str) -> str:
    """Canonical comparison key for a URL (host + path only)."""
    text = (url or "").strip().rstrip(_TRAILING_PUNCT)
    try:
        parts = urlsplit(text)
        host = (parts.hostname or "").lower()
    except ValueError:
        return text.lower()
    if host.startswith("www."):
        host = host[4:]
    if host == "old.reddit.com":
        host = "reddit.com"
    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    return f"{host}{path}"


def extract_urls(text: str) -> list[str]:
    """Unique URLs in text (first spelling wins), in order of appearance."""
    seen: dict[str, str] = {}
    for match in _URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(_TRAILING_PUNCT)
        key = normalize_url(url)
        if key and key not in seen:
            seen[key] = url
    return list(seen.values())


def build_evidence_index(messages: list[BaseMessage]) -> dict[str, Evidence]:
    """Map normalized URL -> Evidence, parsed from tool observations.

    Understands the exact formats this project's tools emit:
    - ``web_search``:   "   URL: <url>" result lines
    - ``browse_page``:  "### Content from: <url>" header (+ body text)
    - ``spawn_*``:      researcher report text (any URL inside counts)
    """
    index: dict[str, Evidence] = {}

    def add(url: str, source: str, body: str = "") -> None:
        key = normalize_url(url)
        if not key:
            return
        entry = index.get(key)
        if entry is None:
            index[key] = Evidence(url=url.rstrip(_TRAILING_PUNCT), sources=[source], body=body)
            return
        if source not in entry.sources:
            entry.sources.append(source)
        if not entry.body and body:
            entry.body = body

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = getattr(message, "name", "") or ""
        text = message_text(message)
        if not text:
            continue
        if name == "web_search":
            for match in _SEARCH_URL_RE.finditer(text):
                add(match.group(1), "web_search")
        elif name == "browse_page":
            header = _BROWSE_HEADER_RE.search(text)
            if header:
                add(header.group(1), "browse_page", body=text[header.end() :])
            else:
                # Blocked / timeout / failed-extract message: the requested
                # URL is still real provenance, just with no usable content.
                first = _URL_RE.search(text)
                if first:
                    add(first.group(0), "browse_page (no content)")
        elif _is_spawn_tool(name):
            for url in extract_urls(text):
                add(url, "researcher report")
    return index


def _number_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(text or ""):
        token = match.group(0).replace(",", "").rstrip("%")
        if len(token) >= 2:  # ignore single digits (match noise)
            tokens.add(token)
    return tokens


def _line_around(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def audit_citations(
    text: str,
    index: dict[str, Evidence],
    *,
    stage: str,
    grounding: bool = True,
) -> CitationAudit:
    """Audit every URL cited in ``text`` against the evidence index.

    Number grounding only applies when the evidence entry carries page text
    (researcher level); the planner level has no per-URL bodies.
    """
    audit = CitationAudit(stage=stage)
    body_tokens: dict[str, set[str]] = {}
    seen: set[str] = set()
    for match in _URL_RE.finditer(text or ""):
        raw = match.group(0).rstrip(_TRAILING_PUNCT)
        key = normalize_url(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        evidence = index.get(key)
        row = CitationRow(
            url=raw,
            verified=evidence is not None,
            sources=list(evidence.sources) if evidence else [],
        )
        if row.verified and grounding and evidence is not None and evidence.body:
            row.grounding_checked = True
            if key not in body_tokens:
                body_tokens[key] = _number_tokens(evidence.body)
            cited = _number_tokens(_line_around(text, match.start(), match.end()))
            row.numbers_total = len(cited)
            row.numbers_found = len(cited & body_tokens[key])
        audit.rows.append(row)
    return audit


def citation_check_line(text: str, index: dict[str, Evidence]) -> str:
    """One-line verdict appended to a researcher report."""
    audit = audit_citations(text, index, stage="researcher report", grounding=True)
    if audit.total == 0:
        return "**Citation check:** no source URLs cited."
    if audit.all_verified:
        return f"**Citation check:** {audit.verified_count}/{audit.total} URLs verified against tool output."
    bad = "; ".join(row.url for row in audit.unverified)
    return (
        f"**Citation check:** {audit.verified_count}/{audit.total} URLs verified — "
        f"NOT in any tool output: {bad}. Do not cite unsourced URLs."
    )


def summarize_audit(audit: CitationAudit) -> str:
    """One-line terminal summary."""
    if audit.total == 0:
        return "citations: none cited"
    if audit.all_verified:
        return f"citations: {audit.verified_count}/{audit.total} verified"
    bad = ", ".join(row.url for row in audit.unverified[:3])
    more = f" (+{len(audit.unverified) - 3} more)" if len(audit.unverified) > 3 else ""
    return f"citations: {audit.verified_count}/{audit.total} verified — unverified: {bad}{more}"


def _cell(text: str) -> str:
    return (text or "—").replace("|", "\\|")


def audit_to_markdown(audit: CitationAudit) -> str:
    """Table for the run report (section heading is added by report.py)."""
    if audit.total == 0:
        return "_No source URLs cited._"
    lines = ["| URL | Verified | Seen in | Numbers |", "|---|---|---|---|"]
    for row in audit.rows:
        if row.grounding_checked:
            numbers = (
                f"{row.numbers_found}/{row.numbers_total}"
                if row.numbers_total
                else "no numbers cited"
            )
        else:
            numbers = "—"
        status = "yes" if row.verified else "**no**"
        lines.append(
            f"| {_cell(row.url)} | {status} | {_cell(', '.join(row.sources))} | {numbers} |"
        )
    lines.append("")
    lines.append(f"{audit.verified_count}/{audit.total} URLs traced to tool output.")
    return "\n".join(lines)
