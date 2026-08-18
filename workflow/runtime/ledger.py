"""Run-scoped shared context: search cache, in-flight locks, findings ledger.

Same-role agents (especially researchers) publish compact notes here so later
siblings can skip repeated searches/URLs and treat already-sourced facts as
gathered. Parallel first-wave overlap is handled by the caches and locks;
the digest helps retries and a second spawn.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from workflow.runtime.citations import extract_urls, normalize_url

_CITATION_LINE = re.compile(r"^\*\*Citation check:\*\*.*$", re.MULTILINE)
_DIGEST_HEADER = (
    "Already gathered by other agents this run. "
    "Treat sourced facts below as collected. Do not repeat these queries or "
    "URLs unless your task needs a fact they do not cover."
)


def normalize_search_query(query: str) -> str:
    return " ".join((query or "").lower().split())


def search_cache_key(query: str, max_results: int) -> str:
    return f"{int(max_results)}|{normalize_search_query(query)}"


def brief_text(text: str, limit: int = 220) -> str:
    """One-line summary of a report, minus citation/evaluator tails."""
    cleaned = _CITATION_LINE.sub("", text or "")
    if "\n---\n" in cleaned:
        cleaned = cleaned.split("\n---\n", 1)[0]
    line = " ".join(cleaned.split())
    if len(line) <= limit:
        return line
    return line[: limit - 1] + "…"


@dataclass
class LedgerEntry:
    role: str
    agent_id: str
    kind: str  # search | browse | report
    title: str = ""
    urls: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def line(self) -> str:
        who = self.agent_id or self.role or "agent"
        if self.kind == "search":
            query = self.queries[0] if self.queries else self.title
            return f"- {who} searched “{query}”"
        if self.kind == "browse":
            url = self.urls[0] if self.urls else self.title
            return f"- {who} browsed {url}"
        summary = self.title or "(report)"
        line = f"- {who} report: {summary}"
        if self.urls:
            shown = ", ".join(self.urls[:4])
            extra = f" (+{len(self.urls) - 4} more)" if len(self.urls) > 4 else ""
            line += f"\n  sources: {shown}{extra}"
        return line


class SharedContext:
    """Thread-safe caches + ledger for one run."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._url_cache: dict[str, str] = {}
        self._search_cache: dict[str, str] = {}
        self._search_inflight: dict[str, threading.Event] = {}
        self._url_inflight: dict[str, threading.Event] = {}
        self.entries: list[LedgerEntry] = []

    # --- URL cache / lock -------------------------------------------------

    def get_url(self, url: str) -> str | None:
        key = normalize_url(url)
        if not key:
            return None
        with self._lock:
            return self._url_cache.get(key)

    def put_url(self, url: str, content: str) -> None:
        key = normalize_url(url)
        if not key:
            return
        with self._lock:
            self._url_cache[key] = content

    def acquire_url(self, url: str) -> bool:
        key = normalize_url(url)
        if not key:
            return True
        with self._lock:
            if key in self._url_cache or key in self._url_inflight:
                return False
            self._url_inflight[key] = threading.Event()
            return True

    def wait_url(self, url: str, timeout: float = 60.0) -> None:
        key = normalize_url(url)
        with self._lock:
            event = self._url_inflight.get(key)
        if event is not None:
            event.wait(timeout=timeout)

    def release_url(self, url: str) -> None:
        key = normalize_url(url)
        with self._lock:
            event = self._url_inflight.pop(key, None)
        if event is not None:
            event.set()

    # --- search cache / lock ----------------------------------------------

    def get_search(self, query: str, max_results: int) -> str | None:
        with self._lock:
            return self._search_cache.get(search_cache_key(query, max_results))

    def put_search(self, query: str, max_results: int, content: str) -> None:
        with self._lock:
            self._search_cache[search_cache_key(query, max_results)] = content

    def acquire_search(self, query: str, max_results: int) -> bool:
        key = search_cache_key(query, max_results)
        with self._lock:
            if key in self._search_cache or key in self._search_inflight:
                return False
            self._search_inflight[key] = threading.Event()
            return True

    def wait_search(self, query: str, max_results: int, timeout: float = 90.0) -> None:
        key = search_cache_key(query, max_results)
        with self._lock:
            event = self._search_inflight.get(key)
        if event is not None:
            event.wait(timeout=timeout)

    def release_search(self, query: str, max_results: int) -> None:
        key = search_cache_key(query, max_results)
        with self._lock:
            event = self._search_inflight.pop(key, None)
        if event is not None:
            event.set()

    # --- ledger -----------------------------------------------------------

    def publish(self, entry: LedgerEntry) -> None:
        with self._lock:
            self.entries.append(entry)

    def digest(self, role: str, *, exclude_agent: str = "", max_chars: int = 1800) -> str:
        with self._lock:
            chosen = [
                entry
                for entry in self.entries
                if entry.role == role and entry.agent_id != exclude_agent
            ]
        if not chosen:
            return ""
        lines = [_DIGEST_HEADER, ""]
        used = 0
        emitted = 0
        for entry in chosen:
            block = entry.line()
            extra = len(block) + 1
            if emitted and used + extra > max_chars:
                lines.append(f"- … {len(chosen) - emitted} more notes omitted")
                break
            lines.append(block)
            used += extra
            emitted += 1
        return "\n".join(lines).strip()

    def to_markdown(self) -> str:
        with self._lock:
            entries = list(self.entries)
        if not entries:
            return ""
        lines = ["## Shared context", ""]
        for entry in entries:
            lines.append(entry.line())
        lines.append("")
        return "\n".join(lines)
