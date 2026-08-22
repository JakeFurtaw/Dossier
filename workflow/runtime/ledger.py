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

from workflow.runtime.citations import normalize_url

_CITATION_LINE = re.compile(r"^\*\*Citation check:\*\*.*$", re.MULTILINE)
_DIGEST_HEADER = (
    "Already gathered by other agents this run. "
    "Treat sourced facts below as collected. Do not repeat these queries or "
    "URLs unless your task needs a fact they do not cover."
)


def normalize_search_query(query: str) -> str:
    """Lowercase + collapse whitespace so cache keys are order-of-spaces stable."""
    return " ".join((query or "").lower().split())


def search_cache_key(query: str, max_results: int) -> str:
    """Canonical key for the in-run search cache (max_results + normalized query)."""
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
        """One ledger line for digest / run-report rendering."""
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


class _CachedKV:
    """Thread-safe value cache with in-flight dedup for parallel fetchers.

    Instantiated twice by SharedContext: the URL cache and the search cache.
    """

    def __init__(self) -> None:
        """One lock guarding the value map and in-flight events."""
        self._lock = threading.RLock()
        self._values: dict[str, str] = {}
        self._inflight: dict[str, threading.Event] = {}

    def get(self, key: str | None) -> str | None:
        """Cached value for this key, or None."""
        if not key:
            return None
        with self._lock:
            return self._values.get(key)

    def put(self, key: str | None, value: str) -> None:
        """Store the result under this key."""
        if not key:
            return
        with self._lock:
            self._values[key] = value

    def acquire(self, key: str | None) -> bool:
        """Claim fetch ownership of this key; False when cached or in flight elsewhere."""
        if not key:
            return True
        with self._lock:
            if key in self._values or key in self._inflight:
                return False
            self._inflight[key] = threading.Event()
            return True

    def wait(self, key: str | None, timeout: float) -> None:
        """Block until the owner of this key finishes (or the timeout)."""
        if not key:
            return
        with self._lock:
            event = self._inflight.get(key)
        if event is not None:
            event.wait(timeout=timeout)

    def release(self, key: str | None) -> None:
        """Drop ownership and wake any waiters on this key."""
        if not key:
            return
        with self._lock:
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()


class SharedContext:
    """Thread-safe caches + findings ledger for one run.

    Exposed to tools and agents through the TraceBus pass-throughs.
    """

    def __init__(self) -> None:
        """URL cache, search cache (each a _CachedKV), and the findings ledger."""
        self._urls = _CachedKV()
        self._searches = _CachedKV()
        self._lock = threading.RLock()
        self.entries: list[LedgerEntry] = []

    # --- URL cache / lock -------------------------------------------------

    def _url_key(self, url: str, kind: str) -> str | None:
        """Canonical URL cache key (kind + normalized URL)."""
        key = normalize_url(url)
        if not key:
            return None
        return f"{kind}|{key}"

    def get_url(self, url: str, kind: str = "page") -> str | None:
        """Cached page/raw body for this URL, or None."""
        return self._urls.get(self._url_key(url, kind))

    def put_url(self, url: str, content: str, kind: str = "page") -> None:
        """Cache the fetched body under this URL."""
        self._urls.put(self._url_key(url, kind), content)

    def acquire_url(self, url: str, kind: str = "page") -> bool:
        """True when this thread may fetch the URL (not cached/in flight elsewhere)."""
        return self._urls.acquire(self._url_key(url, kind))

    def wait_url(self, url: str, timeout: float = 60.0, kind: str = "page") -> None:
        """Wait for another agent's in-flight fetch of this URL."""
        self._urls.wait(self._url_key(url, kind), timeout=timeout)

    def release_url(self, url: str, kind: str = "page") -> None:
        """Finish fetching this URL and wake waiters."""
        self._urls.release(self._url_key(url, kind))

    # --- search cache / lock ----------------------------------------------

    def get_search(self, query: str, max_results: int) -> str | None:
        """Cached web_search result for this query, or None."""
        return self._searches.get(search_cache_key(query, max_results))

    def put_search(self, query: str, max_results: int, content: str) -> None:
        """Cache a web_search result under its canonical key."""
        self._searches.put(search_cache_key(query, max_results), content)

    def acquire_search(self, query: str, max_results: int) -> bool:
        """True when this thread may run the search (not cached/in flight elsewhere)."""
        return self._searches.acquire(search_cache_key(query, max_results))

    def wait_search(self, query: str, max_results: int, timeout: float = 90.0) -> None:
        """Wait for another agent's in-flight run of this search."""
        self._searches.wait(search_cache_key(query, max_results), timeout=timeout)

    def release_search(self, query: str, max_results: int) -> None:
        """Finish the search and wake waiters."""
        self._searches.release(search_cache_key(query, max_results))

    # --- ledger -----------------------------------------------------------

    def publish(self, entry: LedgerEntry) -> None:
        """Append a note (search/browse/report) for peer digests and the run report."""
        with self._lock:
            self.entries.append(entry)

    def digest(self, role: str, *, exclude_agent: str = "", max_chars: int = 1800) -> str:
        """Bounded 'already gathered by your peers' block for a specialist's prompt."""
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
        """The full ledger as a '## Shared context' block for the run report."""
        with self._lock:
            entries = list(self.entries)
        if not entries:
            return ""
        lines = ["## Shared context", ""]
        for entry in entries:
            lines.append(entry.line())
        lines.append("")
        return "\n".join(lines)
