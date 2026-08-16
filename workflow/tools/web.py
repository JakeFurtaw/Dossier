"""Web search and page browse tools.

Extracted from backend.py (DDGS + Playwright + Trafilatura) so this demo does
not import the FastAPI / Nemotron stack. Search is cheap (snippets only);
browse_page does the expensive full-page extract.

One Chromium instance is launched per run (``BrowserPool``) and pages are
opened against it. Thread-safe in-run URL and search caches on the
``TraceBus`` deduplicate parallel researchers hitting the same query or page.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, List, Optional

import playwright.async_api as pwa
from ddgs import DDGS
from ddgs.exceptions import RatelimitException
from langchain_core.tools import tool
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

import trafilatura

logger = logging.getLogger(__name__)


@dataclass
class WebConfig:
    max_results: int = 5
    max_chars_per_page: int = 6000
    browser_timeout: int = 20
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )


web_config = WebConfig()

_browser_pool: ContextVar[BrowserPool | None] = ContextVar("browser_pool", default=None)


class BrowserPool:
    """One headless Chromium per run, owned by a dedicated asyncio thread.

    Playwright objects are bound to the event loop that created them, and
    researchers run in worker threads. Every ``fetch_page`` is scheduled onto
    this loop so pages share a single browser instead of launching Chromium
    per call.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._playwright: Any = None
        self._browser: Any = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._closed = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name="browser-pool", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=45):
            raise TimeoutError("Chromium did not start within 45s")
        if self._error is not None:
            raise self._error

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._launch())
        except BaseException as exc:
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        loop.run_forever()
        loop.run_until_complete(self._shutdown())
        loop.close()

    async def _launch(self) -> None:
        self._playwright = await pwa.async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)

    async def _shutdown(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                logger.debug("browser close failed", exc_info=True)
            self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                logger.debug("playwright stop failed", exc_info=True)
            self._playwright = None

    def fetch(
        self,
        url: str,
        *,
        max_chars: int | None = None,
        timeout: int | None = None,
        wait_until: str | None = None,
    ) -> str:
        if self._loop is None or self._browser is None:
            raise RuntimeError("BrowserPool is not running")
        future = asyncio.run_coroutine_threadsafe(
            fetch_page(
                url,
                max_chars=max_chars,
                timeout=timeout,
                browser=self._browser,
                wait_until=wait_until,
            ),
            self._loop,
        )
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=15)


def get_browser_pool() -> BrowserPool | None:
    return _browser_pool.get()


def set_browser_pool(pool: BrowserPool | None):
    return _browser_pool.set(pool)


def _run_async(coro):
    """Run an async coroutine from a sync LangChain tool."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _truncate_at_sentence(text: str, max_chars: int | None = None) -> str:
    if max_chars is None:
        max_chars = web_config.max_chars_per_page
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_stop = max(
        truncated.rfind(". "),
        truncated.rfind("! "),
        truncated.rfind("? "),
        truncated.rfind("\n\n"),
    )
    return truncated[: last_stop + 1] if last_stop > 0 else truncated[:max_chars]


def _normalize_url(url: str) -> str:
    if "reddit.com" in url and "old.reddit.com" not in url:
        return url.replace("reddit.com", "old.reddit.com")
    return url


def _result_url(row: dict) -> str:
    return (row.get("href") or row.get("url") or "").strip()


_BLOCKED_MARKERS = (
    "check for humans",
    "captcha",
    "are you a robot",
    "access has to be checked",
    "please verify you are a human",
    "unusual traffic from your computer",
    "cf-challenge",
)


def _blocked_page_message(url: str, content: str) -> str | None:
    sample = (content or "").lower()
    if not any(marker in sample for marker in _BLOCKED_MARKERS):
        return None
    return (
        f"Blocked: {url} returned a CAPTCHA or bot-check, not article text. "
        "Do not retry this URL. Pick a different source or report the facts you already have."
    )


def sync_search(query: str, max_results: int | None = None) -> List[dict]:
    """Return DDGS result dicts that have a usable URL."""
    if max_results is None:
        max_results = web_config.max_results
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [r for r in results if _result_url(r)]
    except RatelimitException:
        logger.warning("DDGS rate limited")
        return []
    except Exception as exc:
        logger.error("Search error: %s", exc)
        return []


async def fetch_page(
    url: str,
    max_chars: Optional[int] = None,
    timeout: Optional[int] = None,
    browser: Optional[Any] = None,
    wait_until: Optional[str] = None,
) -> str:
    """Playwright + Trafilatura extract of a single URL (from backend.py)."""
    if timeout is None:
        timeout = web_config.browser_timeout

    url = _normalize_url(url)
    page = None
    playwright = None
    local_browser = None

    try:
        if browser is None:
            playwright = await pwa.async_playwright().start()
            local_browser = await playwright.chromium.launch(headless=True)
            page = await local_browser.new_page()
        else:
            page = await browser.new_page()

        await page.set_extra_http_headers({"User-Agent": web_config.user_agent})

        effective_wait = wait_until or "domcontentloaded"
        html = ""
        try:
            await page.goto(url, wait_until=effective_wait, timeout=timeout * 1000)
            html = await page.content()
        except PlaywrightTimeoutError:
            logger.warning("Timeout fetching %s (wait=%s)", url, effective_wait)
            try:
                html = await page.content()
            except Exception:
                pass
        except Exception as exc:
            if "page is navigating" in str(exc) and effective_wait != "networkidle":
                logger.warning("Navigation conflict on %s, retrying with networkidle", url)
                try:
                    await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                    html = await page.content()
                except PlaywrightTimeoutError:
                    try:
                        html = await page.content()
                    except Exception:
                        pass
            else:
                raise

        if html:
            try:
                await page.evaluate("window.scrollBy(0, 400)")
                await asyncio.sleep(0.8)
                html = await page.content()
            except Exception:
                pass

        if not html or len(html.strip()) < 200:
            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=min(timeout * 1000, 15000),
                )
                html = await page.content()
            except Exception:
                pass

        content = trafilatura.extract(
            html,
            include_links=False,
            include_tables=True,
            include_comments=False,
            no_fallback=True,
        ) or ""

        if len(content.strip()) < 300:
            content = trafilatura.extract(html, no_fallback=False) or ""

        content = content.strip()
        if not content:
            return f"Failed to extract usable content from: {url}"

        truncated = _truncate_at_sentence(content, max_chars)
        return f"### Content from: {url}\n\n{truncated}"

    except PlaywrightTimeoutError:
        logger.warning("Timeout fetching %s", url)
        partial = ""
        try:
            if page:
                partial = (await page.content())[:3000]
        except Exception:
            pass
        if partial:
            return (
                f"Timeout while loading the full page ({url}). Partial HTML:\n\n{partial}"
            )
        return f"Timeout while fetching the page at {url}."
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return f"Error fetching {url}: {exc}"
    finally:
        if page:
            await page.close()
        if local_browser:
            await local_browser.close()
        if playwright:
            await playwright.stop()


def _format_search_results(query: str, rows: list[dict]) -> str:
    if not rows:
        return (
            f'No search results found for "{query}". '
            "Try a different, more specific or simpler query."
        )
    lines = [f'Search results for "{query}" ({len(rows)} hits):', ""]
    for i, row in enumerate(rows, start=1):
        title = (row.get("title") or "").strip() or "(untitled)"
        url = _result_url(row)
        snippet = (row.get("body") or row.get("snippet") or "").strip()
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   Snippet: {snippet}")
        lines.append("")
    lines.append(
        "If snippets already contain the needed fact, call report_findings. "
        "Otherwise browse_page the 1–2 most relevant URLs. "
        "If this missed the entity, try a different query."
    )
    return "\n".join(lines)


def _run_search(query: str, max_results: int) -> str:
    try:
        rows = sync_search(query, max_results=max_results)
    except Exception as exc:
        return f"Search failed: {exc}. Try a different query."
    return _format_search_results(query, rows)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current or external facts.

    Use for news, populations, events, and anything that could be stale.
    Returns titles, URLs, and snippets. After results, browse_page the 1–2
    most relevant URLs if the snippets are truncated or too shallow.
    One well-chosen search is enough unless it missed the entity.
    """
    q = (query or "").strip()
    if not q:
        return "Error: web_search requires a non-empty query."

    n = max(1, min(int(max_results or 5), 8))
    bus = _trace_bus()
    if bus is None:
        return _run_search(q, n)

    cached = bus.get_cached_search(q, n)
    if cached is not None:
        from workflow.runtime.metrics import record

        record("search_cache_hit")
        return cached

    owned = bus.acquire_search(q, n)
    if not owned:
        bus.wait_search(q, n)
        cached = bus.get_cached_search(q, n)
        if cached is not None:
            from workflow.runtime.metrics import record

            record("search_cache_hit")
            return cached
        return _run_search(q, n)

    try:
        cached = bus.get_cached_search(q, n)
        if cached is not None:
            from workflow.runtime.metrics import record

            record("search_cache_hit")
            return cached
        result = _run_search(q, n)
        bus.put_cached_search(q, n, result)
        if not result.startswith(("Error:", "Search failed:")):
            _publish_search(bus, q, result)
        return result
    finally:
        bus.release_search(q, n)


@tool
def browse_page(url: str, instructions: str = "") -> str:
    """Fetch the full, clean content of ONE specific webpage.

    Use this for a concrete URL (article, Wikipedia, official stats page) and
    to fully load a promising search hit whose snippet is insufficient.
    Prefer this over another web_search when you already have a URL.
    """
    target = (url or "").strip()
    if not target.startswith(("http://", "https://")):
        return "Error: browse_page requires a valid full http(s) URL."

    cached = _cache_get(target)
    if cached is not None:
        from workflow.runtime.metrics import record

        record("url_cache_hit")
        return _with_focus(cached, instructions)

    bus = _trace_bus()
    owned = bus.acquire_url_fetch(target) if bus is not None else True
    if bus is not None and not owned:
        bus.wait_url_fetch(target)
        cached = _cache_get(target)
        if cached is not None:
            from workflow.runtime.metrics import record

            record("url_cache_hit")
            return _with_focus(cached, instructions)

    try:
        content = _fetch_page_sync(target)
    except Exception as exc:
        if bus is not None and owned:
            bus.release_url_fetch(target)
        return f"Failed to load page at {target}: {exc}"

    try:
        blocked = _blocked_page_message(target, content)
        if blocked:
            from workflow.runtime.metrics import record

            record("blocked_page")
            _cache_put(target, blocked)
            _publish_browse(bus, target, blocked=True)
            return blocked

        _cache_put(target, content)
        _publish_browse(bus, target, blocked=False)
        return _with_focus(content, instructions)
    finally:
        if bus is not None and owned:
            bus.release_url_fetch(target)


def _fetch_page_sync(url: str) -> str:
    pool = get_browser_pool()
    if pool is not None:
        return pool.fetch(url, max_chars=web_config.max_chars_per_page, timeout=45)
    return _run_async(
        fetch_page(url, max_chars=web_config.max_chars_per_page, timeout=45)
    )


def _with_focus(content: str, instructions: str) -> str:
    focus = (instructions or "").strip()
    if focus:
        return f"Focus requested: {focus}\n\n{content}"
    return content


def _cache_get(url: str) -> str | None:
    bus = _trace_bus()
    if bus is None:
        return None
    return bus.get_cached_url(url)


def _cache_put(url: str, content: str) -> None:
    bus = _trace_bus()
    if bus is None:
        return
    bus.put_cached_url(url, content)


def _publish_search(bus, query: str, result: str) -> None:
    from workflow.runtime.citations import extract_urls
    from workflow.runtime.ledger import LedgerEntry
    from workflow.runtime.tracing import current_agent

    agent_id, role = current_agent()
    bus.publish_entry(
        LedgerEntry(
            role=role or "researcher",
            agent_id=agent_id,
            kind="search",
            title=query,
            queries=[query],
            urls=extract_urls(result),
        )
    )


def _publish_browse(bus, url: str, *, blocked: bool) -> None:
    if bus is None:
        return
    from workflow.runtime.ledger import LedgerEntry
    from workflow.runtime.tracing import current_agent

    agent_id, role = current_agent()
    title = f"blocked {url}" if blocked else url
    bus.publish_entry(
        LedgerEntry(
            role=role or "researcher",
            agent_id=agent_id,
            kind="browse",
            title=title,
            urls=[url],
        )
    )


def _trace_bus():
    # Lazy import: start_trace owns the pool and would otherwise cycle with this module.
    # Read the ContextVar directly so we do not create the process-wide fallback bus.
    try:
        from workflow.runtime.tracing import try_get_bus

        return try_get_bus()
    except Exception:
        return None
