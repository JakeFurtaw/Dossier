"""Web search and page browse tools.

Search is cheap (DDGS snippets only). ``browse_page`` tries a capped httpx
GET + Trafilatura first and only falls back to Playwright for thin, JS-heavy,
or anti-bot pages.

One fetch worker is started per run (``BrowserPool``). Chromium is launched
lazily on the first Playwright fallback. Each browser fetch uses an isolated
``BrowserContext`` so parallel specialists do not share cookies or headers.
Thread-safe in-run URL and search caches on the ``TraceBus`` deduplicate
parallel agents hitting the same query or page.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, List, Optional
from urllib.parse import urlparse

import httpx
import playwright.async_api as pwa
from ddgs import DDGS
from ddgs.exceptions import RatelimitException
from langchain_core.tools import tool
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

import trafilatura

logger = logging.getLogger(__name__)

MIN_HTML_CHARS = 200
MIN_EXTRACTED_CHARS = 300
MAX_HTTP_BODY_BYTES = 20 * 1024 * 1024

_TEXT_CONTENT_TYPES = (
    "text/html",
    "application/xhtml",
    "text/plain",
    "text/xml",
    "text/markdown",
    "application/xml",
    "application/json",
    "application/javascript",
    "text/css",
)
_BINARY_CONTENT_TYPES = (
    "application/octet-stream",
    "application/zip",
    "application/gzip",
    "application/x-gzip",
    "application/x-tar",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
)
_FILE_SUFFIXES = (
    ".pdf",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".mp3",
    ".gz",
    ".tar",
    ".docx",
    ".xlsx",
    ".pptx",
)
_CF_CHALLENGE_MARKERS = (
    "cf-browser-verification",
    "cdn-cgi/challenge-platform",
    "challenge-platform",
    "window._cf_chl_opt",
    "checking your browser before accessing",
    'id="challenge-form"',
    "just a moment...",
    "attention required! | cloudflare",
)
_BLOCKED_MARKERS = (
    "check for humans",
    "captcha",
    "are you a robot",
    "access has to be checked",
    "please verify you are a human",
    "unusual traffic from your computer",
    "cf-challenge",
    *_CF_CHALLENGE_MARKERS,
)


@dataclass
class WebConfig:
    max_results: int = 5
    max_chars_per_page: int = 6000
    browser_timeout: int = 20
    http_timeout: int = 8
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )


web_config = WebConfig()

_browser_pool: ContextVar[BrowserPool | None] = ContextVar("browser_pool", default=None)


class _BodyTooLarge(Exception):
    pass


class BrowserPool:
    """Per-run fetch worker: shared httpx client + lazy Chromium.

    Playwright objects are bound to the event loop that created them, and
    specialists run in worker threads. Playwright fallbacks are scheduled onto
    this loop. Isolation is a fresh BrowserContext per fetch, not a shared page.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._playwright: Any = None
        self._browser: Any = None
        self._browser_lock: asyncio.Lock | None = None
        self._http_client: Any = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._closed = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name="browser-pool", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=45):
            raise TimeoutError("Fetch worker did not start within 45s")
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
        self._browser_lock = asyncio.Lock()
        self._http_client = _make_http_client()

    async def _ensure_browser(self) -> Any:
        if self._browser is not None:
            return self._browser
        if self._browser_lock is None:
            raise RuntimeError("BrowserPool is not running")
        async with self._browser_lock:
            if self._browser is not None:
                return self._browser
            logger.info("Launching Playwright for JS-heavy or blocked pages")
            self._playwright = await pwa.async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            return self._browser

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
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                logger.debug("http client close failed", exc_info=True)
            self._http_client = None

    def fetch(
        self,
        url: str,
        *,
        max_chars: int | None = None,
        timeout: int | None = None,
        wait_until: str | None = None,
    ) -> str:
        """Playwright fallback only. HTTP is attempted on the caller thread first."""
        if self._loop is None:
            raise RuntimeError("BrowserPool is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._playwright_fetch(
                url, max_chars=max_chars, timeout=timeout, wait_until=wait_until
            ),
            self._loop,
        )
        return future.result()

    def fetch_http(
        self,
        url: str,
        *,
        max_chars: int | None = None,
    ) -> str | None:
        """Cheap HTTP GET + extract on the pool loop (one client, keep-alive).

        Returns the formatted extract, a file/size notice, or None when the
        page is not usable over plain HTTP (caller then uses the Playwright
        fallback).
        """
        if self._loop is None or self._http_client is None:
            raise RuntimeError("Fetch worker is not running")
        future = asyncio.run_coroutine_threadsafe(
            _http_extract_or_none(url, max_chars=max_chars, http_client=self._http_client),
            self._loop,
        )
        return future.result()

    async def _playwright_fetch(
        self,
        url: str,
        *,
        max_chars: int | None,
        timeout: int | None,
        wait_until: str | None,
    ) -> str:
        browser = await self._ensure_browser()
        return await _fetch_page_with_playwright(
            url,
            max_chars=max_chars,
            timeout=timeout,
            browser=browser,
            wait_until=wait_until,
        )

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


def _extract_text(html: str) -> str:
    if not html or not html.strip():
        return ""
    content = trafilatura.extract(
        html,
        include_links=False,
        include_tables=True,
        include_comments=False,
        fast=True,
    ) or ""
    if len(content.strip()) < MIN_EXTRACTED_CHARS:
        content = trafilatura.extract(html, fast=False) or ""
    return content.strip()


def _format_extract(url: str, body: str) -> str:
    return f"### Content from: {url}\n\n{body}"


def _looks_like_file_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(suffix) for suffix in _FILE_SUFFIXES)


def _file_notice(url: str) -> str:
    return _format_extract(
        url,
        "This URL points to a file rather than a webpage, and no extractable "
        "text was retrieved. Open the link directly.",
    )


def _is_protected_or_unusable(
    html: str,
    status_code: int | None = None,
    headers: Any | None = None,
) -> bool:
    """True when HTML is empty/tiny or looks like a Cloudflare / anti-bot wall."""
    if not html or len(html.strip()) < MIN_HTML_CHARS:
        return True
    if status_code is not None and status_code in (401, 403, 429, 503):
        return True
    if headers:
        mitigated = str(headers.get("cf-mitigated") or "").lower()
        if mitigated == "challenge":
            return True
    lowered = html.lower()
    return any(marker in lowered for marker in _BLOCKED_MARKERS)


def _blocked_page_message(url: str, content: str) -> str | None:
    sample = (content or "").lower()
    if not any(marker in sample for marker in _BLOCKED_MARKERS):
        return None
    return (
        f"Blocked: {url} returned a CAPTCHA or bot-check, not article text. "
        "Do not retry this URL. Pick a different source or report the facts you already have."
    )


def _http_client_headers() -> dict[str, str]:
    return {
        "User-Agent": web_config.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


def _make_http_client(timeout: float | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout if timeout is not None else web_config.http_timeout,
        headers=_http_client_headers(),
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
    )


def _classify_http_body(content_type: str, url: str) -> str:
    """Return 'html', 'pdf', or 'binary' from Content-Type (URL suffix as fallback)."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if "application/pdf" in ct:
        return "pdf"
    if any(token in ct for token in _TEXT_CONTENT_TYPES):
        return "html"
    path = urlparse(url).path.lower()
    if path.endswith(".pdf") and (not ct or ct == "application/octet-stream"):
        return "pdf"
    if ct.startswith(("image/", "audio/", "video/")) or ct in _BINARY_CONTENT_TYPES:
        return "binary"
    if not ct:
        return "html"
    return "binary"


async def _read_capped_body(resp: httpx.Response) -> bytes:
    """Read a response body, aborting if Content-Length or streamed size exceeds the cap."""
    cl = resp.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_HTTP_BODY_BYTES:
        raise _BodyTooLarge()
    buf = bytearray()
    async for chunk in resp.aiter_bytes():
        buf.extend(chunk)
        if len(buf) > MAX_HTTP_BODY_BYTES:
            raise _BodyTooLarge()
    return bytes(buf)


async def _fast_http_fetch(
    url: str,
    timeout: float,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, int | None, Any, str | None]:
    """Lightweight GET. Returns (text, status_code, headers, non_html_kind).

    non_html_kind is 'pdf', 'binary', or 'too_large' when the response is not a
    webpage (callers must not fall back to Playwright). None means treat as HTML.
    """
    owns_client = client is None
    if client is None:
        client = _make_http_client(timeout=timeout)
    try:
        async with client.stream("GET", url, timeout=timeout) as resp:
            kind = _classify_http_body(
                resp.headers.get("content-type", ""),
                str(resp.url) or url,
            )
            if kind == "binary":
                return "", resp.status_code, resp.headers, "binary"
            try:
                data = await _read_capped_body(resp)
            except _BodyTooLarge:
                logger.info("Response exceeded %s bytes for %s", MAX_HTTP_BODY_BYTES, url)
                return "", resp.status_code, resp.headers, "too_large"
            if kind == "pdf":
                return "", resp.status_code, resp.headers, "pdf"
            encoding = resp.encoding or "utf-8"
            try:
                return data.decode(encoding, errors="replace"), resp.status_code, resp.headers, None
            except Exception:
                return "", resp.status_code, resp.headers, "binary"
    except _BodyTooLarge:
        logger.info("Response exceeded %s bytes for %s", MAX_HTTP_BODY_BYTES, url)
        return "", None, {}, "too_large"
    except Exception as exc:
        logger.info("Fast HTTP fetch failed for %s: %s", url, exc)
        return "", None, {}, None
    finally:
        if owns_client:
            await client.aclose()


async def _http_extract_or_none(
    url: str,
    max_chars: int | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> str | None:
    """Return a formatted extract on a cheap HTTP hit, else None (caller uses Playwright).

    PDF / binary / oversized responses return a notice and never trigger Playwright.
    """
    from workflow.runtime.metrics import record

    html, status, headers, kind = await _fast_http_fetch(
        url, timeout=web_config.http_timeout, client=http_client
    )
    if kind == "binary":
        record("http_binary_skip")
        return _format_extract(
            url,
            "This URL is a binary file (not a webpage), so no page text was extracted. "
            "Open the link directly if you need the file.",
        )
    if kind == "too_large":
        record("http_body_too_large")
        limit_mb = MAX_HTTP_BODY_BYTES // (1024 * 1024)
        return _format_extract(
            url,
            f"This URL returned more than {limit_mb} MB and was skipped.",
        )
    if kind == "pdf":
        record("http_pdf_skip")
        return _format_extract(
            url,
            "This URL is a PDF. No extractable text layer was read "
            "(open the link directly if you need the file).",
        )

    blocked = _is_protected_or_unusable(html, status, headers)
    extracted = _extract_text(html) if html and not blocked else ""
    if extracted and len(extracted) >= MIN_EXTRACTED_CHARS:
        record("http_fetch_hit")
        return _format_extract(url, _truncate_at_sentence(extracted, max_chars))
    if blocked or not html:
        reason = "empty, tiny, or Cloudflare-protected HTML"
    else:
        reason = f"thin extract ({len(extracted)} chars)"
    logger.info("Falling back to Playwright for %s (%s)", url, reason)
    return None


async def _page_html_capped(page: Any) -> str:
    """Copy page HTML into Python, but avoid page.content() when the DOM is already huge.

    Chromium still holds the full rendered page in memory; this only limits the
    Python-side copy forwarded to Trafilatura.
    """
    try:
        size = await page.evaluate(
            "() => (document.documentElement && document.documentElement.outerHTML) "
            "? document.documentElement.outerHTML.length : 0"
        )
    except Exception:
        size = 0
    if size and int(size) > MAX_HTTP_BODY_BYTES:
        logger.info("Playwright DOM is %s chars; taking innerText instead of full HTML", size)
        try:
            from workflow.runtime.metrics import record

            record("dom_capped")
        except Exception:
            pass
        try:
            return (await page.evaluate("() => document.body ? document.body.innerText : ''")) or ""
        except Exception:
            return ""
    try:
        html = await page.content()
    except Exception:
        return ""
    if html and len(html) > MAX_HTTP_BODY_BYTES:
        return html[:MAX_HTTP_BODY_BYTES]
    return html or ""


async def fetch_page(
    url: str,
    max_chars: Optional[int] = None,
    timeout: Optional[int] = None,
    browser: Optional[Any] = None,
    wait_until: Optional[str] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> str:
    """Hybrid fetch: cheap httpx first, Playwright only if that is not usable."""
    if timeout is None:
        timeout = web_config.browser_timeout

    url = _normalize_url(url)
    http_result = await _http_extract_or_none(url, max_chars, http_client=http_client)
    if http_result is not None:
        return http_result
    if _looks_like_file_url(url):
        return _file_notice(url)

    from workflow.runtime.metrics import record

    record("playwright_fallback")
    return await _fetch_page_with_playwright(
        url,
        max_chars=max_chars,
        timeout=timeout,
        browser=browser,
        wait_until=wait_until,
    )


async def _fetch_page_with_playwright(
    url: str,
    max_chars: Optional[int] = None,
    timeout: Optional[int] = None,
    browser: Optional[Any] = None,
    wait_until: Optional[str] = None,
    context: Optional[Any] = None,
) -> str:
    if timeout is None:
        timeout = web_config.browser_timeout

    page = None
    playwright = None
    local_browser = None
    owns_context = context is None

    try:
        if context is None:
            if browser is None:
                playwright = await pwa.async_playwright().start()
                local_browser = await playwright.chromium.launch(headless=True)
                target_browser = local_browser
            else:
                target_browser = browser
            # Isolated context: UA applies to HTTP + navigator.userAgent, and
            # concurrent tasks do not share cookies, sessions, or header state.
            context = await target_browser.new_context(user_agent=web_config.user_agent)

        page = await context.new_page()

        effective_wait = wait_until or "domcontentloaded"
        html = ""
        try:
            await page.goto(url, wait_until=effective_wait, timeout=timeout * 1000)
            html = await _page_html_capped(page)
        except PlaywrightTimeoutError:
            logger.warning("Timeout fetching %s (wait=%s)", url, effective_wait)
            try:
                html = await _page_html_capped(page)
            except Exception:
                pass
        except Exception as exc:
            if "page is navigating" in str(exc) and effective_wait != "networkidle":
                logger.warning("Navigation conflict on %s, retrying with networkidle", url)
                try:
                    await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
                    html = await _page_html_capped(page)
                except PlaywrightTimeoutError:
                    try:
                        html = await _page_html_capped(page)
                    except Exception:
                        pass
            else:
                raise

        if html:
            try:
                await page.evaluate("window.scrollBy(0, 400)")
                await asyncio.sleep(0.8)
                html = await _page_html_capped(page)
            except Exception:
                pass

        if not html or len(html.strip()) < MIN_HTML_CHARS:
            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=min(timeout * 1000, 15000),
                )
                html = await _page_html_capped(page)
            except Exception:
                pass

        content = _extract_text(html)
        if not content:
            return f"Failed to extract usable content from: {url}"

        truncated = _truncate_at_sentence(content, max_chars)
        return _format_extract(url, truncated)

    except PlaywrightTimeoutError:
        logger.warning("Timeout fetching %s", url)
        partial = ""
        try:
            if page:
                partial = (await _page_html_capped(page))[:3000]
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
            try:
                await page.close()
            except Exception:
                logger.debug("page close failed", exc_info=True)
        if owns_context and context is not None:
            try:
                await context.close()
            except Exception:
                logger.debug("context close failed", exc_info=True)
        if local_browser:
            await local_browser.close()
        if playwright:
            await playwright.stop()


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
    Tries a fast HTTP GET first; Playwright only runs if that is not usable.
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
    """HTTP via the run's BrowserPool (shared client, keep-alive) when present.

    Playwright only via the pool (or a local launch) when HTTP was not usable.
    """
    url = _normalize_url(url)
    pool = get_browser_pool()
    if pool is not None:
        try:
            http_result = pool.fetch_http(url, max_chars=web_config.max_chars_per_page)
        except Exception as exc:
            logger.info("Pooled HTTP fetch failed for %s: %s; retrying locally", url, exc)
            http_result = _run_async(
                _http_extract_or_none(url, max_chars=web_config.max_chars_per_page)
            )
    else:
        http_result = _run_async(
            _http_extract_or_none(url, max_chars=web_config.max_chars_per_page)
        )
    if http_result is not None:
        return http_result
    if _looks_like_file_url(url):
        return _file_notice(url)

    from workflow.runtime.metrics import record

    record("playwright_fallback")
    if pool is not None:
        return pool.fetch(url, max_chars=web_config.max_chars_per_page, timeout=45)
    return _run_async(
        _fetch_page_with_playwright(
            url, max_chars=web_config.max_chars_per_page, timeout=45
        )
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
