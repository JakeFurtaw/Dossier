"""Web search and page browse tools.

Extracted from backend.py (DDGS + Playwright + Trafilatura) so this demo does
not import the FastAPI / Nemotron stack. Search is cheap (snippets only);
browse_page does the expensive full-page extract.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
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
    try:
        rows = sync_search(q, max_results=n)
    except Exception as exc:
        return f"Search failed: {exc}. Try a different query."

    if not rows:
        return (
            f'No search results found for "{q}". '
            "Try a different, more specific or simpler query."
        )

    lines = [f'Search results for "{q}" ({len(rows)} hits):', ""]
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

    try:
        content = _run_async(
            fetch_page(target, max_chars=web_config.max_chars_per_page, timeout=45)
        )
    except Exception as exc:
        return f"Failed to load page at {target}: {exc}"

    blocked = _blocked_page_message(target, content)
    if blocked:
        return blocked

    focus = (instructions or "").strip()
    if focus:
        return f"Focus requested: {focus}\n\n{content}"
    return content
