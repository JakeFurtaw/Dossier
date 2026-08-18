from __future__ import annotations

import asyncio

import httpx
import pytest

from workflow.runtime.metrics import snapshot
from workflow.tools.web import (
    MAX_HTTP_BODY_BYTES,
    _BodyTooLarge,
    _classify_http_body,
    _fast_http_fetch,
    _fetch_page_with_playwright,
    _http_extract_or_none,
    _is_protected_or_unusable,
    _looks_like_file_url,
    _page_html_capped,
    _read_capped_body,
    fetch_page,
    web_config,
)


ARTICLE_HTML = """<!DOCTYPE html>
<html><head><title>Lisbon population</title></head>
<body>
<article>
<h1>Lisbon city proper</h1>
<p>The city of Lisbon has a population of 545,796 as of the 2021 census,
according to official statistics published by Statistics Portugal. The
figure refers to the city proper, not the wider metropolitan area.</p>
<p>The Lisbon metropolitan area is larger and is often cited separately.
Researchers should prefer the city-proper number when comparing to a
town of 5,000 people.</p>
<p>Source notes and methodology are described on the official statistics
portal and on the municipal pages that republish the census release.</p>
</article>
</body></html>
"""


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes], headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}
        self._chunks = chunks

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakePage:
    def __init__(self, *, size: int = 0, html: str = "", inner: str = "") -> None:
        self.size = size
        self.html = html
        self.inner = inner
        self.content_calls = 0
        self.closed = False
        self.gotos: list[str] = []

    async def evaluate(self, script: str):
        if "outerHTML.length" in script:
            return self.size
        if "innerText" in script:
            return self.inner
        return None

    async def content(self) -> str:
        self.content_calls += 1
        return self.html

    async def goto(self, url: str, wait_until: str = "", timeout: int = 0) -> None:
        del wait_until, timeout
        self.gotos.append(url)

    async def close(self) -> None:
        self.closed = True


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.closed = False
        self.user_agent = ""

    async def new_page(self) -> _FakePage:
        return self.page

    async def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.contexts: list[_FakeContext] = []
        self.new_page_calls = 0

    async def new_context(self, user_agent: str | None = None) -> _FakeContext:
        ctx = _FakeContext(self.page)
        ctx.user_agent = user_agent or ""
        self.contexts.append(ctx)
        return ctx

    async def new_page(self) -> _FakePage:
        self.new_page_calls += 1
        raise AssertionError("must not open pages on the shared browser")


def _client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_classify_http_body() -> None:
    assert _classify_http_body("text/html; charset=utf-8", "https://x.test/a") == "html"
    assert _classify_http_body("application/pdf", "https://x.test/a") == "pdf"
    assert _classify_http_body("application/zip", "https://x.test/a") == "binary"
    assert _classify_http_body("image/png", "https://x.test/a.png") == "binary"
    assert _classify_http_body("application/octet-stream", "https://x.test/doc.pdf") == "pdf"
    assert _classify_http_body("", "https://x.test/article") == "html"
    assert _looks_like_file_url("https://x.test/a.PDF")
    assert not _looks_like_file_url("https://x.test/a")


def test_read_capped_body_respects_content_length() -> None:
    resp = _FakeStreamResponse(
        [b"ignored"],
        headers={"content-length": str(MAX_HTTP_BODY_BYTES + 1)},
    )
    with pytest.raises(_BodyTooLarge):
        asyncio.run(_read_capped_body(resp))


def test_read_capped_body_aborts_mid_stream() -> None:
    resp = _FakeStreamResponse([b"a" * (MAX_HTTP_BODY_BYTES // 2), b"b" * (MAX_HTTP_BODY_BYTES // 2 + 8)])
    with pytest.raises(_BodyTooLarge):
        asyncio.run(_read_capped_body(resp))


def test_read_capped_body_allows_small_payload() -> None:
    resp = _FakeStreamResponse([b"hello"], headers={"content-length": "5"})
    assert asyncio.run(_read_capped_body(resp)) == b"hello"


def test_protected_or_unusable() -> None:
    assert _is_protected_or_unusable("tiny")
    assert _is_protected_or_unusable("<html>" + "x" * 300, status_code=403)
    wall = "<html><body>" + "Just a moment... cf-browser-verification " + ("pad " * 80) + "</body></html>"
    assert _is_protected_or_unusable(wall, status_code=200)
    assert not _is_protected_or_unusable(ARTICLE_HTML, status_code=200)


def test_http_extract_returns_article_and_skips_playwright(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, text=ARTICLE_HTML, headers={"content-type": "text/html"})

    async def boom(*_args, **_kwargs):
        raise AssertionError("Playwright must not run on a usable HTTP extract")

    monkeypatch.setattr("workflow.tools.web._fetch_page_with_playwright", boom)
    client = _client_for(handler)
    result = asyncio.run(fetch_page("https://example.com/lisbon", http_client=client))
    asyncio.run(client.aclose())
    assert result.startswith("### Content from: https://example.com/lisbon")
    assert "545,796" in result
    assert snapshot()["http_fetch_hit"] == 1
    assert "playwright_fallback" not in snapshot()


def test_http_extract_falls_back_on_thin_or_protected(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            403,
            text="<html><body>Just a moment... checking your browser before accessing</body></html>",
            headers={"content-type": "text/html", "cf-mitigated": "challenge"},
        )

    async def fake_pw(url: str, **_kwargs) -> str:
        return f"### Content from: {url}\n\nplaywright body"

    monkeypatch.setattr("workflow.tools.web._fetch_page_with_playwright", fake_pw)
    client = _client_for(handler)
    result = asyncio.run(fetch_page("https://walled.example/x", http_client=client))
    asyncio.run(client.aclose())
    assert "playwright body" in result
    assert snapshot()["playwright_fallback"] == 1


def test_binary_and_oversize_never_use_playwright(monkeypatch) -> None:
    async def boom(*_args, **_kwargs):
        raise AssertionError("Playwright must not run for binary or oversized bodies")

    monkeypatch.setattr("workflow.tools.web._fetch_page_with_playwright", boom)

    def binary_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=b"PK\x03\x04", headers={"content-type": "application/zip"})

    client = _client_for(binary_handler)
    binary = asyncio.run(_http_extract_or_none("https://x.test/a.zip", http_client=client))
    asyncio.run(client.aclose())
    assert binary is not None
    assert "binary file" in binary
    assert snapshot()["http_binary_skip"] == 1

    def huge_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            content=b"<html>x</html>",
            headers={
                "content-type": "text/html",
                "content-length": str(MAX_HTTP_BODY_BYTES + 10),
            },
        )

    client = _client_for(huge_handler)
    huge = asyncio.run(_http_extract_or_none("https://x.test/huge", http_client=client))
    asyncio.run(client.aclose())
    assert huge is not None
    assert "MB and was skipped" in huge
    assert snapshot()["http_body_too_large"] == 1


def test_pdf_notice_does_not_fallback(monkeypatch) -> None:
    async def boom(*_args, **_kwargs):
        raise AssertionError("Playwright must not run for PDFs")

    monkeypatch.setattr("workflow.tools.web._fetch_page_with_playwright", boom)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})

    client = _client_for(handler)
    result = asyncio.run(fetch_page("https://x.test/a.pdf", http_client=client))
    asyncio.run(client.aclose())
    assert "PDF" in result
    assert "playwright_fallback" not in snapshot()


def test_page_html_capped_uses_inner_text_when_dom_is_huge() -> None:
    page = _FakePage(size=MAX_HTTP_BODY_BYTES + 50, html="<unused>", inner="visible text only")
    assert asyncio.run(_page_html_capped(page)) == "visible text only"
    assert page.content_calls == 0
    assert snapshot()["dom_capped"] == 1


def test_page_html_capped_uses_content_when_small() -> None:
    page = _FakePage(size=1200, html="<html>ok</html>")
    assert asyncio.run(_page_html_capped(page)) == "<html>ok</html>"
    assert page.content_calls == 1


def test_playwright_uses_isolated_context(monkeypatch) -> None:
    page = _FakePage(size=400, html=ARTICLE_HTML, inner="ignored")
    browser = _FakeBrowser(page)
    monkeypatch.setattr("workflow.tools.web.asyncio.sleep", lambda *_a, **_k: asyncio.sleep(0))
    result = asyncio.run(
        _fetch_page_with_playwright(
            "https://example.com/lisbon",
            browser=browser,
            timeout=1,
        )
    )
    assert browser.new_page_calls == 0
    assert len(browser.contexts) == 1
    assert browser.contexts[0].user_agent == web_config.user_agent
    assert browser.contexts[0].closed
    assert page.closed
    assert "545,796" in result or result.startswith("### Content from:")


def test_browse_page_http_hit_never_touches_pool(monkeypatch) -> None:
    from workflow.runtime.metrics import snapshot as snap
    from workflow.runtime.tracing import start_trace
    from workflow.tools.web import browse_page

    async def fake_http(url: str, max_chars=None, http_client=None):
        del max_chars, http_client
        return f"### Content from: {url}\n\n" + ("Lisbon has 545,796 residents. " * 8)

    def boom(*_args, **_kwargs):
        raise AssertionError("BrowserPool.fetch must not run on an HTTP hit")

    monkeypatch.setattr("workflow.tools.web._http_extract_or_none", fake_http)
    monkeypatch.setattr("workflow.tools.web.BrowserPool.fetch", boom)
    with start_trace(goal="g", save=False, render=False, browser=False):
        result = browse_page.invoke({"url": "https://example.com/lisbon"})
    assert "545,796" in result
    assert "http_fetch_hit" in snap() or "Lisbon" in result


def test_browser_pool_does_not_launch_chromium_on_start() -> None:
    from workflow.tools.web import BrowserPool

    pool = BrowserPool()
    pool.start()
    try:
        assert pool._browser is None
        assert pool._playwright is None
        assert pool._loop is not None
    finally:
        pool.close()


def test_fetch_page_sync_uses_pool_http_client_when_available(monkeypatch) -> None:
    from workflow.tools.web import BrowserPool, _browser_pool, _fetch_page_sync, set_browser_pool

    pool = BrowserPool()
    pool.start()
    seen: dict = {}

    async def fake_http(url: str, max_chars=None, http_client=None):
        seen["client"] = http_client
        return f"### Content from: {url}\n\npooled body text"

    monkeypatch.setattr("workflow.tools.web._http_extract_or_none", fake_http)
    token = set_browser_pool(pool)
    try:
        client_ref = pool._http_client
        result = _fetch_page_sync("https://example.com/pooled")
    finally:
        _browser_pool.reset(token)
        pool.close()
    assert "pooled body text" in result
    assert seen["client"] is client_ref
    assert pool._browser is None  # Chromium stays lazy on an HTTP hit


def test_fast_http_fetch_returns_none_kind_on_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectError("nope")

    client = _client_for(handler)
    text, status, _headers, kind = asyncio.run(
        _fast_http_fetch("https://down.example/", timeout=1, client=client)
    )
    asyncio.run(client.aclose())
    assert text == ""
    assert status is None
    assert kind is None
