"""Web tools: DuckDuckGo search (no API key) and URL fetching with readable-text extraction.

Tool design notes for small models:
- short, explicit docstrings (the docstring IS the tool description the model sees)
- one or two simple string args
- bounded output (timeouts + char caps) so a page can't blow the context window
- never raise: return an error string the model can react to
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from langchain_core.tools import BaseTool, tool

from research_agent.config import Settings
from research_agent.logging import get_logger
from research_agent.schemas import Source

log = get_logger(__name__)

SearchFn = Callable[[str, int], list[dict[str, Any]]]


def ddgs_search(query: str, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS  # imported lazily: keeps test startup fast

    return DDGS(timeout=10).text(query, max_results=max_results)


def _is_public_http_url(url: str) -> bool:
    """Basic SSRF guard: only http(s) to hosts that resolve to public addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


def make_web_tools(
    settings: Settings,
    *,
    search_fn: SearchFn = ddgs_search,
    http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
) -> list[BaseTool]:
    """Build web tools bound to settings. `search_fn`/`http_client_factory` are test seams."""

    def client() -> httpx.AsyncClient:
        if http_client_factory:
            return http_client_factory()
        return httpx.AsyncClient(
            timeout=settings.fetch_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "research-agent/0.1 (learning project)"},
        )

    # `@tool` turns a function into a LangChain Tool: name + description + JSON schema from
    # the signature. response_format="content_and_artifact" lets a tool return
    # (text_for_the_model, machine_readable_artifact). The artifact rides along on the
    # ToolMessage but is NOT sent to the LLM; we use it to collect citations reliably.
    # Parallel researchers searching simultaneously get rate-limited by the free search
    # backends (observed: "No results found" for every backend). Serialize + retry once.
    search_gate = asyncio.Semaphore(1)

    @tool(response_format="content_and_artifact")
    async def web_search(query: str) -> tuple[str, list[Source]]:
        """Search the web. Returns titles, URLs and short snippets for the query."""
        results: list[dict[str, Any]] = []
        for attempt in range(settings.search_retries + 1):
            try:
                async with search_gate:
                    results = await asyncio.to_thread(search_fn, query, settings.search_max_results)
                break
            except Exception as exc:
                log.warning(
                    "tool.web_search.failed", query=query, attempt=attempt, error=repr(exc)[:200]
                )
                if attempt == settings.search_retries:
                    return f"Search failed ({type(exc).__name__}). Try a different query.", []
                await asyncio.sleep(settings.search_retry_delay_s * (attempt + 1))
        if not results:
            return "No results found. Try a broader query.", []
        lines: list[str] = []
        sources: list[Source] = []
        for i, r in enumerate(results, 1):
            title, url = r.get("title", "").strip(), r.get("href", "").strip()
            body = (r.get("body") or "").strip()[:300]
            lines.append(f"{i}. {title}\n   URL: {url}\n   {body}")
            if url:
                sources.append(Source(title=title or url, url=url, kind="web"))
        return "\n".join(lines), sources

    @tool(response_format="content_and_artifact")
    async def fetch_url(url: str) -> tuple[str, list[Source]]:
        """Fetch a web page by URL and return its main text (truncated)."""
        if not await asyncio.to_thread(_is_public_http_url, url):
            return "Refused: only public http(s) URLs can be fetched.", []
        try:
            async with client() as c:
                resp = await c.get(url)
                resp.raise_for_status()
                html = resp.text[:2_000_000]  # hard cap before parsing
        except Exception as exc:
            log.warning("tool.fetch_url.failed", url=url, error=repr(exc)[:200])
            return f"Fetch failed ({type(exc).__name__}). Use search snippets instead.", []
        text = await asyncio.to_thread(trafilatura.extract, html, include_comments=False)
        if not text:
            return "Could not extract readable text from that page.", []
        meta = await asyncio.to_thread(trafilatura.extract_metadata, html)
        title = (meta.title if meta and meta.title else None) or url
        cap = settings.fetch_max_chars
        body = text[:cap] + ("\n...[truncated]" if len(text) > cap else "")
        return body, [Source(title=title, url=url, kind="web")]

    return [web_search, fetch_url]
