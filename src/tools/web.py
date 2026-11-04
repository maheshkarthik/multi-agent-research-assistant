"""Web search through DuckDuckGo (no API key) and safe page fetching.

Fetched pages are untrusted. The fetcher only follows http(s) URLs to public
addresses, caps download size and time, and returns plain text only.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import trafilatura

from src.sources import Evidence
from src.text import best_excerpt
from src.tools import USER_AGENT, ToolError

logger = logging.getLogger(__name__)

MAX_PAGE_BYTES = 2_000_000
FETCH_TIMEOUT_SECONDS = 10
RESULTS_PER_QUERY = 5
PAGES_PER_QUESTION = 3
MIN_PAGE_CHARS = 200  # pages with less extracted text than this fall back to the search snippet


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


SearchFn = Callable[[str, int], list[SearchHit]]
FetchFn = Callable[[str], str]


def duckduckgo_search(query: str, max_results: int = RESULTS_PER_QUERY) -> list[SearchHit]:
    """Search the web with DuckDuckGo. Raises ToolError on failure."""
    from ddgs import DDGS  # imported lazily so tests and offline use don't need it loaded

    try:
        results = DDGS(timeout=FETCH_TIMEOUT_SECONDS).text(query, max_results=max_results)
    except Exception as exc:  # ddgs raises several exception types for network and rate-limit errors
        raise ToolError(f"Web search failed for {query!r}: {exc}") from exc
    return [
        SearchHit(title=_clean_title(r.get("title", "")), url=r.get("href", ""), snippet=r.get("body", "").strip())
        for r in results or []
        if r.get("href", "").startswith(("http://", "https://"))
    ]


def _clean_title(title: str, limit: int = 100) -> str:
    title = " ".join(title.split())
    return title if len(title) <= limit else title[: limit - 3].rstrip() + "..."


def fetch_page_text(url: str) -> str:
    """Download ``url`` and return its main text. Raises ToolError if unsafe or unreadable."""
    check_url_is_public(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.5"})
    try:
        with _OPENER.open(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "")
            if "html" not in content_type and "text/plain" not in content_type:
                raise ToolError(f"Skipped {url}: unsupported content type {content_type or 'unknown'}")
            raw = response.read(MAX_PAGE_BYTES + 1)
    except ToolError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise ToolError(f"Could not fetch {url}: {exc}") from exc
    if len(raw) > MAX_PAGE_BYTES:
        raw = raw[:MAX_PAGE_BYTES]

    html = raw.decode("utf-8", errors="replace")
    text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_precision=True)
    if not text:
        raise ToolError(f"No readable text found at {url}")
    return text


def check_url_is_public(url: str) -> None:
    """Raise ToolError unless ``url`` is http(s) and its host resolves only to public addresses."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError(f"Refused to fetch {url!r}: only http and https URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ToolError(f"Could not resolve {parsed.hostname}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ToolError(f"Refused to fetch {url!r}: it points to a private or local address")


class _PublicOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Checks every redirect target before following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - stdlib signature
        check_url_is_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_PublicOnlyRedirectHandler)


class WebSearchTool:
    """Searches the web and reads the top pages for each sub-question."""

    name = "web"

    def __init__(
        self,
        search: SearchFn = duckduckgo_search,
        fetch: FetchFn = fetch_page_text,
        pages_per_question: int = PAGES_PER_QUESTION,
    ) -> None:
        self._search = search
        self._fetch = fetch
        self.pages_per_question = pages_per_question

    def gather(self, sub_question: str, queries: list[str]) -> list[Evidence]:
        hits: list[SearchHit] = []
        seen: set[str] = set()
        errors: list[str] = []
        for query in queries or [sub_question]:
            try:
                results = self._search(query, RESULTS_PER_QUERY)
            except ToolError as exc:
                errors.append(str(exc))
                continue
            for hit in results:
                # Wikipedia has its own tool with cleaner extracts.
                if hit.url not in seen and "wikipedia.org" not in hit.url:
                    seen.add(hit.url)
                    hits.append(hit)
            if len(hits) >= self.pages_per_question * 2:
                break
        if not hits:
            if errors:
                raise ToolError("; ".join(errors))
            return []

        # Fetch a few spare pages in parallel, since some will fail or be empty.
        candidates = hits[: self.pages_per_question * 2]
        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            texts = list(pool.map(self._safe_fetch, [hit.url for hit in candidates]))

        evidence: list[Evidence] = []
        for hit, text in zip(candidates, texts):
            body = text if text and len(text) >= MIN_PAGE_CHARS else hit.snippet
            if not body:
                continue
            evidence.append(
                Evidence(kind="web", title=hit.title or hit.url, location=hit.url, text=best_excerpt(body, sub_question))
            )
            if len(evidence) == self.pages_per_question:
                break
        return evidence

    def _safe_fetch(self, url: str) -> str:
        try:
            return self._fetch(url)
        except ToolError as exc:
            logger.info("%s", exc)
            return ""
