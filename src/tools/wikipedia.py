"""Wikipedia search and article text through the public MediaWiki API (no API key)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from src.sources import Evidence
from src.text import best_excerpt
from src.tools import USER_AGENT, ToolError

API_URL = "https://en.wikipedia.org/w/api.php"
ARTICLE_URL = "https://en.wikipedia.org/wiki/"
TIMEOUT_SECONDS = 10
ARTICLES_PER_QUESTION = 2

ApiFn = Callable[[dict[str, str]], dict[str, Any]]


def call_api(params: dict[str, str]) -> dict[str, Any]:
    """Call the MediaWiki API and return the decoded JSON. Raises ToolError on failure."""
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"Wikipedia request failed: {exc}") from exc


class WikipediaTool:
    """Finds the most relevant Wikipedia articles and returns their best passages."""

    name = "wikipedia"

    def __init__(self, api: ApiFn = call_api, articles_per_question: int = ARTICLES_PER_QUESTION) -> None:
        self._api = api
        self.articles_per_question = articles_per_question

    def gather(self, sub_question: str, queries: list[str]) -> list[Evidence]:
        query = (queries or [sub_question])[0]
        found = self._api({"action": "query", "list": "search", "srsearch": query, "srlimit": str(self.articles_per_question)})
        titles = [item["title"] for item in found.get("query", {}).get("search", [])]
        if not titles:
            return []

        evidence = []
        for title in titles:  # one request per article: full extracts can't be batched
            pages = self._api(
                {"action": "query", "prop": "extracts", "explaintext": "1", "redirects": "1", "titles": title}
            ).get("query", {}).get("pages", [])
            text = pages[0].get("extract", "") if pages else ""
            if text:
                url = ARTICLE_URL + urllib.parse.quote(title.replace(" ", "_"))
                evidence.append(
                    Evidence(kind="wikipedia", title=f"{title} (Wikipedia)", location=url, text=best_excerpt(text, sub_question))
                )
        return evidence
