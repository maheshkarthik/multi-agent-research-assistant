"""Tests for the web and Wikipedia tools, with the network replaced by fakes."""

import unittest
from unittest import mock

from src.tools import ToolError
from src.tools.web import SearchHit, WebSearchTool, check_url_is_public
from src.tools.wikipedia import WikipediaTool

LONG_TEXT = "The printing press changed science in Europe. " * 20


class UrlSafetyTests(unittest.TestCase):
    def test_rejects_non_http_schemes(self):
        for url in ("file:///etc/passwd", "ftp://example.com/x", "javascript:alert(1)"):
            with self.assertRaises(ToolError):
                check_url_is_public(url)

    def test_rejects_private_and_local_addresses(self):
        for address in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"):
            with mock.patch("socket.getaddrinfo", return_value=[(None, None, None, None, (address, 80))]):
                with self.assertRaises(ToolError, msg=address):
                    check_url_is_public("http://innocent-looking.example/")

    def test_allows_public_addresses(self):
        with mock.patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 443))]):
            check_url_is_public("https://example.com/page")


class WebSearchToolTests(unittest.TestCase):
    def test_reads_pages_and_skips_wikipedia_results(self):
        hits = [
            SearchHit("Wiki", "https://en.wikipedia.org/wiki/X", "wiki snippet"),
            SearchHit("Page A", "https://a.example", "snippet a"),
            SearchHit("Page B", "https://b.example", "snippet b"),
        ]
        tool = WebSearchTool(search=lambda q, n: hits, fetch=lambda url: LONG_TEXT, pages_per_question=3)
        evidence = tool.gather("How did printing change science?", ["printing science"])
        self.assertEqual([e.location for e in evidence], ["https://a.example", "https://b.example"])
        self.assertTrue(all(e.kind == "web" for e in evidence))
        self.assertLessEqual(len(evidence[0].text), 1504)

    def test_falls_back_to_snippet_when_page_cannot_be_read(self):
        def fetch(url):
            raise ToolError("blocked")

        tool = WebSearchTool(search=lambda q, n: [SearchHit("A", "https://a.example", "useful snippet")], fetch=fetch)
        self.assertEqual(tool.gather("Q?", ["q"])[0].text, "useful snippet")

    def test_deduplicates_urls_across_queries(self):
        hit = SearchHit("A", "https://a.example", "s")
        tool = WebSearchTool(search=lambda q, n: [hit], fetch=lambda url: LONG_TEXT)
        self.assertEqual(len(tool.gather("Q?", ["one", "two"])), 1)

    def test_raises_when_every_search_fails(self):
        def search(q, n):
            raise ToolError("rate limited")

        with self.assertRaises(ToolError):
            WebSearchTool(search=search, fetch=lambda url: "").gather("Q?", ["a", "b"])

    def test_one_failed_query_is_tolerated(self):
        def search(q, n):
            if q == "bad":
                raise ToolError("rate limited")
            return [SearchHit("A", "https://a.example", "s")]

        self.assertEqual(len(WebSearchTool(search=search, fetch=lambda url: LONG_TEXT).gather("Q?", ["bad", "good"])), 1)


class WikipediaToolTests(unittest.TestCase):
    def fake_api(self, params):
        if params.get("list") == "search":
            return {"query": {"search": [{"title": "Printing press"}, {"title": "Empty page"}]}}
        extract = LONG_TEXT if params["titles"] == "Printing press" else ""
        return {"query": {"pages": [{"title": params["titles"], "extract": extract}]}}

    def test_returns_articles_with_urls(self):
        evidence = WikipediaTool(api=self.fake_api).gather("Printing and science?", ["printing press"])
        self.assertEqual(len(evidence), 1)  # the article with no text is skipped
        self.assertEqual(evidence[0].location, "https://en.wikipedia.org/wiki/Printing_press")
        self.assertEqual(evidence[0].title, "Printing press (Wikipedia)")
        self.assertEqual(evidence[0].kind, "wikipedia")

    def test_no_results(self):
        self.assertEqual(WikipediaTool(api=lambda p: {"query": {"search": []}}).gather("Q?", ["q"]), [])


if __name__ == "__main__":
    unittest.main()
