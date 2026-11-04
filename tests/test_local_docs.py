"""Tests for local document search, indexing and the source switch."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.index import index_pdfs
from src.local_docs.vector_store import SearchResult, VectorStore, VectorStoreError
from src.tools import ToolError, make_tools
from src.tools.local import LocalDocsTool, is_relevant

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def result(chunk_id, source="doc.pdf", page=1, similarity=0.6, keyword_rank=None, text="Relevant text.", chunk_index=0):
    return SearchResult(
        id=chunk_id,
        text=text,
        metadata={"source": source, "page": page, "chunk_index": chunk_index},
        distance=1.0 - similarity,
        keyword_rank=keyword_rank,
    )


class FakeStore:
    def __init__(self, results_by_query=None, error=None):
        self.results_by_query = results_by_query or {}
        self.error = error
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.results_by_query.get(query, [])


class RelevanceTests(unittest.TestCase):
    def test_semantic_match_is_relevant(self):
        self.assertTrue(is_relevant(result("a", similarity=0.4)))

    def test_top_keyword_match_needs_some_similarity(self):
        self.assertTrue(is_relevant(result("a", similarity=0.31, keyword_rank=1)))
        self.assertFalse(is_relevant(result("a", similarity=0.28, keyword_rank=1)))  # e.g. "Rust vs Go" vs an outage report
        self.assertFalse(is_relevant(result("a", similarity=0.31, keyword_rank=5)))


class LocalDocsToolTests(unittest.TestCase):
    def test_merges_chunks_from_the_same_page_in_reading_order(self):
        store = FakeStore({"Q?": [
            result("c2", page=1, similarity=0.7, text="Second part.", chunk_index=2),
            result("c1", page=1, similarity=0.5, text="First part.", chunk_index=1),
            result("c9", page=4, similarity=0.6, text="Other page."),
        ]})
        evidence = LocalDocsTool(store).gather("Q?", [])
        self.assertEqual([e.location for e in evidence], ["doc.pdf, page 1", "doc.pdf, page 4"])
        self.assertEqual(evidence[0].text, "First part.\nSecond part.")
        self.assertEqual(evidence[0].kind, "local")

    def test_searches_question_and_queries_once_each(self):
        store = FakeStore()
        LocalDocsTool(store).gather("Q?", ["q one", "Q?", "q two"])
        self.assertEqual(store.queries, ["Q?", "q one", "q two"])

    def test_filters_irrelevant_chunks_and_caps_pages(self):
        store = FakeStore({"Q?": [result(f"c{i}", page=i, similarity=0.9 - i * 0.1) for i in range(1, 8)]})
        evidence = LocalDocsTool(store, pages_per_question=2).gather("Q?", [])
        self.assertEqual([e.location for e in evidence], ["doc.pdf, page 1", "doc.pdf, page 2"])
        store = FakeStore({"Q?": [result("x", similarity=0.1)]})
        self.assertEqual(LocalDocsTool(store).gather("Q?", []), [])

    def test_store_errors_become_tool_errors(self):
        with self.assertRaises(ToolError):
            LocalDocsTool(FakeStore(error=VectorStoreError("disk full"))).gather("Q?", [])


class IndexAndSwitchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_indexes_pdfs_and_skips_already_indexed(self):
        pdf = SAMPLES / "travel-guide-isle-of-carran.pdf"
        first = index_pdfs([pdf], persist_directory=self.db)
        self.assertRegex(first[0], r"travel-guide-isle-of-carran.pdf: 3 pages, \d+ chunks")
        self.assertIn("already indexed", index_pdfs([pdf], persist_directory=self.db)[0])
        self.assertIn("chunks", index_pdfs([pdf], reindex=True, persist_directory=self.db)[0])

        evidence = LocalDocsTool(VectorStore(persist_directory=self.db)).gather("How long is the ferry crossing to Carran?", [])
        self.assertTrue(evidence)
        self.assertTrue(evidence[0].location.startswith("travel-guide-isle-of-carran.pdf, page"))

    def test_source_switch(self):
        with mock.patch.dict(os.environ, {"LOCAL_DOCS_DB": str(self.db)}):
            web, warnings = make_tools("web")
            self.assertEqual([t.name for t in web], ["web", "wikipedia"])
            self.assertEqual(warnings, [])

            both, warnings = make_tools("both")  # empty index: web only, with a warning
            self.assertEqual([t.name for t in both], ["web", "wikipedia"])
            self.assertIn("python -m src.index", warnings[0])
            with self.assertRaises(ValueError):
                make_tools("local")

            index_pdfs([SAMPLES / "travel-guide-isle-of-carran.pdf"], persist_directory=self.db)
            self.assertEqual([t.name for t in make_tools("both")[0]], ["web", "wikipedia", "local"])
            self.assertEqual([t.name for t in make_tools("local")[0]], ["local"])
        with self.assertRaises(ValueError):
            make_tools("everything")


if __name__ == "__main__":
    unittest.main()
