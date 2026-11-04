"""Tests for src.local_docs.vector_store (copied from Project 01).

These use ChromaDB's real local embedding model (no network after the first
download). Each test gets its own temporary database directory.

Run from the project root with:  python -m unittest -v
"""

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from chromadb.errors import InternalError

from src.local_docs.document_processor import DocumentChunk, DocumentProcessor
from src.local_docs.vector_store import (
    BM25Index,
    DocumentInfo,
    SearchResult,
    VectorStore,
    VectorStoreError,
    tokenize,
)
from tests.test_document_processor import make_pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COOKING = [
    "Preheat the oven to 220 degrees and knead the bread dough for ten minutes.",
    "Simmer the tomato sauce with garlic, basil and olive oil before serving with pasta.",
]
ASTRONOMY = [
    "Jupiter is the largest planet in the solar system and has dozens of moons.",
    "A black hole forms when a massive star collapses under its own gravity.",
    "The Andromeda galaxy is the nearest large spiral galaxy to the Milky Way.",
]
FINANCE = [
    "Quarterly revenue increased by twelve percent, driven by subscription sales.",
]


def chunks_for(source: str, texts: list[str], page: int = 1) -> list[DocumentChunk]:
    return [DocumentChunk(text=t, source=source, page=page, chunk_index=i) for i, t in enumerate(texts)]


class VectorStoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "chroma_db"
        self.store = VectorStore(persist_directory=self.db_path)

    def tearDown(self):
        self._tmp.cleanup()

    def add_corpus(self):
        self.store.add_documents(chunks_for("cookbook.pdf", COOKING))
        self.store.add_documents(chunks_for("space.pdf", ASTRONOMY, page=4))
        self.store.add_documents(chunks_for("report.pdf", FINANCE, page=2))


class InsertTests(VectorStoreTestCase):
    def test_insert_chunks(self):
        written = self.store.add_documents(chunks_for("space.pdf", ASTRONOMY))
        self.assertEqual(written, 3)
        self.assertEqual(self.store.count(), 3)

    def test_empty_input_is_a_no_op(self):
        self.assertEqual(self.store.add_documents([]), 0)
        self.assertEqual(self.store.count(), 0)

    def test_blank_chunks_are_skipped(self):
        chunks = chunks_for("a.pdf", ["Real content here.", "   "])
        self.assertEqual(self.store.add_documents(chunks), 1)

    def test_non_chunk_input_raises(self):
        with self.assertRaises(TypeError):
            self.store.add_documents(["just a string"])
        self.assertEqual(self.store.count(), 0)

    def test_list_documents(self):
        self.add_corpus()
        self.assertEqual(
            self.store.list_documents(),
            [
                DocumentInfo("cookbook.pdf", 2),
                DocumentInfo("report.pdf", 1),
                DocumentInfo("space.pdf", 3),
            ],
        )

    def test_list_documents_on_empty_store(self):
        self.assertEqual(self.store.list_documents(), [])


class SearchTests(VectorStoreTestCase):
    def test_semantic_search_finds_related_chunk(self):
        self.add_corpus()
        cases = {
            "Which planet is biggest?": "space.pdf",
            "How do I bake a loaf?": "cookbook.pdf",
            "How did sales perform financially?": "report.pdf",
        }
        for question, expected_source in cases.items():
            with self.subTest(question=question):
                top = self.store.search(question, top_k=1)[0]
                self.assertEqual(top.source, expected_source)

    def test_results_ordered_by_similarity_with_scores(self):
        self.add_corpus()
        results = self.store.search("stars and galaxies in space", top_k=6)
        self.assertEqual(len(results), 6)
        distances = [r.distance for r in results]
        self.assertEqual(distances, sorted(distances))
        for result in results:
            self.assertIsInstance(result, SearchResult)
            self.assertAlmostEqual(result.similarity, 1 - result.distance)
            self.assertTrue(-1.0 <= result.similarity <= 1.0)
        # All three astronomy chunks should outrank cooking and finance.
        self.assertEqual({r.source for r in results[:3]}, {"space.pdf"})

    def test_top_k_defaults_to_five(self):
        self.add_corpus()
        self.assertEqual(len(self.store.search("anything")), 5)

    def test_top_k_larger_than_store_returns_everything(self):
        self.store.add_documents(chunks_for("cookbook.pdf", COOKING))
        self.assertEqual(len(self.store.search("food", top_k=50)), 2)

    def test_search_filtered_by_source(self):
        self.add_corpus()
        results = self.store.search("planets", top_k=5, source="cookbook.pdf")
        self.assertEqual({r.source for r in results}, {"cookbook.pdf"})

    def test_blank_question_returns_nothing(self):
        self.add_corpus()
        self.assertEqual(self.store.search("   "), [])

    def test_empty_store_returns_nothing(self):
        self.assertEqual(self.store.search("Is anything here?"), [])

    def test_invalid_top_k_raises(self):
        for bad in (0, -3):
            with self.subTest(top_k=bad), self.assertRaises(ValueError):
                self.store.search("question", top_k=bad)


class MetadataTests(VectorStoreTestCase):
    def test_text_and_metadata_round_trip(self):
        chunk = DocumentChunk(text=ASTRONOMY[1], source="space.pdf", page=7, chunk_index=12)
        self.store.add_documents([chunk])
        result = self.store.search("black hole", top_k=1)[0]
        self.assertEqual(result.id, chunk.id)
        self.assertEqual(result.text, chunk.text)
        self.assertEqual(result.metadata, {"source": "space.pdf", "page": 7, "chunk_index": 12})
        self.assertEqual(result.page, 7)
        self.assertIsInstance(result.metadata["page"], int)

    def test_metadata_from_processed_pdf(self):
        pdf = make_pdf([["Page one talks about volcanoes and lava."], [], ["Page three covers ocean tides."]])
        document = DocumentProcessor().process_pdf(pdf, source_name="earth.pdf")
        self.store.add_documents(document.chunks)

        result = self.store.search("What causes tides?", top_k=1)[0]
        self.assertEqual(result.metadata, {"source": "earth.pdf", "page": 3, "chunk_index": 1})


class DeleteTests(VectorStoreTestCase):
    def test_delete_document_removes_only_its_chunks(self):
        self.add_corpus()
        self.assertEqual(self.store.delete_document("space.pdf"), 3)
        self.assertEqual(self.store.count(), 3)
        self.assertNotIn("space.pdf", [d.source for d in self.store.list_documents()])
        self.assertNotIn("space.pdf", {r.source for r in self.store.search("planets", top_k=10)})

    def test_delete_unknown_document_returns_zero(self):
        self.add_corpus()
        self.assertEqual(self.store.delete_document("missing.pdf"), 0)
        self.assertEqual(self.store.count(), 6)

    def test_delete_blank_source_raises(self):
        with self.assertRaises(ValueError):
            self.store.delete_document("")


class DuplicateIngestionTests(VectorStoreTestCase):
    def test_adding_same_document_twice_does_not_duplicate(self):
        chunks = chunks_for("space.pdf", ASTRONOMY)
        self.store.add_documents(chunks)
        self.store.add_documents(chunks)
        self.assertEqual(self.store.count(), 3)
        self.assertEqual(self.store.list_documents(), [DocumentInfo("space.pdf", 3)])

    def test_reingesting_updated_document_replaces_old_chunks(self):
        self.store.add_documents(chunks_for("space.pdf", ASTRONOMY))
        self.store.add_documents(chunks_for("space.pdf", ["Saturn has spectacular rings of ice."]))
        self.assertEqual(self.store.count(), 1)
        self.assertIn("Saturn", self.store.search("planets", top_k=5)[0].text)

    def test_reingesting_one_document_leaves_others_alone(self):
        self.add_corpus()
        self.store.add_documents(chunks_for("cookbook.pdf", COOKING))
        self.assertEqual(self.store.count(), 6)

    def test_duplicate_ids_within_one_call_are_collapsed(self):
        chunk = chunks_for("space.pdf", ASTRONOMY[:1])[0]
        self.assertEqual(self.store.add_documents([chunk, chunk]), 1)
        self.assertEqual(self.store.count(), 1)


class PersistenceTests(VectorStoreTestCase):
    def test_new_instance_sees_existing_data(self):
        self.add_corpus()
        reopened = VectorStore(persist_directory=self.db_path)
        self.assertEqual(reopened.count(), 6)
        self.assertEqual(reopened.search("Which planet is biggest?", top_k=1)[0].source, "space.pdf")

    def test_data_survives_a_separate_process(self):
        """A fresh interpreter rules out any in-process client caching."""
        self.add_corpus()
        script = textwrap.dedent(
            f"""
            from src.local_docs.vector_store import VectorStore
            store = VectorStore(persist_directory={str(self.db_path)!r})
            top = store.search("Which planet is biggest?", top_k=1)[0]
            print(store.count(), top.source, top.page)
            """
        )
        output = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        self.assertEqual(output, ["6", "space.pdf", "4"])

    def test_separate_collections_are_isolated(self):
        self.add_corpus()
        other = VectorStore(persist_directory=self.db_path, collection_name="other_chunks")
        self.assertEqual(other.count(), 0)


# --- Hybrid retrieval -------------------------------------------------------

# A ticket ID buried in a chunk about something else, surrounded by chunks that
# sound like the question. Embeddings rank it last; BM25 finds the exact ID.
BURIED_ID_CHUNK = DocumentChunk(
    "The office relocation to the fourth floor begins on Monday. Movers will pack desks, monitors and "
    "filing cabinets over the weekend, and the kitchen will be closed until Wednesday. Network cabling "
    "for the new floor is tracked under QX-7781. Parking permits remain unchanged.",
    source="ops.pdf", page=12, chunk_index=0,
)
TICKET_CHUNKS = chunks_for("itsm.pdf", [
    "Support tickets are assigned to the on-call engineer, who is responsible for resolving them within one business day.",
    "Each ticket has an owner responsible for updates; unassigned tickets are reviewed by the team lead every morning.",
    "Escalated tickets are handed to the infrastructure manager, who becomes responsible until the issue is closed.",
    "Ticket priorities are set by the service desk and the responsible team is notified automatically.",
    "Who is responsible for a ticket is shown in the Assignee field of the tracking system.",
])
HR_CHUNKS = chunks_for("hr.pdf", [
    "Employees accrue 1.5 days of paid vacation per month and must request leave two weeks in advance.",
    "Unused vacation days, up to a maximum of 10, may be carried over into the next calendar year.",
], page=7)
HYBRID_CORPUS = [BURIED_ID_CHUNK, *TICKET_CHUNKS, *HR_CHUNKS]


class TokenizeAndBM25Tests(unittest.TestCase):
    def test_tokenize_lowercases_splits_and_drops_stop_words(self):
        self.assertEqual(tokenize("What is the status of INV-20431?"), ["status", "inv", "20431"])
        self.assertEqual(tokenize("the and of"), [])

    def setUp(self):
        self.index = BM25Index(
            ids=["a", "b", "c"],
            texts=[
                "apple banana apple",
                "banana cherry durian elderberry fig grape honeydew",
                "cherry",
            ],
            metadatas=[{"source": "x.pdf"}, {"source": "y.pdf"}, {"source": "y.pdf"}],
        )

    def test_rare_and_repeated_terms_score_higher(self):
        # "apple" appears only in "a" (twice): it should be the clear winner.
        hits = self.index.search("apple banana", limit=10)
        self.assertEqual([self.index.ids[i] for i, _ in hits], ["a", "b"])
        self.assertGreater(hits[0][1], hits[1][1])

    def test_shorter_chunk_wins_for_same_term(self):
        hits = self.index.search("cherry", limit=10)
        self.assertEqual([self.index.ids[i] for i, _ in hits], ["c", "b"])

    def test_no_shared_terms_gives_no_hits(self):
        self.assertEqual(self.index.search("zucchini", limit=10), [])
        self.assertEqual(self.index.search("the of and", limit=10), [])

    def test_source_filter_and_limit(self):
        self.assertEqual([self.index.ids[i] for i, _ in self.index.search("banana", 10, source="y.pdf")], ["b"])
        self.assertEqual(len(self.index.search("banana cherry", limit=1)), 1)

    def test_empty_index(self):
        self.assertEqual(BM25Index([], [], []).search("anything", limit=5), [])


class HybridSearchTestCase(VectorStoreTestCase):
    def setUp(self):
        super().setUp()
        self.store.add_documents(HYBRID_CORPUS)

    def rank_of(self, results: list[SearchResult], chunk: DocumentChunk) -> int | None:
        ids = [r.id for r in results]
        return ids.index(chunk.id) + 1 if chunk.id in ids else None


class HybridRetrievalTests(HybridSearchTestCase):
    def test_hybrid_is_the_default_mode(self):
        self.assertEqual(self.store.search_mode, "hybrid")
        result = self.store.search("Who is responsible for ticket QX-7781?")[0]
        self.assertIsNotNone(result.fusion_score)

    def test_keyword_match_ranks_highly_despite_low_semantic_similarity(self):
        question = "Who is responsible for ticket QX-7781?"
        semantic = self.store.search(question, top_k=3, mode="semantic")
        hybrid = self.store.search(question, top_k=3)

        self.assertIsNone(self.rank_of(semantic, BURIED_ID_CHUNK), "semantic alone misses the exact ID")
        self.assertIsNotNone(self.rank_of(hybrid, BURIED_ID_CHUNK), "hybrid brings it into the top 3")
        found = next(r for r in hybrid if r.id == BURIED_ID_CHUNK.id)
        self.assertEqual(found.keyword_rank, 1)
        self.assertLess(found.similarity, min(r.similarity for r in semantic))

    def test_semantic_match_ranks_highly_without_keyword_overlap(self):
        question = "Can I keep holiday I didn't use?"
        self.assertEqual(self.store.search(question, mode="keyword"), [])  # no shared terms at all
        top = self.store.search(question, top_k=3)[0]
        self.assertEqual(top.id, HR_CHUNKS[1].id)
        self.assertIsNone(top.keyword_rank)
        self.assertEqual(top.semantic_rank, 1)

    def test_keyword_only_results_report_true_semantic_similarity(self):
        question = "Who is responsible for ticket QX-7781?"
        hybrid = next(r for r in self.store.search(question, top_k=3) if r.id == BURIED_ID_CHUNK.id)
        keyword = self.store.search(question, top_k=1, mode="keyword")[0]
        semantic = next(
            r for r in self.store.search(question, top_k=len(HYBRID_CORPUS), mode="semantic")
            if r.id == BURIED_ID_CHUNK.id
        )
        self.assertAlmostEqual(hybrid.distance, semantic.distance, places=5)
        self.assertAlmostEqual(keyword.distance, semantic.distance, places=5)

    def test_text_and_metadata_preserved(self):
        result = next(
            r for r in self.store.search("QX-7781 cabling", top_k=3) if r.id == BURIED_ID_CHUNK.id
        )
        self.assertEqual(result.text, BURIED_ID_CHUNK.text)
        self.assertEqual(result.metadata, BURIED_ID_CHUNK.metadata)
        self.assertEqual((result.source, result.page), ("ops.pdf", 12))


class FusionTests(HybridSearchTestCase):
    QUESTION = "Who is responsible for ticket QX-7781?"

    def test_fusion_score_is_reciprocal_rank_sum(self):
        for result in self.store.search(self.QUESTION, top_k=len(HYBRID_CORPUS)):
            expected = sum(
                1 / (60 + rank) for rank in (result.semantic_rank, result.keyword_rank) if rank is not None
            )
            self.assertAlmostEqual(result.fusion_score, expected)

    def test_results_ordered_by_fusion_score(self):
        scores = [r.fusion_score for r in self.store.search(self.QUESTION, top_k=len(HYBRID_CORPUS))]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_agreement_between_retrievers_beats_single_retriever(self):
        results = self.store.search(self.QUESTION, top_k=len(HYBRID_CORPUS))
        both = [r for r in results if r.semantic_rank and r.keyword_rank]
        one = [r for r in results if not (r.semantic_rank and r.keyword_rank)]
        best_both = min(both, key=lambda r: r.semantic_rank + r.keyword_rank)
        self.assertTrue(all(best_both.fusion_score > r.fusion_score for r in one))

    def test_rrf_constant_is_configurable(self):
        store = VectorStore(persist_directory=self.db_path, rrf_k=1)
        results = store.search(self.QUESTION, top_k=len(HYBRID_CORPUS))
        for result in results:
            ranks = [r for r in (result.semantic_rank, result.keyword_rank) if r is not None]
            self.assertAlmostEqual(result.fusion_score, sum(1 / (1 + r) for r in ranks))
        default = self.store.search(self.QUESTION, top_k=1)[0]
        self.assertGreater(results[0].fusion_score, default.fusion_score)

    def test_results_are_unique_by_chunk_id(self):
        results = self.store.search(self.QUESTION, top_k=len(HYBRID_CORPUS))
        ids = [r.id for r in results]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(any(r.semantic_rank and r.keyword_rank for r in results), "some chunks found by both")
        self.assertEqual(len(results), len(HYBRID_CORPUS))

    def test_top_k_limits_fused_results(self):
        self.assertEqual(len(self.store.search(self.QUESTION, top_k=2)), 2)

    def test_top_k_larger_than_candidate_pool(self):
        store = VectorStore(persist_directory=self.db_path, candidate_pool=2)
        self.assertEqual(len(store.search(self.QUESTION, top_k=5)), 5)


class HybridFilteringAndEdgeCaseTests(HybridSearchTestCase):
    def test_source_filter_applies_to_both_retrievers(self):
        for mode in ("hybrid", "keyword", "semantic"):
            with self.subTest(mode=mode):
                results = self.store.search("responsible ticket QX-7781", top_k=10, source="itsm.pdf", mode=mode)
                self.assertTrue(results)
                self.assertEqual({r.source for r in results}, {"itsm.pdf"})

    def test_source_filter_excludes_keyword_only_match(self):
        results = self.store.search("QX-7781", top_k=10, source="hr.pdf")
        self.assertNotIn(BURIED_ID_CHUNK.id, [r.id for r in results])

    def test_empty_store_in_every_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = VectorStore(persist_directory=Path(tmp) / "db")
            for mode in ("hybrid", "keyword", "semantic"):
                with self.subTest(mode=mode):
                    self.assertEqual(empty.search("anything", mode=mode), [])

    def test_blank_query_and_invalid_top_k_in_every_mode(self):
        for mode in ("hybrid", "keyword", "semantic"):
            with self.subTest(mode=mode):
                self.assertEqual(self.store.search("   ", mode=mode), [])
                with self.assertRaises(ValueError):
                    self.store.search("tickets", top_k=0, mode=mode)

    def test_query_of_only_stop_words_falls_back_to_semantic_ranking(self):
        results = self.store.search("what is the", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.keyword_rank is None for r in results))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            self.store.search("tickets", mode="fuzzy")
        for kwargs in ({"search_mode": "fuzzy"}, {"rrf_k": 0}, {"candidate_pool": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VectorStore(persist_directory=self.db_path, **kwargs)

    def test_chromadb_errors_are_wrapped(self):
        for mode in ("hybrid", "keyword", "semantic"):
            with self.subTest(mode=mode):
                with mock.patch.object(self.store._collection, "query", side_effect=InternalError("boom")):
                    with self.assertRaises(VectorStoreError):
                        self.store.search("QX-7781", mode=mode)
        self.store._keyword_index_cache = None  # force an index rebuild, which reads from ChromaDB
        with mock.patch.object(self.store._collection, "get", side_effect=InternalError("boom")):
            with self.assertRaises(VectorStoreError):
                self.store.search("QX-7781", mode="keyword")


class KeywordIndexFreshnessTests(HybridSearchTestCase):
    def test_index_sees_newly_added_chunks(self):
        self.assertEqual(self.store.search("Zephyrine", mode="keyword"), [])
        self.store.add_documents(chunks_for("security.pdf", ["The Zephyrine protocol needs two-person approval."]))
        self.assertEqual(self.store.search("Zephyrine", mode="keyword")[0].source, "security.pdf")

    def test_index_forgets_deleted_documents(self):
        self.assertTrue(self.store.search("QX-7781", mode="keyword"))
        self.store.delete_document("ops.pdf")
        self.assertEqual(self.store.search("QX-7781", mode="keyword"), [])

    def test_index_sees_reingested_text(self):
        self.store.add_documents([DocumentChunk("Cabling is now tracked under ZZ-1000.", "ops.pdf", 12, 0)])
        self.assertEqual(self.store.search("QX-7781", mode="keyword"), [])
        self.assertEqual(self.store.search("ZZ-1000", mode="keyword")[0].id, BURIED_ID_CHUNK.id)

    def test_index_notices_writes_from_another_instance(self):
        self.store.search("anything", mode="keyword")  # build the cache
        VectorStore(persist_directory=self.db_path).add_documents(
            chunks_for("security.pdf", ["The Zephyrine protocol needs two-person approval."])
        )
        self.assertEqual(self.store.search("Zephyrine", mode="keyword")[0].source, "security.pdf")


class SemanticOnlyModeTests(HybridSearchTestCase):
    QUESTION = "Who is responsible for ticket QX-7781?"

    def test_semantic_mode_orders_by_similarity_without_keyword_fields(self):
        results = self.store.search(self.QUESTION, top_k=5, mode="semantic")
        self.assertEqual([r.distance for r in results], sorted(r.distance for r in results))
        self.assertEqual([r.semantic_rank for r in results], [1, 2, 3, 4, 5])
        for result in results:
            self.assertIsNone(result.keyword_rank)
            self.assertIsNone(result.keyword_score)
            self.assertIsNone(result.fusion_score)

    def test_semantic_default_mode_store(self):
        store = VectorStore(persist_directory=self.db_path, search_mode="semantic")
        default = store.search(self.QUESTION, top_k=3)
        explicit = self.store.search(self.QUESTION, top_k=3, mode="semantic")
        self.assertEqual([r.id for r in default], [r.id for r in explicit])
        self.assertIsNone(default[0].fusion_score)

    def test_search_result_still_constructible_with_four_fields(self):
        result = SearchResult(id="x", text="t", metadata={"source": "a.pdf", "page": 2, "chunk_index": 0}, distance=0.25)
        self.assertEqual((result.similarity, result.source, result.page), (0.75, "a.pdf", 2))


if __name__ == "__main__":
    unittest.main()
