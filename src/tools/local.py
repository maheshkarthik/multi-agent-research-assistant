"""Research tool that searches local PDFs with hybrid (semantic + BM25) search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.sources import Evidence
from src.text import best_excerpt
from src.tools import ToolError

if TYPE_CHECKING:  # chromadb is slow to import; only load it when local search is used
    from src.local_docs.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

PAGES_PER_QUESTION = 3
# PDF pages are dense and trusted to be on-topic once they pass the relevance check,
# so they get more room than web pages (at most 3 pages, so the prompt stays small).
EXCERPT_CHARS = 2500
CANDIDATES_PER_QUERY = 6
# Local documents are often unrelated to a web-style question. A chunk is used only
# if it is semantically close to the question, or is a top keyword match that is
# at least loosely related (pure keyword overlap such as "science" isn't enough).
MIN_SIMILARITY = 0.35
KEYWORD_MIN_SIMILARITY = 0.3
KEYWORD_RANK_CUTOFF = 2


class LocalDocsTool:
    """Finds the most relevant pages in the indexed PDFs.

    Chunks from the same page are merged, so each piece of evidence is one page
    and can be cited as "file.pdf, page N".
    """

    name = "local"

    def __init__(self, store: "VectorStore", pages_per_question: int = PAGES_PER_QUESTION) -> None:
        self.store = store
        self.pages_per_question = pages_per_question

    def gather(self, sub_question: str, queries: list[str]) -> list[Evidence]:
        from src.local_docs.vector_store import VectorStoreError

        best: dict[str, "SearchResult"] = {}
        try:
            for query in dict.fromkeys([sub_question, *queries]):
                for result in self.store.search(query, top_k=CANDIDATES_PER_QUERY):
                    if is_relevant(result) and (result.id not in best or result.similarity > best[result.id].similarity):
                        best[result.id] = result
        except VectorStoreError as exc:
            raise ToolError(f"Local document search failed: {exc}") from exc

        pages: dict[tuple[str, int], list["SearchResult"]] = {}
        for result in sorted(best.values(), key=lambda r: -r.similarity):
            pages.setdefault((result.source, result.page), []).append(result)

        evidence = []
        for (source, page), chunks in list(pages.items())[: self.pages_per_question]:
            chunks.sort(key=lambda r: int(r.metadata.get("chunk_index", 0)))
            text = best_excerpt("\n".join(chunk.text for chunk in chunks), sub_question, limit=EXCERPT_CHARS)
            label = f"{source}, page {page}"
            evidence.append(Evidence(kind="local", title=label, location=label, text=text))
        return evidence


def is_relevant(result: "SearchResult") -> bool:
    if result.similarity >= MIN_SIMILARITY:
        return True
    return (
        result.keyword_rank is not None
        and result.keyword_rank <= KEYWORD_RANK_CUTOFF
        and result.similarity >= KEYWORD_MIN_SIMILARITY
    )
