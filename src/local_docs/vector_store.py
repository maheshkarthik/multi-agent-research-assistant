"""Persist document chunks in ChromaDB and retrieve them with hybrid search.

Typical usage::

    store = VectorStore()                      # ./chroma_db, collection "document_chunks"
    store.add_documents(document.chunks)       # chunks from DocumentProcessor
    results = store.search("What was Q3 revenue?", top_k=5)
    for result in results:
        print(result.similarity, result.source, result.page, result.text[:80])

    store.search("INV-20431", mode="semantic")  # semantic-only, for comparison

Search combines two rankings with Reciprocal Rank Fusion (RRF):

* semantic: ChromaDB nearest neighbours by embedding cosine distance, which
  finds paraphrases and related wording;
* keyword: Okapi BM25 over the stored chunk text, which finds exact terms such
  as names, codes and numbers that embeddings tend to blur.

Embeddings are computed locally by ChromaDB's default embedding function
(all-MiniLM-L6-v2 via ONNX); no external API is called. The model is
downloaded to ~/.cache/chroma on first use. The BM25 index is built in memory
from the stored chunks and uses only the standard library.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union, get_args

import chromadb
from chromadb.errors import ChromaError
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.local_docs.document_processor import ChunkMetadata, DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIRECTORY = "./chroma_db"
DEFAULT_COLLECTION_NAME = "document_chunks"
DEFAULT_TOP_K = 5

SearchMode = Literal["hybrid", "semantic", "keyword"]
DEFAULT_SEARCH_MODE: SearchMode = "hybrid"
DEFAULT_RRF_K = 60
"""RRF constant: larger values flatten the advantage of top ranks. 60 is the standard choice."""
DEFAULT_CANDIDATE_POOL = 20
"""How many candidates each retriever contributes before fusion."""


class VectorStoreError(Exception):
    """Raised when a ChromaDB operation fails."""


@dataclass(frozen=True)
class SearchResult:
    """One chunk returned by a search.

    ``distance`` (and therefore ``similarity``) is always the semantic cosine
    distance to the query, whichever retriever found the chunk, so similarity
    thresholds keep the same meaning in every search mode.

    Attributes:
        id: The chunk's ID in the store.
        text: The original chunk text.
        metadata: The chunk's metadata (``source``, ``page``, ``chunk_index``).
        distance: Cosine distance to the query; 0 is identical, lower is closer.
        semantic_rank: 1-based rank among semantic candidates, or None if the
            chunk was not a semantic candidate.
        keyword_rank: 1-based rank among BM25 candidates, or None if the chunk
            was not a keyword candidate.
        keyword_score: BM25 score, or None if not a keyword candidate.
        fusion_score: Reciprocal Rank Fusion score; None outside hybrid mode.
    """

    id: str
    text: str
    metadata: ChunkMetadata
    distance: float
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    fusion_score: float | None = None

    @property
    def similarity(self) -> float:
        """Cosine similarity to the query (``1 - distance``); higher is closer."""
        return 1.0 - self.distance

    @property
    def source(self) -> str:
        """Filename of the document this chunk came from."""
        return str(self.metadata["source"])

    @property
    def page(self) -> int:
        """1-based page number this chunk came from."""
        return int(self.metadata["page"])


@dataclass(frozen=True)
class DocumentInfo:
    """Summary of one document held in the store.

    Attributes:
        source: The document's filename.
        chunk_count: Number of chunks stored for it.
    """

    source: str
    chunk_count: int


_TOKEN_PATTERN = re.compile(r"\w+")
_STOP_WORDS = frozenset(
    """a about an and any are as at be been but by can could did do does for from had has
    have how i if in into is it its may me might must my of on or our shall should so
    than that the their them then there these they this those to was we were what when
    where which who whom why will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens for keyword search, dropping common stop words.

    ``\\w+`` keeps letters and digits from any script, so "INV-20431" becomes
    ``["inv", "20431"]`` both in documents and in queries.
    """
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _STOP_WORDS]


class BM25Index:
    """An in-memory Okapi BM25 index over chunk text.

    Memory use is roughly proportional to the stored text (about 10 MB for
    10,000 chunks of 1,000 characters), which is fine for a local corpus.
    """

    def __init__(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[ChunkMetadata],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Index the given chunks.

        Args:
            ids: Chunk IDs.
            texts: Chunk texts, parallel to ``ids``.
            metadatas: Chunk metadata, parallel to ``ids``.
            k1: Term-frequency saturation; higher lets repeated terms count more.
            b: Length normalisation; 1 fully normalises by chunk length, 0 ignores it.
        """
        self.ids = list(ids)
        self.texts = list(texts)
        self.metadatas = [dict(m) for m in metadatas]
        self.k1 = k1
        self.b = b

        self._term_freqs = [Counter(tokenize(text)) for text in self.texts]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        self._postings: dict[str, list[int]] = defaultdict(list)
        for index, term_freqs in enumerate(self._term_freqs):
            for term in term_freqs:
                self._postings[term].append(index)
        n = len(self.ids)
        self._idf = {
            term: math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, limit: int, source: str | None = None) -> list[tuple[int, float]]:
        """Score chunks against a query.

        Args:
            query: Free-text query.
            limit: Maximum number of hits.
            source: If given, only score chunks from this document.

        Returns:
            ``(position, score)`` pairs for chunks sharing at least one term
            with the query, highest score first. Ties are broken by chunk ID.
        """
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index in self._postings[term]:
                if source and self.metadatas[index].get("source") != source:
                    continue
                tf = self._term_freqs[index][term]
                norm = 1 - self.b + self.b * self._lengths[index] / self._avg_length
                scores[index] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.ids[item[0]]))
        return ranked[:limit]


@dataclass(frozen=True)
class _Candidate:
    """A chunk found by one retriever, before fusion."""

    id: str
    text: str
    metadata: ChunkMetadata
    rank: int
    distance: float | None = None
    score: float | None = None


class VectorStore:
    """A persistent ChromaDB collection of document chunks with hybrid search."""

    def __init__(
        self,
        persist_directory: Union[str, Path] = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        search_mode: SearchMode = DEFAULT_SEARCH_MODE,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    ) -> None:
        """Open (or create) the on-disk database and its chunk collection.

        Args:
            persist_directory: Folder where ChromaDB stores its files.
            collection_name: Name of the collection holding the chunks.
            search_mode: Default mode for :meth:`search`: ``hybrid``,
                ``semantic`` or ``keyword``.
            rrf_k: Reciprocal Rank Fusion constant (must be positive).
            candidate_pool: Candidates each retriever contributes to fusion;
                raised to ``top_k`` automatically when ``top_k`` is larger.

        Raises:
            ValueError: If ``search_mode``, ``rrf_k`` or ``candidate_pool`` is invalid.
            VectorStoreError: If the database or collection cannot be opened.
        """
        _validate_mode(search_mode)
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")
        if candidate_pool < 1:
            raise ValueError(f"candidate_pool must be at least 1, got {candidate_pool}")
        self.search_mode: SearchMode = search_mode
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self._keyword_index_cache: BM25Index | None = None

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        with _wrap_errors(f"open collection {collection_name!r} in {self.persist_directory}"):
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=DefaultEmbeddingFunction(),
                configuration={"hnsw": {"space": "cosine"}},
            )
        logger.info(
            "Opened collection %r in %s (%d chunks)",
            collection_name,
            self.persist_directory,
            self.count(),
        )

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        with _wrap_errors("count chunks"):
            return self._collection.count()

    def add_documents(self, chunks: Iterable[DocumentChunk]) -> int:
        """Store chunks, replacing any earlier version of the same documents.

        Chunks are written with upsert, so adding the same document twice
        overwrites it instead of duplicating it. Chunks left over from an
        earlier, longer version of a document are removed, so the store always
        matches the latest ingestion of each source.

        Args:
            chunks: Chunks from :class:`~src.local_docs.document_processor.DocumentProcessor`.
                Typically all chunks of one or more documents.

        Returns:
            The number of chunks written.

        Raises:
            TypeError: If any item is not a DocumentChunk.
            VectorStoreError: If ChromaDB rejects the write.
        """
        unique = self._deduplicate(chunks)
        if not unique:
            logger.info("add_documents called with no chunks; nothing to store")
            return 0

        new_ids = {chunk.id for chunk in unique}
        sources = sorted({chunk.source for chunk in unique})
        self._keyword_index_cache = None  # rebuilt on the next keyword search
        with _wrap_errors(f"add chunks for {sources}"):
            for source in sources:
                stale = [i for i in self._ids_for_source(source) if i not in new_ids]
                if stale:
                    self._collection.delete(ids=stale)
                    logger.info("Removed %d stale chunks of %s", len(stale), source)

            batch_size = self._client.get_max_batch_size()
            for start in range(0, len(unique), batch_size):
                batch = unique[start : start + batch_size]
                self._collection.upsert(
                    ids=[chunk.id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch],
                )

        logger.info("Stored %d chunks from %s", len(unique), ", ".join(sources))
        return len(unique)

    def search(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        source: str | None = None,
        mode: SearchMode | None = None,
    ) -> list[SearchResult]:
        """Find the chunks most relevant to a question.

        Args:
            question: The natural-language query.
            top_k: Maximum number of results to return.
            source: If given, only search chunks from this document.
            mode: ``hybrid`` (semantic + BM25, fused with RRF), ``semantic``
                (embeddings only) or ``keyword`` (BM25 only). Defaults to the
                store's ``search_mode``.

        Returns:
            Results ordered from most to least relevant, each chunk at most
            once. Semantic mode orders by similarity, keyword mode by BM25
            score, hybrid mode by RRF score. Empty if the question is blank or
            nothing is stored.

        Raises:
            ValueError: If ``top_k`` is less than 1 or ``mode`` is unknown.
            VectorStoreError: If the query fails.
        """
        mode = mode or self.search_mode
        _validate_mode(mode)
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        if not question or not question.strip():
            logger.warning("search called with a blank question; returning no results")
            return []

        available = self.count()
        if available == 0:
            logger.info("search called on an empty store; returning no results")
            return []

        if mode == "semantic":
            results = [
                SearchResult(c.id, c.text, c.metadata, c.distance, semantic_rank=c.rank)
                for c in self._semantic_candidates(question, min(top_k, available), source)
            ]
        elif mode == "keyword":
            hits = self._keyword_candidates(question, top_k, source)
            distances = self._distances(question, [hit.id for hit in hits])
            results = [
                SearchResult(
                    hit.id, hit.text, hit.metadata, distances[hit.id],
                    keyword_rank=hit.rank, keyword_score=hit.score,
                )
                for hit in hits
            ]
        else:
            results = self._hybrid_search(question, top_k, source, available)

        logger.debug(
            "search(%r, top_k=%d, mode=%s) returned %d results", question, top_k, mode, len(results)
        )
        return results

    def _hybrid_search(
        self, question: str, top_k: int, source: str | None, available: int
    ) -> list[SearchResult]:
        """Fuse semantic and BM25 rankings with Reciprocal Rank Fusion.

        Each chunk scores ``sum(1 / (rrf_k + rank))`` over the rankings it
        appears in, so chunks ranked well by both retrievers rise to the top,
        and a chunk found by only one retriever can still make the cut.
        """
        pool = max(self.candidate_pool, top_k)
        semantic = {c.id: c for c in self._semantic_candidates(question, min(pool, available), source)}
        keyword = {c.id: c for c in self._keyword_candidates(question, pool, source)}

        fused: dict[str, float] = defaultdict(float)
        for candidates in (semantic, keyword):
            for chunk_id, candidate in candidates.items():
                fused[chunk_id] += 1.0 / (self.rrf_k + candidate.rank)

        def sort_key(chunk_id: str) -> tuple[float, float, str]:
            distance = semantic[chunk_id].distance if chunk_id in semantic else math.inf
            return (-fused[chunk_id], distance, chunk_id)

        winners = sorted(fused, key=sort_key)[:top_k]
        # Keyword-only winners have no semantic distance yet; measure it so
        # similarity means the same thing for every result.
        distances = self._distances(question, [i for i in winners if i not in semantic])

        results: list[SearchResult] = []
        for chunk_id in winners:
            sem, kw = semantic.get(chunk_id), keyword.get(chunk_id)
            base = sem or kw
            assert base is not None
            results.append(
                SearchResult(
                    id=chunk_id,
                    text=base.text,
                    metadata=base.metadata,
                    distance=sem.distance if sem else distances[chunk_id],
                    semantic_rank=sem.rank if sem else None,
                    keyword_rank=kw.rank if kw else None,
                    keyword_score=kw.score if kw else None,
                    fusion_score=fused[chunk_id],
                )
            )
        logger.debug(
            "Hybrid search: %d semantic + %d keyword candidates -> %d unique, returning %d",
            len(semantic), len(keyword), len(fused), len(results),
        )
        return results

    def _semantic_candidates(self, question: str, n: int, source: str | None) -> list[_Candidate]:
        with _wrap_errors("search"):
            response = self._collection.query(
                query_texts=[question],
                n_results=n,
                where={"source": source} if source else None,
                include=["documents", "metadatas", "distances"],
            )
        return [
            _Candidate(id=chunk_id, text=text, metadata=dict(metadata), rank=rank, distance=float(distance))
            for rank, (chunk_id, text, metadata, distance) in enumerate(
                zip(
                    response["ids"][0],
                    response["documents"][0],
                    response["metadatas"][0],
                    response["distances"][0],
                ),
                start=1,
            )
        ]

    def _keyword_candidates(self, question: str, n: int, source: str | None) -> list[_Candidate]:
        index = self._keyword_index()
        return [
            _Candidate(
                id=index.ids[position],
                text=index.texts[position],
                metadata=index.metadatas[position],
                rank=rank,
                score=score,
            )
            for rank, (position, score) in enumerate(index.search(question, n, source), start=1)
        ]

    def _distances(self, question: str, ids: list[str]) -> dict[str, float]:
        """Return the semantic cosine distance from the question to specific chunks."""
        if not ids:
            return {}
        with _wrap_errors("measure similarity"):
            response = self._collection.query(
                query_texts=[question], ids=ids, n_results=len(ids), include=["distances"]
            )
        return dict(zip(response["ids"][0], map(float, response["distances"][0])))

    def _keyword_index(self) -> BM25Index:
        """Return the BM25 index, rebuilding it if the collection has changed.

        The index is rebuilt after this instance writes, or when the chunk
        count differs (another process or instance changed the store).
        """
        cached = self._keyword_index_cache
        if cached is not None and len(cached) == self.count():
            return cached
        with _wrap_errors("build keyword index"):
            data = self._collection.get(include=["documents", "metadatas"])
        index = BM25Index(data["ids"], data["documents"] or [], data["metadatas"] or [])
        logger.info("Built keyword index over %d chunks", len(index))
        self._keyword_index_cache = index
        return index

    def delete_document(self, source: str) -> int:
        """Delete every chunk belonging to a document.

        Args:
            source: The document's filename, as stored in chunk metadata.

        Returns:
            The number of chunks deleted; 0 if the document was not in the store.

        Raises:
            ValueError: If ``source`` is blank.
            VectorStoreError: If the deletion fails.
        """
        if not source or not source.strip():
            raise ValueError("source must be a non-empty filename")

        with _wrap_errors(f"delete {source}"):
            ids = self._ids_for_source(source)
            if not ids:
                logger.warning("delete_document: %s is not in the store", source)
                return 0
            self._keyword_index_cache = None
            self._collection.delete(ids=ids)

        logger.info("Deleted %d chunks of %s", len(ids), source)
        return len(ids)

    def list_documents(self) -> list[DocumentInfo]:
        """Return every stored document and its chunk count, sorted by filename."""
        with _wrap_errors("list documents"):
            metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        counts = Counter(str(metadata["source"]) for metadata in metadatas)
        return [DocumentInfo(source=name, chunk_count=counts[name]) for name in sorted(counts)]

    def _ids_for_source(self, source: str) -> list[str]:
        return self._collection.get(where={"source": source}, include=[])["ids"]

    @staticmethod
    def _deduplicate(chunks: Iterable[DocumentChunk]) -> list[DocumentChunk]:
        """Validate chunks, drop blank ones, and keep the last chunk for each repeated ID."""
        by_id: dict[str, DocumentChunk] = {}
        for chunk in chunks:
            if not isinstance(chunk, DocumentChunk):
                raise TypeError(f"Expected DocumentChunk, got {type(chunk).__name__}")
            if not chunk.text.strip():
                logger.warning("Skipping blank chunk %s", chunk.id)
                continue
            if chunk.id in by_id:
                logger.warning("Duplicate chunk ID %s in input; keeping the last one", chunk.id)
            by_id[chunk.id] = chunk
        return list(by_id.values())


def _validate_mode(mode: str) -> None:
    if mode not in get_args(SearchMode):
        raise ValueError(f"mode must be one of {get_args(SearchMode)}, got {mode!r}")


@contextmanager
def _wrap_errors(action: str) -> Iterator[None]:
    """Re-raise ChromaDB errors as VectorStoreError with context about what failed."""
    try:
        yield
    except ChromaError as exc:
        raise VectorStoreError(f"Could not {action}: {exc}") from exc
