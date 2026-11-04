"""Load PDF files and split their text into chunks ready for a vector store.

Typical usage::

    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    document = processor.process_pdf("data/report.pdf")
    records = document.to_chroma_records()  # ids, documents, metadatas

Chunks never span page boundaries, so every chunk maps to exactly one page
and its page number can be cited precisely in answers.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Union

from pypdf import PdfReader

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

PdfSource = Union[str, Path, bytes, BinaryIO]
"""A file path, raw PDF bytes, or a binary file-like object (e.g. a Streamlit upload)."""

ChunkMetadata = dict[str, Union[str, int]]


class DocumentProcessingError(Exception):
    """Raised when a file cannot be read as a PDF (missing, corrupt, or encrypted)."""


@dataclass(frozen=True)
class DocumentChunk:
    """A piece of text from one page of a document, plus where it came from.

    Attributes:
        text: The chunk's text content.
        source: Filename of the document the chunk came from.
        page: 1-based page number the chunk was taken from.
        chunk_index: 0-based position of the chunk within the whole document.
    """

    text: str
    source: str
    page: int
    chunk_index: int

    @property
    def id(self) -> str:
        """A stable identifier, unique within a document, for use as a vector store ID."""
        return f"{self.source}::page-{self.page}::chunk-{self.chunk_index}"

    @property
    def metadata(self) -> ChunkMetadata:
        """Metadata as a flat dict of primitive values, as ChromaDB requires."""
        return {
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


@dataclass
class ProcessedDocument:
    """The result of processing one PDF.

    Attributes:
        source: Filename of the processed document.
        page_count: Total number of pages in the PDF.
        chunks: Text chunks in document order.
        skipped_pages: 1-based page numbers that yielded no text (blank,
            scanned images, or extraction failures).
    """

    source: str
    page_count: int
    chunks: list[DocumentChunk] = field(default_factory=list)
    skipped_pages: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if no text could be extracted from any page."""
        return not self.chunks

    def to_chroma_records(self) -> dict[str, list]:
        """Return chunks as parallel lists matching ChromaDB's ``collection.add()`` arguments.

        Returns:
            A dict with ``ids``, ``documents`` and ``metadatas`` keys, each a
            list with one entry per chunk.
        """
        return {
            "ids": [chunk.id for chunk in self.chunks],
            "documents": [chunk.text for chunk in self.chunks],
            "metadatas": [chunk.metadata for chunk in self.chunks],
        }


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into fixed-size character chunks that overlap.

    Each chunk starts ``chunk_size - chunk_overlap`` characters after the
    previous one, so consecutive chunks share exactly ``chunk_overlap``
    characters. The last chunk may be shorter than ``chunk_size``.

    Args:
        text: The text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared by consecutive chunks.

    Returns:
        The chunks in order, or an empty list if ``text`` is blank.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or ``chunk_overlap`` is
            negative or not smaller than ``chunk_size``.
    """
    _validate_chunk_settings(chunk_size, chunk_overlap)

    text = text.strip()
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
    return chunks


class DocumentProcessor:
    """Extracts text from PDFs and splits it into metadata-tagged chunks."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Create a processor.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters shared by consecutive chunks.

        Raises:
            ValueError: If the chunk settings are invalid (see :func:`split_text`).
        """
        _validate_chunk_settings(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_pdf(self, source: PdfSource, source_name: str | None = None) -> ProcessedDocument:
        """Read a PDF and return its text as chunks with metadata.

        Pages without extractable text are skipped and listed in
        ``skipped_pages`` rather than raising, so a partly scanned PDF still
        yields chunks for its text pages. A PDF with no text at all returns a
        document whose ``is_empty`` is True.

        Args:
            source: A file path, raw PDF bytes, or a binary file-like object.
            source_name: Filename to record in chunk metadata. If omitted, it
                is taken from the path or the file object's ``name`` attribute.

        Returns:
            The processed document.

        Raises:
            DocumentProcessingError: If the file is missing, is not a valid
                PDF, or is password-protected.
        """
        name = source_name or _infer_source_name(source)
        reader = _open_reader(source, name)
        page_texts = _extract_page_texts(reader, name)

        document = ProcessedDocument(source=name, page_count=len(page_texts))
        for page_number, page_text in enumerate(page_texts, start=1):
            page_chunks = split_text(page_text, self.chunk_size, self.chunk_overlap)
            if not page_chunks:
                document.skipped_pages.append(page_number)
                continue
            for text in page_chunks:
                document.chunks.append(
                    DocumentChunk(
                        text=text,
                        source=name,
                        page=page_number,
                        chunk_index=len(document.chunks),
                    )
                )

        if document.is_empty:
            logger.warning("No extractable text found in %s (%d pages)", name, document.page_count)
        elif document.skipped_pages:
            logger.info("Skipped pages without text in %s: %s", name, document.skipped_pages)
        return document


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"chunk_overlap must be between 0 and chunk_size - 1, got {chunk_overlap}"
        )


def _infer_source_name(source: PdfSource) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return Path(name).name
    return "unknown.pdf"


def _open_reader(source: PdfSource, name: str) -> PdfReader:
    """Open a PDF, converting every failure into DocumentProcessingError."""
    if isinstance(source, (str, Path)) and not Path(source).is_file():
        raise DocumentProcessingError(f"File not found: {source}")
    stream = io.BytesIO(source) if isinstance(source, bytes) else source

    try:
        reader = PdfReader(stream)
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentProcessingError(f"{name} is password-protected")
        # Touch the page tree now so structural corruption surfaces here.
        len(reader.pages)
    except DocumentProcessingError:
        raise
    except Exception as exc:  # pypdf raises many exception types for malformed files
        raise DocumentProcessingError(f"{name} is not a valid PDF: {exc}") from exc
    return reader


def _extract_page_texts(reader: PdfReader, name: str) -> list[str]:
    """Return cleaned text for every page, using "" for pages that fail to extract."""
    texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # a single bad page shouldn't sink the document
            logger.warning("Could not extract text from %s page %d: %s", name, page_number, exc)
            raw = ""
        texts.append(_clean_text(raw))
    return texts


def _clean_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
