"""Index PDFs for local research: python -m src.index [folder-or-files ...]

With no arguments, indexes the sample documents in data/samples/. Re-running is
safe: documents already in the index are skipped unless --reindex is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.local_docs import db_dir

if TYPE_CHECKING:
    from src.local_docs.document_processor import PdfSource
    from src.local_docs.vector_store import VectorStore

SAMPLES_DIR = Path("data/samples")


def index_pdfs(paths: list[Path], reindex: bool = False, persist_directory: Path | None = None) -> list[str]:
    """Index every PDF in ``paths`` (files or folders). Returns one status line per file."""
    from src.local_docs.vector_store import VectorStore

    store = VectorStore(persist_directory=persist_directory or db_dir())
    pdfs: list[Path] = []
    for path in paths:
        pdfs += sorted(path.glob("*.pdf")) if path.is_dir() else [path]
    return [index_document(store, str(pdf), pdf.name, reindex=reindex) for pdf in pdfs]


def index_document(store: "VectorStore", source: "PdfSource", name: str, reindex: bool = False) -> str:
    """Add one PDF (a path, bytes or file object) to ``store``. Returns a status line.

    Documents already in the index are skipped unless ``reindex`` is True, in
    which case the old chunks are replaced.
    """
    from src.local_docs.document_processor import DocumentProcessingError, DocumentProcessor

    existing = {doc.source for doc in store.list_documents()}
    if name in existing and not reindex:
        return f"{name}: already indexed, skipped"
    try:
        document = DocumentProcessor().process_pdf(source, source_name=name)
    except DocumentProcessingError as exc:
        return f"{name}: could not be read ({exc})"
    if document.is_empty:
        return f"{name}: no extractable text (scanned PDF?), skipped"
    if name in existing:
        store.delete_document(name)
    stored = store.add_documents(document.chunks)
    return f"{name}: {document.page_count} pages, {stored} chunks"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index PDFs for local research.")
    parser.add_argument("paths", nargs="*", type=Path, help="PDF files or folders (default: data/samples)")
    parser.add_argument("--reindex", action="store_true", help="re-index documents that are already indexed")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    paths = args.paths or [SAMPLES_DIR]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"Not found: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    print(f"Indexing into {db_dir()}")
    for line in index_pdfs(paths, reindex=args.reindex):
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
