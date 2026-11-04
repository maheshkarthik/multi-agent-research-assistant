"""Index PDFs for local research: python -m src.index [folder-or-files ...]

With no arguments, indexes the sample documents in data/samples/. Re-running is
safe: documents already in the index are skipped unless --reindex is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.local_docs import db_dir

SAMPLES_DIR = Path("data/samples")


def index_pdfs(paths: list[Path], reindex: bool = False, persist_directory: Path | None = None) -> list[str]:
    """Index every PDF in ``paths`` (files or folders). Returns one status line per file."""
    from src.local_docs.document_processor import DocumentProcessor
    from src.local_docs.vector_store import VectorStore

    store = VectorStore(persist_directory=persist_directory or db_dir())
    processor = DocumentProcessor()
    existing = {doc.source for doc in store.list_documents()}

    pdfs: list[Path] = []
    for path in paths:
        pdfs += sorted(path.glob("*.pdf")) if path.is_dir() else [path]

    lines = []
    for pdf in pdfs:
        if pdf.name in existing and not reindex:
            lines.append(f"{pdf.name}: already indexed, skipped")
            continue
        document = processor.process_pdf(str(pdf))
        if document.is_empty:
            lines.append(f"{pdf.name}: no extractable text (scanned PDF?), skipped")
            continue
        if reindex and pdf.name in existing:
            store.delete_document(pdf.name)
        stored = store.add_documents(document.chunks)
        lines.append(f"{pdf.name}: {document.page_count} pages, {stored} chunks")
    return lines


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
