"""PDF processing and hybrid search over local documents.

``document_processor.py`` and ``vector_store.py`` are copied from Project 01
(Local RAG Document Assistant, https://github.com/maheshkarthik/local-rag-document-assistant)
so that this repository runs on its own. Changes from the original are limited
to import paths.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DB_DIR = "./chroma_db"


def db_dir() -> Path:
    """Where the local document index lives. Set LOCAL_DOCS_DB to reuse another index, such as Project 01's."""
    return Path(os.getenv("LOCAL_DOCS_DB", DEFAULT_DB_DIR))
