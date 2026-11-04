"""Research tools. Each tool turns a sub-question and search queries into evidence."""

from __future__ import annotations

from typing import Protocol

from src.sources import Evidence

USER_AGENT = "ai-engineering-lab-research-assistant/0.1 (+https://github.com/maheshkarthik)"


class ToolError(Exception):
    """A tool could not complete its search."""


class ResearchTool(Protocol):
    """Anything that can gather evidence for a sub-question."""

    name: str

    def gather(self, sub_question: str, queries: list[str]) -> list[Evidence]:
        """Return evidence relevant to ``sub_question``, best first."""
        ...


SOURCE_CHOICES = ("both", "web", "local")


def make_tools(sources: str = "both", store: object | None = None) -> tuple[list[ResearchTool], list[str]]:
    """Build the tools for a source setting: "web", "local" or "both".

    Returns the tools and any setup warnings, such as an empty local index.
    Imports are lazy so web-only runs never load ChromaDB. Pass ``store`` to
    reuse an open VectorStore instead of opening the default index.
    """
    if sources not in SOURCE_CHOICES:
        raise ValueError(f"sources must be one of {', '.join(SOURCE_CHOICES)}")
    tools: list[ResearchTool] = []
    warnings: list[str] = []
    if sources in ("web", "both"):
        from src.tools.web import WebSearchTool
        from src.tools.wikipedia import WikipediaTool

        tools += [WebSearchTool(), WikipediaTool()]
    if sources in ("local", "both"):
        from src.local_docs import db_dir
        from src.local_docs.vector_store import VectorStore
        from src.tools.local import LocalDocsTool

        if store is None:
            store = VectorStore(persist_directory=db_dir())
        if store.count() == 0:
            warnings.append("No local documents are indexed yet. Run: python -m src.index (indexes data/samples)")
            if sources == "local":
                raise ValueError(warnings[0])
        else:
            tools.append(LocalDocsTool(store))
    return tools, warnings
