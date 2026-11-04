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
