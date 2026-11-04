"""Checks the writer's citations against the sources that were actually gathered."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches [3], [1, 4] and [1][4] (the last as two matches).
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
MIN_PARAGRAPH_CHARS = 60  # shorter lines (headings, labels) don't need citations


@dataclass
class CitationCheck:
    cited_ids: list[int] = field(default_factory=list)  # valid ids, in order of first use
    unknown_ids: list[int] = field(default_factory=list)  # cited but never gathered
    uncited_paragraphs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unknown_ids and not self.uncited_paragraphs


def cited_ids(text: str) -> list[int]:
    """All citation numbers in ``text``, in order of first appearance, without duplicates."""
    ids: dict[int, None] = {}
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            ids[int(part)] = None
    return list(ids)


def check_citations(report: str, valid_ids: set[int]) -> CitationCheck:
    """Find citations to unknown sources and substantial paragraphs with no citation.

    Headings and the summary-style conclusion are still expected to cite, except
    for short lines, which are ignored.
    """
    result = CitationCheck()
    for source_id in cited_ids(report):
        (result.cited_ids if source_id in valid_ids else result.unknown_ids).append(source_id)

    for block in re.split(r"\n\s*\n", report):
        block = block.strip()
        if not block or block.startswith("#") or len(block) < MIN_PARAGRAPH_CHARS:
            continue
        if not _CITATION.search(block):
            result.uncited_paragraphs.append(block)
    return result
