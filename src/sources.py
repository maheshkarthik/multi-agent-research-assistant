"""Evidence gathered by tools, and the registry that numbers it for citation."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["web", "wikipedia", "local"]


@dataclass(frozen=True)
class Evidence:
    """One excerpt returned by a tool, before it has a citation number."""

    kind: SourceKind
    title: str
    location: str  # URL for web sources, "file.pdf, page 3" for local ones
    text: str


@dataclass(frozen=True)
class Source:
    """A numbered source that agents can cite as [id]."""

    id: int
    kind: SourceKind
    title: str
    location: str
    text: str

    @property
    def citation_label(self) -> str:
        if self.kind == "local":
            return f"{self.location} (local document)"
        return f"{self.title}. {self.location}"


class SourceRegistry:
    """Gives every distinct source a stable number, shared by all agents in a run.

    The same URL or file page found by two researchers gets one number, so the
    final source list has no duplicates. Safe to use from several threads.
    """

    def __init__(self) -> None:
        self._by_id: dict[int, Source] = {}
        self._by_location: dict[str, int] = {}
        self._lock = threading.Lock()

    def register(self, evidence: Evidence) -> Source:
        key = _normalize_location(evidence.location)
        with self._lock:
            if key in self._by_location:
                return self._by_id[self._by_location[key]]
            source = Source(
                id=len(self._by_id) + 1,
                kind=evidence.kind,
                title=evidence.title,
                location=evidence.location,
                text=evidence.text,
            )
            self._by_id[source.id] = source
            self._by_location[key] = source.id
            return source

    def get(self, source_id: int) -> Source | None:
        return self._by_id.get(source_id)

    def all(self) -> list[Source]:
        return [self._by_id[i] for i in sorted(self._by_id)]

    def __len__(self) -> int:
        return len(self._by_id)


def _normalize_location(location: str) -> str:
    key = location.strip().lower()
    key = re.sub(r"^https?://(www\.)?", "", key)
    key = key.split("#", 1)[0]
    return key.rstrip("/")
