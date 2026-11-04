"""Formatting helpers for showing a research result in Markdown (UI and downloads)."""

from __future__ import annotations

import re

from src.sources import Source

_CITATION_GROUP = re.compile(r"(?<!\[)\[(\d+(?:\s*,\s*\d+)*)\](?!\()")
KIND_LABELS = {"web": "Web", "wikipedia": "Wikipedia", "local": "Local PDF"}


def link_citations(report: str, sources: list[Source]) -> str:
    """Turn [n] citations into links to the source's URL.

    Groups such as [1, 3] become separate links. Local documents have no URL,
    so their citations stay as plain [n]. Citations to unknown sources are left
    unchanged, so the citation checker's warning still matches the text.
    """
    by_id = {source.id: source for source in sources}

    def replace(match: re.Match[str]) -> str:
        parts = []
        for raw in match.group(1).split(","):
            source = by_id.get(int(raw))
            if source is not None and source.kind != "local":
                parts.append(f"[[{source.id}]]({_markdown_url(source.location)})")
            else:
                parts.append(f"[{raw.strip()}]")
        return "".join(parts)

    return _CITATION_GROUP.sub(replace, report)


def source_line(source: Source) -> str:
    """One Markdown line for the source list."""
    label = KIND_LABELS.get(source.kind, source.kind)
    if source.kind == "local":
        return f"**[{source.id}]** {source.location} · {label}"
    title = source.title.replace("[", "(").replace("]", ")")
    return f"**[{source.id}]** [{title}]({_markdown_url(source.location)}) · {label}"


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:limit].rstrip("-") or "research")


def _markdown_url(url: str) -> str:
    # Parentheses and spaces would end a Markdown link early.
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")
