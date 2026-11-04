"""Text helpers for trimming long pages down to the parts that matter."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or that the this to "
    "was were what when where which who why will with".split()
)
DEFAULT_EXCERPT_CHARS = 1500


def keywords(text: str) -> set[str]:
    """Lower-case content words of ``text``, without stopwords and very short tokens."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def best_excerpt(text: str, query: str, limit: int = DEFAULT_EXCERPT_CHARS) -> str:
    """Return up to ``limit`` characters of ``text``, preferring paragraphs that match ``query``.

    Paragraphs are scored by how many query keywords they contain. The best ones
    are kept, in their original order, so the excerpt still reads naturally. If
    no paragraph matches, the start of the text is returned.
    """
    text = text.strip()
    if len(text) <= limit:
        return text

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if len(p.strip()) > 40]
    terms = keywords(query)
    scored = [(len(terms & keywords(p)), i, p) for i, p in enumerate(paragraphs)]
    if not terms or not any(score for score, _, _ in scored):
        return _truncate(text, limit)

    chosen: list[tuple[int, str]] = []
    used = 0
    for score, index, paragraph in sorted(scored, key=lambda item: (-item[0], item[1])):
        if score == 0:
            break
        if used + len(paragraph) > limit:
            if not chosen:  # the best paragraph alone is too long: keep its start
                chosen.append((index, _truncate(paragraph, limit)))
                used = limit
            continue
        chosen.append((index, paragraph))
        used += len(paragraph) + 1
    return "\n".join(p for _, p in sorted(chosen))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit * 0.8 else cut).rstrip() + " ..."
