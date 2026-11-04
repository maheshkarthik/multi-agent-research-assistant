"""Test doubles for the model and research tools."""

from __future__ import annotations

from typing import Any, Callable

from src.llm import LLMError, LLMStats
from src.sources import Evidence
from src.tools import ToolError

Reply = dict[str, Any] | str | Exception | Callable[[str, str], dict[str, Any] | str]


AGENT_MARKERS = {
    "planner": "planning agent",
    "conflicts": "check numbered findings for contradictions",
    "reviewer": "reviewing agent",
    "researcher": "research agent",
}
NO_CONFLICTS = {"topics": [], "contradictions": []}


class FakeLLM:
    """Returns scripted replies, chosen by which agent is calling.

    The agent is recognised from its system prompt, so tests don't depend on the
    order of calls. Each reply may be a dict (JSON reply), a string, an exception
    to raise, or a function of (system, user).
    """

    def __init__(
        self,
        planner: Reply | None = None,
        researcher: Reply | None = None,
        writer: Reply | None = None,
        reviewer: Reply | None = None,
        conflicts: Reply | None = NO_CONFLICTS,
    ) -> None:
        self.replies = {
            "planner": planner, "researcher": researcher, "reviewer": reviewer, "conflicts": conflicts, "writer": writer,
        }
        self.calls: list[tuple[str, str, str]] = []  # (agent, system, user)
        self.stats = LLMStats()

    def _reply(self, system: str, user: str) -> dict[str, Any] | str:
        agent = next((name for name, marker in AGENT_MARKERS.items() if marker in system), "writer")
        self.calls.append((agent, system, user))
        self.stats.record(0.0, 0, 0)
        reply = self.replies[agent]
        if callable(reply) and not isinstance(reply, Exception):
            reply = reply(system, user)
        if isinstance(reply, Exception):
            raise reply
        if reply is None:
            raise AssertionError(f"FakeLLM has no reply for the {agent}")
        return reply

    def generate(self, system: str, user: str, *, schema=None, num_ctx: int = 4096) -> str:
        reply = self._reply(system, user)
        if isinstance(reply, dict):
            raise AssertionError("generate() called where a text reply was expected")
        return reply

    def generate_json(self, system: str, user: str, *, schema, num_ctx: int = 4096) -> dict[str, Any]:
        reply = self._reply(system, user)
        if not isinstance(reply, dict):
            raise LLMError("not JSON")
        return reply

    def agents_called(self) -> list[str]:
        return [agent for agent, _, _ in self.calls]


class FakeTool:
    """Returns evidence from a function of the sub-question, or raises."""

    def __init__(self, name: str = "fake", results: Callable[[str], list[Evidence]] | Exception | None = None) -> None:
        self.name = name
        self._results = results
        self.calls: list[tuple[str, list[str]]] = []

    def gather(self, sub_question: str, queries: list[str]) -> list[Evidence]:
        self.calls.append((sub_question, queries))
        if isinstance(self._results, Exception):
            raise self._results
        if self._results is None:
            return [evidence(f"{self.name} page about {sub_question}", f"https://{self.name}.example/{abs(hash(sub_question))}")]
        return self._results(sub_question)


def evidence(title: str, location: str, text: str = "Some relevant text.", kind: str = "web") -> Evidence:
    return Evidence(kind=kind, title=title, location=location, text=text)  # type: ignore[arg-type]


def plan_reply(*questions: str) -> dict[str, Any]:
    return {"sub_questions": [{"question": q, "search_queries": [q.lower()]} for q in questions]}


def review_reply(*follow_ups: str) -> dict[str, Any]:
    return {"complete": not follow_ups, "follow_ups": [{"question": q, "search_queries": [q.lower()]} for q in follow_ups]}


def conflicts_reply(*pairs: tuple[int, int]) -> dict[str, Any]:
    return {"topics": [], "contradictions": [{"finding_a": a, "finding_b": b, "difference": "They differ."} for a, b in pairs]}


def cite_all_researcher(system: str, user: str) -> dict[str, Any]:
    """A researcher reply that makes one finding citing every source it was shown."""
    import re

    ids = [int(i) for i in re.findall(r'<source id="(\d+)"', user)]
    return {"findings": [{"claim": "A supported fact.", "source_ids": ids}], "gaps": ""}


__all__ = ["FakeLLM", "FakeTool", "ToolError", "evidence", "plan_reply", "review_reply", "conflicts_reply", "cite_all_researcher"]
