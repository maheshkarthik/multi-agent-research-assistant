"""Runs the research pipeline: plan, gather evidence, research, write, check.

Plain Python, no agent framework:

    Planner ─► [gather evidence for every sub-question in parallel]
            ─► Researcher per sub-question (one model call each, in order)
            ─► Writer ─► citation check

Evidence gathering is network-bound, so all sub-questions are searched at once.
Model calls run one after another because a local Ollama on an 8 GB machine
serves one request at a time. Progress is reported through ``on_event`` so the
command line and the UI can show what each agent is doing.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agents import Findings, Plan, Planner, Researcher, SubQuestion, Writer
from src.citations import CitationCheck, check_citations
from src.llm import LLM, LLMConnectionError, LLMError
from src.sources import Evidence, Source, SourceRegistry
from src.tools import ResearchTool, ToolError

logger = logging.getLogger(__name__)

MAX_SOURCES_PER_QUESTION = 5
GATHER_WORKERS = 6

NO_FINDINGS_REPORT = (
    "The research agents could not find information that answers this question in the selected sources. "
    "Try rephrasing the question, increasing the depth, or enabling another source."
)


@dataclass(frozen=True)
class Event:
    """A progress update. ``kind`` is one of the constants below."""

    kind: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


PLAN_READY = "plan_ready"
SEARCHING = "searching"
SOURCES_FOUND = "sources_found"
RESEARCHER_DONE = "researcher_done"
WRITING = "writing"
DONE = "done"
WARNING = "warning"

EventHandler = Callable[[Event], None]


@dataclass
class ResearchResult:
    question: str
    plan: Plan
    findings: list[Findings]
    report: str  # the writer's Markdown, without the source list
    sources: list[Source]  # every source gathered, numbered
    citations: CitationCheck
    seconds: float
    model_calls: int
    warnings: list[str] = field(default_factory=list)

    @property
    def cited_sources(self) -> list[Source]:
        by_id = {source.id: source for source in self.sources}
        return [by_id[i] for i in self.citations.cited_ids if i in by_id]

    def to_markdown(self) -> str:
        """The full report with a numbered source list, ready to save."""
        lines = [f"# {self.question}", "", self.report.strip(), "", "## Sources", ""]
        cited = sorted(self.cited_sources, key=lambda s: s.id)
        lines += [f"[{s.id}] {s.citation_label}  " for s in cited] or ["No sources were cited."]
        uncited = len(self.sources) - len(cited)
        if uncited:
            lines += ["", f"_{uncited} further source(s) were read but not cited._"]
        lines += ["", f"_Generated locally in {self.seconds:.0f} s with {self.model_calls} model calls. "
                  "AI-generated: check important facts against the sources._"]
        return "\n".join(lines)


class ResearchRun:
    """One research question, end to end."""

    def __init__(
        self,
        llm: LLM,
        tools: list[ResearchTool],
        depth: int = 3,
        on_event: EventHandler | None = None,
        max_sources_per_question: int = MAX_SOURCES_PER_QUESTION,
    ) -> None:
        if not tools:
            raise ValueError("At least one research tool is required.")
        self.llm = llm
        self.tools = tools
        self.depth = depth
        self.max_sources = max_sources_per_question
        self._on_event = on_event or (lambda event: None)
        self.registry = SourceRegistry()
        self.warnings: list[str] = []

    def run(self, question: str) -> ResearchResult:
        """Research ``question`` and return the report.

        Raises:
            LLMConnectionError: If Ollama is unreachable (nothing useful can be done without it).
            ValueError: If the question is empty.
        """
        question = question.strip()
        if not question:
            raise ValueError("Please enter a research question.")
        started = time.monotonic()
        calls_before = _model_calls(self.llm)

        plan = Planner(self.llm).run(question, self.depth)
        if plan.fallback:
            self._warn("The planner could not split the question, so it was researched as a whole.")
        self._emit(PLAN_READY, f"Plan ready: {len(plan.sub_questions)} sub-questions", plan=plan)

        all_findings: list[Findings] = []
        researcher = Researcher(self.llm)
        with ThreadPoolExecutor(max_workers=GATHER_WORKERS) as pool:
            futures = {sq.number: pool.submit(self._gather, sq) for sq in plan.sub_questions}
            for sq in plan.sub_questions:
                all_findings.append(self._research(researcher, sq, futures[sq.number]))

        if any(f.findings for f in all_findings):
            self._emit(WRITING, "Writing the report")
            try:
                report = Writer(self.llm).run(question, all_findings)
            except LLMConnectionError:
                raise
            except LLMError as exc:
                self._warn(f"The writer failed ({exc}); showing the raw findings instead.")
                report = _findings_as_report(all_findings)
        else:
            report = NO_FINDINGS_REPORT

        sources = self.registry.all()
        citations = check_citations(report, {s.id for s in sources})
        if citations.unknown_ids:
            self._warn(f"The report cites sources that don't exist: {citations.unknown_ids}")

        result = ResearchResult(
            question=question,
            plan=plan,
            findings=all_findings,
            report=report,
            sources=sources,
            citations=citations,
            seconds=time.monotonic() - started,
            model_calls=_model_calls(self.llm) - calls_before,
            warnings=list(self.warnings),
        )
        self._emit(DONE, f"Done in {result.seconds:.0f} s", result=result)
        return result

    # -------------------------------------------------------------- steps

    def _gather(self, sq: SubQuestion) -> list[Source]:
        """Run every tool for one sub-question and register up to max_sources sources."""
        self._emit(SEARCHING, f"Searching for sub-question {sq.number}", sub_question=sq)
        per_tool: list[list[Evidence]] = []
        for tool in self.tools:
            try:
                per_tool.append(tool.gather(sq.text, sq.queries))
            except ToolError as exc:
                self._warn(f"{tool.name} search failed for sub-question {sq.number}: {exc}")
            except Exception as exc:  # a broken tool must not stop the whole run
                logger.exception("Unexpected error in tool %s", tool.name)
                self._warn(f"{tool.name} search failed for sub-question {sq.number}: {exc}")

        unique: list[Source] = []
        for evidence in _interleave(per_tool):
            if len(unique) == self.max_sources:
                break
            source = self.registry.register(evidence)  # same URL from another sub-question keeps its number
            if source not in unique:
                unique.append(source)
        self._emit(SOURCES_FOUND, f"Sub-question {sq.number}: {len(unique)} sources", sub_question=sq, sources=unique)
        return unique

    def _research(self, researcher: Researcher, sq: SubQuestion, gathered: Future) -> Findings:
        sources = gathered.result()  # _gather never raises
        try:
            findings = researcher.run(sq, sources)
        except LLMConnectionError:
            raise
        except LLMError as exc:
            self._warn(f"The researcher for sub-question {sq.number} failed: {exc}")
            findings = Findings(sub_question=sq, sources=sources, error=str(exc))
        if findings.dropped_source_ids:
            self._warn(
                f"Sub-question {sq.number}: ignored citations to unknown sources {findings.dropped_source_ids}"
            )
        self._emit(
            RESEARCHER_DONE,
            f"Sub-question {sq.number}: {len(findings.findings)} findings",
            sub_question=sq,
            findings=findings,
        )
        return findings

    def _emit(self, kind: str, message: str, **data: Any) -> None:
        try:
            self._on_event(Event(kind, message, data))
        except Exception:  # a broken progress display must not stop the research
            logger.exception("Event handler failed")

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        self._emit(WARNING, message)


def _interleave(lists: list[list[Evidence]]) -> list[Evidence]:
    """Take results from each tool in turn, so every enabled source is represented."""
    result = []
    for i in range(max((len(items) for items in lists), default=0)):
        result += [items[i] for items in lists if i < len(items)]
    return result


def _findings_as_report(all_findings: list[Findings]) -> str:
    parts = []
    for item in all_findings:
        parts.append(f"## {item.sub_question.text}")
        parts += [f"- {f.claim} " + "".join(f"[{i}]" for i in f.source_ids) for f in item.findings] or ["- No findings."]
    return "\n\n".join(parts)


def _model_calls(llm: LLM) -> int:
    stats = getattr(llm, "stats", None)
    return getattr(stats, "calls", 0)
