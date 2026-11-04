"""Runs the research pipeline: plan, gather evidence, research, review, write, check.

Plain Python, no agent framework:

    Planner ─► [gather evidence for every sub-question in parallel]
            ─► Researcher per sub-question (one model call each, in order)
            ─► Reviewer, optional: finds gaps ─► one follow-up research round
                                   ─► finds conflicts between sources
            ─► Writer ─► citation check

Evidence gathering is network-bound, so all sub-questions are searched at once.
Model calls run one after another because a local Ollama on an 8 GB machine
serves one request at a time. Progress is reported through ``on_event`` so the
command line and the UI can show what each agent is doing.

All events are emitted on the thread that called ``run()``, never on a search
thread, so UI frameworks that only allow updates from their own thread (such as
Streamlit) can render them directly. Sources are numbered on that thread too, in
sub-question order, so numbering is the same from run to run.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agents import Findings, Plan, Planner, Researcher, Review, Reviewer, SubQuestion, Writer
from src.citations import CitationCheck, check_citations
from src.llm import LLM, LLMConnectionError, LLMError
from src.sources import Evidence, Source, SourceRegistry
from src.text import count
from src.tools import ResearchTool, ToolError

logger = logging.getLogger(__name__)

MAX_SOURCES_PER_QUESTION = 5
GATHER_WORKERS = 6

NO_FINDINGS_REPORT = (
    "The research agents could not find information that answers this question in the selected sources. "
    "Try rephrasing the question, increasing the depth, or enabling another source."
)


NO_NEW_SOURCES = "Found no sources beyond those already read."


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
REVIEWING = "reviewing"
REVIEW_DONE = "review_done"
CHECKING_CONFLICTS = "checking_conflicts"
CONFLICTS_DONE = "conflicts_done"
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
    review: Review | None = None  # None when the reviewer was off or failed

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
        review: bool = False,
    ) -> None:
        if not tools:
            raise ValueError("At least one research tool is required.")
        self.llm = llm
        self.tools = tools
        self.depth = depth
        self.max_sources = max_sources_per_question
        self.review = review
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

        all_findings = self._research_round(plan.sub_questions)
        review = None
        if self.review:
            follow_ups = self._find_gaps(question, all_findings)
            if follow_ups:
                all_findings += self._research_round(follow_ups, already_read={s.id for s in self.registry.all()})
            review = Review(follow_ups, self._find_conflicts(question, all_findings))

        if any(f.findings for f in all_findings):
            self._emit(WRITING, "Writing the report")
            try:
                useful = [f for f in all_findings if f.findings or not f.sub_question.follow_up]
                report = Writer(self.llm).run(question, useful, review.conflicts if review else [])
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
            review=review,
        )
        self._emit(DONE, f"Done in {result.seconds:.0f} s", result=result)
        return result

    # -------------------------------------------------------------- steps

    def _research_round(self, sub_questions: list[SubQuestion], already_read: set[int] = frozenset()) -> list[Findings]:
        """Search for every sub-question at once, then run the researchers one by one.

        A sub-question whose sources were all in ``already_read`` gets no model call:
        its researcher would only repeat findings from the earlier round.
        """
        researcher = Researcher(self.llm)
        with ThreadPoolExecutor(max_workers=GATHER_WORKERS) as pool:
            futures = {sq.number: pool.submit(self._gather, sq) for sq in sub_questions}
            for sq in sub_questions:
                self._emit(SEARCHING, f"Searching for sub-question {sq.number}", sub_question=sq)
            return [self._research(researcher, sq, futures[sq.number], already_read) for sq in sub_questions]

    def _find_gaps(self, question: str, all_findings: list[Findings]) -> list[SubQuestion]:
        """Ask the reviewer what is missing. A failed check is skipped, not fatal."""
        self._emit(REVIEWING, "Reviewing the findings for gaps")
        try:
            follow_ups = Reviewer(self.llm).find_gaps(question, all_findings)
        except LLMConnectionError:
            raise
        except LLMError as exc:
            self._warn(f"The reviewer failed to check for gaps ({exc}); continuing without follow-ups.")
            self._emit(REVIEW_DONE, "Gap check skipped", follow_ups=None)
            return []
        message = f"Review: {count(len(follow_ups), 'follow-up question')}" if follow_ups else "Review: no gaps found"
        self._emit(REVIEW_DONE, message, follow_ups=follow_ups)
        return follow_ups

    def _find_conflicts(self, question: str, all_findings: list[Findings]) -> list[str]:
        """Ask the reviewer which findings disagree. A failed check is skipped, not fatal."""
        self._emit(CHECKING_CONFLICTS, "Checking the findings for conflicts")
        try:
            conflicts = Reviewer(self.llm).find_conflicts(question, all_findings)
        except LLMConnectionError:
            raise
        except LLMError as exc:
            self._warn(f"The reviewer failed to check for conflicts ({exc}); continuing without.")
            self._emit(CONFLICTS_DONE, "Conflict check skipped", conflicts=None)
            return []
        message = f"Review: {count(len(conflicts), 'conflict')} between sources" if conflicts else "Review: no conflicts"
        self._emit(CONFLICTS_DONE, message, conflicts=conflicts)
        return conflicts

    def _gather(self, sq: SubQuestion) -> tuple[list[list[Evidence]], list[str]]:
        """Run every tool for one sub-question. Runs on a search thread, so it doesn't emit events."""
        per_tool: list[list[Evidence]] = []
        warnings: list[str] = []
        for tool in self.tools:
            try:
                per_tool.append(tool.gather(sq.text, sq.queries))
            except ToolError as exc:
                warnings.append(f"{tool.name} search failed for sub-question {sq.number}: {exc}")
            except Exception as exc:  # a broken tool must not stop the whole run
                logger.exception("Unexpected error in tool %s", tool.name)
                warnings.append(f"{tool.name} search failed for sub-question {sq.number}: {exc}")
        return per_tool, warnings

    def _register(self, per_tool: list[list[Evidence]]) -> list[Source]:
        """Number up to max_sources sources, taking results from each tool in turn."""
        unique: list[Source] = []
        for evidence in _interleave(per_tool):
            if len(unique) == self.max_sources:
                break
            source = self.registry.register(evidence)  # same URL from another sub-question keeps its number
            if source not in unique:
                unique.append(source)
        return unique

    def _research(self, researcher: Researcher, sq: SubQuestion, gathered: Future, already_read: set[int]) -> Findings:
        per_tool, warnings = gathered.result()  # _gather never raises
        for warning in warnings:
            self._warn(warning)
        sources = self._register(per_tool)
        self._emit(SOURCES_FOUND, f"Sub-question {sq.number}: {len(sources)} sources", sub_question=sq, sources=sources)
        try:
            if sources and all(s.id in already_read for s in sources):
                findings = Findings(sub_question=sq, sources=sources, gaps=NO_NEW_SOURCES)
            else:
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
            f"Sub-question {sq.number}: {count(len(findings.findings), 'finding')}",
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
