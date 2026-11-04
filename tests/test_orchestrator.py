"""Tests for the research pipeline, using a fake model and fake tools."""

import unittest

from src.llm import LLMConnectionError, LLMError
from src.orchestrator import (
    CHECKING_CONFLICTS,
    CONFLICTS_DONE,
    DONE,
    NO_FINDINGS_REPORT,
    NO_NEW_SOURCES,
    PLAN_READY,
    RESEARCHER_DONE,
    REVIEW_DONE,
    REVIEWING,
    SEARCHING,
    SOURCES_FOUND,
    WARNING,
    WRITING,
    ResearchRun,
)
from src.tools import ToolError
from tests.fakes import FakeLLM, FakeTool, cite_all_researcher, conflicts_reply, evidence, plan_reply, review_reply


def run(llm, tools, depth=2, **kwargs):
    events = []
    result = ResearchRun(llm, tools, depth=depth, on_event=events.append, **kwargs).run("Main question?")
    return result, events


class PipelineTests(unittest.TestCase):
    def test_full_run_calls_each_agent_and_numbers_sources(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher,
                      writer="Summary that is long enough to be checked for citations [1][2].")
        result, events = run(llm, [FakeTool("web"), FakeTool("wiki")])

        self.assertEqual(llm.agents_called(), ["planner", "researcher", "researcher", "writer"])
        self.assertEqual(result.model_calls, 4)
        self.assertEqual([s.id for s in result.sources], [1, 2, 3, 4])
        self.assertEqual(result.citations.cited_ids, [1, 2])
        self.assertEqual(result.warnings, [])

        kinds = [e.kind for e in events]
        self.assertEqual(kinds[0], PLAN_READY)
        self.assertEqual(kinds.count(SOURCES_FOUND), 2)
        self.assertEqual(kinds.count(RESEARCHER_DONE), 2)
        self.assertEqual(kinds[-2:], [WRITING, DONE])

    def test_events_are_emitted_on_the_calling_thread_in_order(self):
        import threading

        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher, writer="Report [1].")
        result, events = run(llm, [FakeTool("web"), FakeTool("broken", ToolError("down"))])
        kinds = [e.kind for e in events]
        self.assertEqual(kinds[:3], [PLAN_READY, SEARCHING, SEARCHING])
        self.assertEqual(kinds[3:6], [WARNING, SOURCES_FOUND, RESEARCHER_DONE])  # sub-question 1, its warning first
        self.assertEqual([s.id for s in result.findings[0].sources], [1])  # numbered in sub-question order
        self.assertEqual([s.id for s in result.findings[1].sources], [2])

        seen = []
        handler = lambda event: seen.append(threading.current_thread())
        ResearchRun(FakeLLM(planner=plan_reply("A?", "B?"), researcher=cite_all_researcher, writer="R [1]."),
                    [FakeTool("web")], depth=2, on_event=handler).run("Q?")
        self.assertEqual(set(seen), {threading.current_thread()})

    def test_markdown_lists_only_cited_sources(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher,
                      writer="Summary that is long enough to be checked for citations [2].")
        markdown = run(llm, [FakeTool("web")])[0].to_markdown()
        self.assertTrue(markdown.startswith("# Main question?"))
        self.assertIn("## Sources", markdown)
        self.assertIn("[2] web page about Second?", markdown)
        self.assertNotIn("[1] web page", markdown)
        self.assertIn("1 further source(s) were read but not cited", markdown)

    def test_sources_are_interleaved_across_tools_and_capped(self):
        many = lambda name: (lambda sq: [evidence(f"{name}{i}", f"https://{name}.example/{sq}/{i}") for i in range(5)])
        llm = FakeLLM(planner=plan_reply("Only?", "Other?"), researcher=cite_all_researcher, writer="Report [1].")
        result, events = run(llm, [FakeTool("a", many("a")), FakeTool("b", many("b"))], max_sources_per_question=3)
        first = next(e for e in events if e.kind == SOURCES_FOUND)
        self.assertEqual([s.title for s in first.data["sources"]], ["a0", "b0", "a1"])
        self.assertEqual(len(result.sources), 6)  # sources beyond the cap are never registered

    def test_same_page_found_twice_gets_one_number(self):
        shared = lambda sq: [evidence("Shared page", "https://shared.example/page")]
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher, writer="Report [1].")
        result, _ = run(llm, [FakeTool("web", shared)])
        self.assertEqual(len(result.sources), 1)


class FailureTests(unittest.TestCase):
    def test_failing_tool_is_reported_and_others_still_used(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher, writer="Report [1].")
        result, events = run(llm, [FakeTool("broken", ToolError("rate limited")), FakeTool("web")])
        self.assertEqual(len(result.sources), 2)
        self.assertEqual(sum("broken search failed" in w for w in result.warnings), 2)
        self.assertIn(WARNING, [e.kind for e in events])

    def test_unexpected_tool_exception_does_not_stop_run(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher, writer="Report [1].")
        with self.assertLogs("src.orchestrator", level="ERROR"):
            result, _ = run(llm, [FakeTool("buggy", RuntimeError("boom")), FakeTool("web")])
        self.assertEqual(result.report, "Report [1].")

    def test_one_researcher_failing_still_produces_report(self):
        def researcher(system, user):
            if "Second?" in user:
                raise LLMError("garbled")
            return cite_all_researcher(system, user)

        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=researcher, writer="Report [1].")
        result, _ = run(llm, [FakeTool("web")])
        self.assertEqual(result.findings[1].error, "garbled")
        self.assertEqual(result.report, "Report [1].")
        self.assertTrue(any("sub-question 2 failed" in w for w in result.warnings))

    def test_writer_failure_falls_back_to_raw_findings(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher, writer=LLMError("bad"))
        result, _ = run(llm, [FakeTool("web")])
        self.assertIn("## First?", result.report)
        self.assertIn("A supported fact. [1]", result.report)

    def test_writer_timeout_still_returns_the_findings(self):
        from src.llm import LLMTimeoutError

        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher,
                      writer=LLMTimeoutError("slow"))
        result, _ = run(llm, [FakeTool("web")])
        self.assertIn("A supported fact. [1]", result.report)
        self.assertTrue(any("writer failed" in w for w in result.warnings))

    def test_no_findings_skips_writer(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher={"findings": [], "gaps": "Nothing"})
        result, _ = run(llm, [FakeTool("web")])
        self.assertEqual(result.report, NO_FINDINGS_REPORT)
        self.assertNotIn("writer", llm.agents_called())

    def test_ollama_down_stops_the_run(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=LLMConnectionError("Ollama is down"))
        with self.assertRaises(LLMConnectionError):
            run(llm, [FakeTool("web")])

    def test_invented_citation_in_report_is_warned(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher, writer="Claim [1][42].")
        result, _ = run(llm, [FakeTool("web")])
        self.assertEqual(result.citations.unknown_ids, [42])
        self.assertTrue(any("don't exist" in w for w in result.warnings))

    def test_broken_event_handler_does_not_stop_run(self):
        def handler(event):
            raise RuntimeError("UI crashed")

        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher, writer="Report [1].")
        with self.assertLogs("src.orchestrator", level="ERROR"):
            result = ResearchRun(llm, [FakeTool("web")], depth=2, on_event=handler).run("Q?")
        self.assertEqual(result.report, "Report [1].")

    def test_unclear_question_stops_before_any_search(self):
        tool = FakeTool("web")
        llm = FakeLLM(planner={"is_research_question": False, "reason": "gibberish", "sub_questions": []})
        with self.assertRaises(ValueError):
            run(llm, [tool])
        self.assertEqual(tool.calls, [])

    def test_reviewer_off_by_default(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher, writer="Report [1].")
        result, _ = run(llm, [FakeTool("web")])
        self.assertNotIn("reviewer", llm.agents_called())
        self.assertIsNone(result.review)


class ReviewTests(unittest.TestCase):
    def test_follow_ups_get_an_extra_research_round_then_conflicts_are_checked(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher,
                      reviewer=review_reply("Third?"), conflicts=conflicts_reply((1, 3)), writer="Report [1][3].")
        tool = FakeTool("web")
        result, events = run(llm, [tool], review=True)

        self.assertEqual(llm.agents_called(),
                         ["planner", "researcher", "researcher", "reviewer", "researcher", "conflicts", "writer"])
        self.assertEqual([c[0] for c in tool.calls], ["First?", "Second?", "Third?"])
        self.assertEqual([f.sub_question.number for f in result.findings], [1, 2, 3])
        self.assertEqual([s.id for s in result.findings[2].sources], [3])
        self.assertIn("F3. A supported fact. [3]", llm.calls[-2][2])  # follow-up findings are compared too
        writer_prompt = llm.calls[-1][2]
        self.assertIn('Conflicts between sources:\n- "A supported fact." [1] versus "A supported fact." [3]', writer_prompt)
        self.assertIn("Follow-up 3: Third?", writer_prompt)
        self.assertEqual([sq.text for sq in result.review.follow_ups], ["Third?"])
        self.assertEqual(len(result.review.conflicts), 1)

        kinds = [e.kind for e in events]
        review_at = kinds.index(REVIEWING)
        self.assertEqual(kinds[review_at:], [REVIEWING, REVIEW_DONE, SEARCHING, SOURCES_FOUND, RESEARCHER_DONE,
                                             CHECKING_CONFLICTS, CONFLICTS_DONE, WRITING, DONE])

    def test_complete_review_goes_straight_to_the_conflict_check(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher,
                      reviewer=review_reply(), writer="Report [1].")
        result, _ = run(llm, [FakeTool("web")], review=True)
        self.assertEqual(llm.agents_called(), ["planner", "researcher", "researcher", "reviewer", "conflicts", "writer"])
        self.assertEqual(result.review.follow_ups, [])
        self.assertEqual(result.review.conflicts, [])
        self.assertNotIn("Conflicts", llm.calls[-1][2])

    def test_failed_checks_are_skipped_with_warnings(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher,
                      reviewer=LLMError("garbled"), conflicts=LLMError("garbled"), writer="Report [1].")
        result, events = run(llm, [FakeTool("web")], review=True)
        self.assertEqual(result.report, "Report [1].")
        self.assertEqual(sum("reviewer failed" in w for w in result.warnings), 2)
        self.assertIsNone(next(e for e in events if e.kind == REVIEW_DONE).data["follow_ups"])
        self.assertIsNone(next(e for e in events if e.kind == CONFLICTS_DONE).data["conflicts"])

    def test_ollama_down_during_review_stops_the_run(self):
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher,
                      reviewer=LLMConnectionError("down"))
        with self.assertRaises(LLMConnectionError):
            run(llm, [FakeTool("web")], review=True)

    def test_follow_up_with_no_new_sources_skips_its_researcher(self):
        same_page = FakeTool("web", lambda sq: [evidence("Same page", "https://same.example")])
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher,
                      reviewer=review_reply("Third?"), writer="Report [1].")
        result, _ = run(llm, [same_page], review=True)
        self.assertEqual(llm.agents_called(), ["planner", "researcher", "researcher", "reviewer", "conflicts", "writer"])
        self.assertEqual(result.findings[2].gaps, NO_NEW_SOURCES)
        self.assertNotIn("Third?", llm.calls[-1][2])  # an empty follow-up isn't given to the writer

    def test_follow_ups_can_rescue_a_run_with_no_findings(self):
        def researcher(system, user):
            return cite_all_researcher(system, user) if "Better?" in user else {"findings": [], "gaps": "Nothing"}

        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=researcher,
                      reviewer=review_reply("Better?"), writer="Report [3].")
        result, _ = run(llm, [FakeTool("web")], review=True)
        self.assertEqual(result.report, "Report [3].")


class ValidationTests(unittest.TestCase):
    def test_empty_question_and_no_tools_are_rejected(self):
        with self.assertRaises(ValueError):
            ResearchRun(FakeLLM(), [FakeTool()]).run("   ")
        with self.assertRaises(ValueError):
            ResearchRun(FakeLLM(), [])


if __name__ == "__main__":
    unittest.main()
