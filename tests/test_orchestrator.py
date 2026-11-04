"""Tests for the research pipeline, using a fake model and fake tools."""

import unittest

from src.llm import LLMConnectionError, LLMError
from src.orchestrator import (
    DONE,
    NO_FINDINGS_REPORT,
    PLAN_READY,
    RESEARCHER_DONE,
    SOURCES_FOUND,
    WARNING,
    WRITING,
    ResearchRun,
)
from src.tools import ToolError
from tests.fakes import FakeLLM, FakeTool, cite_all_researcher, evidence, plan_reply


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
        llm = FakeLLM(planner=plan_reply("First?", "Second?"), researcher=cite_all_researcher, writer="Claim [42].")
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

    def test_empty_question_and_no_tools_are_rejected(self):
        with self.assertRaises(ValueError):
            ResearchRun(FakeLLM(), [FakeTool()]).run("   ")
        with self.assertRaises(ValueError):
            ResearchRun(FakeLLM(), [])


if __name__ == "__main__":
    unittest.main()
