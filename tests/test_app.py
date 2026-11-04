"""UI tests for app.py with Streamlit's AppTest, using a fake model, fake tools and a fake index."""

import unittest
from pathlib import Path
from unittest import mock

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.llm import LLMConnectionError
from tests.fakes import FakeLLM, FakeTool, cite_all_researcher, evidence, plan_reply

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class AppLLM(FakeLLM):
    """FakeLLM plus what the app's status check needs."""

    model = "qwen3:4b"
    available = True

    def list_models(self):
        if not self.available:
            raise LLMConnectionError("Could not connect to Ollama at http://localhost:11434.")
        return [self.model]


class FakeStore:
    def __init__(self, documents=()):
        self.documents = [mock.Mock(source=name) for name in documents]

    def list_documents(self):
        return self.documents

    def count(self):
        return 10 * len(self.documents)


def web_tool():
    return FakeTool("web", lambda sq: [evidence(f"Page about {sq}", f"https://example.com/{sq.strip('?').lower()}")])


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.llm = AppLLM(
            planner=plan_reply("Why?", "How?"),
            researcher=cite_all_researcher,
            writer="Printing spread knowledge quickly across Europe and changed how scholars worked [1].\n\n"
                   "## Conclusion\n\nIt mattered a great deal for science and learning [2].",
        )
        self.tools = [web_tool()]
        patches = [
            mock.patch("src.llm.OllamaLLM", side_effect=lambda *a, **k: self.llm),
            mock.patch("src.tools.make_tools", side_effect=lambda sources, store=None: (self.tools, [])),
            mock.patch("src.local_docs.vector_store.VectorStore", return_value=FakeStore(["guide.pdf"])),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def app(self):
        return AppTest.from_file(APP, default_timeout=30).run()

    def research(self, at, question="How did printing change science?"):
        at.text_input(key="question").set_value(question)
        next(b for b in at.button if b.label == "Research").click()
        return at.run()

    def test_initial_page(self):
        at = self.app()
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "Research assistant")
        self.assertIn("qwen3:4b", at.sidebar.success[0].value)
        self.assertIn("1 documents", at.sidebar.caption[0].value)

    def test_research_shows_linked_report_sources_and_download(self):
        at = self.research(self.app())
        self.assertFalse(at.exception)
        self.assertEqual(self.llm.agents_called(), ["planner", "researcher", "researcher", "writer"])

        report = next(m.value for m in at.markdown if "Printing spread knowledge" in m.value)
        self.assertIn("[[1]](https://example.com/", report)  # citations link to their source
        self.assertEqual([m.value for m in at.metric][1:], ["4", "2", "2"])  # calls, read, cited
        self.assertTrue(any("Page about Why?" in m.value for m in at.markdown))
        self.assertEqual(len(at.get("download_button")), 1)
        self.assertEqual(at.status[0].label[:17], "Research complete")

    def test_empty_question_is_rejected(self):
        at = self.research(self.app(), question="   ")
        self.assertIn("Please enter a research question", at.warning[0].value)
        self.assertEqual(self.llm.calls, [])

    def test_ollama_down(self):
        AppLLM.available = False
        self.addCleanup(setattr, AppLLM, "available", True)
        at = self.research(self.app())
        self.assertIn("Could not connect to Ollama", at.sidebar.error[0].value)
        self.assertTrue(any("Could not connect" in e.value for e in at.error))
        self.assertEqual(self.llm.calls, [])

    def test_unclear_question_shows_a_warning(self):
        self.llm.replies["planner"] = {"is_research_question": False, "reason": "placeholder text", "sub_questions": []}
        at = self.research(self.app(), question="Your question here")
        self.assertFalse(at.exception)
        self.assertTrue(any("placeholder text" in w.value for w in at.warning))
        self.assertEqual(at.status[0].label, "Research stopped")

    def test_failing_researcher_is_marked_in_progress(self):
        def researcher(system, user):
            if "How?" in user:
                from src.llm import LLMError
                raise LLMError("garbled JSON")
            return cite_all_researcher(system, user)

        self.llm.replies["researcher"] = researcher
        at = self.research(self.app())
        self.assertFalse(at.exception)
        labels = [s.label for s in at.status]
        self.assertIn("Researcher 2: failed", labels)

    def test_example_fills_the_question(self):
        at = self.app()
        at.pills[0].set_value("What are the health effects of intermittent fasting?").run()
        self.assertEqual(at.text_input(key="question").value, "What are the health effects of intermittent fasting?")


if __name__ == "__main__":
    unittest.main()
