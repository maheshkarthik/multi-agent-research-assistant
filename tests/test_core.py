"""Tests for the model-output helpers, text excerpts, source registry and citation checks."""

import unittest

from src.citations import check_citations, cited_ids
from src.llm import LLMError, parse_json_object, strip_reasoning
from src.sources import SourceRegistry
from src.text import best_excerpt, keywords
from tests.fakes import evidence


class StripReasoningTests(unittest.TestCase):
    def test_removes_full_think_block(self):
        self.assertEqual(strip_reasoning("<think>hmm</think>\nAnswer"), "Answer")

    def test_removes_text_before_lone_closing_tag(self):
        # qwen3 sometimes writes its reasoning as plain text ending in </think>, even with thinking off
        self.assertEqual(strip_reasoning("Let me plan this.\n</think>\n\nThe report."), "The report.")

    def test_leaves_normal_text_alone(self):
        self.assertEqual(strip_reasoning("  Plain answer.  "), "Plain answer.")


class ParseJsonTests(unittest.TestCase):
    def test_parses_object_inside_code_fence(self):
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_rejects_non_json(self):
        with self.assertRaises(LLMError):
            parse_json_object("no json here")

    def test_rejects_malformed_json(self):
        with self.assertRaises(LLMError):
            parse_json_object('{"a": }')


class OllamaErrorTests(unittest.TestCase):
    def call_with(self, error):
        from unittest import mock

        from src.llm import OllamaLLM

        with mock.patch("urllib.request.urlopen", side_effect=error):
            OllamaLLM(timeout=1).generate("system", "user")

    def test_timeout_is_not_a_connection_error(self):
        from src.llm import LLMConnectionError, LLMTimeoutError

        with self.assertRaises(LLMTimeoutError) as caught:
            self.call_with(TimeoutError("timed out"))
        self.assertNotIsInstance(caught.exception, LLMConnectionError)

    def test_every_call_uses_the_same_context_size_and_caps_output(self):
        from unittest import mock

        from src.llm import NUM_CTX, OllamaLLM

        llm = OllamaLLM()
        reply = {"message": {"content": "{}"}, "done_reason": "stop"}
        with mock.patch.object(llm, "_post", return_value=reply) as post:
            llm.generate("s", "u", max_tokens=2000)
            llm.generate_json("s", "u", schema={"type": "object"})
        options = [call.args[1]["options"] for call in post.call_args_list]
        self.assertEqual([o["num_ctx"] for o in options], [NUM_CTX, NUM_CTX])  # a change would reload the model
        self.assertEqual([o["num_predict"] for o in options], [2000, 1024])

    def test_refused_connection_is_a_connection_error(self):
        import urllib.error

        from src.llm import LLMConnectionError

        with self.assertRaises(LLMConnectionError):
            self.call_with(urllib.error.URLError(ConnectionRefusedError()))


class ExcerptTests(unittest.TestCase):
    def test_short_text_is_returned_unchanged(self):
        self.assertEqual(best_excerpt("Short text.", "anything"), "Short text.")

    def test_prefers_matching_paragraphs_in_original_order(self):
        filler = "Unrelated filler sentence about gardening and weather patterns in spring. " * 3
        text = "\n".join([filler, "The printing press spread science quickly across Europe.", filler,
                          "Science journals relied on the printing press for distribution.", filler])
        excerpt = best_excerpt(text, "printing press science", limit=200)
        self.assertIn("spread science quickly", excerpt)
        self.assertIn("journals", excerpt)
        self.assertNotIn("gardening", excerpt)
        self.assertLess(excerpt.index("spread"), excerpt.index("journals"))

    def test_never_exceeds_limit_by_much(self):
        text = ("printing press " * 400 + "\n") * 3
        self.assertLessEqual(len(best_excerpt(text, "printing press", limit=500)), 504)

    def test_falls_back_to_start_when_nothing_matches(self):
        text = "Alpha beta gamma delta epsilon zeta eta theta. " * 100
        self.assertTrue(best_excerpt(text, "printing press", limit=100).startswith("Alpha beta"))

    def test_keywords_drop_stopwords(self):
        self.assertEqual(keywords("How did the printing press change science?"), {"printing", "press", "change", "science"})


class SourceRegistryTests(unittest.TestCase):
    def test_numbers_sources_from_one(self):
        registry = SourceRegistry()
        first = registry.register(evidence("A", "https://a.example"))
        second = registry.register(evidence("B", "https://b.example"))
        self.assertEqual((first.id, second.id), (1, 2))
        self.assertEqual(len(registry), 2)

    def test_same_page_keeps_its_number(self):
        registry = SourceRegistry()
        first = registry.register(evidence("A", "https://www.a.example/page/"))
        again = registry.register(evidence("A again", "http://a.example/page#section"))
        self.assertEqual(first.id, again.id)
        self.assertEqual(len(registry), 1)


class CitationTests(unittest.TestCase):
    def test_extracts_all_citation_styles(self):
        self.assertEqual(cited_ids("Fact [2]. Another [1, 3]. More [4][2]."), [2, 1, 3, 4])

    def test_flags_unknown_ids(self):
        check = check_citations("A long enough sentence that states a fact about something important [1][9].", {1, 2})
        self.assertEqual(check.cited_ids, [1])
        self.assertEqual(check.unknown_ids, [9])
        self.assertFalse(check.ok)

    def test_flags_long_uncited_paragraphs_but_not_headings(self):
        report = "## A heading without citations\n\nThis paragraph makes a claim without any citation at all, sadly.\n\nCited claim that is long enough to be checked by the citation checker [1]."
        check = check_citations(report, {1})
        self.assertEqual(len(check.uncited_paragraphs), 1)
        self.assertIn("without any citation", check.uncited_paragraphs[0])


if __name__ == "__main__":
    unittest.main()


class ReportFormattingTests(unittest.TestCase):
    def setUp(self):
        from src.sources import Source

        self.sources = [
            Source(1, "web", "Page (2024)", "https://a.example/x_(y)", "t"),
            Source(2, "local", "guide.pdf, page 3", "guide.pdf, page 3", "t"),
        ]

    def test_web_citations_become_links_and_local_stay_plain(self):
        from src.report import link_citations

        linked = link_citations("Fact [1]. Local [2]. Both [1, 2]. Unknown [9].", self.sources)
        self.assertEqual(
            linked,
            "Fact [[1]](https://a.example/x_%28y%29). Local [2]. Both [[1]](https://a.example/x_%28y%29)[2]. Unknown [9].",
        )

    def test_already_linked_citations_are_untouched(self):
        from src.report import link_citations

        text = "See [[1]](https://a.example)."
        self.assertEqual(link_citations(text, self.sources), text)

    def test_source_lines_and_slug(self):
        from src.report import slugify, source_line

        self.assertEqual(source_line(self.sources[1]), "**[2]** guide.pdf, page 3 · Local PDF")
        self.assertIn("[Page (2024)](https://a.example/x_%28y%29) · Web", source_line(self.sources[0]))
        self.assertEqual(slugify("How did the printing press change science?"), "how-did-the-printing-press-change-science")
        self.assertEqual(slugify("???"), "research")

    def test_headings_are_demoted(self):
        from src.report import demote_headings

        self.assertEqual(demote_headings("## Causes\ntext ## not a heading\n##### Deep"), "#### Causes\ntext ## not a heading\n###### Deep")
