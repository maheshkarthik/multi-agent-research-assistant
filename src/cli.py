"""Command-line research: python -m src.cli "your question" [--depth 3] [--output report.md]"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.llm import LLMConnectionError, LLMError, OllamaLLM
from src.orchestrator import PLAN_READY, RESEARCHER_DONE, SOURCES_FOUND, WARNING, WRITING, Event, ResearchRun
from src.tools.web import WebSearchTool
from src.tools.wikipedia import WikipediaTool


def print_event(event: Event) -> None:
    if event.kind == PLAN_READY:
        print(f"\n{event.message}")
        for sq in event.data["plan"].sub_questions:
            print(f"  {sq.number}. {sq.text}")
            print(f"     queries: {'; '.join(sq.queries)}")
        print()
    elif event.kind in (SOURCES_FOUND, RESEARCHER_DONE, WRITING):
        print(f"- {event.message}")
    elif event.kind == WARNING:
        print(f"! {event.message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research a question with a team of local AI agents.")
    parser.add_argument("question", help="the research question, in quotes")
    parser.add_argument("--depth", type=int, default=3, help="number of sub-questions, 2 to 5 (default 3)")
    parser.add_argument("--output", type=Path, help="save the report as Markdown to this file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    llm = OllamaLLM()
    run = ResearchRun(llm, [WebSearchTool(), WikipediaTool()], depth=args.depth, on_event=print_event)
    try:
        result = run.run(args.question)
    except (LLMConnectionError, LLMError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    markdown = result.to_markdown()
    print("\n" + "=" * 72 + "\n")
    print(markdown)
    if result.citations.uncited_paragraphs:
        print(f"\n! {len(result.citations.uncited_paragraphs)} paragraph(s) have no citation.", file=sys.stderr)
    if args.output:
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
