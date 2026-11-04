"""Streamlit UI for the multi-agent research assistant.

Run with:  .venv/bin/streamlit run app.py

Environment variables:
    OLLAMA_URL      where Ollama is listening (default http://localhost:11434)
    OLLAMA_MODEL    which model the agents use (default qwen3:4b)
    LOCAL_DOCS_DB   where the local PDF index lives (default ./chroma_db)
"""

from __future__ import annotations

import logging

import streamlit as st

from src.agents import MAX_DEPTH, MIN_DEPTH, UnclearQuestionError
from src.index import SAMPLES_DIR, index_document, index_pdfs
from src.llm import LLMConnectionError, LLMError, OllamaLLM
from src.local_docs import db_dir
from src.orchestrator import (
    DONE,
    PLAN_READY,
    RESEARCHER_DONE,
    SOURCES_FOUND,
    WARNING,
    WRITING,
    Event,
    ResearchResult,
    ResearchRun,
)
from src.report import KIND_LABELS, demote_headings, link_citations, slugify, source_line
from src.tools import make_tools

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

SOURCE_OPTIONS = {"Both": "both", "Web": "web", "Local PDFs": "local"}
EXAMPLES = [
    "How did the printing press change European science?",
    "What are the health effects of intermittent fasting?",
    "How does Quillmark Analytics handle passwords, and why?",
]

st.set_page_config(page_title="Research assistant", page_icon=":material/travel_explore:", layout="centered")


@st.cache_resource
def get_store():
    """The local PDF index, opened once per server process."""
    from src.local_docs.vector_store import VectorStore

    return VectorStore(persist_directory=db_dir())


@st.cache_data(ttl=30, show_spinner=False)
def ollama_status() -> tuple[bool, str]:
    llm = OllamaLLM(timeout=3)
    try:
        models = llm.list_models()
    except LLMError as exc:
        return False, str(exc)
    if llm.model not in models:
        return False, f"Model `{llm.model}` is not downloaded. Run `ollama pull {llm.model}`."
    return True, f"Ollama is running with `{llm.model}`."


def use_example() -> None:
    if st.session_state.example:
        st.session_state.question = st.session_state.example
    st.session_state.example = None


st.session_state.setdefault("result", None)

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Research settings")
    source_label = st.segmented_control(
        "Sources", list(SOURCE_OPTIONS), default="Both", required=True, key="sources",
        help="Web searches DuckDuckGo and Wikipedia. Local PDFs searches the documents indexed below.",
    )
    depth = st.slider(
        "Sub-questions", MIN_DEPTH, MAX_DEPTH, 3,
        help="How many parts the planner splits your question into. Each adds about 20 to 40 seconds.",
    )

    st.subheader("Local PDFs")
    store = get_store()
    documents = store.list_documents()
    if documents:
        st.caption(f"{len(documents)} documents, {store.count()} chunks indexed")
        with st.expander("Indexed documents"):
            for doc in documents:
                st.markdown(f"- {doc.source}")
    else:
        st.caption("No documents indexed yet.")
        if st.button("Index sample documents", icon=":material/library_add:", width="stretch"):
            with st.spinner("Indexing sample PDFs (the first run downloads an 80 MB embedding model)..."):
                lines = index_pdfs([SAMPLES_DIR], persist_directory=db_dir())
            st.toast(f"Indexed {len(lines)} sample documents", icon=":material/check:")
            st.rerun()

    with st.form("upload", clear_on_submit=True, border=False):
        uploads = st.file_uploader("Add your own PDFs", type="pdf", accept_multiple_files=True)
        if st.form_submit_button("Add to index", icon=":material/upload:", width="stretch") and uploads:
            with st.spinner("Indexing..."):
                for upload in uploads:
                    st.caption(index_document(store, upload.getvalue(), upload.name))

    st.subheader("Status")
    ok, status_message = ollama_status()
    (st.success if ok else st.error)(status_message, icon=":material/check_circle:" if ok else ":material/error:")

# ---------------------------------------------------------------- question

st.title("Research assistant")
st.caption(
    "A team of local AI agents plans your question, researches each part on the web and in your PDFs, "
    "and writes a report where every claim cites its source. Runs on your machine with Ollama."
)

st.pills("Examples", EXAMPLES, key="example", on_change=use_example, label_visibility="collapsed")
with st.form("ask"):
    question = st.text_input(
        "Research question", key="question", placeholder="What do you want to research?",
    )
    submitted = st.form_submit_button("Research", icon=":material/travel_explore:", type="primary")

# ---------------------------------------------------------------- run


class ProgressView:
    """Renders orchestrator events as a step timeline inside one status box.

    Child steps are created with ``self.box.status(...)`` rather than ``with self.box:``,
    because leaving a ``with`` block marks a running status as complete.
    """

    def __init__(self) -> None:
        self.box = st.status(":shimmer[Planning the research]", expanded=True)
        self.plan_step = self.box.status(":shimmer[Planner: splitting the question]", type="step")
        self.steps: dict[int, object] = {}
        self.writing_step = None
        self.warnings: list[str] = []

    def __call__(self, event: Event) -> None:
        data = event.data
        if event.kind == PLAN_READY:
            plan = data["plan"]
            self.plan_step.update(label=f"Planner: {len(plan.sub_questions)} sub-questions", state="complete")
            self.plan_step.markdown("\n".join(f"{sq.number}. {sq.text}" for sq in plan.sub_questions))
            self.box.update(label=":shimmer[Researching]", expanded=True)
            for sq in plan.sub_questions:
                self.steps[sq.number] = self.box.status(f":shimmer[Researcher {sq.number}: searching]", type="step")
        elif event.kind == SOURCES_FOUND:
            sq, sources = data["sub_question"], data["sources"]
            step = self.steps[sq.number]
            step.update(label=f":shimmer[Researcher {sq.number}: reading {len(sources)} sources]")
            step.markdown(f"**{sq.text}**")
            step.markdown("\n".join(f"- {source_line(s)}" for s in sources) or "No sources found.")
        elif event.kind == RESEARCHER_DONE:
            sq, findings = data["sub_question"], data["findings"]
            failed = findings.error is not None
            label = f"Researcher {sq.number}: " + ("failed" if failed else f"{len(findings.findings)} findings")
            self.steps[sq.number].update(label=label, state="error" if failed else "complete")
        elif event.kind == WRITING:
            self.box.update(label=":shimmer[Writing the report]", expanded=True)
            self.writing_step = self.box.status(":shimmer[Writer: drafting the report]", type="step")
        elif event.kind == WARNING:
            self.warnings.append(event.message)
        elif event.kind == DONE:
            result = data["result"]
            if self.writing_step is not None:
                self.writing_step.update(label="Writer: report ready", state="complete")
            self.box.update(
                label=f"Research complete in {result.seconds:.0f} s ({result.model_calls} model calls)",
                state="complete", expanded=False,
            )

    def fail(self, label: str) -> None:
        self.box.update(label=label, state="error", expanded=False)


if submitted:
    if not question.strip():
        st.warning("Please enter a research question.", icon=":material/edit:")
    elif not ok:
        st.error(status_message, title="The agents need Ollama", icon=":material/error:")
    else:
        st.session_state.result = None
        try:
            tools, setup_warnings = make_tools(SOURCE_OPTIONS[source_label], store=store)
        except ValueError as exc:
            st.warning(str(exc), icon=":material/folder_off:")
        else:
            for warning in setup_warnings:
                st.info(warning, icon=":material/info:")
            progress = ProgressView()
            try:
                st.session_state.result = ResearchRun(OllamaLLM(), tools, depth=depth, on_event=progress).run(question)
            except UnclearQuestionError as exc:
                progress.fail("Research stopped")
                st.warning(str(exc), icon=":material/help:")
            except LLMConnectionError as exc:
                progress.fail("Research stopped")
                ollama_status.clear()
                st.error(str(exc), title="Lost connection to Ollama", icon=":material/error:")
            except LLMError as exc:
                progress.fail("Research failed")
                st.error(str(exc), title="The model returned an unusable response", icon=":material/error:")

# ---------------------------------------------------------------- result

result: ResearchResult | None = st.session_state.result
if result is not None:
    st.subheader(result.question)
    st.markdown(demote_headings(link_citations(result.report, result.sources)))

    check = result.citations
    if check.unknown_ids:
        st.warning(
            f"The report cites source numbers that don't exist: {', '.join(map(str, check.unknown_ids))}.",
            title="Unverified citations", icon=":material/report:",
        )
    if check.uncited_paragraphs:
        st.caption(
            f":material/info: {len(check.uncited_paragraphs)} paragraph(s) have no citation, "
            "usually the summary or conclusion. Treat them as the writer's synthesis."
        )

    with st.container(horizontal=True):
        st.metric("Time", f"{result.seconds:.0f} s", border=True)
        st.metric("Model calls", result.model_calls, border=True)
        st.metric("Sources read", len(result.sources), border=True)
        st.metric("Sources cited", len(result.cited_sources), border=True)

    st.subheader("Sources")
    cited = sorted(result.cited_sources, key=lambda s: s.id)
    st.markdown("\n".join(f"- {source_line(s)}" for s in cited) or "No sources were cited.")

    with st.expander("How the agents worked", icon=":material/account_tree:"):
        for item in result.findings:
            sq = item.sub_question
            st.markdown(f"**Researcher {sq.number}: {sq.text}**")
            st.caption("Searched for: " + "; ".join(sq.queries))
            kinds = sorted({KIND_LABELS[s.kind] for s in item.sources})
            st.markdown(f"Read {len(item.sources)} sources ({', '.join(kinds) or 'none'}).")
            if item.error:
                st.error(item.error, title="This researcher failed")
            for finding in item.findings:
                st.markdown(f"- {finding.claim} " + "".join(f"[{i}]" for i in finding.source_ids))
            if item.gaps:
                st.caption(f"Gaps: {item.gaps}")
            if item.dropped_source_ids:
                st.caption(f"Ignored citations to sources it wasn't given: {item.dropped_source_ids}")
        if result.warnings:
            st.markdown("**Warnings**")
            for warning in result.warnings:
                st.caption(f"- {warning}")

    st.download_button(
        "Download report (Markdown)", result.to_markdown(), file_name=f"{slugify(result.question)}.md",
        mime="text/markdown", icon=":material/download:",
    )
