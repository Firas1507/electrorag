"""Chat page."""
from __future__ import annotations

import streamlit as st

from gui import config, runtime
from gui.backend import safe_ask
from gui.components import evidence_block, md_safe, page_header, source_chips
from gui.models import Answer

_HISTORY_KEY = "chat_history"
_PENDING_KEY = "pending_question"


def _render_answer(answer: Answer) -> None:
    if answer.status == "answered":
        st.markdown(md_safe(answer.text))
        source_chips(answer.citations)
        with st.expander(f"Supporting evidence ({len(answer.citations)})"):
            for i, citation in enumerate(answer.citations, start=1):
                evidence_block(i, citation)
        meta = [answer.model, f"{answer.latency_s:.1f}s" if answer.latency_s is not None else None,
                f"{len(answer.citations)} validated citation(s)"]
        st.caption(" · ".join(m for m in meta if m))
    elif answer.status == "unsupported":
        st.info(answer.message or "I couldn't find supporting evidence for this in the indexed datasheets.",
                icon=":material/search_off:")
    elif answer.status == "validation_failed":
        st.warning(
            (answer.message or "Citation validation failed.")
            + " No answer is shown because it could not be verified against the datasheets.",
            icon=":material/gpp_maybe:",
        )
    else:
        st.error(answer.message or "Something went wrong while answering.", icon=":material/error:")


def render() -> None:
    page_header(
        "Ask the datasheet library",
        "Get grounded answers about electronic components from your indexed datasheets.",
    )
    backend = runtime.get_backend()

    if backend.mode == "mock":
        st.warning("**Mock mode:** answers, filenames and citations below are fabricated demo data. "
                   "Do not rely on them.", icon=":material/science:")
    else:
        ollama = runtime.get_ollama_status()
        if not ollama.reachable:
            st.error("Ollama is not reachable. Start the Ollama app (or run `ollama serve`) and try again.",
                     icon=":material/power_off:")

    history: list[dict] = st.session_state.setdefault(_HISTORY_KEY, [])

    if history:
        with st.sidebar:
            if st.button("Clear conversation", icon=":material/delete:"):
                st.session_state[_HISTORY_KEY] = []
                st.rerun()

    for turn in history:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.markdown(md_safe(turn["text"]))
            else:
                _render_answer(turn["answer"])

    if not history and config.EXAMPLE_QUESTIONS:
        with st.container(border=True):
            st.subheader("Start with a question")
            st.caption("Choose an example, or enter your own question below.")
            columns = st.columns(len(config.EXAMPLE_QUESTIONS), gap="small")
            for i, example in enumerate(config.EXAMPLE_QUESTIONS):
                with columns[i]:
                    if st.button(
                        example,
                        key=f"example_{i}",
                        icon=":material/arrow_forward:",
                        width="stretch",
                        wrap=True,
                    ):
                        st.session_state[_PENDING_KEY] = example
                        st.rerun()

    question = st.chat_input("Ask about a component or datasheet…", max_chars=config.MAX_QUESTION_CHARS)
    if question is None:
        question = st.session_state.pop(_PENDING_KEY, None)

    if question:
        history.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(md_safe(question))
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating a validated answer…"):
                answer = safe_ask(backend, question)
            _render_answer(answer)
        history.append({"role": "assistant", "answer": answer})
