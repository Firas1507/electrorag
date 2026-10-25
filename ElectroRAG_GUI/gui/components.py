"""Small reusable UI helpers. All dynamic text is HTML-escaped before it is injected."""
from __future__ import annotations

import html
import re

import streamlit as st

from gui.models import Citation


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def md_safe(text: str) -> str:
    """Prevent '$' from being parsed as LaTeX in Streamlit markdown."""
    return text.replace("$", r"\$")


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="er-header"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def badge(kind: str, text: str) -> str:
    """kind: ok | warn | err. Returns an HTML string for st.markdown(..., unsafe_allow_html=True)."""
    return f'<span class="er-badge er-badge-{esc(kind)}">{esc(text)}</span>'


def source_chips(citations: tuple[Citation, ...]) -> None:
    labels = list(dict.fromkeys(c.label for c in citations))  # unique, order preserved
    if labels:
        chips = "".join(f'<span class="er-chip">{esc(label)}</span>' for label in labels)
        st.markdown(
            f'<div class="er-source-list"><span class="er-score">VALIDATED SOURCES</span>'
            f'<div class="er-source-chips">{chips}</div></div>',
            unsafe_allow_html=True,
        )


def evidence_block(index: int, citation: Citation) -> None:
    text = re.sub(r"\n\s*\n+", "\n", citation.evidence.strip()) or "(no evidence text provided)"
    body = esc(text).replace("\n", "<br>")  # single line: blank lines would break markdown HTML blocks
    score = f'<span class="er-score">relevance {citation.score:.3f}</span>' if citation.score is not None else ""
    st.markdown(
        f'<div class="er-ev-head"><span class="er-chip">{index}. {esc(citation.label)}</span>{score}</div>'
        f'<div class="er-evidence">{body}</div>',
        unsafe_allow_html=True,
    )
