
"""Shared visual theme for the Streamlit interface."""
import streamlit as st

_CSS = """
<style>
:root {
  --er-card: #111A2B;
  --er-border: #1E2B42;
  --er-teal: #2DD4BF;
  --er-teal-dim: rgba(45, 212, 191, .12);
  --er-text: #E6EDF3;
  --er-muted: #8B98A9;
  --er-warn: #F5B94C;
  --er-err: #F87171;
}

/* --------------------------------------------------
   General layout and typography
   -------------------------------------------------- */

html,
body,
.stApp {
  font-family: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
}

#MainMenu,
footer {
  visibility: hidden;
}

/* Keep Streamlit's header and toolbar available. */
[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stToolbar"] {
  visibility: visible !important;
  opacity: 1 !important;
}

/* Keep the sidebar reopen controls accessible. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
  z-index: 999999 !important;
}

/* Preserve the layout and prevent horizontal overflow. */
.block-container {
  box-sizing: border-box;
  width: 100%;
  max-width: 1100px;
  padding: clamp(1.25rem, 3vw, 2.2rem)
           clamp(1rem, 3vw, 2rem)
           7.5rem;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  min-width: 0;
  max-width: 100%;
}

/* --------------------------------------------------
   Sidebar
   -------------------------------------------------- */

[data-testid="stSidebar"] {
  border-right: 1px solid var(--er-border);
}

[data-testid="stSidebar"] > div:first-child {
  min-width: 0;
}

[data-testid="stSidebarNav"]::before {
  content: "ElectroRAG";
  display: block;
  padding: 1.1rem 1rem .4rem;
  font-size: 1.25rem;
  font-weight: 650;
  letter-spacing: .02em;
  color: var(--er-teal);
}

/* --------------------------------------------------
   Page headings
   -------------------------------------------------- */

.er-header {
  min-width: 0;
  margin-bottom: 1.2rem;
  padding-bottom: .9rem;
  border-bottom: 1px solid var(--er-border);
}

.er-eyebrow {
  font-size: .72rem;
  letter-spacing: .18em;
  color: var(--er-teal);
  font-weight: 600;
}

.er-header h1 {
  max-width: 100%;
  margin: .15rem 0 .2rem;
  padding: 0;
  font-size: 1.75rem;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.er-header p {
  max-width: 100%;
  margin: 0;
  color: var(--er-muted);
  font-size: .95rem;
  overflow-wrap: anywhere;
}

/* --------------------------------------------------
   Source chips and badges
   -------------------------------------------------- */

.er-chip {
  display: inline-flex;
  box-sizing: border-box;
  max-width: 100%;
  min-width: 0;
  margin: .15rem .35rem .15rem 0;
  padding: .15rem .6rem;
  border: 1px solid rgba(45, 212, 191, .35);
  border-radius: 999px;
  background: var(--er-teal-dim);
  color: var(--er-teal);
  font-family: Consolas, "Cascadia Mono", monospace;
  font-size: .78rem;
  white-space: normal;
  overflow-wrap: anywhere;
}

.er-badge {
  display: inline-block;
  padding: .12rem .6rem;
  border-radius: 6px;
  font-size: .75rem;
  font-weight: 600;
  letter-spacing: .04em;
}

.er-badge-ok {
  background: rgba(45, 212, 191, .14);
  color: var(--er-teal);
}

.er-badge-warn {
  background: rgba(245, 185, 76, .14);
  color: var(--er-warn);
}

.er-badge-err {
  background: rgba(248, 113, 113, .14);
  color: var(--er-err);
}

/* --------------------------------------------------
   Source lists and supporting evidence
   -------------------------------------------------- */

.er-source-list,
.er-source-chips {
  min-width: 0;
  max-width: 100%;
}

.er-source-chips {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: .25rem .35rem;
}

.er-source-chips .er-chip {
  margin: 0;
}

.er-ev-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: .35rem .6rem;
  min-width: 0;
  max-width: 100%;
  margin: .6rem 0 .25rem;
}

.er-score {
  min-width: 0;
  max-width: 100%;
  color: var(--er-muted);
  font-family: Consolas, monospace;
  font-size: .78rem;
  overflow-wrap: anywhere;
}

.er-evidence {
  box-sizing: border-box;
  min-width: 0;
  max-width: 100%;
  padding: .65rem .9rem;
  border-left: 3px solid var(--er-teal);
  border-radius: 0 6px 6px 0;
  background: var(--er-card);
  color: #C9D4E0;
  font-family: Consolas, "Cascadia Mono", monospace;
  font-size: .83rem;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: normal;
}

/* --------------------------------------------------
   Metrics and chat
   -------------------------------------------------- */

[data-testid="stMetric"] {
  padding: .8rem 1rem;
  border: 1px solid var(--er-border);
  border-radius: 8px;
  background: var(--er-card);
}

[data-testid="stMetricLabel"] {
  color: var(--er-muted);
}

[data-testid="stChatMessage"] {
  min-width: 0;
  max-width: 100%;
  background: transparent;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
  min-width: 0;
  max-width: 100%;
  overflow-wrap: anywhere;
  word-break: normal;
}

[data-testid="stChatMessage"] pre {
  box-sizing: border-box;
  max-width: 100%;
  overflow-x: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: normal;
}

[data-testid="stChatMessage"] code {
  overflow-wrap: anywhere;
  word-break: normal;
}

/* --------------------------------------------------
   Responsive layout
   -------------------------------------------------- */

@media (max-width: 640px) {
  .block-container {
    padding: 1.25rem 1rem 7.5rem;
  }

  .er-header h1 {
    font-size: 1.5rem;
  }
}
</style>
"""


def inject_css() -> None:
    """Inject the shared ElectroRAG theme."""
    st.markdown(_CSS, unsafe_allow_html=True)
