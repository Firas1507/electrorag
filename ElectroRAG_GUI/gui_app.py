"""ElectroRAG GUI entry point. Run with:  python -m streamlit run gui_app.py"""
from __future__ import annotations

import streamlit as st

from gui import config
from gui.theme import inject_css
from gui.views import chat, library, status

st.set_page_config(
    page_title=config.APP_NAME,
    page_icon=":material/memory:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

navigation = st.navigation(
    [
        st.Page(chat.render, title="Chat", icon=":material/forum:", url_path="chat", default=True),
        st.Page(library.render, title="Datasheet Library", icon=":material/library_books:", url_path="library"),
        st.Page(status.render, title="System Status", icon=":material/monitor_heart:", url_path="status"),
    ]
)

with st.sidebar:
    mode_label = "MOCK DATA" if config.MODE == "mock" else "LIVE"
    st.caption(f"Mode: **{mode_label}** · Model: `{config.OLLAMA_MODEL}`")
    st.caption("Runs fully offline on this machine.")

navigation.run()
