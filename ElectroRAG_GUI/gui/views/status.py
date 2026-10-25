"""System Status page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from gui import config, runtime
from gui.components import badge, page_header


def render() -> None:
    page_header("System Status", "Local model, index and environment diagnostics.")

    if st.button("Refresh", icon=":material/refresh:"):
        runtime.clear_data_caches()
        st.rerun()

    # --- Ollama -------------------------------------------------------------------------
    st.subheader("Language model (Ollama)")
    ollama = runtime.get_ollama_status()
    if ollama.reachable:
        st.markdown(badge("ok", "OLLAMA REACHABLE"), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Ollama version", ollama.version or "—")
        c2.metric("Configured model", config.OLLAMA_MODEL)
        c3.metric("Models installed", len(ollama.installed_models))
        if not ollama.model_installed:
            st.error(f"Model `{config.OLLAMA_MODEL}` is not installed. Run:  `ollama pull {config.OLLAMA_MODEL}`")
        elif ollama.model_loaded:
            st.success(f"`{config.OLLAMA_MODEL}` is installed and currently loaded in memory.")
        else:
            st.info(f"`{config.OLLAMA_MODEL}` is installed; it will load on the first question.")
        with st.expander("Installed models"):
            st.write(", ".join(ollama.installed_models) or "None")
    else:
        st.markdown(badge("err", "OLLAMA NOT REACHABLE"), unsafe_allow_html=True)
        st.error("Start the Ollama app or run `ollama serve`, then press Refresh.")
        st.caption(f"Endpoint: {config.OLLAMA_URL} · {ollama.error}")

    # --- Index --------------------------------------------------------------------------
    st.subheader("Document index")
    try:
        summary = runtime.get_index_summary()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the index: {type(exc).__name__}: {exc}")
    else:
        stats = summary.stats
        if stats.available:
            st.markdown(badge("warn" if config.MODE == "mock" else "ok",
                              "MOCK INDEX" if config.MODE == "mock" else "INDEX READABLE"),
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Documents", f"{stats.document_count:,}")
            c2.metric("Chunks", f"{stats.chunk_count:,}")
            c3.metric("Collection", stats.collection)
            if stats.detail:
                st.caption(stats.detail)
        else:
            st.markdown(badge("err", "INDEX UNAVAILABLE"), unsafe_allow_html=True)
            st.warning(stats.detail)

    # --- Diagnostics --------------------------------------------------------------------
    st.subheader("Diagnostics")
    st.dataframe(pd.DataFrame(runtime.get_diagnostics()), hide_index=True)
    st.caption("Diagnostics are read-only and never include environment variables or credentials.")
