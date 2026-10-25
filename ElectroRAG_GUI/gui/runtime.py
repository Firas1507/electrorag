"""Streamlit-aware glue: caching of backend, index metadata, and status checks."""
from __future__ import annotations

import streamlit as st

from gui import config, diagnostics, ollama_client
from gui.backend import RagBackend, create_backend
from gui.models import IndexSummary
from gui.ollama_client import OllamaStatus


@st.cache_resource(show_spinner=False)
def get_backend() -> RagBackend:
    return create_backend()


@st.cache_data(ttl=60, show_spinner="Reading index metadata…")
def get_index_summary() -> IndexSummary:
    return get_backend().index_summary()


@st.cache_data(ttl=5, show_spinner=False)
def get_ollama_status() -> OllamaStatus:
    return ollama_client.get_status(config.OLLAMA_URL, config.OLLAMA_MODEL)


@st.cache_data(ttl=60, show_spinner=False)
def get_diagnostics() -> list[dict[str, str]]:
    return diagnostics.collect()


def clear_data_caches() -> None:
    """Refresh displayed data without reloading the (expensive) pipeline."""
    get_index_summary.clear()
    get_ollama_status.clear()
    get_diagnostics.clear()


def clear_backend_cache() -> None:
    """Reload the live pipeline after datasheets have been added to the index."""
    get_backend.clear()
