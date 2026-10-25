"""Datasheet Library page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from gui import runtime
from gui.components import page_header
from gui.datasheet_upload import index_uploaded_pdfs


def render() -> None:
    page_header("Datasheet Library", "Upload datasheet PDFs and browse indexed documents.")

    if st.button("Refresh", icon=":material/refresh:"):
        runtime.clear_data_caches()
        st.rerun()

    uploads = st.file_uploader(
        "Choose datasheet PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs are saved locally and added to the searchable index.",
    )
    if st.button("Upload and index", disabled=not uploads, icon=":material/upload_file:"):
        try:
            with st.spinner("Saving PDFs and adding their text to the index…"):
                filenames = index_uploaded_pdfs(uploads)
        except Exception as exc:  # noqa: BLE001 - report upload/index errors in the page
            st.error(f"Could not upload datasheets: {type(exc).__name__}: {exc}")
        else:
            runtime.clear_backend_cache()
            runtime.clear_data_caches()
            st.success(f"Uploaded and indexed {len(filenames)} datasheet(s): {', '.join(filenames)}")
            st.rerun()

    try:
        summary = runtime.get_index_summary()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the library: {type(exc).__name__}: {exc}")
        return

    if not summary.stats.available:
        st.warning(summary.stats.detail or "The index is not available.")
        return

    left, right = st.columns(2)
    left.metric("Documents", f"{summary.stats.document_count:,}")
    right.metric("Chunks", f"{summary.stats.chunk_count:,}")
    if summary.stats.detail:
        st.caption(summary.stats.detail)

    query = st.text_input("Filter", placeholder="Filter by filename or title…", label_visibility="collapsed")
    needle = query.strip().lower()
    docs = [d for d in summary.documents
            if not needle or needle in d.filename.lower() or needle in (d.title or "").lower()]

    if not docs:
        st.info("No documents match your filter." if needle else "The index contains no documents.")
        return

    frame = pd.DataFrame(
        [
            {
                "Filename": d.filename,
                "Title": d.title or "—",
                "PDF pages": d.page_count,
                "Pages indexed": d.pages_indexed,
                "Chunks": d.chunk_count,
                "Size (MB)": d.size_mb,
                "Modified": d.modified or "—",
            }
            for d in docs
        ]
    )
    st.dataframe(frame, hide_index=True)
