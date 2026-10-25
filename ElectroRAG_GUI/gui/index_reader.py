"""
Read-only access to the existing ChromaDB index plus optional PDF metadata via PyMuPDF.

This module never creates, modifies, or deletes anything in the index.
ADAPT: collection name and metadata keys live in gui/config.py.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from gui import config
from gui.models import DocumentInfo, IndexStats, IndexSummary

_BATCH = 5000


def _unavailable(detail: str) -> IndexSummary:
    return IndexSummary(IndexStats(False, 0, 0, config.COLLECTION_NAME, detail), ())


def _pdf_lookup() -> dict[str, Path]:
    if not config.PDF_DIR.is_dir():
        return {}
    return {p.name: p for p in config.PDF_DIR.rglob("*.pdf")}


def _pdf_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        stat = path.stat()
        info["size_mb"] = round(stat.st_size / (1024 * 1024), 2)
        info["modified"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
    except OSError:
        return info
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            info["page_count"] = doc.page_count
            title = (doc.metadata or {}).get("title")
            info["title"] = title.strip() if isinstance(title, str) and title.strip() else None
    except Exception:
        pass  # metadata is best-effort; never fail the page over one odd PDF
    return info


def read_index_summary() -> IndexSummary:
    chroma_dir = config.CHROMA_DIR
    # Check first: PersistentClient would CREATE the directory if it did not exist.
    if not chroma_dir.is_dir():
        return _unavailable(f"Index directory not found: {chroma_dir}")

    try:
        import chromadb
    except ImportError:
        return _unavailable("The 'chromadb' package is not installed in this environment.")

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(config.COLLECTION_NAME)  # NOT get_or_create
        chunk_count = collection.count()
    except Exception as exc:
        return _unavailable(f"Could not open collection '{config.COLLECTION_NAME}': {type(exc).__name__}: {exc}")

    chunks_per_file: Counter[str] = Counter()
    pages_per_file: dict[str, set] = defaultdict(set)
    offset = 0
    try:
        while offset < chunk_count:
            batch = collection.get(limit=_BATCH, offset=offset, include=["metadatas"])
            metas = batch.get("metadatas") or []
            if not metas:
                break
            for meta in metas:
                meta = meta or {}
                raw_name = meta.get(config.META_FILENAME_KEY)
                name = Path(str(raw_name)).name if raw_name else "(unknown)"
                chunks_per_file[name] += 1
                page = meta.get(config.META_PAGE_KEY)
                if page is not None:
                    pages_per_file[name].add(page)
            offset += len(metas)
    except Exception as exc:
        return _unavailable(f"Failed while reading chunk metadata: {type(exc).__name__}: {exc}")

    pdfs = _pdf_lookup()
    documents = []
    for name in sorted(chunks_per_file, key=str.lower):
        extra = _pdf_info(pdfs[name]) if name in pdfs else {}
        documents.append(
            DocumentInfo(
                filename=name,
                chunk_count=chunks_per_file[name],
                pages_indexed=len(pages_per_file[name]) or None,
                page_count=extra.get("page_count"),
                size_mb=extra.get("size_mb"),
                modified=extra.get("modified"),
                title=extra.get("title"),
            )
        )

    detail = ""
    if not any(n in pdfs for n in chunks_per_file):
        detail = f"No matching PDFs found under {config.PDF_DIR}; file metadata (size, pages) is unavailable."
    stats = IndexStats(True, len(documents), chunk_count, config.COLLECTION_NAME, detail)
    return IndexSummary(stats, tuple(documents))
