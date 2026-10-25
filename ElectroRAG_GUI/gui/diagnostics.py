"""Safe, read-only system diagnostics. Never reports environment variables or secrets."""
from __future__ import annotations

import os
import platform
import shutil
import sys
from importlib import metadata

from gui import config

_PACKAGES = ["streamlit", "chromadb", "pymupdf", "rank-bm25", "sentence-transformers", "torch"]


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _dir_size_mb(path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return round(total / (1024 * 1024), 1)


def collect() -> list[dict[str, str]]:
    rows = [
        ("Python", sys.version.split()[0]),
        ("Operating system", platform.platform()),
        ("GUI mode", config.MODE),
        ("Ollama endpoint", config.OLLAMA_URL),
        ("Configured model", config.OLLAMA_MODEL),
    ]
    rows += [(f"Package: {p}", _version(p)) for p in _PACKAGES]

    for label, path in (("Index directory", config.CHROMA_DIR), ("PDF directory", config.PDF_DIR)):
        rows.append((label, f"{path}  ({'found' if path.is_dir() else 'NOT FOUND'})"))
    if config.CHROMA_DIR.is_dir():
        rows.append(("Index size on disk", f"{_dir_size_mb(config.CHROMA_DIR)} MB"))
    try:
        free_gb = shutil.disk_usage(config.PROJECT_ROOT).free / (1024 ** 3)
        rows.append(("Free disk space (project drive)", f"{free_gb:.1f} GB"))
    except OSError:
        pass
    return [{"Item": k, "Value": v} for k, v in rows]
