"""GUI configuration. Items marked ADAPT must match your real project."""
from __future__ import annotations

import os
from pathlib import Path

# Turn off ChromaDB telemetry (set via env var so we don't create a conflicting Settings object
# if your own pipeline also opens the same index in this process).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


APP_NAME = "ElectroRAG"
APP_TAGLINE = "Local AI assistant for electronics datasheets"

# "live" (default, real pipeline) or "mock" (fabricated demo data)
MODE = os.environ.get("ELECTRORAG_GUI_MODE", "live").strip().lower()
if MODE not in {"mock", "live"}:
    MODE = "mock"

# Ollama (local only)
OLLAMA_URL = os.environ.get("ELECTRORAG_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("ELECTRORAG_OLLAMA_MODEL", "qwen3:8b")

# ---- ADAPT: where your existing index and PDFs live -----------------------------------------
CHROMA_DIR = _env_path("ELECTRORAG_CHROMA_DIR", PROJECT_ROOT / "data" / "chroma_db")
PDF_DIR = _env_path("ELECTRORAG_PDF_DIR", PROJECT_ROOT / "data" / "datasheets")
COLLECTION_NAME = os.environ.get("ELECTRORAG_COLLECTION", "electrorag")

# ---- ADAPT: metadata keys your indexer stored on each chunk ---------------------------------
META_FILENAME_KEY = os.environ.get("ELECTRORAG_META_FILENAME_KEY", "source")
META_PAGE_KEY = os.environ.get("ELECTRORAG_META_PAGE_KEY", "page")
# ----------------------------------------------------------------------------------------------

MAX_QUESTION_CHARS = 2000

EXAMPLE_QUESTIONS = [
    "What is the absolute maximum supply voltage of this op-amp?",
    "Compare the dropout voltage of the regulators in the library.",
    "What is the recommended decoupling capacitor value on the VDD pin?",
]
