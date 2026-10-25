"""Backend boundary: the GUI only ever talks to a `RagBackend`."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from typing import Any, Optional, Protocol

from gui import config, index_reader, pipeline_adapter
from gui.mock_backend import MockBackend
from gui.models import Answer, IndexSummary

log = logging.getLogger("electrorag.gui")


class RagBackend(Protocol):
    mode: str

    def ask(self, question: str) -> Answer: ...

    def index_summary(self) -> IndexSummary: ...


class LiveBackend:
    """Real backend. The pipeline is loaded lazily so Library/Status pages work even before wiring."""

    mode = "live"

    def __init__(self) -> None:
        self._pipeline: Optional[Any] = None
        self._lock = threading.Lock()

    def _get_pipeline(self) -> Any:
        with self._lock:
            if self._pipeline is None:
                self._pipeline = pipeline_adapter.load_pipeline()
            return self._pipeline

    def ask(self, question: str) -> Answer:
        return pipeline_adapter.run_query(self._get_pipeline(), question)

    def index_summary(self) -> IndexSummary:
        return index_reader.read_index_summary()


def create_backend() -> RagBackend:
    return LiveBackend() if config.MODE == "live" else MockBackend()


def safe_ask(backend: RagBackend, question: str) -> Answer:
    """Run a question and ALWAYS return an Answer; enforce the citation invariants."""
    q = (question or "").strip()
    if not q:
        return Answer(q, "error", message="Please enter a question.")
    if len(q) > config.MAX_QUESTION_CHARS:
        return Answer(q[:80] + "…", "error", message=f"Question is too long (max {config.MAX_QUESTION_CHARS} characters).")

    started = time.perf_counter()
    try:
        answer = backend.ask(q)
    except NotImplementedError as exc:
        return Answer(q, "error", message=str(exc) or "Live pipeline is not connected. Edit gui/pipeline_adapter.py.")
    except Exception as exc:  # noqa: BLE001 - GUI must never crash on pipeline errors
        log.exception("Pipeline error")
        return Answer(q, "error", message=f"Pipeline error ({type(exc).__name__}): {str(exc)[:300]}")

    # Invariant 1: an "answered" result without validated citations is never shown as an answer.
    if answer.status == "answered" and not answer.citations:
        return Answer(
            q,
            "validation_failed",
            message="The pipeline returned an answer without validated citations, so it was withheld.",
        )
    # Invariant 2: text from a failed validation is never displayed.
    if answer.status == "validation_failed" and answer.text:
        answer = replace(answer, text="")
    if answer.latency_s is None:
        answer = replace(answer, latency_s=time.perf_counter() - started)
    return answer
