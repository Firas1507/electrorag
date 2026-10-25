"""Plain data containers shared between the GUI and the backend adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

AnswerStatus = Literal["answered", "unsupported", "validation_failed", "error"]


@dataclass(frozen=True)
class Citation:
    """A citation that has ALREADY passed your pipeline's validation. The GUI never creates these."""

    filename: str
    page: Optional[int]
    evidence: str = ""
    score: Optional[float] = None
    chunk_id: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.filename} · p.{self.page}" if self.page is not None else self.filename


@dataclass(frozen=True)
class Answer:
    question: str
    status: AnswerStatus
    text: str = ""
    citations: tuple[Citation, ...] = ()
    latency_s: Optional[float] = None
    model: Optional[str] = None
    message: str = ""  # human-readable reason for unsupported / failed / error states


@dataclass(frozen=True)
class DocumentInfo:
    filename: str
    chunk_count: int
    pages_indexed: Optional[int] = None
    page_count: Optional[int] = None
    size_mb: Optional[float] = None
    modified: Optional[str] = None
    title: Optional[str] = None


@dataclass(frozen=True)
class IndexStats:
    available: bool
    document_count: int
    chunk_count: int
    collection: str
    detail: str = ""


@dataclass(frozen=True)
class IndexSummary:
    stats: IndexStats
    documents: tuple[DocumentInfo, ...] = ()
