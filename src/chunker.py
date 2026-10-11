"""Chunker for ElectroRAG: splits page documents into metadata-rich chunks.

Chunks never cross page boundaries, so every chunk maps to exactly one
(source, page) pair and can be cited precisely.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

Document = Dict[str, Any]
Chunk = Dict[str, Any]

DEFAULT_CHUNK_SIZE = 1000      # max characters per chunk
DEFAULT_CHUNK_OVERLAP = 150    # characters shared between consecutive chunks
DEFAULT_MIN_CHUNK_CHARS = 30   # drop fragments shorter than this
MAX_HEADING_CHARS = 80

# Common datasheet section titles (matched case-insensitively, exact title).
SECTION_KEYWORDS = {
    "features", "features and benefits", "applications", "description",
    "general description", "detailed description", "functional description",
    "block diagram", "functional block diagram", "specifications",
    "pin configuration", "pin configuration and functions", "pin description",
    "pin descriptions", "pin assignments", "pinout",
    "absolute maximum ratings", "recommended operating conditions",
    "electrical characteristics", "electrical specifications",
    "dc electrical characteristics", "ac electrical characteristics",
    "switching characteristics", "timing characteristics", "timing diagrams",
    "typical characteristics", "typical performance characteristics",
    "typical application", "typical applications", "application information",
    "thermal information", "thermal characteristics", "register map",
    "package information", "package outline", "ordering information",
    "revision history",
}

_NUMBERING = re.compile(r"^\d+(?:\.\d+)*\.?\s+")          # "1 ", "1.", "7.3 "
_TITLE_START = re.compile(r"^[A-Z][A-Za-z]{3,}")          # "Recommended ...", not "V supply"
_TITLE_SPLIT = re.compile(r"[:(\-\u2013\u2014,]")          # cut "Title: VCC = 5 V" at the colon


# --------------------------------------------------------------------------- #
# Section detection
# --------------------------------------------------------------------------- #
def _is_heading(line: str) -> bool:
    """Heuristic: is this line a datasheet section heading?"""
    if len(line) > MAX_HEADING_CHARS:
        return False

    numbering = _NUMBERING.match(line)
    title = line[numbering.end():] if numbering else line

    base = _TITLE_SPLIT.split(title)[0].strip().lower()
    if base in SECTION_KEYWORDS:
        return True

    # Generic dotted-number headings such as "7.3 Thermal Considerations".
    return bool(numbering and "." in numbering.group(0) and _TITLE_START.match(title))


def find_headings(text: str) -> List[Tuple[int, str]]:
    """Return (char_offset, heading_text) for each heading found in the text."""
    headings: List[Tuple[int, str]] = []
    pos = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and _is_heading(stripped):
            headings.append((pos, stripped))
        pos += len(line) + 1
    return headings


def _section_at(headings: List[Tuple[int, str]], pos: int, carried: Optional[str]) -> Optional[str]:
    """Section in effect at a character offset (falls back to the carried-over section)."""
    current = carried
    for heading_pos, title in headings:
        if heading_pos > pos:
            break
        current = title
    return current


# --------------------------------------------------------------------------- #
# Text splitting
# --------------------------------------------------------------------------- #
def _snap_end(text: str, start: int, end: int) -> int:
    """Pull a chunk end back to a natural break (paragraph > line > sentence > word)."""
    floor = start + (end - start) // 2
    for sep in ("\n\n", "\n", ". ", " "):
        idx = text.rfind(sep, floor, end)
        if idx != -1:
            return idx + len(sep)
    return end


def _snap_start(text: str, pos: int, limit: int) -> int:
    """Move a chunk start forward so it doesn't begin in the middle of a word."""
    if pos > 0 and not text[pos - 1].isspace():
        while pos < limit and not text[pos].isspace():
            pos += 1
    return pos


def split_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    """Split text into (start, end) character spans with overlap and clean boundaries."""
    spans: List[Tuple[int, int]] = []
    n = len(text)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _snap_end(text, start, end)

        # Trim whitespace so offsets point at real content.
        s, e = start, end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((s, e))

        if end >= n:
            break
        start = _snap_start(text, max(end - overlap, start + 1), end)

    return spans


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> List[Chunk]:
    """Turn page documents ({text, source, page, ...}) into metadata-rich chunks.

    Each chunk contains:
        chunk_id          unique id, e.g. "LM358.pdf::p3::c1"
        text              chunk content
        source            PDF filename
        doc_title         filename without extension
        page              1-indexed page number
        total_pages       page count of the source PDF (if provided by the loader)
        section           last datasheet heading seen (carried across pages), or None
        chunk_index       running index within the source file (for neighbor lookup)
        page_chunk_index  index within the page
        char_start/end    character span within the page text
        char_count        length of the chunk text
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    chunks: List[Chunk] = []
    source_counts: Dict[str, int] = {}
    current_source: Optional[str] = None
    carried_section: Optional[str] = None

    for doc in sorted(documents, key=lambda d: (d["source"], d["page"])):
        source, page = doc["source"], doc["page"]
        text = doc.get("text") or ""
        if not text.strip():
            continue

        if source != current_source:  # new file: reset section tracking
            current_source = source
            carried_section = None

        headings = find_headings(text)
        page_chunk_index = 0

        for start, end in split_text(text, chunk_size, chunk_overlap):
            chunk_text = text[start:end]
            if len(chunk_text) < min_chunk_chars:
                continue

            chunks.append(
                {
                    "chunk_id": f"{source}::p{page}::c{page_chunk_index}",
                    "text": chunk_text,
                    "source": source,
                    "doc_title": Path(source).stem,
                    "page": page,
                    "total_pages": doc.get("total_pages"),
                    "section": _section_at(headings, start, carried_section),
                    "chunk_index": source_counts.get(source, 0),
                    "page_chunk_index": page_chunk_index,
                    "char_start": start,
                    "char_end": end,
                    "char_count": len(chunk_text),
                }
            )
            page_chunk_index += 1
            source_counts[source] = source_counts.get(source, 0) + 1

        if headings:
            carried_section = headings[-1][1]

    return chunks


# --------------------------------------------------------------------------- #
# Testing helpers
# --------------------------------------------------------------------------- #
TEST_KEYWORDS = [
    "Features",
    "Applications",
    "Pin Configuration",
    "Electrical Characteristics",
    "Application Information",
]


def find_chunks_with_keywords(chunks: List[Chunk], keywords: List[str]) -> List[Chunk]:
    """Return chunks whose text contains any keyword (case-insensitive)."""
    lowered = [k.lower() for k in keywords]
    return [c for c in chunks if any(k in c["text"].lower() for k in lowered)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Works as `python src/chunker.py` or `python -m src.chunker`.
    try:
        from src.pdf_loader import load_all_pdfs
    except ImportError:
        from pdf_loader import load_all_pdfs

    pages = load_all_pdfs()
    chunks = chunk_documents(pages)
    print(f"\n{len(pages)} pages -> {len(chunks)} chunks\n")

    for chunk in chunks[:10]:
        preview = chunk["text"][:80].replace("\n", " ")
        print(
            f"{chunk['chunk_id']} | section={chunk['section']} "
            f"| {chunk['char_count']} chars | {preview}..."
        )
    if len(chunks) > 10:
        print(f"... and {len(chunks) - 10} more chunks")

    # Print full text of chunks containing the test keywords.
    matches = find_chunks_with_keywords(chunks, TEST_KEYWORDS)
    print(f"\n{'=' * 80}")
    print(f"{len(matches)} chunks contain keywords: {', '.join(TEST_KEYWORDS)}")
    print("=" * 80)

    for chunk in matches:
        print(f"\nchunk_id: {chunk['chunk_id']}")
        print(f"section:  {chunk['section']}")
        print(f"page:     {chunk['page']}")
        print("text:")
        print(chunk["text"])
        print("-" * 80)


if __name__ == "__main__":
    main()