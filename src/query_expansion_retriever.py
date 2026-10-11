"""
Query-expansion experiment for ElectroRAG (TPS63020).

For each test question a small, hand-written set of retrieval queries is run through
semantic search (ChromaDB, read-only) and BM25. All returned chunks are merged by
chunk ID and ranked by how often they were retrieved. No LLM / Ollama is used.

Run (from the ElectroRAG root):
    python -m src.query_expansion_retriever
"""

import re
import sys

from src.bm25_retriever import build_index, search as bm25_search
from src.chunker import chunk_documents
from src.embeddings import load_model
from src.pdf_loader import load_all_pdfs
from src.retriever import open_collection, retrieve


# ============================================================
# Configuration
# ============================================================

TARGET_FILE = "TPS63020.pdf"

TOP_K_PER_METHOD = 10
RRF_K = 60

SHOW_TOP = 10

PREVIEW_CHARS = 150


# ============================================================
# Query expansion definitions
# ============================================================

# Format:
#
# question -> (
#     relevant pages,
#     expanded queries
# )
#
# Keep this structure consistent because rag.py expects:
#
#     for question, (_pages, queries) in EXPANSIONS.items():
#
EXPANSIONS = {

    # --------------------------------------------------------
    # Switching frequency
    # --------------------------------------------------------

    "What is the switching frequency of the TPS63020?": (
        {4},
        [
            "What is the switching frequency of the TPS63020?",
            "TPS63020 switching frequency",
            "TPS63020 oscillator frequency",
            "TPS63020 oscillator frequency 2200 2400 2600 kHz",
            "TPS63020 PWM frequency",
        ],
    ),

    # --------------------------------------------------------
    # EN pin
    # --------------------------------------------------------

    "What is the function of the EN pin of the TPS63020?": (
        {4, 9},
        [
            "What is the function of the EN pin of the TPS63020?",
            "TPS63020 EN pin enable input",
            "TPS63020 EN high low shutdown",
            "TPS63020 device enable",
            "TPS63020 enable input pin 12",
        ],
    ),

    # --------------------------------------------------------
    # PS/SYNC pin
    # --------------------------------------------------------

    "What is the function of the PS/SYNC pin of the TPS63020?": (
        {4},
        [
            "What is the function of the PS/SYNC pin of the TPS63020?",
            "TPS63020 PS SYNC pin",
            "TPS63020 power save mode synchronization",
            "TPS63020 PS/SYNC enable disable power save",
            "TPS63020 clock synchronization pin",
        ],
    ),
}


# ============================================================
# Collecting and merging results
# ============================================================

def better(old, new):
    """
    Keep the better (smaller) rank.

    None means the chunk has not been seen yet.
    """

    return new if old is None else min(old, new)


def get_record(
    merged: dict,
    chunk_id: str,
    text: str,
    source,
    page,
) -> dict:
    """
    Create or retrieve a merged chunk record.
    """

    if chunk_id not in merged:

        merged[chunk_id] = {
            "chunk_id": chunk_id,
            "text": text,
            "source": source,
            "page": page,
            "best_semantic_rank": None,
            "best_bm25_rank": None,
            "appearances": 0,
            "hybrid_score": 0.0,
        }

    return merged[chunk_id]


def _specification_queries(queries: list) -> list:
    """Add generic datasheet terminology for electrical-specification queries."""

    if not queries:
        return []

    question = queries[0]
    normalized = question.lower()
    parts = re.findall(
        r"\b(?=[a-z0-9-]*\d)[a-z][a-z0-9-]{2,}\b",
        normalized,
    )
    part = next(
        (
            token
            for token in parts
            if token not in {"maximum", "minimum"}
        ),
        "",
    )
    if not part:
        return list(queries)

    additions = []

    if re.search(r"\b(voltage|supply|vdd|vcc)\b", normalized):
        additions.extend(
            (
                f"{part} recommended operating conditions supply voltage "
                "operating voltage range",
                f"{part} operating voltage VDD external power supply",
                f"{part} electrical characteristics VDD supply rail "
                "power supply rejection ratio PSRR min max unit",
                f"{part} device features external power supply VDD "
                "power supply scheme",
            )
        )

    if re.search(r"\b(interface|interfaces|communication|protocol)\b", normalized):
        additions.append(
            f"{part} communication interfaces peripherals USART UART SPI "
            "I2C I3C I2S USB CAN"
        )

    if re.search(r"\b(impedance|speaker|load)\b", normalized):
        additions.append(
            f"{part} speaker load impedance output load ohms"
        )

    expanded = []
    seen = set()
    for query in (*queries, *additions):
        normalized_query = query.casefold().strip()
        if normalized_query and normalized_query not in seen:
            expanded.append(query)
            seen.add(normalized_query)

    return expanded


def expanded_search(
    queries: list,
    collection,
    embedding_model,
    bm25_index,
    chunks: list,
) -> list:
    """
    Run every query through both semantic search and BM25.

    Returned chunks are deduplicated by chunk ID and ranked by:

    1. Number of appearances
    2. Best BM25 rank
    3. Best semantic rank
    """

    merged = {}
    queries = _specification_queries(queries)

    for query in queries:

        # ----------------------------------------------------
        # Semantic search
        # ----------------------------------------------------

        for rank, hit in enumerate(
            retrieve(
                collection,
                embedding_model,
                query,
                top_k=TOP_K_PER_METHOD,
            ),
            start=1,
        ):

            meta = hit["metadata"]

            record = get_record(
                merged,
                hit["chunk_id"],
                hit["text"],
                meta.get("source"),
                meta.get("page"),
            )

            record["appearances"] += 1

            record["best_semantic_rank"] = better(
                record["best_semantic_rank"],
                rank,
            )
            record["hybrid_score"] += 1.0 / (RRF_K + rank)

        # ----------------------------------------------------
        # BM25 search
        # ----------------------------------------------------

        # Score 0 means no query keyword in the chunk.
        # We do not count such results as real BM25 matches.

        bm25_rank = 0

        for score, chunk in bm25_search(
            bm25_index,
            chunks,
            query,
            top_k=TOP_K_PER_METHOD,
        ):

            if score <= 0:
                continue

            bm25_rank += 1

            record = get_record(
                merged,
                chunk["chunk_id"],
                chunk["text"],
                chunk["source"],
                chunk["page"],
            )

            record["appearances"] += 1

            record["best_bm25_rank"] = better(
                record["best_bm25_rank"],
                bm25_rank,
            )
            record["hybrid_score"] += 1.0 / (RRF_K + bm25_rank)

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    def sort_key(record: dict):
        return (
            -record["hybrid_score"],
            -record["appearances"],
            record["best_bm25_rank"] or 999,
            record["best_semantic_rank"] or 999,
        )

    return sorted(
        merged.values(),
        key=sort_key,
    )


# ============================================================
# Judging and printing
# ============================================================

def is_relevant(
    record: dict,
    relevant_pages: set,
) -> bool:
    """
    Determine whether a retrieved chunk belongs to the
    expected source file and relevant page.
    """

    same_file = (
        (record["source"] or "").lower()
        == TARGET_FILE.lower()
    )

    return (
        same_file
        and record["page"] in relevant_pages
    )


def first_relevant_rank(
    ranked: list,
    relevant_pages: set,
    limit: int,
):
    """
    Return the 1-based position of the first relevant chunk
    within the first `limit` results.

    Return None if no relevant chunk is found.
    """

    for position, record in enumerate(
        ranked[:limit],
        start=1,
    ):

        if is_relevant(
            record,
            relevant_pages,
        ):

            return position

    return None


def preview(text: str) -> str:
    """
    Create a short one-line preview of chunk text.
    """

    flat = " ".join(
        text.split()
    )

    return (
        flat[:PREVIEW_CHARS]
        + (
            "..."
            if len(flat) > PREVIEW_CHARS
            else ""
        )
    )


def rank_text(rank) -> str:
    """
    Format a rank for display.
    """

    return f"#{rank}" if rank else "-"


def print_question(
    question: str,
    relevant_pages: set,
    queries: list,
    ranked: list,
) -> None:
    """
    Print detailed retrieval results for one question.
    """

    pages_text = ", ".join(
        str(p)
        for p in sorted(relevant_pages)
    )

    print(
        "\n" + "=" * 100
    )

    print(
        f"QUESTION: {question}"
    )

    print(
        f"KNOWN RELEVANT PAGE(S): "
        f"{pages_text}   ('*' marks a relevant chunk)"
    )

    print(
        "Expanded queries:"
    )

    for number, query in enumerate(
        queries,
        start=1,
    ):

        print(
            f"  {number}. {query}"
        )

    print(
        f"Unique chunks retrieved: {len(ranked)}"
    )

    print(
        "=" * 100
    )

    for position, record in enumerate(
        ranked[:SHOW_TOP],
        start=1,
    ):

        mark = (
            "*"
            if is_relevant(
                record,
                relevant_pages,
            )
            else " "
        )

        print(
            f" {mark}{position:>2}. "
            f"seen {record['appearances']}x"
            f" | best semantic "
            f"{rank_text(record['best_semantic_rank'])}"
            f" | best BM25 "
            f"{rank_text(record['best_bm25_rank'])}"
            f" | p.{record['page']}"
            f" | {record['chunk_id']}"
        )

        print(
            f"        {preview(record['text'])}"
        )


def hit_text(rank) -> str:
    """
    Format a hit rank for the summary.
    """

    return (
        f"YES (#{rank})"
        if rank
        else "NO"
    )


def print_summary(rows: list) -> None:
    """
    Print retrieval summary.
    """

    width = 58

    print(
        "\n" + "=" * 100
    )

    print(
        "SUMMARY   "
        "(#N = position of the first relevant chunk "
        "in the final ranking)"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Question':<{width}} "
        f"{'Relevant page in top 5':<24} "
        f"{'Relevant page in top 10':<24}"
    )

    print(
        "-" * 100
    )

    for question, top5, top10 in rows:

        shown = (
            question
            if len(question) <= width - 1
            else question[: width - 4] + "..."
        )

        print(
            f"{shown:<{width}} "
            f"{hit_text(top5):<24} "
            f"{hit_text(top10):<24}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Load documents and chunks
    # --------------------------------------------------------

    chunks = chunk_documents(
        load_all_pdfs()
    )

    if not chunks:

        sys.exit(
            "No chunks found. "
            "Put the PDF(s) in data/datasheets/."
        )

    # --------------------------------------------------------
    # Build BM25 index
    # --------------------------------------------------------

    bm25_index = build_index(
        chunks
    )

    # --------------------------------------------------------
    # Open ChromaDB
    # --------------------------------------------------------

    collection = open_collection()

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedding_model = load_model()

    print(
        f"Chunks: {len(chunks)} "
        f"| Target file: {TARGET_FILE}"
    )

    # --------------------------------------------------------
    # Run every expansion test
    # --------------------------------------------------------

    rows = []

    for question, (
        relevant_pages,
        queries,
    ) in EXPANSIONS.items():

        ranked = expanded_search(
            queries,
            collection,
            embedding_model,
            bm25_index,
            chunks,
        )

        print_question(
            question,
            relevant_pages,
            queries,
            ranked,
        )

        rows.append(
            (
                question,
                first_relevant_rank(
                    ranked,
                    relevant_pages,
                    5,
                ),
                first_relevant_rank(
                    ranked,
                    relevant_pages,
                    10,
                ),
            )
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        rows
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()