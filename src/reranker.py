"""
Cross-encoder reranking for ElectroRAG.

The reranker takes a candidate pool produced by semantic + BM25 retrieval
and scores every candidate against the ORIGINAL user question.

It is read-only:
    - It does not modify ChromaDB.
    - It does not call an LLM.
    - It does not generate answers.

Run directly:
    python -m src.reranker
"""

import sys
import re

from sentence_transformers import CrossEncoder

from src.bm25_retriever import build_index
from src.chunker import chunk_documents
from src.embeddings import load_model
from src.pdf_loader import load_all_pdfs
from src.query_expansion_retriever import (
    EXPANSIONS,
    TARGET_FILE,
    expanded_search,
    first_relevant_rank,
    hit_text,
    is_relevant,
)
from src.retriever import open_collection


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SHOW_TOP = 10
PREVIEW_CHARS = 200

# Print per-candidate specification diagnostics from rerank().
# Off by default so benchmark output stays readable; rag.py enables it with DEBUG.
VERBOSE = False


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

_INTERFACE_TERMS = (
    "usart",
    "uart",
    "spi",
    "i2c",
    "i3c",
    "i2s",
    "usb",
    "can",
    "lin",
    "ethernet",
    "wifi",
    "wi-fi",
    "bluetooth",
    "swd",
    "jtag",
)

_TEXT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "do",
    "for", "from", "how", "in", "is", "it", "its", "of", "on", "or", "the",
    "this", "to", "what", "which", "with",
}


def _specification_intent(question: str) -> str:
    """Identify spec categories where datasheets commonly use different terms."""

    normalized = question.lower()
    if re.search(r"\b(voltage|supply|vdd|vcc)\b", normalized):
        return "voltage"
    if re.search(r"\b(interface|interfaces|communication|protocol)\b", normalized):
        return "interface"
    if re.search(r"\b(impedance|speaker|load)\b", normalized):
        return "impedance"
    return ""


def _specification_relevance(
    intent: str,
    text: str,
    question: str = "",
) -> float:
    """Score explicit specification evidence without using benchmark answers."""

    normalized = text.lower()

    if intent == "voltage":
        score = 0.0
        parameter_relevance = _explicit_parameter_relevance(question, text)
        if re.search(r"\b(recommended operating conditions)\b", normalized):
            score += 0.45
        if re.search(r"\b(operating voltage|supply voltage)\b", normalized):
            score += 0.4
        elif re.search(r"\b(power[- ]supply|input voltage)\b", normalized):
            score += 0.2
        generic_rail = bool(
            re.search(r"\b(vdd|vcc)\b", normalized)
        )
        specific_rail = bool(
            re.search(r"\b(vdd[a-z0-9_]+|vcc[a-z0-9_]+)\b", normalized)
        )
        if generic_rail:
            score += 0.3
        if re.search(r"\bvoltage\b", normalized):
            score += 0.1
        if re.search(
            r"(?<![\w.])\d+(?:\.\d+)?\s*(?:to|[-–—])\s*"
            r"\d+(?:\.\d+)?\s*v\b",
            normalized,
        ):
            score += 0.35
        if re.search(
            r"\b(min|minimum)\b[\s\S]{0,100}\b(max|maximum)\b"
            r"[\s\S]{0,50}\b(unit|v)\b",
            normalized,
        ):
            score += 0.25
        if specific_rail and not generic_rail and not parameter_relevance:
            score -= 0.45
        if parameter_relevance:
            score += 0.4
            if re.search(r"(?<![\w.])\d+(?:\.\d+)?\s*v\b", normalized):
                score += 0.35
        return min(score, 1.0)

    if intent == "interface":
        found = {
            term
            for term in _INTERFACE_TERMS
            if re.search(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                normalized,
            )
        }
        score = min(len(found) / 3.0, 1.0) * 0.85
        if re.search(r"\b(interface|peripheral|communication)\b", normalized):
            score += 0.15
        return min(score, 1.0)

    if intent == "impedance":
        score = 0.0
        if re.search(r"\b(speaker|load|output)\b", normalized):
            score += 0.25
        if re.search(r"\b(impedance|zspk)\b", normalized):
            score += 0.35
        normalized = normalized.replace("ω", "ohm").replace("Ω", "ohm")
        if re.search(r"(?:[ΩΩ]|\bohms?\b)", normalized):
            score += 0.25
        if re.search(r"\d+(?:\.\d+)?\s*(?:[ΩΩ]|\bohms?\b)", normalized):
            score += 0.3
        return min(score, 1.0)

    return 0.0


def _explicit_parameter_relevance(question: str, text: str) -> float:
    """Reward exact electrical parameter labels named by the user."""

    normalized_question = question.lower()
    normalized_text = text.lower()

    question_parameters = re.findall(
        r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",
        question,
    )

    if any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(parameter.lower())}(?![a-z0-9])",
                normalized_text,
            )
            for parameter in question_parameters
    ):
        return 1.0

    if not question_parameters and re.search(
        r"\b(voltage|supply|vdd|vcc)\b",
        normalized_question,
    ) and (
        re.search(r"\bvdd\s*=", normalized_text)
        or re.search(r"\boperating voltage\s*\(\s*vdd\s*\)", normalized_text)
    ):
        return 0.75

    return 0.0


def _entity_relevance(question: str, record: dict) -> float:
    """Prefer chunks from the datasheet for the part named in the question."""

    entity = next(
        iter(
            re.findall(
                r"\b(?=[a-z0-9-]*\d)[a-z][a-z0-9-]{2,}\b",
                question.lower(),
            )
        ),
        "",
    )
    if not entity:
        return 0.0

    entity_key = re.sub(r"[^a-z0-9]", "", entity)
    source_key = re.sub(
        r"[^a-z0-9]",
        "",
        str(record.get("source", "")).lower(),
    )
    text_key = re.sub(
        r"[^a-z0-9]",
        "",
        record["text"].lower(),
    )

    if entity_key in source_key:
        return 1.0
    if entity_key in text_key:
        return 0.5
    return 0.0


def _token_similarity(first: str, second: str) -> float:
    """Jaccard similarity over informative words for near-duplicate detection."""

    first_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", first.lower())
        if len(token) > 2 and token not in _TEXT_STOPWORDS
    }
    second_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", second.lower())
        if len(token) > 2 and token not in _TEXT_STOPWORDS
    }
    if not first_tokens or not second_tokens:
        return 0.0

    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


def _diversify(ranked: list, intent: str) -> list:
    """Greedily order evidence, limiting same-page and near-duplicate chunks."""

    if not intent or len(ranked) < 2:
        return ranked

    remaining = list(ranked)
    selected = []
    page_counts = {}

    while remaining:
        eligible = [
            record
            for record in remaining
            if page_counts.get(
                (record.get("source"), record.get("page")),
                0,
            ) < 2
        ]
        if not eligible:
            eligible = remaining

        best = max(
            eligible,
            key=lambda record: (
                record["rerank_relevance"]
                - 0.2
                * max(
                    (
                        _token_similarity(
                            record["text"],
                            prior["text"],
                        )
                        for prior in selected
                    ),
                    default=0.0,
                ),
                record["rerank_score"],
            ),
        )
        selected.append(best)
        remaining.remove(best)
        page_key = (best.get("source"), best.get("page"))
        page_counts[page_key] = page_counts.get(page_key, 0) + 1

    return selected


def rerank(
    model: CrossEncoder,
    question: str,
    candidates: list,
) -> list:
    """
    Rerank candidate chunks against the ORIGINAL question.

    Parameters
    ----------
    model:
        Loaded sentence-transformers CrossEncoder.

    question:
        Original user question.

    candidates:
        List of retrieved chunk dictionaries. Each candidate must contain
        a "text" field.

    Returns
    -------
    list
        Candidates sorted from highest cross-encoder score to lowest.

        Each candidate receives:
            "rerank_score": float
    """

    # Nothing to rerank.
    if not candidates:
        return []

    # Validate the candidate structure before sending anything
    # to the cross-encoder.
    valid_candidates = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        text = candidate.get("text")

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        # Keep the original candidate metadata unchanged.
        valid_candidates.append(candidate)

    if not valid_candidates:
        return []

    # CrossEncoder expects pairs:
    #
    # [
    #     ("question", "candidate text"),
    #     ("question", "candidate text"),
    #     ...
    # ]
    pairs = [
        (question, candidate["text"])
        for candidate in valid_candidates
    ]

    # Run the local cross-encoder.
    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    # Attach raw cross-encoder scores and normalized ranks. Raw scores are
    # retained for diagnostics; normalized ranks make the combined signal
    # comparable across different question candidate pools.
    reranked = []

    for candidate, score in zip(valid_candidates, scores):
        record = {
            **candidate,
            "rerank_score": float(score),
        }

        reranked.append(record)

    # Highest score = most relevant.
    reranked.sort(key=lambda record: record["rerank_score"], reverse=True)

    intent = _specification_intent(question)
    if not intent:
        return reranked

    denominator = max(len(reranked) - 1, 1)
    for rank, record in enumerate(reranked):
        cross_encoder_relevance = 1.0 - rank / denominator
        specification_relevance = _specification_relevance(
            intent,
            record["text"],
            question,
        )
        entity_relevance = _entity_relevance(question, record)
        parameter_relevance = _explicit_parameter_relevance(
            question,
            record["text"],
        )
        record["rerank_relevance"] = (
            0.1 * cross_encoder_relevance
            + 0.25 * specification_relevance
            + 0.45 * entity_relevance
            + 0.2 * parameter_relevance
        )
        record["specification_relevance"] = specification_relevance
        record["parameter_relevance"] = parameter_relevance

    reranked.sort(
        key=lambda record: (
            record["rerank_relevance"],
            record["rerank_score"],
        ),
        reverse=True,
    )

    diversified = _diversify(reranked, intent)
    if not VERBOSE:
        return diversified

    print("Specification reranking diagnostics:")
    for rank, record in enumerate(diversified, start=1):
        if (
            rank > 5
            and record["specification_relevance"] == 0
        ):
            continue
        print(
            f"  Rerank #{rank} | top-5: "
            f"{'YES' if rank <= 5 else 'NO'} | "
            f"cross-encoder score: {record['rerank_score']:.4f} | "
            f"spec relevance: "
            f"{record['specification_relevance']:.2f} | "
            f"parameter match: {record['parameter_relevance']:.0f} | "
            f"{record.get('source', 'N/A')} p.{record.get('page', 'N/A')} | "
            f"{record.get('chunk_id', 'N/A')}"
        )

    return diversified


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def preview(text: str) -> str:
    """Create a short one-line preview of a chunk."""

    flat = " ".join(text.split())

    if len(flat) > PREVIEW_CHARS:
        return flat[:PREVIEW_CHARS] + "..."

    return flat


def print_question(
    question: str,
    relevant_pages: set,
    candidate_count: int,
    reranked: list,
) -> None:
    """Print reranking results for one question."""

    pages_text = ", ".join(
        str(page)
        for page in sorted(relevant_pages)
    )

    print("\n" + "=" * 100)
    print(f"QUESTION: {question}")
    print(
        f"KNOWN RELEVANT PAGE(S): {pages_text}   "
        "('*' marks a relevant chunk)"
    )
    print(
        f"Candidates reranked: {candidate_count}   "
        "(score = raw cross-encoder output, higher = better)"
    )
    print("=" * 100)

    for rank, record in enumerate(
        reranked[:SHOW_TOP],
        start=1,
    ):
        mark = (
            "*"
            if is_relevant(record, relevant_pages)
            else " "
        )

        page = record.get("page", "?")
        chunk_id = record.get("chunk_id", "?")
        score = record.get("rerank_score", 0.0)
        text = record.get("text", "")

        print(
            f" {mark}{rank:>2}. "
            f"score {score:>8.3f} | "
            f"p.{page} | "
            f"{chunk_id}"
        )

        print(
            f"        {preview(text)}"
        )


def print_summary(rows: list) -> None:
    """Print the final reranking summary."""

    width = 58

    print("\n" + "=" * 100)
    print(
        "SUMMARY   "
        "(#N = rank of the first relevant chunk after reranking)"
    )
    print("=" * 100)

    print(
        f"{'Question':<{width}} "
        f"{'Relevant page in top 5':<24} "
        f"{'Relevant page in top 10':<24}"
    )

    print("-" * 100)

    for question, top5, top10 in rows:

        shown = question

        if len(shown) > width - 1:
            shown = shown[: width - 4] + "..."

        print(
            f"{shown:<{width}} "
            f"{hit_text(top5):<24} "
            f"{hit_text(top10):<24}"
        )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the standalone reranking experiment."""

    print("Loading PDF documents...")

    documents = load_all_pdfs()

    print("Creating chunks...")

    chunks = chunk_documents(documents)

    if not chunks:
        sys.exit(
            "No chunks found. "
            "Put the PDF files in data/datasheets/."
        )

    print("Building BM25 index...")

    bm25_index = build_index(chunks)

    print("Opening ChromaDB collection...")

    collection = open_collection()

    print("Loading embedding model...")

    embedding_model = load_model()

    print(
        f"Loading cross-encoder: {RERANKER_MODEL}"
    )

    reranker = CrossEncoder(RERANKER_MODEL)

    print()
    print(
        f"Chunks: {len(chunks)} | "
        f"Target file: {TARGET_FILE} | "
        f"Reranker: {RERANKER_MODEL}"
    )

    rows = []

    for question, (
        relevant_pages,
        queries,
    ) in EXPANSIONS.items():

        print()
        print(f"Processing: {question}")

        # ---------------------------------------------------------------
        # 1. Retrieve candidate pool
        # ---------------------------------------------------------------

        candidates = expanded_search(
            queries,
            collection,
            embedding_model,
            bm25_index,
            chunks,
        )

        # ---------------------------------------------------------------
        # 2. Cross-encoder reranking
        # ---------------------------------------------------------------

        reranked = rerank(
            reranker,
            question,
            candidates,
        )

        # ---------------------------------------------------------------
        # 3. Display results
        # ---------------------------------------------------------------

        print_question(
            question,
            relevant_pages,
            len(candidates),
            reranked,
        )

        # ---------------------------------------------------------------
        # 4. Evaluate relevant page position
        # ---------------------------------------------------------------

        rows.append(
            (
                question,
                first_relevant_rank(
                    reranked,
                    relevant_pages,
                    5,
                ),
                first_relevant_rank(
                    reranked,
                    relevant_pages,
                    10,
                ),
            )
        )

    # -------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------

    print_summary(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()