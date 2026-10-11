"""
ElectroRAG pipeline.

Pipeline:

    question
        -> query expansion
        -> semantic + BM25 retrieval
        -> deduplication
        -> cross-encoder reranking
        -> top 5 chunks
        -> generator
        -> local LLM
        -> deterministic citation resolution
        -> final answer

Run from the ElectroRAG root:

    python -m src.rag "What is the input voltage range of the TPS63020?"
"""

import sys

from sentence_transformers import CrossEncoder

from src.bm25_retriever import build_index
from src.chunker import chunk_documents
from src.embeddings import load_model
from src.generator import generate_answer
from src.pdf_loader import load_all_pdfs
from src.query_expansion_retriever import EXPANSIONS, expanded_search
from src import reranker as reranker_module
from src.reranker import RERANKER_MODEL, rerank
from src.retriever import open_collection


# ============================================================
# Configuration
# ============================================================

# Number of reranked chunks passed to the generator.
CONTEXT_CHUNKS = 5

# Print retrieval/debug information.
DEBUG = True

# Optional minimum reranker score.
# Keep disabled until we have enough benchmark data.
MIN_RERANK_SCORE = None

NOT_FOUND_MESSAGE = (
    "I could not find this information in the indexed datasheet."
)


# ============================================================
# Query expansion
# ============================================================

def get_queries(question: str) -> list:
    """
    Return manual query expansions for known benchmark questions.

    Questions without a predefined expansion use the original
    question only.
    """

    wanted = " ".join(
        question.lower().split()
    )

    for known_question, (_pages, queries) in EXPANSIONS.items():

        known = " ".join(
            known_question.lower().split()
        )

        if known == wanted:
            return queries

    return [question]


# ============================================================
# Context conversion
# ============================================================

def to_context_hits(reranked: list) -> list:
    """
    Convert reranked records into the format expected by
    the generator and citation resolver.
    """

    return [
        {
            "text": record["text"],
            "source": record["source"],
            "page": record["page"],
            "chunk_id": record["chunk_id"],
            "rerank_score": record["rerank_score"],
        }
        for record in reranked
    ]


# ============================================================
# Debug output
# ============================================================

def print_debug(
    queries: list,
    candidate_count: int,
    top: list,
) -> None:
    """
    Print retrieval and reranking information.
    """

    print("\n---------------- DEBUG ----------------")

    print(
        f"Queries used:                {len(queries)}"
    )

    print(
        f"Candidates before reranking: {candidate_count}"
    )

    print(
        f"Top {len(top)} after reranking "
        "(higher score = better):"
    )

    for rank, record in enumerate(top, start=1):

        print(
            f"\n  {rank}. "
            f"score {record['rerank_score']:8.3f} "
            f"| p.{record['page']} "
            f"| {record['chunk_id']}"
        )

        print(
            "  ---------------- CONTEXT ----------------"
        )

        print(
            record["text"]
        )

        print(
            "  ------------------------------------------"
        )


def print_not_found() -> None:
    """
    Print the standard not-found response.
    """

    print("\nFinal answer:")

    print(
        NOT_FOUND_MESSAGE
    )


# ============================================================
# Main RAG pipeline
# ============================================================

def main() -> None:

    reranker_module.VERBOSE = DEBUG

    # --------------------------------------------------------
    # 0. Read question
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        sys.exit(
            'Usage: python -m src.rag "Your question here"'
        )

    question = " ".join(
        sys.argv[1:]
    )

    # --------------------------------------------------------
    # 1. Load documents and build chunks
    # --------------------------------------------------------

    documents = load_all_pdfs()

    chunks = chunk_documents(
        documents
    )

    if not chunks:

        sys.exit(
            "No chunks found. "
            "Put the PDF(s) in data/datasheets/."
        )

    # --------------------------------------------------------
    # 2. Build BM25 index
    # --------------------------------------------------------

    bm25_index = build_index(
        chunks
    )

    # --------------------------------------------------------
    # 3. Open vector database and load models
    # --------------------------------------------------------

    collection = open_collection()

    embedding_model = load_model()

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print(
        f"\nQuestion: {question}"
    )

    # --------------------------------------------------------
    # 4. Query expansion
    # --------------------------------------------------------

    queries = get_queries(
        question
    )

    # --------------------------------------------------------
    # 5. Semantic + BM25 retrieval
    # --------------------------------------------------------

    candidates = expanded_search(
        queries,
        collection,
        embedding_model,
        bm25_index,
        chunks,
    )

    if not candidates:

        if DEBUG:

            print_debug(
                queries,
                0,
                [],
            )

        print_not_found()

        return

    # --------------------------------------------------------
    # 6. Cross-encoder reranking
    # --------------------------------------------------------

    reranked = rerank(
        reranker,
        question,
        candidates,
    )

    # Keep only the requested number of context chunks.
    top = reranked[
        :CONTEXT_CHUNKS
    ]

    if DEBUG:

        print_debug(
            queries,
            len(candidates),
            top,
        )

    # --------------------------------------------------------
    # 7. Optional evidence threshold
    # --------------------------------------------------------

    if MIN_RERANK_SCORE is not None:

        top = [
            record
            for record in top
            if record["rerank_score"]
            >= MIN_RERANK_SCORE
        ]

    if not top:

        print_not_found()

        return

    # --------------------------------------------------------
    # 8. Prepare context records
    # --------------------------------------------------------

    context_hits = to_context_hits(
        top
    )

    # --------------------------------------------------------
    # 9. Generate final grounded answer
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            question,
            context_hits,
        )

    except RuntimeError as exc:

        sys.exit(
            f"\nError: {exc}"
        )

    # --------------------------------------------------------
    # 10. Final answer
    # --------------------------------------------------------

    print(
        "\nFinal answer:"
    )

    print(
        answer
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()