"""
LEGACY: superseded by src/benchmark.py (which uses the real generator prompt).
This file keeps its own older prompt and is not representative of production.

ElectroRAG V1 - End-to-End RAG Evaluation

Run:

    python -m src.evaluate_rag

Pipeline:

    Question
        ↓
    Query expansion
        ↓
    Semantic + BM25 retrieval
        ↓
    Deduplication
        ↓
    CrossEncoder reranking
        ↓
    Top 5 chunks
        ↓
    Local Qwen3:8b
        ↓
    [C1], [C2], ...
        ↓
    Deterministic citation resolver
        ↓
    Final answer with PDF/page citations
"""

import re

from sentence_transformers import CrossEncoder

from src.bm25_retriever import build_index
from src.chunker import chunk_documents
from src.embeddings import load_model
from src.llm import stream_answer
from src.pdf_loader import load_all_pdfs
from src.query_expansion_retriever import (
    EXPANSIONS,
    expanded_search,
)
from src.reranker import (
    RERANKER_MODEL,
    rerank,
)
from src.retriever import open_collection
from src.citation_resolver import resolve_citations


# ============================================================
# Configuration
# ============================================================

CONTEXT_CHUNKS = 5

NOT_FOUND_MESSAGE = (
    "I could not find this information in the indexed datasheet."
)


# ============================================================
# LLM instructions
# ============================================================

INSTRUCTIONS = f"""
You are a technical assistant specialized in electronics datasheets.

STRICT RULES:

1. Answer ONLY using the provided context.

2. Do NOT use outside knowledge.

3. Do NOT guess or invent specifications.

4. If the context does not contain enough information to answer
the question, reply exactly:

{NOT_FOUND_MESSAGE}

5. Keep the answer concise and technically precise.

6. EVERY factual statement must be supported by one or more
context citations.

7. Cite evidence ONLY using context IDs:

[C1]
[C2]
[C3]
[C4]
[C5]

8. NEVER write PDF filenames or page numbers yourself.

9. NEVER create citations such as:

[TPS63020.pdf, p. 6]

10. ONLY use context IDs that actually exist.

11. Prefer the context that most directly supports the statement.

12. Do not cite a context that does not support the statement.

13. If the answer cannot be supported by the context,
use the exact NOT FOUND response.

14. Do not explain your reasoning process.

15. Give only the concise technical answer with citations.
"""


# ============================================================
# Benchmark questions
# ============================================================

TESTS = [

    {
        "question":
            "What is the input voltage range of the TPS63020?",
        "acceptable_pages": {1, 6},
        "keywords": ["1.8", "5.5"],
        "should_refuse": False,
    },

    {
        "question":
            "What is the adjustable output voltage range of the TPS63020?",
        "acceptable_pages": {1, 6, 15},
        "keywords": ["1.2", "5.5"],
        "should_refuse": False,
    },

    {
        "question":
            "What is the switching frequency of the TPS63020?",
        "acceptable_pages": {1, 6},
        "keywords": ["2.4", "MHz"],
        "should_refuse": False,
    },

    {
        "question":
            "What is the function of the EN pin of the TPS63020?",
        "acceptable_pages": {4, 9},
        "keywords": ["enable"],
        "should_refuse": False,
    },

    {
        "question":
            "What is the function of the PS/SYNC pin of the TPS63020?",
        "acceptable_pages": {4, 12},
        "keywords": ["power", "save"],
        "should_refuse": False,
    },

    {
        "question":
            "What is the function of the FB pin of the TPS63020?",
        "acceptable_pages": {4},
        "keywords": ["feedback"],
        "should_refuse": False,
    },

    {
        "question":
            "What is the function of the PG pin of the TPS63020?",
        "acceptable_pages": {4},
        "keywords": ["power", "good"],
        "should_refuse": False,
    },

    {
        "question":
            "How many pins does the TPS63020 have?",
        "acceptable_pages": {1, 4},
        "keywords": ["14"],
        "should_refuse": False,
    },

    {
        "question":
            "What inductance value is recommended for the TPS63020?",
        "acceptable_pages": {14, 15},
        "keywords": ["1", "µH"],
        "should_refuse": False,
    },

    {
        "question":
            "How does the TPS63020 behave at light load?",
        "acceptable_pages": {1, 9, 12},
        "keywords": ["power", "save"],
        "should_refuse": False,
    },

    {
        "question":
            "How does the TPS63020 behave at heavy load?",
        "acceptable_pages": {1, 12},
        "keywords": ["PWM"],
        "should_refuse": False,
    },

    {
        "question":
            "What is the recommended input voltage for the TPS63020?",
        "acceptable_pages": {1, 6},
        "keywords": ["1.8", "5.5"],
        "should_refuse": False,
    },

    {
        "question":
            "What happens when EN is connected to GND?",
        "acceptable_pages": {9},
        "keywords": ["shutdown"],
        "should_refuse": False,
    },

    {
        "question":
            "What is the weight of the TPS63020?",
        "acceptable_pages": set(),
        "keywords": [],
        "should_refuse": True,
    },

    {
        "question":
            "What is the price of the TPS63020?",
        "acceptable_pages": set(),
        "keywords": [],
        "should_refuse": True,
    },
]


# ============================================================
# Helpers
# ============================================================

def normalize(text: str) -> str:
    """Normalize text for keyword comparisons."""

    return " ".join(
        text.lower().split()
    )


def get_queries(question: str) -> list:
    """
    Use the project's manual query expansions when available.
    Otherwise use the original question.
    """

    normalized_question = normalize(
        question
    )

    for known_question, expansion_data in EXPANSIONS.items():

        if isinstance(expansion_data, tuple):
            queries = expansion_data[-1]
        else:
            queries = expansion_data

        if normalize(known_question) == normalized_question:
            return queries

    return [question]


def build_context(records: list) -> str:
    """
    Build numbered context blocks.

    C1 corresponds to records[0],
    C2 corresponds to records[1], etc.
    """

    blocks = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        blocks.append(
            f"[C{index}]\n"
            f"Source: {record['source']}\n"
            f"Page: {record['page']}\n"
            f"Content:\n{record['text']}"
        )

    return "\n\n---\n\n".join(
        blocks
    )


def build_prompt(
    question: str,
    records: list,
) -> str:
    """Build grounded prompt."""

    context = build_context(
        records
    )

    return f"""
{INSTRUCTIONS}

================ CONTEXT ================

{context}

================ QUESTION ================

{question}

================ ANSWER ================
""".strip()


def extract_context_ids(
    answer: str,
) -> set:
    """
    Extract LLM-generated context IDs such as C1, C2, C5.
    """

    matches = re.findall(
        r"\[C(\d+)\]",
        answer,
        re.IGNORECASE,
    )

    return {
        f"C{number}"
        for number in matches
    }


def context_ids_to_pages(
    context_ids: set,
    records: list,
) -> set:
    """
    Convert C1/C2/... into actual PDF page numbers.
    """

    pages = set()

    for context_id in context_ids:

        match = re.fullmatch(
            r"C(\d+)",
            context_id,
            re.IGNORECASE,
        )

        if not match:
            continue

        index = int(
            match.group(1)
        ) - 1

        if 0 <= index < len(records):

            pages.add(
                int(records[index]["page"])
            )

    return pages


def extract_resolved_pages(
    answer: str,
) -> set:
    """
    Extract resolved citations from the final answer.

    Example:

        [TPS63020.pdf, p. 6]

    -> {6}
    """

    pattern = (
        r"\["
        r"[^\],]+"
        r",\s*p\.\s*"
        r"(\d+)"
        r"\]"
    )

    return {
        int(page)
        for page in re.findall(
            pattern,
            answer,
            re.IGNORECASE,
        )
    }


# ============================================================
# Single benchmark test
# ============================================================

def run_test(
    test,
    collection,
    embedding_model,
    bm25_index,
    chunks,
    reranker,
):

    question = test["question"]
    acceptable_pages = test["acceptable_pages"]
    keywords = test["keywords"]
    should_refuse = test["should_refuse"]

    # --------------------------------------------------------
    # 1. Query expansion
    # --------------------------------------------------------

    queries = get_queries(
        question
    )

    # --------------------------------------------------------
    # 2. Semantic + BM25 retrieval
    # --------------------------------------------------------

    candidates = expanded_search(
        queries,
        collection,
        embedding_model,
        bm25_index,
        chunks,
    )

    # --------------------------------------------------------
    # 3. CrossEncoder reranking
    # --------------------------------------------------------

    if candidates:

        ranked = rerank(
            reranker,
            question,
            candidates,
        )

        top = ranked[
            :CONTEXT_CHUNKS
        ]

    else:

        top = []

    # --------------------------------------------------------
    # 4. Generate answer
    # --------------------------------------------------------

    if not top:

        raw_answer = NOT_FOUND_MESSAGE

    else:

        prompt = build_prompt(
            question,
            top,
        )

        pieces = []

        for piece in stream_answer(
            prompt
        ):
            pieces.append(
                piece
            )

        raw_answer = "".join(
            pieces
        ).strip()

        if not raw_answer:

            raw_answer = (
                NOT_FOUND_MESSAGE
            )

    # --------------------------------------------------------
    # 5. Resolve citations
    # --------------------------------------------------------

    if raw_answer == NOT_FOUND_MESSAGE:

        answer = raw_answer

    else:

        answer = resolve_citations(
            raw_answer,
            [
                {
                    "source": record["source"],
                    "page": record["page"],
                    "text": record["text"],
                }
                for record in top
            ],
        )

    # --------------------------------------------------------
    # 6. Retrieval evaluation
    # --------------------------------------------------------

    top_pages = {
        int(record["page"])
        for record in top
    }

    if should_refuse:

        retrieval_pass = True

    else:

        retrieval_pass = bool(
            acceptable_pages
            & top_pages
        )

    # --------------------------------------------------------
    # 7. Answer evaluation
    # --------------------------------------------------------

    answer_normalized = normalize(
        answer
    )

    if should_refuse:

        answer_pass = (
            answer
            == NOT_FOUND_MESSAGE
        )

    else:

        answer_pass = all(
            normalize(keyword)
            in answer_normalized
            for keyword in keywords
        )

    # --------------------------------------------------------
    # 8. Citation evaluation
    # --------------------------------------------------------

    citation_pages = extract_resolved_pages(
        answer
    )

    if should_refuse:

        citation_pass = True

    else:

        citation_pass = bool(
            acceptable_pages
            & citation_pages
        )

    # --------------------------------------------------------
    # 9. Citation ID validity
    # --------------------------------------------------------

    context_ids = extract_context_ids(
        raw_answer
    )

    valid_context_ids = {
        f"C{index}"
        for index in range(
            1,
            len(top) + 1,
        )
    }

    invalid_context_ids = (
        context_ids
        - valid_context_ids
    )

    context_id_pass = (
        not invalid_context_ids
    )

    # --------------------------------------------------------
    # 10. Return result
    # --------------------------------------------------------

    return {
        "question": question,
        "queries": queries,
        "candidates": len(candidates),
        "top": top,
        "top_pages": top_pages,
        "raw_answer": raw_answer,
        "answer": answer,
        "context_ids": context_ids,
        "invalid_context_ids":
            invalid_context_ids,
        "citation_pages":
            citation_pages,
        "retrieval_pass":
            retrieval_pass,
        "answer_pass":
            answer_pass,
        "citation_pass":
            citation_pass,
        "context_id_pass":
            context_id_pass,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("ElectroRAG V1 - END-TO-END EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load everything once
    # --------------------------------------------------------

    print(
        "\n[1/5] Loading datasheets..."
    )

    documents = load_all_pdfs()

    print(
        "\n[2/5] Creating chunks..."
    )

    chunks = chunk_documents(
        documents
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        "\n[3/5] Building BM25 index..."
    )

    bm25_index = build_index(
        chunks
    )

    print(
        "\n[4/5] Opening ChromaDB + embedding model..."
    )

    collection = open_collection()

    embedding_model = load_model()

    print(
        "\n[5/5] Loading CrossEncoder..."
    )

    print(
        f"Model: {RERANKER_MODEL}"
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print(
        "\nAll components loaded successfully."
    )

    # --------------------------------------------------------
    # Run tests
    # --------------------------------------------------------

    results = []

    for index, test in enumerate(
        TESTS,
        start=1,
    ):

        print("\n")
        print("=" * 80)
        print(
            f"TEST {index}/{len(TESTS)}"
        )
        print("=" * 80)

        print(
            f"Question: {test['question']}"
        )

        result = run_test(
            test,
            collection,
            embedding_model,
            bm25_index,
            chunks,
            reranker,
        )

        results.append(
            result
        )

        print(
            f"\nQueries used: "
            f"{len(result['queries'])}"
        )

        print(
            f"Candidates before reranking: "
            f"{result['candidates']}"
        )

        print(
            "\nTop reranked chunks:"
        )

        for rank, record in enumerate(
            result["top"],
            start=1,
        ):

            score = record.get(
                "rerank_score",
                record.get(
                    "score",
                    "N/A",
                ),
            )

            print(
                f"  {rank}. "
                f"page={record['page']} "
                f"score={score}"
            )

        print(
            "\nRaw LLM answer:"
        )

        print(
            result["raw_answer"]
        )

        print(
            "\nFinal resolved answer:"
        )

        print(
            result["answer"]
        )

        print(
            "\nTop pages: "
            f"{sorted(result['top_pages'])}"
        )

        print(
            "Citation pages: "
            f"{sorted(result['citation_pages'])}"
        )

        print(
            "Context IDs: "
            f"{sorted(result['context_ids'])}"
        )

        if result["invalid_context_ids"]:

            print(
                "INVALID context IDs: "
                f"{sorted(result['invalid_context_ids'])}"
            )

        print(
            "\nEvaluation:"
        )

        print(
            "  Retrieval : "
            + (
                "PASS"
                if result["retrieval_pass"]
                else "FAIL"
            )
        )

        print(
            "  Answer    : "
            + (
                "PASS"
                if result["answer_pass"]
                else "FAIL"
            )
        )

        print(
            "  Citation  : "
            + (
                "PASS"
                if result["citation_pass"]
                else "FAIL"
            )
        )

        print(
            "  Context ID: "
            + (
                "PASS"
                if result["context_id_pass"]
                else "FAIL"
            )
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    retrieval_total = sum(
        result["retrieval_pass"]
        for result in results
    )

    answer_total = sum(
        result["answer_pass"]
        for result in results
    )

    citation_total = sum(
        result["citation_pass"]
        for result in results
    )

    context_id_total = sum(
        result["context_id_pass"]
        for result in results
    )

    overall_total = sum(
        result["retrieval_pass"]
        and result["answer_pass"]
        and result["citation_pass"]
        and result["context_id_pass"]
        for result in results
    )

    total = len(results)

    print("\n\n")
    print("=" * 80)
    print(
        "ElectroRAG V1 - FINAL BENCHMARK"
    )
    print("=" * 80)

    print(
        f"\nRetrieval : "
        f"{retrieval_total}/{total}"
    )

    print(
        f"Answer    : "
        f"{answer_total}/{total}"
    )

    print(
        f"Citation  : "
        f"{citation_total}/{total}"
    )

    print(
        f"Context ID: "
        f"{context_id_total}/{total}"
    )

    print(
        f"Overall   : "
        f"{overall_total}/{total}"
    )

    print("\n")

    print(
        f"{'#':<4}"
        f"{'Retrieval':<12}"
        f"{'Answer':<10}"
        f"{'Citation':<10}"
        f"{'C-ID':<8}"
        f"Question"
    )

    print(
        "-" * 90
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        retrieval = (
            "PASS"
            if result["retrieval_pass"]
            else "FAIL"
        )

        answer = (
            "PASS"
            if result["answer_pass"]
            else "FAIL"
        )

        citation = (
            "PASS"
            if result["citation_pass"]
            else "FAIL"
        )

        context_id = (
            "PASS"
            if result["context_id_pass"]
            else "FAIL"
        )

        print(
            f"{index:<4}"
            f"{retrieval:<12}"
            f"{answer:<10}"
            f"{citation:<10}"
            f"{context_id:<8}"
            f"{result['question']}"
        )

    # --------------------------------------------------------
    # Failure analysis
    # --------------------------------------------------------

    failures = [
        (
            index,
            result,
        )
        for index, result in enumerate(
            results,
            start=1,
        )
        if not (
            result["retrieval_pass"]
            and result["answer_pass"]
            and result["citation_pass"]
            and result["context_id_pass"]
        )
    ]

    print("\n")
    print("=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)

    if not failures:

        print(
            "\nNo failures detected."
        )

    else:

        for index, result in failures:

            print(
                f"\nTEST {index}: "
                f"{result['question']}"
            )

            print(
                f"Answer: "
                f"{result['answer']}"
            )

            print(
                f"Top pages: "
                f"{sorted(result['top_pages'])}"
            )

            print(
                f"Citation pages: "
                f"{sorted(result['citation_pages'])}"
            )

            print(
                f"Context IDs: "
                f"{sorted(result['context_ids'])}"
            )

            if result["invalid_context_ids"]:

                print(
                    f"Invalid IDs: "
                    f"{sorted(result['invalid_context_ids'])}"
                )

    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":
    main()