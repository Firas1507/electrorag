"""Hybrid retriever for ElectroRAG (experiment): semantic search + BM25, merged with RRF.

Two independent searches are run for the question:
    A. Semantic: top 10 from the ChromaDB collection (read-only)
    B. BM25:     top 10 over the chunks produced by the PDF loader + chunker
Reciprocal Rank Fusion then merges them into one ranked list.
No LLM / Ollama is used.

Run (from the ElectroRAG root):
    python -m src.hybrid_retriever "What is the function of the EN pin of the TPS63020?"
"""

import sys

from src.bm25_retriever import build_index, search as bm25_search, tokenize
from src.chunker import chunk_documents
from src.embeddings import load_model
from src.pdf_loader import load_all_pdfs
from src.retriever import open_collection, retrieve

SEMANTIC_K = 10
BM25_K = 10
FINAL_K = 10
RRF_K = 60  # RRF score = 1 / (RRF_K + rank), rank starting at 1


def rrf_score(rank: int) -> float:
    """Reciprocal Rank Fusion contribution of one result list."""
    return 1.0 / (RRF_K + rank)


def fuse(semantic_hits: list, bm25_results: list) -> list:
    """Merge both ranked lists by chunk ID, adding up the RRF scores of each chunk."""
    merged = {}  # chunk_id -> record

    def get_record(chunk_id: str, text: str, source, page) -> dict:
        if chunk_id not in merged:
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "text": text,
                "source": source,
                "page": page,
                "semantic_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }
        return merged[chunk_id]

    # A. Semantic results (already ordered best first).
    for rank, hit in enumerate(semantic_hits, start=1):
        meta = hit["metadata"]
        record = get_record(hit["chunk_id"], hit["text"], meta.get("source"), meta.get("page"))
        record["semantic_rank"] = rank
        record["rrf_score"] += rrf_score(rank)

    # B. BM25 results, ordered best first. Chunks with score 0 share no keyword
    #    with the question, so they are not counted as a real BM25 match.
    rank = 0
    for score, chunk in bm25_results:
        if score <= 0:
            continue
        rank += 1
        record = get_record(chunk["chunk_id"], chunk["text"], chunk["source"], chunk["page"])
        record["bm25_rank"] = rank
        record["rrf_score"] += rrf_score(rank)

    # Highest fused score first; ties go to the chunk with the better single rank.
    def sort_key(record: dict):
        ranks = [r for r in (record["semantic_rank"], record["bm25_rank"]) if r is not None]
        return (-record["rrf_score"], min(ranks))

    return sorted(merged.values(), key=sort_key)


def print_results(results: list) -> None:
    for position, record in enumerate(results, start=1):
        semantic = record["semantic_rank"] if record["semantic_rank"] is not None else "-"
        bm25 = record["bm25_rank"] if record["bm25_rank"] is not None else "-"

        print("=" * 80)
        print(f"Final rank:    {position}")
        print(f"RRF score:     {record['rrf_score']:.5f}")
        print(f"Semantic rank: {semantic}")
        print(f"BM25 rank:     {bm25}")
        print(f"Source:        {record['source']}")
        print(f"Page:          {record['page']}")
        print(f"Chunk ID:      {record['chunk_id']}")
        print("Text:")
        print(record["text"])
    print("=" * 80)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.hybrid_retriever "Your question here"')
    question = " ".join(sys.argv[1:])

    # Chunks for BM25 (same loader + chunker as the rest of the project).
    chunks = chunk_documents(load_all_pdfs())
    if not chunks:
        sys.exit("No chunks found. Put the PDF(s) in data/datasheets/.")
    bm25_index = build_index(chunks)

    # Existing ChromaDB collection + embedding model (read-only).
    collection = open_collection()
    embedding_model = load_model()

    semantic_hits = retrieve(collection, embedding_model, question, top_k=SEMANTIC_K)
    bm25_results = bm25_search(bm25_index, chunks, question, top_k=BM25_K)

    results = fuse(semantic_hits, bm25_results)[:FINAL_K]

    print(f"\nQuestion:     {question}")
    print(f"BM25 tokens:  {tokenize(question)}")
    print(f"Semantic hits: {len(semantic_hits)} | BM25 hits: {sum(1 for s, _ in bm25_results if s > 0)}")
    print(f"Fused candidates shown: {len(results)}\n")

    print_results(results)


if __name__ == "__main__":
    main()