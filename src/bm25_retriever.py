"""BM25 keyword retriever for ElectroRAG (experiment).

Builds an in-memory BM25 index over the existing datasheet chunks and prints the
top matches for a question. It does not use ChromaDB, embeddings or the LLM.

Install:  python -m pip install rank-bm25
Run (from the ElectroRAG root):
    python -m src.bm25_retriever "What is the function of the EN pin of the TPS63020?"
"""

import re
import sys

from rank_bm25 import BM25Okapi

from src.chunker import chunk_documents
from src.pdf_loader import load_all_pdfs

TOP_K = 10

# Very common words carry no keyword signal; dropping them keeps scores focused.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "do", "for",
    "from", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "with",
}


def tokenize(text: str) -> list:
    """Lowercase and split into alphanumeric tokens (so "TPS63020" and "EN" stay whole)."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def build_index(chunks: list) -> BM25Okapi:
    """Build a BM25 index from the text of every chunk."""
    return BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])


def search(index: BM25Okapi, chunks: list, question: str, top_k: int = TOP_K) -> list:
    """Return the top_k (score, chunk) pairs, best first."""
    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    scores = index.get_scores(query_tokens)  # one BM25 score per chunk
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [(float(scores[i]), chunks[i]) for i in ranked[:top_k]]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.bm25_retriever "Your question here"')
    question = " ".join(sys.argv[1:])

    chunks = chunk_documents(load_all_pdfs())
    if not chunks:
        sys.exit("No chunks found. Put the PDF(s) in data/datasheets/.")

    index = build_index(chunks)
    results = search(index, chunks, question)

    print(f"\nQuestion:     {question}")
    print(f"Query tokens: {tokenize(question)}")
    print(f"Chunks indexed: {len(chunks)}\n")

    if not results:
        sys.exit("The question has no searchable keywords.")

    for rank, (score, chunk) in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank:       {rank}")
        print(f"BM25 score: {score:.4f}")
        print(f"Source:     {chunk['source']}")
        print(f"Page:       {chunk['page']}")
        print(f"Chunk ID:   {chunk['chunk_id']}")
        print("Text:")
        print(chunk["text"])
    print("=" * 80)


if __name__ == "__main__":
    main()