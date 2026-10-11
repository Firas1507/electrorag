"""Semantic retriever for ElectroRAG (M5): finds the chunks most similar to a question.

Read-only: it opens the existing ChromaDB collection and never writes to it.
No LLM is involved yet; this only prints the retrieved chunks.

Run (from the ElectroRAG root):
    python -m src.retriever "What is the input voltage range of the TPS63020?"
"""

import argparse
import sys

import chromadb

from src.embeddings import MODEL_NAME, load_model
from src.vector_store import COLLECTION_NAME, DB_PATH

TOP_K = 5


def open_collection():
    """Open the existing collection (get_collection never creates anything)."""
    if not DB_PATH.exists():
        sys.exit(f"Database not found at {DB_PATH.resolve()}. Run: python -m src.vector_store")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        sys.exit(f"Collection '{COLLECTION_NAME}' not found. Run: python -m src.vector_store")

    if collection.count() == 0:
        sys.exit(f"Collection '{COLLECTION_NAME}' is empty. Run: python -m src.vector_store")
    return collection


def retrieve(collection, model, question: str, top_k: int = TOP_K) -> list:
    """Embed the question and return the top_k most similar chunks."""
    query_embedding = model.encode([question], show_progress_bar=False)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns one list per query; we sent a single query, so take [0].
    hits = []
    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"chunk_id": chunk_id, "text": text, "metadata": metadata or {}, "distance": distance})
    return hits


def print_hits(hits: list) -> None:
    for rank, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        print("=" * 80)
        print(f"Rank:      {rank}")
        print(f"Distance:  {hit['distance']:.4f}  (lower = more similar)")
        print(f"Source:    {meta.get('source')}")
        print(f"Page:      {meta.get('page')}")
        if meta.get("section"):
            print(f"Section:   {meta['section']}")
        print(f"Chunk ID:  {hit['chunk_id']}")
        print("Text:")
        print(hit["text"])
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over the ElectroRAG datasheet chunks.")
    parser.add_argument("question", nargs="+", help="Your question (quotes optional)")
    question = " ".join(parser.parse_args().question)

    collection = open_collection()
    model = load_model()

    print(f"\nQuestion:   {question}")
    print(f"Model:      {MODEL_NAME}")
    print(f"Collection: {collection.name} ({collection.count()} records)\n")

    print_hits(retrieve(collection, model, question))


if __name__ == "__main__":
    main()