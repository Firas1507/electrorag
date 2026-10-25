"""Vector store for ElectroRAG (M4): embeds all chunks and stores them in ChromaDB.

Install:  pip install chromadb sentence-transformers
Run (from the ElectroRAG root):  python -m src.vector_store

Safe to rerun: records are written with upsert, so a chunk ID that already
exists is updated in place instead of being duplicated. Obsolete chunks are
removed only for source documents included in a successful indexing run.
Missing source PDFs require an explicit remove_document call.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import chromadb

from src.embeddings import embed_texts, get_all_chunks, load_model

DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "electrorag"


def get_collection():
    """Open (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def build_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Chunk fields to store as metadata.

    - "text" and "chunk_id" are skipped: they are stored as the document and the ID.
    - None values are skipped: ChromaDB metadata only accepts str/int/float/bool.
    """
    skip = {"text", "chunk_id"}
    return {k: v for k, v in chunk.items() if k not in skip and v is not None}


def remove_document(collection, source: str) -> int:
    """Remove records owned by one explicitly named source document.

    Missing documents are never inferred to be deletions. Call this function
    only when removal of the named source is intentional.
    """
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty document name")

    result = collection.get(where={"source": source}, include=["metadatas"])
    ids = result["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def _as_embedding_rows(embeddings: Any) -> List[List[float]]:
    """Convert array-like embeddings to Chroma-compatible nested lists."""
    if hasattr(embeddings, "tolist"):
        rows = embeddings.tolist()
    else:
        rows = [
            row.tolist() if hasattr(row, "tolist") else list(row)
            for row in embeddings
        ]
    return rows


def synchronize_chunks(
    collection,
    chunks: Sequence[Mapping[str, Any]],
    embeddings: Any,
) -> None:
    """Upsert chunks and prune obsolete records within each represented source.

    All inputs are validated before any collection mutation. Records for
    sources absent from ``chunks`` are deliberately left untouched; deleting
    those requires an explicit ``remove_document`` call.
    """
    if not chunks:
        raise ValueError("Cannot synchronize an empty chunk set")

    embedding_rows = _as_embedding_rows(embeddings)
    if len(chunks) != len(embedding_rows):
        raise ValueError(
            f"Expected one embedding per chunk; got {len(embedding_rows)} "
            f"embeddings for {len(chunks)} chunks"
        )

    grouped: Dict[str, List[Tuple[Mapping[str, Any], List[float]]]] = {}
    seen_ids = set()
    for chunk, embedding in zip(chunks, embedding_rows):
        source = chunk.get("source")
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        if not isinstance(source, str) or not source:
            raise ValueError("Every chunk must have a non-empty source")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Every chunk from {source!r} must have a non-empty chunk_id")
        if not isinstance(text, str):
            raise ValueError(f"Every chunk from {source!r} must have text")
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")
        seen_ids.add(chunk_id)
        grouped.setdefault(source, []).append((chunk, embedding))

    obsolete_ids = set()
    for source, source_items in grouped.items():
        desired_ids = {item[0]["chunk_id"] for item in source_items}
        existing = collection.get(
            where={"source": source},
            include=["metadatas"],
        )
        obsolete_ids.update(set(existing["ids"]) - desired_ids)

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embedding_rows,
        metadatas=[build_metadata(dict(chunk)) for chunk in chunks],
    )

    if obsolete_ids:
        collection.delete(ids=sorted(obsolete_ids))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Finish strict extraction and embedding before changing the collection.
    chunks = get_all_chunks(strict=True)

    model = load_model()
    embeddings = embed_texts(model, [chunk["text"] for chunk in chunks])

    collection = get_collection()
    synchronize_chunks(collection, chunks, embeddings)

    print(f"\nNumber of chunks:           {len(chunks)}")
    print(f"Embedding dimension:        {model.get_embedding_dimension()}")
    print(f"Collection name:            {collection.name}")
    print(f"Records after synchronization: {collection.count()}")
    print(f"Database path:              {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()