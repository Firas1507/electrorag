"""Vector store for ElectroRAG (M4): embeds all chunks and stores them in ChromaDB.

Install:  pip install chromadb sentence-transformers
Run (from the ElectroRAG root):  python -m src.vector_store

Safe to rerun: records are written with upsert, so a chunk ID that already
exists is updated in place instead of being duplicated.
"""

import logging
from pathlib import Path
from typing import Any, Dict

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Load PDFs -> chunk them (via the existing pipeline).
    chunks = get_all_chunks()

    # 2. Embed every chunk text with all-MiniLM-L6-v2.
    model = load_model()
    embeddings = embed_texts(model, [chunk["text"] for chunk in chunks])

    # 3. Upsert into ChromaDB (insert new IDs, replace existing ones).
    collection = get_collection()
    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[build_metadata(chunk) for chunk in chunks],
    )

    print(f"\nNumber of chunks:           {len(chunks)}")
    print(f"Embedding dimension:        {model.get_embedding_dimension()}")
    print(f"Collection name:            {collection.name}")
    print(f"Records after insertion:    {collection.count()}")
    print(f"Database path:              {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()