"""Local embedding test for ElectroRAG (M3.2).

Runs the existing loader + chunker pipeline, then embeds ALL chunks in a
single call with a small local sentence-transformers model.

Install:  pip install sentence-transformers
Run (from the ElectroRAG root):  python -m src.embeddings
"""

import logging
from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer

from src.chunker import chunk_documents
from src.pdf_loader import load_all_pdfs

# Small, fast, general-purpose model: 384-dimensional embeddings, ~90 MB.
# The first run downloads it once into the local Hugging Face cache;
# after that it loads from disk and runs fully offline on your machine.
MODEL_NAME = "all-MiniLM-L6-v2"


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the embedding model (runs locally on CPU/GPU, no API calls)."""
    return SentenceTransformer(model_name)


def get_all_chunks() -> List[Dict[str, Any]]:
    """Run the loader + chunker and return every chunk."""
    pages = load_all_pdfs()
    chunks = chunk_documents(pages)
    if not chunks:
        raise RuntimeError("No chunks produced. Put at least one text PDF in data/datasheets/.")
    return chunks


def embed_texts(model: SentenceTransformer, texts: List[str]):
    """Embed a list of strings in one call; returns a NumPy array (n_texts, dimension)."""
    return model.encode(texts, show_progress_bar=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    chunks = get_all_chunks()
    texts = [chunk["text"] for chunk in chunks]

    model = load_model()
    embeddings = embed_texts(model, texts)

    print(f"\nModel name:             {MODEL_NAME}")
    print(f"Number of chunks:       {len(chunks)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Embedding dimension:    {model.get_embedding_dimension()}")
    print(f"First 5 values of first embedding: {embeddings[0][:5].tolist()}")


if __name__ == "__main__":
    main()