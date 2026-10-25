"""Save uploaded datasheet PDFs and add their chunks to the existing index."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Iterable, Protocol

from gui import config


class UploadedPdf(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def index_uploaded_pdfs(uploads: Iterable[UploadedPdf]) -> list[str]:
    """Validate, store, and index uploaded PDFs; return their saved filenames."""
    files = list(uploads)
    if not files:
        return []

    names = [upload.name for upload in files]
    if any(
        not name
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        for name in names
    ):
        raise ValueError("Each upload must have a valid filename.")
    if any(Path(name).suffix.lower() != ".pdf" for name in names):
        raise ValueError("Only PDF datasheets are supported.")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Selected files must have unique filenames.")

    pdf_dir = config.PDF_DIR
    existing = [name for name in names if (pdf_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "A datasheet with this name already exists: " + ", ".join(existing)
        )

    contents = [bytes(upload.getvalue()) for upload in files]
    pdf_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    published: list[Path] = []
    chunk_ids: list[str] = []
    collection = None
    indexed = False

    try:
        for content in contents:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pdf", prefix=".electrorag-upload-",
                dir=pdf_dir, delete=False,
            ) as temporary:
                temporary.write(content)
                staged.append(Path(temporary.name))

        project_root = str(config.PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.chunker import chunk_documents
        from src.embeddings import load_model
        from src.pdf_loader import load_pdf

        documents = []
        for staged_path, name in zip(staged, names):
            pages = load_pdf(staged_path)
            if not pages:
                raise ValueError(f"{name} contains no extractable PDF text.")
            for page in pages:
                page["source"] = name
            documents.extend(pages)

        chunks = chunk_documents(documents)
        if not chunks:
            raise ValueError("The uploaded PDFs did not contain any indexable text.")

        model = load_model()
        embeddings = model.encode(
            [chunk["text"] for chunk in chunks],
            show_progress_bar=True,
        )

        import chromadb

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        collection = client.get_or_create_collection(name=config.COLLECTION_NAME)
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        if collection.get(ids=chunk_ids).get("ids"):
            raise FileExistsError(
                "The index already contains chunks with these filenames. "
                "Choose different PDF filenames."
            )

        from src.vector_store import build_metadata

        collection.upsert(
            ids=chunk_ids,
            documents=[chunk["text"] for chunk in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[build_metadata(chunk) for chunk in chunks],
        )
        indexed = True

        for staged_path, name in zip(staged, names):
            target = pdf_dir / name
            if target.exists():
                raise FileExistsError(f"{name} appeared in the library during upload.")
            staged_path.rename(target)
            published.append(target)

        return names
    except Exception:
        if indexed and collection is not None:
            collection.delete(ids=chunk_ids)
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
