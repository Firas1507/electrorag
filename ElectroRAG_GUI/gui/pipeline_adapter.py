"""Adapter connecting the Streamlit GUI to ElectroRAG's existing RAG pipeline."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from gui import config
from gui.models import Answer, Citation


@dataclass
class RagPipeline:
    chunks: list[dict[str, Any]]
    bm25_index: Any
    collection: Any
    embedding_model: Any
    reranker: Any
    rerank: Any
    rag_module: Any
    generator_module: Any
    validator_module: Any
    llm_module: Any


def load_pipeline() -> RagPipeline:
    """Load the existing index and local models once, on the first live query."""
    project_root = str(config.PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import chromadb
    from sentence_transformers import CrossEncoder

    from src import citation_validation, generator, llm, rag
    from src.bm25_retriever import build_index
    from src.chunker import chunk_documents
    from src.embeddings import load_model
    from src.pdf_loader import load_all_pdfs
    from src.reranker import RERANKER_MODEL, rerank

    if not config.CHROMA_DIR.is_dir():
        raise FileNotFoundError(f"ChromaDB index not found: {config.CHROMA_DIR}")
    if not config.PDF_DIR.is_dir():
        raise FileNotFoundError(f"Datasheet directory not found: {config.PDF_DIR}")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        collection = client.get_collection(name=config.COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open ChromaDB collection {config.COLLECTION_NAME!r}: {exc}"
        ) from exc
    if collection.count() == 0:
        raise RuntimeError(f"ChromaDB collection {config.COLLECTION_NAME!r} is empty.")

    chunks = chunk_documents(load_all_pdfs(config.PDF_DIR))
    if not chunks:
        raise RuntimeError(f"No text chunks found in {config.PDF_DIR}.")

    return RagPipeline(
        chunks=chunks,
        bm25_index=build_index(chunks),
        collection=collection,
        embedding_model=load_model(),
        reranker=CrossEncoder(RERANKER_MODEL),
        rerank=rerank,
        rag_module=rag,
        generator_module=generator,
        validator_module=citation_validation,
        llm_module=llm,
    )


def run_query(pipeline: RagPipeline, question: str) -> Answer:
    """Run retrieval, reranking and citation-validated generation for one question."""
    candidates = pipeline.rag_module.expanded_search(
        pipeline.rag_module.get_queries(question),
        pipeline.collection,
        pipeline.embedding_model,
        pipeline.bm25_index,
        pipeline.chunks,
    )
    if not candidates:
        return Answer(
            question,
            "unsupported",
            message="No supporting evidence was found in the indexed datasheets.",
        )

    reranked = pipeline.rerank(pipeline.reranker, question, candidates)
    top = reranked[:pipeline.rag_module.CONTEXT_CHUNKS]
    if pipeline.rag_module.MIN_RERANK_SCORE is not None:
        top = [
            record
            for record in top
            if record["rerank_score"] >= pipeline.rag_module.MIN_RERANK_SCORE
        ]
    if not top:
        return Answer(
            question,
            "unsupported",
            message="No supporting evidence was found in the indexed datasheets.",
        )

    context = pipeline.rag_module.to_context_hits(top)
    pipeline.llm_module.MODEL = config.OLLAMA_MODEL
    pipeline.llm_module.OLLAMA_URL = f"{config.OLLAMA_URL}/api/generate"
    answer_text = pipeline.generator_module.generate_answer(question, context)
    if answer_text.strip() == pipeline.generator_module.NOT_FOUND_MESSAGE:
        return Answer(
            question,
            "unsupported",
            message="I couldn't find supporting evidence in the indexed datasheets.",
            model=config.OLLAMA_MODEL,
        )

    report = pipeline.validator_module.validate_citations(
        answer_text,
        context,
        not_found_message=pipeline.generator_module.NOT_FOUND_MESSAGE,
    )
    if not report.valid or not report.citations:
        return Answer(
            question,
            "validation_failed",
            message="The generated answer did not pass citation validation and was withheld.",
            model=config.OLLAMA_MODEL,
        )

    citation_pairs = list(dict.fromkeys(report.citations))
    citations = []
    for filename, page in citation_pairs:
        matching = [
            record
            for record in context
            if str(record["source"]).lower() == filename.lower()
            and int(record["page"]) == page
        ]
        if not matching:
            return Answer(
                question,
                "validation_failed",
                message="A validated citation could not be matched to its retrieved evidence.",
                model=config.OLLAMA_MODEL,
            )

        citations.append(
            Citation(
                filename=filename,
                page=page,
                evidence="\n\n".join(record["text"] for record in matching),
                score=max(
                    (
                        float(record["rerank_score"])
                        for record in matching
                        if record.get("rerank_score") is not None
                    ),
                    default=None,
                ),
                chunk_id=matching[0].get("chunk_id"),
            )
        )

    return Answer(
        question,
        "answered",
        text=answer_text,
        citations=tuple(citations),
        model=config.OLLAMA_MODEL,
    )
