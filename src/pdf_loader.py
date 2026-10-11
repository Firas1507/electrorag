"""PDF loader for ElectroRAG: extracts text page by page from electronics datasheets."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import pymupdf  # modern import name (replaces the legacy `import fitz`)

logger = logging.getLogger(__name__)

DATASHEET_DIR = Path("data/datasheets")

Document = Dict[str, Any]

# Silence MuPDF's low-level stderr output; failures are logged by this module instead.
pymupdf.TOOLS.mupdf_display_errors(False)


def clean_text(text: str) -> str:
    """Normalize whitespace while keeping paragraph/line structure."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces/tabs
    text = re.sub(r" *\n *", "\n", text)           # trim around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse blank lines
    return text.strip()


def _describe_empty_page(page: "pymupdf.Page") -> str:
    """Return a short reason for a page with no extractable text."""
    try:
        return "likely scanned (image-only)" if page.get_images() else "blank"
    except Exception:
        return "empty"


def load_pdf(pdf_path: Union[str, Path]) -> List[Document]:
    """Extract non-empty pages from one PDF as a list of documents.

    Each document has: text, source (filename), page (1-indexed), total_pages.
    Empty or scanned (image-only) pages are skipped with a log message.
    """
    pdf_path = Path(pdf_path)
    documents: List[Document] = []

    try:
        with pymupdf.open(pdf_path) as pdf:
            if pdf.needs_pass:
                logger.error("%s is password-protected, skipping", pdf_path.name)
                return []

            total_pages = pdf.page_count
            for page in pdf:
                page_number = page.number + 1  # PyMuPDF page numbers are 0-indexed

                try:
                    text = clean_text(page.get_text("text"))
                except Exception as exc:  # corrupted page
                    logger.warning("%s p.%d: extraction failed (%s)", pdf_path.name, page_number, exc)
                    continue

                if not text:
                    logger.info(
                        "%s p.%d: no text, %s - skipping",
                        pdf_path.name, page_number, _describe_empty_page(page),
                    )
                    continue

                documents.append(
                    {
                        "text": text,
                        "source": pdf_path.name,
                        "page": page_number,
                        "total_pages": total_pages,
                    }
                )
    except Exception as exc:  # unreadable/corrupt file
        logger.error("Could not open %s: %s", pdf_path.name, exc)

    return documents


def load_all_pdfs(directory: Union[str, Path] = DATASHEET_DIR) -> List[Document]:
    """Load every PDF in a directory (sorted by name) into a flat list of documents."""
    directory = Path(directory)
    if not directory.is_dir():
        logger.error("Directory not found: %s", directory)
        return []

    pdf_files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        logger.warning("No PDF files found in %s", directory)

    documents: List[Document] = []
    for pdf_file in pdf_files:
        pdf_docs = load_pdf(pdf_file)
        logger.info("Loaded %d text pages from %s", len(pdf_docs), pdf_file.name)
        documents.extend(pdf_docs)

    return documents


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    documents = load_all_pdfs()
    print(f"\nExtracted {len(documents)} pages with text.\n")

    for doc in documents:
        preview = doc["text"][:100].replace("\n", " ")
        print(
            f"[{doc['source']}] page {doc['page']}/{doc['total_pages']} "
            f"| {len(doc['text'])} chars | {preview}..."
        )


if __name__ == "__main__":
    main()