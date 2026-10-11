"""TEMPORARY diagnostic: find chunks that mention EN / Enable / enable / shutdown.

Used to check whether the EN-pin information was extracted and chunked correctly.
Reuses the existing loader and chunker; it does not touch ChromaDB.

Run (from the ElectroRAG root):
    python -m src.find_text
"""

import re

from src.chunker import chunk_documents
from src.pdf_loader import load_all_pdfs

SEARCH_TERMS = ["EN", "Enable", "enable", "shutdown"]


def build_patterns(terms: list) -> dict:
    """Compile one case-insensitive pattern per term (skipping case-only duplicates).

    "EN" is matched as a standalone token (not part of another word), otherwise
    it would match the letters "en" inside almost every chunk ("when", "ten", ...).
    Other terms are plain substrings, so "enable" also finds "enabled"/"enables".
    """
    patterns = {}
    seen = set()
    for term in terms:
        if term.lower() in seen:  # "Enable" and "enable" are the same case-insensitive search
            continue
        seen.add(term.lower())

        if term.upper() == "EN":
            regex = r"(?<![A-Za-z0-9])EN(?![A-Za-z0-9])"
        else:
            regex = re.escape(term)
        patterns[term] = re.compile(regex, re.IGNORECASE)
    return patterns


def find_matches(chunk: dict, patterns: dict) -> list:
    """Return the terms found in this chunk's text."""
    return [term for term, pattern in patterns.items() if pattern.search(chunk["text"])]


def main() -> None:
    chunks = chunk_documents(load_all_pdfs())
    patterns = build_patterns(SEARCH_TERMS)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Searching (case-insensitive) for: {', '.join(patterns)}")

    match_count = 0
    for chunk in chunks:
        matched = find_matches(chunk, patterns)
        if not matched:
            continue

        match_count += 1
        print("\n" + "=" * 80)
        print(f"Chunk ID:      {chunk['chunk_id']}")
        print(f"Page:          {chunk['page']}")
        print(f"Section:       {chunk.get('section')}")
        print(f"Matched terms: {', '.join(matched)}")
        print("-" * 80)
        print(chunk["text"])

    print("\n" + "=" * 80)
    print(f"{match_count} of {len(chunks)} chunks matched.")


if __name__ == "__main__":
    main()