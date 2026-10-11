"""
ElectroRAG - Deterministic Citation Resolver (v1.2)

The LLM cites context blocks as [C1], [C2], ... and this module converts
those IDs into real datasheet citations, e.g. [TPS63020.pdf, p. 6].

v1.2 changes
------------
* Handles grouped citations that local LLMs commonly produce:
      [C1, C2]   [C1; C2]   [C1 and C2]   [C1-C3]   [C1][C2]
* Output keeps the existing "[source, p. N]" syntax, so the benchmark's
  citation checker still works unchanged.
* Identical citations produced by one group are de-duplicated.
* Invalid IDs (e.g. [C9] with only 5 context blocks) are left as the
  literal "[C9]" so evaluators and humans can spot them.
* New helpers: find_invalid_citations() and has_citation().
"""

import re

# One bracket group: "[C1]", "[C1, C2]", "[C1-C3]", "[C1 and C2]" ...
_GROUP_PATTERN = re.compile(
    r"\[\s*(C\s*\d+(?:\s*(?:,|;|&|and|-|\u2013|\u2014|to)\s*C?\s*\d+)*)\s*\]",
    re.IGNORECASE,
)
_RANGE_SEPARATORS = re.compile(r"-|\u2013|\u2014|\bto\b", re.IGNORECASE)
_MAX_RANGE = 10

# Kept for backward compatibility with older imports.
CITATION_PATTERN = re.compile(r"\[C(\d+)\]", re.IGNORECASE)


def _expand_group(group: str) -> list:
    """Turn "C1, C3-C4" into [1, 3, 4] (order preserved, no duplicates)."""
    ids = []
    # Split on list separators first; ranges are handled per piece.
    for piece in re.split(r"\s*(?:,|;|&|\band\b)\s*", group, flags=re.IGNORECASE):
        numbers = [int(n) for n in re.findall(r"\d+", piece)]
        if not numbers:
            continue
        if len(numbers) == 2 and _RANGE_SEPARATORS.search(piece):
            low, high = numbers
            if low <= high and high - low < _MAX_RANGE:
                numbers = list(range(low, high + 1))
        for number in numbers:
            if number not in ids:
                ids.append(number)
    return ids


def resolve_citations(answer, context_records):
    """
    Replace [C1], [C1, C2], [C1-C3] ... with real source/page citations.

    context_records must be the exact, ordered records supplied to the LLM
    (each with "source" and "page").
    """

    def replace_group(match):
        pieces = []
        for number in _expand_group(match.group(1)):
            index = number - 1
            if index < 0 or index >= len(context_records):
                pieces.append(f"[C{number}]")  # invalid ID: leave visible
                continue
            record = context_records[index]
            pieces.append(f"[{record['source']}, p. {record['page']}]")

        unique = list(dict.fromkeys(pieces))
        return "".join(unique)

    return _GROUP_PATTERN.sub(replace_group, answer)


def find_invalid_citations(answer, context_records) -> list:
    """Return the sorted context numbers cited by the answer that do not exist."""
    invalid = set()
    for match in _GROUP_PATTERN.finditer(answer):
        for number in _expand_group(match.group(1)):
            if number < 1 or number > len(context_records):
                invalid.add(number)
    return sorted(invalid)


def has_citation(answer) -> bool:
    """True if the raw (unresolved) answer contains at least one [C#] citation."""
    return bool(_GROUP_PATTERN.search(answer))


def get_valid_context_ids(context_records):
    """Return the citation IDs available to the LLM."""
    return [f"C{index}" for index in range(1, len(context_records) + 1)]


if __name__ == "__main__":
    demo_context = [
        {"source": "TPS63020.pdf", "page": 6},
        {"source": "TPS63020.pdf", "page": 9},
        {"source": "TPS63020.pdf", "page": 12},
    ]
    demo = (
        "The input range is 1.8 V to 5.5 V [C1]. "
        "EN low shuts the device down [C2, C3]."
    )
    print("Original:", demo)
    print("Resolved:", resolve_citations(demo, demo_context))
