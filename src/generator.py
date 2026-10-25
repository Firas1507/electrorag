"""
Generator for ElectroRAG.

Responsible for:

1. Building the grounded context.
2. Building the LLM prompt.
3. Sending the prompt to the local LLM.
4. Collecting the generated answer.
5. Resolving [C1], [C2], ... citations deterministically.

The generator does NOT perform retrieval.

Expected input:
    A user question + reranked context chunks.

Run directly:

    python -m src.generator "What is the input voltage range of the TPS63020?"
"""

import re
import sys
import logging

from src.llm import stream_answer
from src.citation_resolver import resolve_citations
from src.citation_validation import validate_citations


# ============================================================
# Configuration
# ============================================================

NOT_FOUND_MESSAGE = (
    "I could not find this information in the indexed datasheet."
)

logger = logging.getLogger(__name__)


# ============================================================
# LLM instructions
# ============================================================

INSTRUCTIONS = f"""
You are ElectroRAG, a technical assistant specialized in
electronics, embedded systems, electrical engineering, and
component datasheets.

Your job is to answer the user's question using ONLY the
provided datasheet context.

STRICT GROUNDING RULES
----------------------

1. Use ONLY information explicitly supported by the provided
   context.

2. Do NOT use outside knowledge.

3. Do NOT guess, estimate, calculate, or invent specifications
   that are not supported by the context.

4. If the provided context genuinely does not contain enough
   information to answer the question, reply exactly:

{NOT_FOUND_MESSAGE}

5. Before drafting, identify every requested item and scan ALL
   context blocks for evidence for each one. Do not stop at the
   first relevant chunk or value.

6. For an explicitly named parameter, such as a pin or supply
   rail, use its exact labeled row or description when available.
   Do not substitute a generic value for that named parameter.

7. Read tables, graphs, feature lists, bullet points, headings,
   and fragmented text. Associate each value with its parameter,
   row/column heading, unit, and condition. If a table gives
   Min, Typ, and Max for the requested parameter, report all
   three with their labels and units, and state the Min-to-Max
   range as well when useful. Use values only from the row for
   the exact requested parameter; do not take a value from a
   neighboring row for another condition or parameter, even when
   both rows concern the same device variant. Never report only
   the endpoints when a relevant typical value is also present.

8. For questions asking for multiple items or a list, explicitly
   enumerate every applicable item stated anywhere in the
   context. Reconcile evidence across blocks; do not omit an item
   because another is easier or more prominent. If the context
   states a count for a list, check the extracted entries against
   that count before answering.

NUMERICAL ACCURACY RULES
------------------------

9. NEVER omit important numerical information.

10. Preserve numerical values exactly as supported by the context.

11. ALWAYS include the unit when the unit is available.

12. NEVER remove units such as:
    V, mV, A, mA, uA, nA, Hz, kHz, MHz, ohm, kOhm,
    W, mW, %, degC, bits, bytes, etc.

13. Preserve both endpoints of every relevant range and keep
    their units. Do not report only one endpoint or replace a
    stated range with a vague description.

    Identify minimum, typical/nominal, and maximum values by
    their stated roles. For a requested parameter, report every
    applicable Min/Typ/Max value supplied in the context, not
    just its range or typical value.

14. Equivalent engineering units may be recognized as the same
    physical quantity when the conversion is exact (for example,
    2400 kHz = 2.4 MHz, 3300 mV = 3.3 V, or 4000 mA = 4 A).
    Prefer preserving the datasheet's original value and unit;
    if a normalized equivalent helps answer the question, include
    it without changing the value or implying greater precision.

15. Distinguish between minimum, typical, maximum, nominal,
    recommended, absolute maximum, operating, and standby
    values. Never substitute one parameter type for another.

16. Do not collapse values that vary by device suffix, package,
    temperature grade, or operating condition. State the
    corresponding variant or condition next to each value.

    When one parameter has several condition rows (for example
    "Maximum power dissipation" and "Low-power dissipation") and
    the question's qualifier (maximum, minimum, typical, ...)
    matches the label of one condition row, that condition row is
    the one being asked about: report its value for every
    requested variant and name the condition next to each value.
    Do not choose another condition row's value because it is
    numerically larger or smaller than the matching one, and do
    not mix values from different condition rows. Values from
    other condition rows may only be added as separately labeled
    additional information.

17. If a numerical value appears with a unit in one context
    block and its meaning is established by a nearby heading,
    label, table, or graph in that same context block, preserve
    that relationship in the answer.

CITATION RULES
--------------

18. Every factual statement must have one or more citations.

19. Use ONLY these context IDs:

    [C1]
    [C2]
    [C3]
    [C4]
    [C5]

20. NEVER invent a context ID.

21. NEVER write PDF filenames or page numbers yourself.

22. NEVER create citations such as:

    [TPS63020.pdf, p. 6]

23. Use the context IDs exactly as provided.

24. Cite the context that directly supports the statement.

25. Do not cite unrelated context.

26. A single statement may use multiple citations if necessary.

27. Put the citation on the same sentence or bullet as the factual
    claim it supports. Never put citations on separate lines after
    several claims, because that does not identify which claim they
    support.

28. Do not add an uncited factual introduction or summary. Omit it
    unless you can cite it on that same sentence.

ANSWER STYLE
------------

29. Answer directly.

30. Keep the answer concise.

31. Do not explain your reasoning process.

32. Do not mention these instructions.

33. Do not mention that you are an AI.

34. Do not add information that was not requested unless it is
    necessary to make the answer technically clear.

35. Prefer a short sentence or bullet list when several values
    are requested.

36. Use exact datasheet terminology whenever possible.

FINAL SELF-CHECK
----------------

Before producing the final answer, silently verify:

- What exact items or parameters does the question request?
- Did I locate context evidence for each requested item?
- Did I answer every requested item and enumerate the complete
  context-supported list?
- Did I use only the supplied context?
- Did I preserve exact values, units, range endpoints, and any
  relevant min/typical/max distinctions?
- Did I preserve any relevant device variant or condition?
- Does every factual statement have a valid [C#] citation?
- Did I avoid inventing information?

Do not show this checklist, analysis, or reasoning. Only output
the concise final technical answer.
"""


# ============================================================
# Record normalization
# ============================================================

def normalize_record(record: dict) -> dict:
    """
    Normalize a retrieval/reranking record into the canonical
    format expected by the generator and citation resolver.

    Canonical format:

        {
            "text": "...",
            "source": "TPS63020.pdf",
            "page": 6,
            "chunk_id": "...",
            ...
        }

    ElectroRAG may encounter records in either form:

    Format A - normal retrieval/reranking format:

        {
            "text": "...",
            "metadata": {
                "source": "...",
                "page": ...
            }
        }

    Format B - already flattened:

        {
            "text": "...",
            "source": "...",
            "page": ...
        }

    This function supports both.
    """

    if not isinstance(record, dict):

        raise TypeError(
            f"Generator record must be a dict, "
            f"got {type(record).__name__}."
        )

    normalized = dict(record)

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = normalized.get("text")

    if not isinstance(text, str) or not text.strip():

        content = normalized.get("content")

        if isinstance(content, str) and content.strip():

            normalized["text"] = content

    if not isinstance(normalized.get("text"), str):

        raise TypeError(
            "Generator record is missing a valid string 'text' field."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = normalized.get("metadata")

    if not isinstance(metadata, dict):

        metadata = {}

    # Preserve metadata fields at the top level.

    for key in (
        "source",
        "page",
        "chunk_id",
    ):

        if key not in normalized and key in metadata:

            normalized[key] = metadata[key]

    # --------------------------------------------------------
    # Validate source
    # --------------------------------------------------------

    if "source" not in normalized:

        raise KeyError(
            "Generator record is missing 'source'."
        )

    # --------------------------------------------------------
    # Validate page
    # --------------------------------------------------------

    if "page" not in normalized:

        raise KeyError(
            "Generator record is missing 'page'."
        )

    return normalized


def normalize_records(records: list) -> list:
    """
    Normalize all records before prompt construction and citation
    resolution.
    """

    if not isinstance(records, list):

        raise TypeError(
            f"Generator records must be a list, "
            f"got {type(records).__name__}."
        )

    normalized_records = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        try:

            normalized_records.append(
                normalize_record(record)
            )

        except Exception as exc:

            raise TypeError(
                f"Invalid generator record C{index}: {exc}"
            ) from exc

    return normalized_records


# ============================================================
# Context construction
# ============================================================

def build_context(records: list) -> str:
    """
    Convert retrieved records into numbered context blocks.

    The LLM sees only C1, C2, C3...

    Python later resolves these IDs into real PDF citations.
    """

    normalized_records = normalize_records(records)

    blocks = []

    for index, record in enumerate(
        normalized_records,
        start=1,
    ):

        source = record["source"]
        page = record["page"]
        text = record["text"]

        blocks.append(
            f"[C{index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

    return "\n\n---\n\n".join(blocks)


def _question_terms(question: str) -> set[str]:
    ignored_terms = {
        "a", "an", "and", "are", "by", "for", "from", "how",
        "is", "of", "on", "or", "the", "to", "what", "which",
        "with",
        "specified", "available", "supported", "ratings", "rating",
        "version", "versions", "ordering", "suffix", "suffixes",
    }
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", question)
        if token.lower() not in ignored_terms
        and not any(character.isdigit() for character in token)
    }


def _normalize_numeric_cell(line: str) -> str:
    return re.sub(r"^[\uFFFD\u2010-\u2015\u2212](?=\d)", "-", line.strip())


def _is_numeric_cell(line: str) -> bool:
    line = _normalize_numeric_cell(line)
    return bool(
        re.fullmatch(
            r"\s*[+\-]?\d+(?:[.,]\d+)?"
            r"(?:\s*(?:to|[-\u2013\u2014])\s*[+\-]?\d+(?:[.,]\d+)?)?"
            r"(?:\s*(?:°\s*)?[A-Za-zµμΩ]+)?\s*",
            line,
            flags=re.IGNORECASE,
        )
    )


def _table_evidence(
    question: str,
    records: list,
) -> list[str]:
    terms = _question_terms(question)
    if not terms:
        return []

    evidence = []
    condition_qualifiers = {
        "maximum", "minimum", "max", "min", "typical", "nominal",
        "recommended", "absolute", "operating", "low", "high",
        "standby", "shutdown",
    }
    requested_qualifiers = terms & condition_qualifiers
    for context_index, record in enumerate(records, start=1):
        lines = [
            line.strip()
            for line in record["text"].splitlines()
            if line.strip()
        ]
        has_table_header = any(
            {"symbol", "parameter", "conditions"}.issubset(
                {
                    token.lower().strip(".,:")
                    for token in re.findall(r"[A-Za-z]+", " ".join(lines[index:index + 7]))
                }
            )
            for index in range(max(0, len(lines) - 6))
        )
        anchors = []
        for index, line in enumerate(lines):
            line_terms = {
                token.lower()
                for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", line)
            }
            if len(line_terms & terms) < min(2, len(terms)):
                continue

            nearby_symbol = any(
                re.fullmatch(r"[A-Z][A-Z0-9_]{0,5}", candidate)
                for candidate in lines[max(0, index - 60):index]
            )
            if not (has_table_header or nearby_symbol):
                continue

            numeric_cells = sum(
                _is_numeric_cell(candidate)
                for candidate in lines[index + 1:min(len(lines), index + 9)]
            )
            if numeric_cells >= 2:
                anchors.append(index)

        for anchor_number, anchor_index in enumerate(anchors):
            end = min(len(lines), anchor_index + 12)
            if anchor_number + 1 < len(anchors):
                end = min(end, anchors[anchor_number + 1])
            for line_index in range(anchor_index + 1, end):
                if (
                    re.fullmatch(r"[A-Z][A-Z0-9_]{0,5}", lines[line_index])
                    and line_index + 1 < len(lines)
                    and not _is_numeric_cell(lines[line_index + 1])
                ):
                    end = line_index
                    break
            parameter_label = lines[anchor_index]
            source_start = anchor_index + 1
            if (
                re.search(r"\b(?:for|of|the|with|at)$", parameter_label, re.IGNORECASE)
                and source_start < end
            ):
                parameter_label = f"{parameter_label} {lines[source_start]}"
                source_start += 1

            source_lines = lines[source_start:end]
            conditions = []
            condition_label = None
            cells = []

            for line in source_lines:
                if _is_numeric_cell(line):
                    cells.append(_normalize_numeric_cell(line))
                    continue
                if cells and re.fullmatch(r"\s*(?:°\s*[A-Za-z]+|[%])\s*", line):
                    cells.append(line)
                    continue

                if condition_label is not None:
                    if cells:
                        conditions.append(
                            (
                                condition_label,
                                ", ".join(cells),
                                requested_qualifiers
                                & {
                                    token.lower()
                                    for token in re.findall(
                                        r"[A-Za-z][A-Za-z0-9_-]*",
                                        condition_label,
                                    )
                                },
                            )
                        )
                    condition_label = None
                    cells = []

                condition_label = line

            if condition_label is not None and cells:
                conditions.append(
                    (
                        condition_label,
                        ", ".join(cells),
                        requested_qualifiers
                        & {
                            token.lower()
                            for token in re.findall(
                                r"[A-Za-z][A-Za-z0-9_-]*",
                                condition_label,
                            )
                        },
                    )
                )

            if conditions:
                best_qualifier_match = max(
                    (len(condition[2]) for condition in conditions),
                    default=0,
                )
                row_lines = [
                    f"[C{context_index}] Parameter row: {parameter_label}"
                ]
                row_lines.extend(
                    (
                        "  Question-matching condition"
                        f" ({', '.join(sorted(matches))}): "
                        f"{label} -> ordered cells: {cells}"
                        if best_qualifier_match and len(matches) == best_qualifier_match
                        else f"  Condition row: {label} -> ordered cells: {cells}"
                    )
                    for label, cells, matches in conditions
                )
                evidence.extend(row_lines)

    return evidence


def build_evidence_index(
    question: str,
    records: list,
) -> str:
    """Create deterministic, citation-linked evidence aids from context text."""
    sections = []

    if re.search(r"\bmodes?\b", question, flags=re.IGNORECASE):
        mode_mentions = []
        for context_index, record in enumerate(records, start=1):
            fragments = re.split(
                r"(?<=[.!?])\s+|[\r\n]+|[•\uF0B7]",
                record["text"],
            )
            for fragment in fragments:
                cleaned = " ".join(fragment.split())
                if cleaned and re.search(
                    r"\bmodes?\b",
                    cleaned,
                    flags=re.IGNORECASE,
                ):
                    mode_mentions.append(
                        f"[C{context_index}] {cleaned}"
                    )
        if mode_mentions:
            sections.append(
                "All context fragments mentioning modes (source order):\n"
                + "\n".join(f"- {mention}" for mention in mode_mentions)
            )

    table_rows = _table_evidence(question, records)
    if table_rows:
        sections.append(
            "Matching table parameter rows and their ordered condition cells "
            "(do not merge cells across condition rows). Where a row is "
            "marked 'Question-matching condition', use that condition's "
            "cells for the requested value and name the condition; the "
            "other condition rows are different operating conditions and "
            "are not the requested value:\n"
            + "\n".join(table_rows)
        )

    if not sections:
        return ""

    return "\n\n================ EVIDENCE INDEX ================\n\n" + (
        "\n\n".join(sections)
    )


# ============================================================
# Prompt construction
# ============================================================

def build_prompt(
    question: str,
    records: list,
) -> str:
    """
    Build the grounded prompt sent to the local LLM.
    """

    context = build_context(records)
    evidence_index = build_evidence_index(question, records)

    return f"""
{INSTRUCTIONS}

================ CONTEXT ================

{context}

================ QUESTION ================

{question}
{evidence_index}

Before answering, make a private evidence inventory from the
supplied context: identify each requested parameter or list item,
its exact value(s) and unit(s), any Min/Typ/Max or operating
condition labels, and the [C#] block that supports it. Read table
values against their column headings in order, and keep each value
bound to the exact parameter row and condition stated by the
question; do not substitute a neighboring row's value. Retain
every populated Min, Typ, and Max entry, including interior/typical
values when also reporting a range. Reconcile complementary
evidence across all context blocks, and include every explicitly
supported requested list item with its supporting citation. If a
value or item is not supported, do not infer it.

================ ANSWER ================
""".strip()


# ============================================================
# Answer generation
# ============================================================

def _collect_answer(prompt: str) -> str:
    answer_parts = []

    try:
        for piece in stream_answer(prompt):
            answer_parts.append(piece)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Generator failed: {exc}"
        ) from exc

    return "".join(answer_parts).strip()


def generate_answer(
    question: str,
    records: list,
) -> str:
    """
    Generate a grounded answer using the local LLM.

    Parameters
    ----------
    question:
        User's original question.

    records:
        Reranked context records.

    Returns
    -------
    str
        Final answer with deterministic citations.

    A draft that fails strict citation validation gets one repair attempt;
    a second invalid draft is discarded in favor of the configured abstention.
    """

    if not records:

        return NOT_FOUND_MESSAGE

    # --------------------------------------------------------
    # Normalize records once.
    # --------------------------------------------------------

    normalized_records = normalize_records(
        records
    )

    # --------------------------------------------------------
    # Build grounded prompt.
    # --------------------------------------------------------

    prompt = build_prompt(
        question,
        normalized_records,
    )

    # --------------------------------------------------------
    # Generate answer with local LLM.
    # --------------------------------------------------------

    answer = _collect_answer(prompt)

    # --------------------------------------------------------
    # Handle empty response.
    # --------------------------------------------------------

    if not answer:

        return (
            "(no visible answer - "
            "the model may have returned an empty response)"
        )

    for attempt in range(2):
        resolved_answer = resolve_citations(
            answer,
            normalized_records,
        )
        report = validate_citations(
            resolved_answer,
            normalized_records,
            not_found_message=NOT_FOUND_MESSAGE,
        )
        if report.valid:
            return resolved_answer

        if attempt == 0:
            logger.info(
                "Retrying generated answer after citation validation failed: %s",
                "; ".join(report.problems),
            )
            prompt = (
                f"{prompt}\n\n"
                "================ CITATION REPAIR ================\n\n"
                "Your previous draft failed strict citation validation. "
                "Rewrite it so every factual sentence or bullet has a valid "
                "[C#] citation on that same sentence or bullet. Put the "
                "citation before the sentence-ending punctuation, as in "
                "\"The pin is enabled [C1].\", not after it. Remove "
                "uncited introductions and summaries. Keep only claims "
                "directly supported by the supplied context, and abstain if "
                "the requested information is not supported.\n\n"
                "Validation problems:\n"
                + "\n".join(f"- {problem}" for problem in report.problems)
                + "\n\nPrevious draft:\n"
                + answer
            )
            answer = _collect_answer(prompt)
            if answer:
                continue

            logger.warning(
                "Discarding generated answer because citation repair returned "
                "an empty response."
            )
            return NOT_FOUND_MESSAGE

        logger.warning(
            "Discarding generated answer with invalid citations: %s",
            "; ".join(report.problems),
        )

    return NOT_FOUND_MESSAGE


# ============================================================
# Standalone test
# ============================================================

def main():
    """
    Standalone generator test.

    This test uses manually created context so that the
    generator can be tested independently from retrieval.
    """

    if len(sys.argv) < 2:

        sys.exit(
            'Usage: python -m src.generator "Your question here"'
        )

    question = " ".join(
        sys.argv[1:]
    )

    # --------------------------------------------------------
    # Example context
    # --------------------------------------------------------

    test_records = [
        {
            "text": (
                "The TPS63020 is a high efficiency buck-boost "
                "converter operating from an input voltage range "
                "of 1.8 V to 5.5 V."
            ),
            "source": "TPS63020.pdf",
            "page": 1,
            "chunk_id": "test_chunk_001",
            "rerank_score": 1.0,
        }
    ]

    print(
        f"Question: {question}"
    )

    print(
        "\nGenerated answer:"
    )

    print(
        "-" * 60
    )

    try:

        answer = generate_answer(
            question,
            test_records,
        )

        print(answer)

    except Exception as exc:

        sys.exit(
            f"Error: {type(exc).__name__}: {exc}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()