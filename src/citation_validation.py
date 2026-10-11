"""
ElectroRAG - Strict citation validation.

Validates a RESOLVED answer (citations already rewritten from [C#] to
"[source.pdf, p. N]" by citation_resolver) against the context records the
LLM was actually given.

Three layers, reported separately so a failure says exactly what is wrong:

1. IDENTITY  - every citation must be well formed and must name a source and
               page that exist in the retrieved context. Unresolved [C#]
               references and fabricated sources/pages are rejected. If an
               expected source is given, at least one citation must use it.
2. COVERAGE  - an answer must contain citations (unless it is the configured
               "not found" message), and every sentence that states a
               measurement (a number with a unit) must carry a citation.
3. SUPPORT   - every number in a cited measurement sentence must literally
               appear in the text of the cited page(s) of the context.

What this does NOT do
---------------------
Layer 3 is a NUMERIC PRESENCE check, not semantic verification. It cannot tell
that a number on the cited page belongs to the right parameter, row or
condition, and it does not judge claims that contain no number-with-unit
(those are only required to be covered by at least one citation somewhere in
the answer). Number-with-unit detection uses a fixed unit list (see _UNIT).
"""

import math
import re
from dataclasses import dataclass, field

from src.citation_resolver import _GROUP_PATTERN as _UNRESOLVED_RE

# A resolved citation, exactly as the benchmark and generator produce it.
_CITATION_RE = re.compile(r"\[([^,\]\[]+),\s*p\.\s*(\d+)\]", re.IGNORECASE)

# Anything that looks like a citation to a PDF but is not in the exact format.
_PDF_BRACKET_RE = re.compile(r"\[[^\[\]]*\.pdf[^\[\]]*\]", re.IGNORECASE)

_PLACEHOLDER_RE = re.compile("\uE000(\\d+)\uE001")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=\S)")

_SIGN = "+\\-\u2212\u2013\u2014"
_NUM = (
    r"(?<![\w.])[" + _SIGN + r"]?"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
)
_UNIT = (
    r"(?:\u00b0\s?[CF]|\u2103|[kKMGT]?Hz|[kMG\u00b5\u03bcmun]?[VAW]"
    r"|[kM]?\u03a9|[kM]?ohms?|%|[m\u00b5\u03bcun]s|[pnu\u00b5\u03bcm]F|dBm?"
    r"|[kKMG]B)"
)
_MEASUREMENT_RE = re.compile(
    rf"(?P<nums>{_NUM}(?:\s*(?:to|[\-\u2013\u2014\u2212~/])\s*{_NUM})*)"
    rf"\s*(?P<unit>{_UNIT})(?![A-Za-z])",
    re.IGNORECASE,
)
_NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)


@dataclass
class CitationReport:
    identity: list = field(default_factory=list)
    coverage: list = field(default_factory=list)
    support: list = field(default_factory=list)
    citations: list = field(default_factory=list)

    @property
    def problems(self):
        return self.identity + self.coverage + self.support

    @property
    def valid(self):
        return not self.problems


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _record_fields(record):
    """Return (source, page, text) for flat or legacy record layouts."""
    if not isinstance(record, dict):
        return None
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    source = record.get("source", metadata.get("source"))
    page = record.get("page", metadata.get("page"))
    text = record.get("text")
    if not isinstance(text, str):
        text = record.get("content") if isinstance(record.get("content"), str) else ""
    try:
        return str(source).strip().lower(), int(page), text
    except (TypeError, ValueError):
        return None


def _context_pages(context_records):
    pages = {}
    for record in context_records:
        fields = _record_fields(record)
        if fields:
            source, page, text = fields
            pages.setdefault((source, page), []).append(text)
    return pages


def _plain_numbers(text):
    """All numbers in a context text (absolute values), comma-decimal tolerant."""
    values = []
    for token in _NUMBER_TOKEN_RE.findall(text.replace("\u2212", "-")):
        if "," in token:
            values.append(float(token.replace(",", ".")))
            values.append(float(token.replace(",", "")))
        else:
            values.append(float(token))
    return values


def _claim_numbers(text):
    """Numbers (absolute values) that belong to a number-with-unit statement."""
    values = []
    for match in _MEASUREMENT_RE.finditer(text):
        for token in re.findall(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?", match.group("nums")):
            if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
                values.append(float(token.replace(",", "")))
            else:
                values.append(float(token.replace(",", ".")))
    return values


def _is_supported(value, page_numbers):
    for scale in _SCALES:
        target = value * scale
        if any(math.isclose(target, known, rel_tol=1e-9, abs_tol=1e-12) for known in page_numbers):
            return True
    return False


def _snippet(text, limit=90):
    text = " ".join(_PLACEHOLDER_RE.sub("[cited]", text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _claim_units(protected_answer):
    """Split into sentence-like units; a citation-only fragment joins the previous unit."""
    units = []
    for line in protected_answer.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in _SENTENCE_SPLIT_RE.split(line):
            part = part.strip()
            if not part:
                continue
            leftover = _PLACEHOLDER_RE.sub("", part).strip(" \t.,;:()[]*_-\u2022>|")
            if not leftover and units:
                units[-1] += " " + part
            else:
                units.append(part)
    return units


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def validate_citations(
    answer,
    context_records,
    expected_source=None,
    not_found_message=None,
):
    """Validate a resolved answer against the context records it was built from."""

    report = CitationReport()
    answer = answer if isinstance(answer, str) else ""
    stripped = answer.strip()

    if not stripped:
        report.coverage.append("answer is empty")
        return report

    if not_found_message and stripped == not_found_message.strip():
        return report  # an explicit "not found" makes no claims

    pages = _context_pages(context_records)

    # --- identity: unresolved references and malformed citations ------------
    for match in _UNRESOLVED_RE.finditer(answer):
        report.identity.append(f"unresolved citation reference [{match.group(1)}]")

    for match in _PDF_BRACKET_RE.finditer(answer):
        if not _CITATION_RE.fullmatch(match.group(0)):
            report.identity.append(f"malformed citation {match.group(0)}")

    # --- identity: every citation must exist in the context -----------------
    protected_parts, last, cited = [], 0, []
    for index, match in enumerate(_CITATION_RE.finditer(answer)):
        source, page = match.group(1).strip(), int(match.group(2))
        cited.append((source, page))
        protected_parts.append(answer[last:match.start()])
        protected_parts.append(f"\uE000{index}\uE001")
        last = match.end()
    protected_parts.append(answer[last:])
    protected = "".join(protected_parts)
    report.citations = cited

    known_sources = {source for source, _ in pages}
    for source, page in cited:
        if (source.lower(), page) in pages:
            continue
        if source.lower() not in known_sources:
            report.identity.append(f"citation [{source}, p. {page}] names a source that is not in the retrieved context")
        else:
            report.identity.append(f"citation [{source}, p. {page}] names a page that is not in the retrieved context")

    if cited and expected_source:
        if not any(
            source.lower() == expected_source.lower() and (source.lower(), page) in pages
            for source, page in cited
        ):
            report.identity.append(f"no valid citation to the expected source {expected_source}")

    # --- coverage -----------------------------------------------------------
    if not cited:
        report.coverage.append("answer contains no citations")

    for unit in _claim_units(protected):
        claim_text = _PLACEHOLDER_RE.sub(" ", unit)
        numbers = _claim_numbers(claim_text)
        if not numbers:
            continue

        indices = [int(i) for i in _PLACEHOLDER_RE.findall(unit)]
        if not indices:
            report.coverage.append(f"uncited claim: \"{_snippet(unit)}\"")
            continue

        # --- support: numbers must appear on the cited page(s) --------------
        page_numbers, found_any_page = [], False
        for i in indices:
            source, page = cited[i]
            texts = pages.get((source.lower(), page))
            if texts is None:
                continue  # invalid citation: already reported under identity
            found_any_page = True
            for text in texts:
                page_numbers.extend(_plain_numbers(text))
        if not found_any_page:
            continue

        missing = sorted({v for v in numbers if not _is_supported(v, page_numbers)})
        if missing:
            shown = ", ".join(f"{v:g}" for v in missing)
            report.support.append(
                f"claim not supported by its cited page(s): \"{_snippet(unit)}\" "
                f"(value(s) {shown} not found there)"
            )

    return report