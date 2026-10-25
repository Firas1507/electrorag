"""
ElectroRAG - Diagnostic RAG Benchmark V3

Evaluates:

1. Document-level Recall@1 / @3 / @5
2. Fact-level retrieval in hybrid candidates and final context
3. Full RAG generation
4. Structured expected-fact evaluation
5. Citation syntax, source, and context-page validity

For generation failures, prints the exact top-5
reranked chunks that were sent to the LLM.

Run from the ElectroRAG root:

    python -m src.benchmark
"""

import re
import sys

from sentence_transformers import CrossEncoder

from src.bm25_retriever import build_index
from src.chunker import chunk_documents
from src.embeddings import load_model
from src.citation_validation import validate_citations
from src.generator import (
    NOT_FOUND_MESSAGE,
    build_evidence_index,
    generate_answer,
    normalize_records,
)
from src.pdf_loader import load_all_pdfs
from src.query_expansion_retriever import EXPANSIONS, expanded_search
from src.reranker import RERANKER_MODEL, rerank
from src.retriever import open_collection, retrieve


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K_RETRIEVAL = 5
CONTEXT_CHUNKS = 5


# ============================================================
# BENCHMARK QUESTIONS
# ============================================================

QUESTIONS = [
    {
        "id": "stm32_01",
        "question": "What is the maximum CPU frequency of the STM32L476?",
        "source": "DS_stm32l476je.pdf",
        "expected_facts": [
            {
                "type": "value",
                "value": 80,
                "unit": "MHz",
            },
        ],
    },

    {
        "id": "stm32_02",
        "question": "What are the low-power operating modes of the STM32L476?",
        "source": "DS_stm32l476je.pdf",
        "expected_facts": [
            {
                "type": "contains_all",
                "values": [
                    "low-power",
                    "stop",
                    "standby",
                ],
            },
        ],
    },

    {
        "id": "stm32_03",
        "question": "What is the supply voltage range of the STM32L476?",
        "source": "DS_stm32l476je.pdf",
        "expected_facts": [
            {
                "type": "range",
                "minimum": 1.71,
                "maximum": 3.6,
                "unit": "V",
            },
        ],
    },

    {
        "id": "stm32_04",
        "question": "What communication interfaces are available on the STM32L476?",
        "source": "DS_stm32l476je.pdf",
        "expected_facts": [
            {
                "type": "contains_all",
                "values": [
                    "USART",
                    "SPI",
                    "I2C",
                ],
            },
        ],
    },

    {
        "id": "stm32_05",
        "question": (
            "What maximum ambient operating temperature ratings are specified "
            "for STM32L476 ordering suffixes 6, 7, and 3?"
        ),
        "source": "DS_stm32l476je.pdf",
        "expected_facts": [
            {
                "type": "list",
                "values": [
                    85,
                    105,
                    125,
                ],
                "unit": "°C",
            },
        ],
    },

    {
        "id": "esp32_01",
        "question": "What is the operating voltage of the ESP32-S3?",
        "source": "ESP32-S3.pdf",
        "expected_facts": [
            {
                "type": "range",
                "minimum": 3.0,
                "maximum": 3.6,
                "unit": "V",
            },
        ],
    },

    {
        "id": "esp32_02",
        "question": "What is the maximum CPU frequency of the ESP32-S3?",
        "source": "ESP32-S3.pdf",
        "expected_facts": [
            {
                "type": "value",
                "value": 240,
                "unit": "MHz",
            },
        ],
    },

    {
        "id": "esp32_03",
        "question": "What voltage can VDD_SPI use on the ESP32-S3?",
        "source": "ESP32-S3.pdf",
        "expected_facts": [
            {
                "type": "list",
                "values": [
                    1.8,
                    3.3,
                ],
                "unit": "V",
            },
        ],
    },

    {
        "id": "esp32_04",
        "question": "How much current does the ESP32-S3 require during operation?",
        "source": "ESP32-S3.pdf",
        "expected_facts": [
            {
                "type": "contains_any",
                "values": [
                    "current",
                    "mA",
                ],
            },
        ],
    },

    {
        "id": "esp32_05",
        "question": "What wireless communication features does the ESP32-S3 support?",
        "source": "ESP32-S3.pdf",
        "expected_facts": [
            {
                "type": "contains_all",
                "values": [
                    "Wi-Fi",
                    "Bluetooth",
                ],
            },
        ],
    },

    {
        "id": "max98357_01",
        "question": "What is the supply voltage range of the MAX98357A?",
        "source": "MAX98357A.pdf",
        "expected_facts": [
            {
                "type": "range",
                "minimum": 2.5,
                "maximum": 5.5,
                "unit": "V",
            },
        ],
    },

    {
        "id": "max98357_02",
        "question": "What digital audio interface does the MAX98357A use?",
        "source": "MAX98357A.pdf",
        "expected_facts": [
            {
                "type": "contains_any",
                "values": [
                    "I2S",
                    "I²S",
                ],
            },
        ],
    },

    {
        "id": "max98357_03",
        "question": "What audio sample rates are supported by the MAX98357A?",
        "source": "MAX98357A.pdf",
        "expected_facts": [
            {
                "type": "range",
                "minimum": 8,
                "maximum": 96,
                "unit": "kHz",
            },
        ],
    },

    {
        "id": "max98357_04",
        "question": "What speaker impedance is supported by the MAX98357A?",
        "source": "MAX98357A.pdf",
        "expected_facts": [
            {
                "type": "list",
                "values": [
                    4,
                    8,
                ],
                "unit": "Ω",
            },
        ],
    },

    {
        "id": "max98357_05",
        "question": "What gain settings are available on the MAX98357A?",
        "source": "MAX98357A.pdf",
        "expected_facts": [
            {
                "type": "list",
                "values": [
                    15,
                    12,
                    9,
                    6,
                    3,
                ],
                "unit": "dB",
            },
        ],
    },

    {
        "id": "tps63020_01",
        "question": "What is the input voltage range of the TPS63020?",
        "source": "TPS63020.pdf",
        "expected_facts": [
            {
                "type": "range",
                "minimum": 1.8,
                "maximum": 5.5,
                "unit": "V",
            },
        ],
    },

    {
        "id": "tps63020_02",
        "question": "What output voltage range can the TPS63020 provide?",
        "source": "TPS63020.pdf",
        "expected_facts": [
            {
                "type": "range",
                "minimum": 1.2,
                "maximum": 5.5,
                "unit": "V",
            },
        ],
    },

    {
        "id": "tps63020_03",
        "question": "What is the typical maximum average current in the switches of the TPS63020?",
        "source": "TPS63020.pdf",
        "expected_facts": [
            {
                "type": "value",
                "value": 4,
                "unit": "A",
            },
        ],
    },

    {
        "id": "tps63020_04",
        "question": "How does the EN pin control the TPS63020?",
        "source": "TPS63020.pdf",
        "expected_facts": [
            {
                "type": "contains_all",
                "values": [
                    "EN",
                    "enable",
                ],
            },
        ],
    },

    {
        "id": "tps63020_05",
        "question": "What is the switching frequency of the TPS63020?",
        "source": "TPS63020.pdf",
        "expected_facts": [
            {
                "type": "value",
                "value": 2.4,
                "unit": "MHz",
            },
        ],
    },
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for robust textual comparison.
    """

    if text is None:
        return ""

    text = str(text)

    replacements = {
        "–": "-",
        "—": "-",
        "−": "-",
        "µ": "u",
        "μ": "u",
        "Ω": "ohm",
        "Ω": "ohm",
        "²": "2",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.lower()


# ============================================================
# ENGINEERING VALUE EXTRACTION
# ============================================================

_NUMBER_CORE = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_UNIT_CORE = (
    r"(?:°\s*c|℃|deg(?:ree)?s?\s*c|ohms?|[ΩΩ]|"
    r"mhz|khz|hz|mv|ma|db|v|a)"
)
_MEASUREMENT_RE = re.compile(
    rf"(?<![\w.])"
    rf"(?P<values>{_NUMBER_CORE}(?:\s*/\s*{_NUMBER_CORE})*)"
    rf"\s*(?P<unit>{_UNIT_CORE})(?![a-z])",
    re.IGNORECASE,
)
_TABLE_UNIT_HEADER_RE = re.compile(
    rf"\(\s*(?P<unit>{_UNIT_CORE})\s*\)",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"(?<![\w.])(?P<between>between\s+)?"
    rf"(?P<first>{_NUMBER_CORE})\s*"
    rf"(?P<unit1>{_UNIT_CORE})?\s*"
    r"(?P<connector>down\s+to|up\s+to|through|to|and|[-–—])\s*"
    rf"(?P<second>{_NUMBER_CORE})\s*"
    rf"(?P<unit2>{_UNIT_CORE})?(?![a-z])",
    re.IGNORECASE,
)


def normalize_unit(unit):
    """
    Normalize common engineering units while preserving dimensions.
    """

    if unit is None:
        return ""

    unit = normalize_text(unit).replace(" ", "")

    aliases = {
        "v": "v",
        "volt": "v",
        "volts": "v",
        "mv": "mv",
        "a": "a",
        "amp": "a",
        "amps": "a",
        "ampere": "a",
        "amperes": "a",
        "ma": "ma",
        "mhz": "mhz",
        "khz": "khz",
        "hz": "hz",
        "ohm": "ohm",
        "ohms": "ohm",
        "ω": "ohm",
        "Ω": "ohm",
        "c": "c",
        "°c": "c",
        "℃": "c",
        "degc": "c",
        "degreec": "c",
        "degrees c": "c",
    }

    return aliases.get(unit, unit)


def _canonical_measurement(value, unit):
    """
    Convert compatible values to a shared base unit for comparison.
    """

    normalized_unit = normalize_unit(unit)
    conversions = {
        "v": ("voltage", 1.0),
        "mv": ("voltage", 0.001),
        "a": ("current", 1.0),
        "ma": ("current", 0.001),
        "hz": ("frequency", 1.0),
        "khz": ("frequency", 1_000.0),
        "mhz": ("frequency", 1_000_000.0),
        "ohm": ("resistance", 1.0),
        "c": ("temperature", 1.0),
    }

    conversion = conversions.get(normalized_unit)
    if conversion is None:
        return normalized_unit, float(value)

    dimension, factor = conversion
    return dimension, float(value) * factor


def extract_measurements(text):
    """
    Extract numeric engineering values and their units.

    A unit applies to numbers in an immediately slash-separated list, such as
    "85/105/125 °C". Word-boundary guards prevent numbers inside part numbers
    (for example, MAX98357A) from becoming measurements.
    """

    measurements = []

    for match in _MEASUREMENT_RE.finditer(normalize_text(text)):
        unit = normalize_unit(match.group("unit"))
        for value in re.findall(_NUMBER_CORE, match.group("values")):
            measurements.append(
                {
                    "value": float(value),
                    "unit": unit,
                }
            )

    return measurements


def extract_numbers(text):
    """Return engineering values, excluding unqualified identifiers."""

    return [
        measurement["value"]
        for measurement in extract_measurements(text)
    ]


def measurement_matches_expected(text, expected_value, expected_unit):
    """Check a value and its unit together, allowing equivalent units."""

    if not expected_unit:
        return any(
            numbers_equal(measurement["value"], expected_value)
            for measurement in extract_measurements(text)
        )

    expected_dimension, expected_canonical = _canonical_measurement(
        expected_value,
        expected_unit,
    )

    return any(
        _canonical_measurement(
            measurement["value"],
            measurement["unit"],
        )[0] == expected_dimension
        and numbers_equal(
            _canonical_measurement(
                measurement["value"],
                measurement["unit"],
            )[1],
            expected_canonical,
        )
        for measurement in extract_measurements(text)
    )


# A range may also be stated as labeled endpoints, e.g.
#   "Minimum: 3.0 V, Typical: 3.3 V, Maximum: 3.6 V"      (label before value)
#   "3.0 V (Min), 3.3 V (Typ), 3.6 V (Max)"               (label after value)
#   "Min 3.0, Typ 3.3, Max 3.6 V"        (one shared unit, as in a Unit column)
# which is the datasheet Min/Typ/Max format the generator is told to use.
# Each endpoint needs its own Min/Max label, and a matching unit (its own, or
# the unit of the next labeled endpoint on the same line). Both endpoints must
# be close together. "Absolute maximum" is never accepted.
_LABEL_CORE = r"(?P<label>min(?:imum)?|max(?:imum)?)\b"
_LABELED_ENDPOINT_RE = re.compile(
    r"(?<![\w-])(?P<absolute>absolute\s+)?"
    + _LABEL_CORE
    + r"(?!\s*[)\]])"  # "(Min)" closes a label-after endpoint; not a label-before
    + r"[^\d\n]{0,25}?"
    + rf"(?P<value>{_NUMBER_CORE})\s*(?P<unit>{_UNIT_CORE})?(?![a-z])",
    re.IGNORECASE,
)
_LABELED_AFTER_RE = re.compile(
    rf"(?<![\w.])(?P<value>{_NUMBER_CORE})\s*(?P<unit>{_UNIT_CORE}(?![a-z]))?"
    r"\s*[(\[]\s*(?P<absolute>absolute\s+)?"
    + _LABEL_CORE
    + r"\s*[)\]]",
    re.IGNORECASE,
)
_LABELED_ENDPOINT_WINDOW = 200  # max characters between the min and max labels


def _labeled_endpoints(text):
    """Return label-before and label-after endpoint matches in text order."""

    found = list(_LABELED_ENDPOINT_RE.finditer(text))
    for match in _LABELED_AFTER_RE.finditer(text):
        # "Min: 3.0 V (min)" style repeats must not be counted twice.
        if not any(
            earlier.start() < match.end() and match.start() < earlier.end()
            for earlier in found
        ):
            found.append(match)
    return sorted(found, key=lambda match: match.start())


def _labeled_range_matches_expected(answer, fact):
    """Accept explicit Min/Max-labeled endpoints that equal the expected range."""

    expected_unit = normalize_unit(fact.get("unit"))
    expected_dimension, expected_min = _canonical_measurement(
        float(fact["minimum"]), expected_unit
    )
    _, expected_max = _canonical_measurement(
        float(fact["maximum"]), expected_unit
    )

    text = normalize_text(answer)
    matches = _labeled_endpoints(text)

    minimums, maximums = [], []
    for index, match in enumerate(matches):
        if match.group("absolute"):
            continue

        unit = match.group("unit")
        if not unit:
            # Shared trailing unit: take it from the next labeled endpoint,
            # but only if it is on the same line.
            for later in matches[index + 1:]:
                if "\n" in text[match.end():later.start()]:
                    break
                if later.group("unit"):
                    unit = later.group("unit")
                    break
        if not unit:
            continue

        dimension, value = _canonical_measurement(
            float(match.group("value")),
            normalize_unit(unit),
        )
        if dimension != expected_dimension:
            continue
        if match.group("label").lower().startswith("min"):
            if numbers_equal(value, expected_min):
                minimums.append(match.start())
        elif numbers_equal(value, expected_max):
            maximums.append(match.start())

    return any(
        abs(low - high) <= _LABELED_ENDPOINT_WINDOW
        for low in minimums
        for high in maximums
    )


# A datasheet table row extracted from a PDF has no "to" or labels, only
# ordered cells, usually one per line:
#
#     VDDA, VDD3P3        <- symbol / parameter text lines
#     3.0                 <- Min
#     3.3                 <- Typ
#     3.6                 <- Max
#     V                   <- Unit
#
# (or a single line: "3.0 3.3 3.6 V", or a pipe-separated row). The endpoints
# of the range are the FIRST and LAST cell of one row. A row is one unbroken
# run of cells: any text line, unit line or row end terminates it, so values
# from different parameter rows are never combined.
_CELL_CORE = rf"(?:{_NUMBER_CORE}(?:\s*{_UNIT_CORE}(?![a-z]))?|-)"
_TABLE_LINE_RE = re.compile(
    rf"\s*\|?\s*(?P<cells>{_CELL_CORE}(?:[\s|]+{_CELL_CORE}){{1,2}})"
    rf"\s*\|?\s*(?P<unit>{_UNIT_CORE}(?![a-z]))?\s*\|?\s*",
    re.IGNORECASE,
)
_SINGLE_CELL_RE = re.compile(
    rf"\s*(?P<value>{_NUMBER_CORE}|-)"
    rf"(?:\s*(?P<unit>{_UNIT_CORE}(?![a-z])))?\s*",
    re.IGNORECASE,
)
_UNIT_ONLY_RE = re.compile(rf"\s*(?P<unit>{_UNIT_CORE}(?![a-z]))\s*", re.IGNORECASE)
_CELL_FINDER_RE = re.compile(
    rf"(?P<value>{_NUMBER_CORE}|-)(?:\s*(?P<unit>{_UNIT_CORE}(?![a-z])))?",
    re.IGNORECASE,
)


def _cell(value_text, unit_text):
    value = None if value_text == "-" else float(value_text)
    return value, normalize_unit(unit_text) if unit_text else None


def _iter_table_rows(text):
    """
    Yield rows as (cells, row_unit); cells is a list of (value|None, unit|None)
    in column order, where None is an empty-cell placeholder.
    """

    run = []

    def flush(row_unit=None):
        nonlocal run
        row = (run, row_unit) if run else None
        run = []
        return row

    for line in normalize_text(text).splitlines():
        line = line.strip()
        if not line:
            continue

        single = _SINGLE_CELL_RE.fullmatch(line)
        if single:
            run.append(_cell(single.group("value"), single.group("unit")))
            continue

        unit_only = _UNIT_ONLY_RE.fullmatch(line)
        if unit_only:
            row = flush(normalize_unit(unit_only.group("unit")))
            if row:
                yield row
            continue

        row = flush()
        if row:
            yield row

        whole_line = _TABLE_LINE_RE.fullmatch(line)
        if whole_line:
            cells = [
                _cell(m.group("value"), m.group("unit"))
                for m in _CELL_FINDER_RE.finditer(whole_line.group("cells"))
            ]
            row_unit = whole_line.group("unit")
            yield cells, normalize_unit(row_unit) if row_unit else None

    row = flush()
    if row:
        yield row


def _table_row_range_matches_expected(text, fact):
    """
    Accept a Min/(Typ)/Max table row whose first and last cells are the
    expected range, in the expected unit, within one row.
    """

    expected_unit = normalize_unit(fact.get("unit"))
    if not expected_unit:
        return False
    expected_dimension, expected_min = _canonical_measurement(
        float(fact["minimum"]), expected_unit
    )
    _, expected_max = _canonical_measurement(
        float(fact["maximum"]), expected_unit
    )

    for cells, row_unit in _iter_table_rows(text):
        if len(cells) not in (2, 3):
            continue  # Min/Typ/Max rows only; longer runs are ambiguous

        low, high = cells[0][0], cells[-1][0]
        if low is None or high is None or low > high:
            continue
        if len(cells) == 3 and cells[1][0] is not None:
            if not low <= cells[1][0] <= high:
                continue

        units = {unit for _, unit in cells if unit}
        if row_unit:
            units.add(row_unit)
        if len(units) != 1:
            continue  # no unit, or conflicting units, inside the row
        unit = units.pop()

        dimension, canonical_low = _canonical_measurement(low, unit)
        _, canonical_high = _canonical_measurement(high, unit)
        if (
            dimension == expected_dimension
            and numbers_equal(canonical_low, expected_min)
            and numbers_equal(canonical_high, expected_max)
        ):
            return True

    return False


def _range_matches_expected(answer, fact):
    """Find both endpoints, with units, in one explicitly expressed range."""

    minimum = float(fact["minimum"])
    maximum = float(fact["maximum"])
    expected_unit = normalize_unit(fact.get("unit"))

    for match in _RANGE_RE.finditer(normalize_text(answer)):
        if (
            match.group("connector").lower() == "and"
            and not match.group("between")
        ):
            continue

        first_unit = normalize_unit(match.group("unit1"))
        second_unit = normalize_unit(match.group("unit2"))

        explicit_units = [
            unit
            for unit in (first_unit, second_unit)
            if unit
        ]
        if not explicit_units:
            continue
        if any(unit != expected_unit for unit in explicit_units):
            continue

        first_unit = first_unit or second_unit
        second_unit = second_unit or first_unit
        first = _canonical_measurement(
            float(match.group("first")),
            first_unit,
        )
        second = _canonical_measurement(
            float(match.group("second")),
            second_unit,
        )
        expected_minimum = _canonical_measurement(
            minimum,
            expected_unit,
        )
        expected_maximum = _canonical_measurement(
            maximum,
            expected_unit,
        )

        if first[0] != second[0] or first[0] != expected_minimum[0]:
            continue

        actual_endpoints = (first[1], second[1])
        expected_endpoints = (
            expected_minimum[1],
            expected_maximum[1],
        )
        if all(
            numbers_equal(actual, expected)
            for actual, expected in zip(
                sorted(actual_endpoints),
                sorted(expected_endpoints),
            )
        ):
            return True

    return (
        _labeled_range_matches_expected(answer, fact)
        or _table_row_range_matches_expected(answer, fact)
    )


# ============================================================
# NUMERIC COMPARISON
# ============================================================

def numbers_equal(actual, expected, tolerance=1e-6):
    """
    Compare numeric values safely.
    """

    return abs(actual - expected) <= tolerance


# ============================================================
# VALUE FACT
# ============================================================

def check_value_fact(answer, fact):
    """
    Check a single expected numeric value.
    """

    expected = float(
        fact["value"]
    )

    expected_unit = fact.get(
        "unit"
    )

    passed = measurement_matches_expected(
        answer,
        expected,
        expected_unit,
    )

    if passed:
        return True, []

    return False, [
        f"value {fact['value']} {expected_unit} not found with a matching unit"
    ]


# ============================================================
# RANGE FACT
# ============================================================

def check_range_fact(answer, fact):
    """
    Check whether an answer contains the expected
    lower and upper values of a range.
    """

    expected_unit = fact.get(
        "unit"
    )

    if _range_matches_expected(answer, fact):
        return True, []

    return False, [
        f"range {fact['minimum']}–{fact['maximum']} "
        f"{expected_unit} not found as one connected range"
    ]


# ============================================================
# CONTAINS ANY
# ============================================================

def check_contains_any(answer, fact):
    """
    Pass if at least one expected textual value appears.
    """

    normalized_answer = normalize_text(
        answer
    )

    values = fact.get(
        "values",
        [],
    )

    for value in values:
        normalized_value = normalize_text(value).strip()
        if len(normalized_value) < 2:
            continue
        pattern = rf"(?<![A-Za-z0-9]){re.escape(normalized_value)}(?![A-Za-z0-9])"
        if re.search(pattern, normalized_answer, flags=re.IGNORECASE):
            return True, []

    return False, [
        "none of: "
        + ", ".join(
            str(value)
            for value in values
        )
    ]


# ============================================================
# CONTAINS ALL
# ============================================================

def check_contains_all(answer, fact):
    """
    Pass if every expected textual value appears.
    """

    normalized_answer = normalize_text(
        answer
    )

    missing = []

    for value in fact.get(
        "values",
        [],
    ):
        if normalize_text(value) not in normalized_answer:
            missing.append(
                str(value)
            )

    return (
        len(missing) == 0,
        missing,
    )


# ============================================================
# TEXT FACT
# ============================================================

def check_text_fact(answer, fact):
    """
    Check whether an expected text phrase appears.
    """

    normalized_answer = normalize_text(
        answer
    )

    expected = normalize_text(
        fact.get(
            "value",
            "",
        )
    )

    if expected in normalized_answer:
        return True, []

    return False, [
        expected
    ]


# ============================================================
# LIST FACT
# ============================================================

def check_list_fact(answer, fact):
    """
    Check a list of expected numeric values.
    """

    expected_values = [
        float(value)
        for value in fact.get(
            "values",
            [],
        )
    ]
    expected_unit = fact.get("unit")

    missing = []

    for expected in expected_values:

        found = measurement_matches_expected(
            answer,
            expected,
            expected_unit,
        )

        if not found:
            missing.append(
                str(expected)
            )

    if (
        missing
        and expected_unit
        and _table_list_matches_expected(
            answer,
            expected_values,
            expected_unit,
        )
    ):
        missing = []

    return (
        len(missing) == 0,
        missing,
    )


def _table_list_matches_expected(text, expected_values, expected_unit):
    """Match a complete list of numeric cells under one explicit unit header."""
    expected_dimension, _ = _canonical_measurement(0, expected_unit)
    for unit, values in _table_measurement_blocks(text):
        dimension, _ = _canonical_measurement(0, unit)
        if dimension != expected_dimension:
            continue

        if all(
            any(
                numbers_equal(value, expected)
                for value in values
            )
            for expected in expected_values
        ):
            return True

    return False


def _table_measurement_blocks(text):
    """Read bare numeric cells only within a heading that explicitly names a unit."""
    lines = [line.strip() for line in normalize_text(text).splitlines()]
    blocks = []

    for header_index, line in enumerate(lines):
        header = _TABLE_UNIT_HEADER_RE.search(line)
        if not header or not line[:header.start()].strip():
            continue

        end = len(lines)
        for index in range(header_index + 1, len(lines)):
            if (
                _TABLE_UNIT_HEADER_RE.search(lines[index])
                or re.match(r"^(?:table|figure)\b", lines[index], re.IGNORECASE)
            ):
                end = index
                break

        values = []
        for cell_line in lines[header_index + 1:end]:
            if re.fullmatch(_NUMBER_CORE, cell_line):
                values.append(float(cell_line))
                continue

            row = _TABLE_LINE_RE.fullmatch(cell_line)
            if row:
                values.extend(
                    float(match.group("value"))
                    for match in _CELL_FINDER_RE.finditer(row.group("cells"))
                    if match.group("value") != "-"
                )

        blocks.append((normalize_unit(header.group("unit")), values))

    return blocks


# ============================================================
# GENERIC FACT EVALUATION
# ============================================================

def evaluate_fact(answer, fact):
    """
    Evaluate one structured expected fact.
    """

    fact_type = fact.get(
        "type"
    )

    if fact_type == "value":
        return check_value_fact(
            answer,
            fact,
        )

    if fact_type == "range":
        return check_range_fact(
            answer,
            fact,
        )

    if fact_type == "contains_any":
        return check_contains_any(
            answer,
            fact,
        )

    if fact_type == "contains_all":
        return check_contains_all(
            answer,
            fact,
        )

    if fact_type == "text":
        return check_text_fact(
            answer,
            fact,
        )

    if fact_type == "list":
        return check_list_fact(
            answer,
            fact,
        )

    return False, [
        f"unsupported fact type: {fact_type}"
    ]


# ============================================================
# ALL FACTS
# ============================================================

def evaluate_facts(answer, expected_facts):
    """
    Evaluate all expected facts for a question.

    A question passes only if every expected fact passes.
    """

    failures = []

    for fact_index, fact in enumerate(
        expected_facts,
        start=1,
    ):

        passed, details = evaluate_fact(
            answer,
            fact,
        )

        if not passed:
            failures.append(
                {
                    "index": fact_index,
                    "type": fact.get(
                        "type"
                    ),
                    "details": details,
                }
            )

    return (
        len(failures) == 0,
        failures,
    )


def describe_missing_numbers(answer, details):
    """
    Diagnostic only (never affects pass/fail): for each missing numeric value
    say whether the number is absent from the answer entirely (the answer is
    incomplete) or present but not in a form the evaluator accepts.
    """

    notes = []
    text = normalize_text(answer)
    for detail in details:
        try:
            number = float(detail)
        except ValueError:
            continue
        pattern = re.compile(
            rf"(?<![\d.]){re.escape(format(number, 'g'))}(?:\.0+)?(?![\d])"
        )
        hits = [
            " ".join(text[max(0, m.start() - 40):m.end() + 20].split())
            for m in pattern.finditer(text)
        ]
        if hits:
            notes.append(
                f"{detail}: present in answer but not accepted: "
                + " | ".join(f"...{hit}..." for hit in hits[:3])
            )
        else:
            notes.append(f"{detail}: absent from the answer (answer incomplete)")
    return notes


# ============================================================
# CITATIONS
# ============================================================

_CITATION_RE = re.compile(
    r"\[([^,\]]+),\s*p\.\s*(\d+)\]",
    re.IGNORECASE,
)


def citation_matches(answer):
    """Extract source/page pairs in the benchmark's citation syntax."""

    if not answer:
        return []

    return [
        (source.strip(), int(page))
        for source, page in _CITATION_RE.findall(answer)
    ]


def citation_sources(answer):
    """
    Extract citation source names from citations such as:

        [TPS63020.pdf, p. 6]
    """

    return [
        match[0]
        for match in citation_matches(answer)
    ]


def citation_present(answer):
    """
    Return True if at least one PDF/page citation exists.
    """

    return bool(
        citation_sources(answer)
    )


def correct_citation_source(
    answer,
    expected_source,
):
    """
    Check whether at least one citation references
    the expected PDF.
    """

    sources = citation_sources(
        answer
    )

    for source in sources:

        if (
            source.strip().lower()
            == expected_source.lower()
        ):
            return True

    return False


def _citation_context(context_records):
    """Flatten benchmark/pipeline records for the citation validator."""

    flattened = []
    for record in context_records:
        metadata = get_record_metadata(record)
        flattened.append(
            {
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "text": get_record_text(record),
            }
        )
    return flattened


def citation_validation_problems(
    answer,
    expected_source,
    context_records,
):
    """
    Strict citation validation. Returns a list of problems (empty = valid).

    Every citation must resolve to a source and page present in the supplied
    context, no unresolved [C#] may remain, and each claim must be cited and
    pass basic numeric/terminology support checks.
    See src/citation_validation.py for exactly what is and is not verified.
    """

    return validate_citations(
        answer,
        _citation_context(context_records),
        expected_source=expected_source,
        not_found_message=NOT_FOUND_MESSAGE,
    ).problems


def is_abstention_answer(answer):
    """Return whether the answer is exactly the configured abstention."""

    return (
        isinstance(answer, str)
        and answer.strip() == NOT_FOUND_MESSAGE.strip()
    )


def citation_page_context_valid(
    answer,
    expected_source,
    context_records,
):
    """
    True only if the answer passes strict citation validation (identity,
    coverage and numeric support), or is the configured abstention. Previously
    this passed whenever ANY single citation was valid, so invalid pages,
    unresolved [C#] references, unrelated pages and uncited claims were not
    detected.
    """

    return not citation_validation_problems(
        answer,
        expected_source,
        context_records,
    )


def summarize_citation_metrics(results):
    """Summarize successful generations; citation denominators exclude abstentions."""

    abstentions = sum(result["abstained"] for result in results)
    answered = [result for result in results if not result["abstained"]]

    return {
        "answer_count": len(results),
        "abstention_count": abstentions,
        "non_abstained_count": len(answered),
        "citation_count": sum(result["citation"] for result in answered),
        "citation_source_count": sum(
            result["citation_source"] for result in answered
        ),
        "citation_page_context_count": sum(
            result["citation_page_context"] for result in answered
        ),
    }


def format_rate(count, denominator):
    """Format a count/rate without dividing by an empty answer set."""

    if not denominator:
        return f"{count}/{denominator} = N/A"
    return f"{count}/{denominator} = {count / denominator * 100:.2f}%"


def run_evaluator_self_tests():
    """Run deterministic evaluator checks before loading the RAG pipeline."""

    range_8_96 = {
        "type": "range",
        "minimum": 8,
        "maximum": 96,
        "unit": "kHz",
    }
    range_voltage = {
        "type": "range",
        "minimum": 1.71,
        "maximum": 3.6,
        "unit": "V",
    }
    range_mhz = {
        "type": "value",
        "value": 2.4,
        "unit": "MHz",
    }
    tests = [
        (
            "attached kHz range",
            evaluate_fact(
                "Sample rates: between 8kHz and 96kHz.",
                range_8_96,
            )[0],
            True,
        ),
        (
            "attached MHz value",
            evaluate_fact("Switching frequency is 2.4MHz.", range_mhz)[0],
            True,
        ),
        (
            "incorrect frequency is rejected",
            evaluate_fact("Switching frequency is 2.2 MHz.", range_mhz)[0],
            False,
        ),
        (
            "incorrect high frequency is rejected",
            evaluate_fact("Switching frequency is 2.6 MHz.", range_mhz)[0],
            False,
        ),
        (
            "equivalent frequency units",
            evaluate_fact("Typical frequency is 2400 kHz.", range_mhz)[0],
            True,
        ),
        (
            "attached voltage values",
            evaluate_fact(
                "VDD_SPI accepts 1.8V or 3.3V.",
                {
                    "type": "list",
                    "values": [1.8, 3.3],
                    "unit": "V",
                },
            )[0],
            True,
        ),
        (
            "attached temperature unit",
            evaluate_fact(
                "Maximum temperature is 85°C.",
                {"type": "value", "value": 85, "unit": "°C"},
            )[0],
            True,
        ),
        (
            "omega resistance unit",
            evaluate_fact(
                "Speaker load is 4Ω.",
                {"type": "value", "value": 4, "unit": "Ω"},
            )[0],
            True,
        ),
        (
            "word resistance unit",
            evaluate_fact(
                "Speaker load is 4 ohm.",
                {"type": "value", "value": 4, "unit": "Ω"},
            )[0],
            True,
        ),
        (
            "attached current unit",
            evaluate_fact(
                "The peak current is 4A.",
                {"type": "value", "value": 4, "unit": "A"},
            )[0],
            True,
        ),
        (
            "standard voltage range",
            evaluate_fact(
                "Supply range: 1.71–3.6 V.",
                range_voltage,
            )[0],
            True,
        ),
        (
            "reverse voltage range",
            evaluate_fact("3.6 V down to 1.71 V.", range_voltage)[0],
            True,
        ),
        (
            "unconnected range endpoints",
            evaluate_fact(
                "1.71 appears in one unrelated specification and 3.6 V appears elsewhere.",
                range_voltage,
            )[0],
            False,
        ),
        (
            "part number is not a measurement",
            extract_numbers("MAX98357A") == [],
            True,
        ),
        (
            "documented STM32 suffix temperatures",
            evaluate_fact(
                "Suffix 6: 85°C; suffix 7: 105 °C; suffix 3: 125°C.",
                {
                    "type": "list",
                    "values": [85, 105, 125],
                    "unit": "°C",
                },
            )[0],
            True,
        ),
        (
            "citation source syntax",
            citation_present("[DS_stm32l476je.pdf, p. 2]")
            and correct_citation_source(
                "[DS_stm32l476je.pdf, p. 2]",
                "DS_stm32l476je.pdf",
            ),
            True,
        ),
        (
            "citation page must be in context",
            not citation_page_context_valid(
                "[DS_stm32l476je.pdf, p. 999]",
                "DS_stm32l476je.pdf",
                [
                    {
                        "source": "DS_stm32l476je.pdf",
                        "page": 2,
                        "text": "datasheet context",
                    }
                ],
            ),
            True,
        ),
        (
            "strict citations: valid answer passes",
            citation_page_context_valid(
                "Range is 1.71 V to 3.6 V [DS_stm32l476je.pdf, p. 2].",
                "DS_stm32l476je.pdf",
                [
                    {
                        "source": "DS_stm32l476je.pdf",
                        "page": 2,
                        "text": "Operating range 1.71 V to 3.6 V",
                    }
                ],
            ),
            True,
        ),
        (
            "strict citations: one invalid page among valid ones fails",
            citation_page_context_valid(
                "Range is 1.71 V to 3.6 V [DS_stm32l476je.pdf, p. 2]. "
                "Also 5 mA [DS_stm32l476je.pdf, p. 999].",
                "DS_stm32l476je.pdf",
                [
                    {
                        "source": "DS_stm32l476je.pdf",
                        "page": 2,
                        "text": "Operating range 1.71 V to 3.6 V",
                    }
                ],
            ),
            False,
        ),
        (
            "strict citations: unresolved [C#] fails",
            citation_page_context_valid(
                "Range is 1.71 V to 3.6 V [DS_stm32l476je.pdf, p. 2] [C9].",
                "DS_stm32l476je.pdf",
                [
                    {
                        "source": "DS_stm32l476je.pdf",
                        "page": 2,
                        "text": "Operating range 1.71 V to 3.6 V",
                    }
                ],
            ),
            False,
        ),
        (
            "strict citations: uncited measurement fails",
            citation_page_context_valid(
                "Range is 1.71 V to 3.6 V. Clock is 80 MHz "
                "[DS_stm32l476je.pdf, p. 2].",
                "DS_stm32l476je.pdf",
                [
                    {
                        "source": "DS_stm32l476je.pdf",
                        "page": 2,
                        "text": "Operating range 1.71 V to 3.6 V",
                    }
                ],
            ),
            False,
        ),
    ]

    failures = []
    for name, actual, expected in tests:
        passed = actual is expected
        print(f"Evaluator self-test [{ 'PASS' if passed else 'FAIL' }]: {name}")
        if not passed:
            failures.append(name)

    if failures:
        raise AssertionError(
            "Evaluator self-tests failed: " + ", ".join(failures)
        )


# ============================================================
# RECORD HELPERS
# ============================================================

def get_record_text(record):
    """
    Return textual content from a record.

    Supports both:

        text

    and older:

        content
    """

    if not isinstance(
        record,
        dict,
    ):
        return ""

    text = record.get(
        "text"
    )

    if (
        isinstance(text, str)
        and text.strip()
    ):
        return text

    content = record.get(
        "content"
    )

    if (
        isinstance(content, str)
        and content.strip()
    ):
        return content

    return ""


def get_record_metadata(record):
    """
    Return metadata from both possible record structures.

    Current canonical records are flat:

        {
            "text": "...",
            "source": "...",
            "page": ...,
            "chunk_id": ...
        }

    Older records may contain:

        {
            "content": "...",
            "metadata": {...}
        }
    """

    if not isinstance(
        record,
        dict,
    ):
        return {}

    metadata = {}

    nested_metadata = record.get(
        "metadata"
    )

    if isinstance(
        nested_metadata,
        dict,
    ):
        metadata.update(
            nested_metadata
        )

    for key in (
        "source",
        "page",
        "chunk_id",
    ):
        if key in record:
            metadata[key] = record[key]

    return metadata


def evaluate_record_facts(records, expected_source, expected_facts):
    """
    Find expected facts within individual retrieved chunks.

    Returns per-fact chunk locations so the benchmark can show whether evidence
    was retrieved and whether it survived reranking into the final context.
    """

    diagnostics = []

    for fact_index, fact in enumerate(expected_facts, start=1):
        matching_chunks = []

        for rank, record in enumerate(records, start=1):
            metadata = get_record_metadata(record)
            if str(metadata.get("source", "")).lower() != expected_source.lower():
                continue

            passed = evaluate_fact(
                get_record_text(record),
                fact,
            )[0]
            if not passed:
                continue

            matching_chunks.append(
                {
                    "rank": rank,
                    "source": metadata.get("source", "N/A"),
                    "page": metadata.get("page", "N/A"),
                    "chunk_id": metadata.get("chunk_id", "N/A"),
                }
            )

        diagnostics.append(
            {
                "index": fact_index,
                "type": fact.get("type"),
                "matches": matching_chunks,
            }
        )

    return (
        all(item["matches"] for item in diagnostics),
        diagnostics,
    )


def print_record_fact_diagnostics(label, passed, diagnostics):
    print(
        f"{label}: {'PASS' if passed else 'FAIL'}"
    )

    for fact in diagnostics:
        if fact["matches"]:
            locations = ", ".join(
                (
                    f"rank {match['rank']} "
                    f"{match['source']} p.{match['page']} "
                    f"chunk {match['chunk_id']}"
                )
                for match in fact["matches"]
            )
        else:
            locations = "no matching fact-bearing chunk"

        print(
            f"  Fact {fact['index']} ({fact['type']}): {locations}"
        )


def configure_console_output():
    """Prevent unsupported console characters from crashing benchmark output."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


# ============================================================
# DIAGNOSTIC CONTEXT
# ============================================================

def print_context(records):
    """
    Print the exact top-5 context sent to the LLM.
    """

    print()
    print("=" * 70)
    print("DIAGNOSTIC CONTEXT")
    print("=" * 70)

    if not records:
        print()
        print("NO CONTEXT RECORDS")
        return

    for i, record in enumerate(
        records,
        start=1,
    ):

        metadata = get_record_metadata(
            record
        )

        content = get_record_text(
            record
        )

        print()
        print(
            f"[C{i}]"
        )

        print(
            f"Source: "
            f"{metadata.get('source', 'N/A')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page', 'N/A')}"
        )

        print(
            f"Chunk ID: "
            f"{metadata.get('chunk_id', 'N/A')}"
        )

        if "rerank_score" in record:
            print(
                f"Rerank score: "
                f"{record['rerank_score']}"
            )

        print(
            "Content:"
        )

        print(content)


# ============================================================
# EXPANSION HELPER
# ============================================================

def get_expanded_queries(question):
    """
    Return the original question plus expansion queries.

    EXPANSIONS entries use the structure:

        question: (
            relevant_pages,
            [
                query1,
                query2,
                ...
            ],
        )

    The page set is metadata for the expansion system.
    Only the actual query list should be passed to
    expanded_search().
    """

    queries = [
        question
    ]

    expansion = EXPANSIONS.get(
        question
    )

    if not expansion:
        return queries

    # Current EXPANSIONS format:
    #
    # (
    #     pages,
    #     queries,
    # )
    #
    # Example:
    #
    # (
    #     {4},
    #     [
    #         "TPS63020 switching frequency",
    #         ...
    #     ],
    # )

    if (
        isinstance(expansion, tuple)
        and len(expansion) == 2
    ):
        _pages, expansion_queries = expansion

        if isinstance(
            expansion_queries,
            list,
        ):
            queries.extend(
                expansion_queries
            )

        return queries

    # Defensive compatibility with
    # older EXPANSIONS format where
    # the value may simply be a list.

    if isinstance(
        expansion,
        list,
    ):
        queries.extend(
            expansion
        )

    return queries


# ============================================================
# MAIN
# ============================================================

def main():

    configure_console_output()
    run_evaluator_self_tests()

    print()
    print("=" * 70)
    print(
        "ElectroRAG - Diagnostic RAG Benchmark V3"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    print()
    print("Loading PDFs...")

    documents = load_all_pdfs()

    print(
        f"Documents loaded: {len(documents)}"
    )

    available_sources = {
        str(document.get("source", "")).lower()
        for document in documents
    }
    source_available_count = sum(
        1
        for item in QUESTIONS
        if item["source"].lower() in available_sources
    )

    # --------------------------------------------------------
    # Chunk documents
    # --------------------------------------------------------

    print()
    print("Building chunks...")

    chunks = chunk_documents(
        documents
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print()
    print("Building BM25 index...")

    bm25 = build_index(
        chunks
    )

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    print()
    print("Opening ChromaDB collection...")

    collection = open_collection()

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print()
    print("Loading embedding model...")

    model = load_model()

    # --------------------------------------------------------
    # Cross encoder
    # --------------------------------------------------------

    print()
    print(
        "Loading cross-encoder reranker..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print()
    print(
        "Collection: electrorag"
    )

    print(
        f"Records:    {collection.count()}"
    )

    print(
        f"Reranker:   {RERANKER_MODEL}"
    )

    # --------------------------------------------------------
    # Result counters
    # --------------------------------------------------------

    retrieval_r1 = 0
    retrieval_r3 = 0
    retrieval_r5 = 0
    fact_candidate_count = 0
    fact_top5_count = 0

    generation_nonempty = 0
    generation_content = 0

    citation_results = []

    failed_retrieval = []
    failed_generation = []

    # ========================================================
    # QUESTIONS
    # ========================================================

    for index, item in enumerate(
        QUESTIONS,
        start=1,
    ):

        question_id = item[
            "id"
        ]

        question = item[
            "question"
        ]

        expected_source = item[
            "source"
        ]

        expected_facts = item[
            "expected_facts"
        ]

        print()
        print("=" * 70)

        print(
            f"[{index}/{len(QUESTIONS)}] "
            f"{question}"
        )

        print("=" * 70)

        print(
            f"Expected source: "
            f"{expected_source}"
        )
        print(
            "Source PDF available: "
            f"{'PASS' if expected_source.lower() in available_sources else 'FAIL'}"
        )

        # ====================================================
        # RETRIEVAL TEST
        # ====================================================

        print()
        print("Retrieval:")

        retrieval_results = retrieve(
            collection,
            model,
            question,
            top_k=TOP_K_RETRIEVAL,
        )

        retrieved_sources = []

        for result in retrieval_results:

            metadata = get_record_metadata(
                result
            )

            source = metadata.get(
                "source",
                "",
            )

            retrieved_sources.append(str(source))

        r1 = (
            len(retrieved_sources) >= 1
            and retrieved_sources[0].lower()
            == expected_source.lower()
        )

        r3 = any(
            source.lower()
            == expected_source.lower()
            for source in retrieved_sources[:3]
        )

        r5 = any(
            source.lower()
            == expected_source.lower()
            for source in retrieved_sources[:5]
        )

        if r1:
            retrieval_r1 += 1

        if r3:
            retrieval_r3 += 1

        if r5:
            retrieval_r5 += 1

        print(
            f"Document Recall@1: "
            f"{'PASS' if r1 else 'FAIL'} | "
            f"Document Recall@3: "
            f"{'PASS' if r3 else 'FAIL'} | "
            f"Document Recall@5: "
            f"{'PASS' if r5 else 'FAIL'}"
        )
        print(
            "These document-level metrics check PDF identity only; "
            "they do not prove fact retrieval."
        )

        for rank, result in enumerate(
            retrieval_results,
            start=1,
        ):

            metadata = get_record_metadata(
                result
            )

            source = metadata.get(
                "source",
                "",
            )

            page = metadata.get(
                "page",
                "",
            )

            distance = result.get(
                "distance",
                "",
            )

            marker = (
                "[PASS]"
                if str(source).lower() == expected_source.lower()
                else "[FAIL]"
            )

            if isinstance(
                distance,
                (int, float),
            ):
                distance_text = (
                    f"{distance:.4f}"
                )
            else:
                distance_text = str(
                    distance
                )

            print(
                f"  {rank}. {marker} "
                f"{source} | "
                f"page {page} | "
                f"distance {distance_text}"
            )

        if not r5:
            failed_retrieval.append(
                question_id
            )

        # ====================================================
        # GENERATION TEST
        # ====================================================

        print()
        print("Generation:")

        top_records = []
        candidates = []
        answer = ""
        candidate_fact_pass = False
        candidate_fact_diagnostics = []
        top5_fact_pass = False
        top5_fact_diagnostics = []

        try:

            # ------------------------------------------------
            # Query expansion
            # ------------------------------------------------

            queries = get_expanded_queries(
                question
            )

            print(
                f"Search queries: {len(queries)}"
            )

            # ------------------------------------------------
            # Hybrid retrieval
            # ------------------------------------------------

            candidates = expanded_search(
                queries,
                collection,
                model,
                bm25,
                chunks,
            )

            if not isinstance(
                candidates,
                list,
            ):
                raise TypeError(
                    "expanded_search() must return a list."
                )

            (
                candidate_fact_pass,
                candidate_fact_diagnostics,
            ) = evaluate_record_facts(
                candidates,
                expected_source,
                expected_facts,
            )
            if candidate_fact_pass:
                fact_candidate_count += 1

            print_record_fact_diagnostics(
                "Fact-bearing evidence in hybrid candidates",
                candidate_fact_pass,
                candidate_fact_diagnostics,
            )

            # ------------------------------------------------
            # Reranking
            # ------------------------------------------------

            reranked = rerank(
                reranker,
                question,
                candidates,
            )

            if not isinstance(
                reranked,
                list,
            ):
                raise TypeError(
                    "rerank() must return a list."
                )

            # ------------------------------------------------
            # Top 5 context
            # ------------------------------------------------

            top_records = reranked[
                :CONTEXT_CHUNKS
            ]

            (
                top5_fact_pass,
                top5_fact_diagnostics,
            ) = evaluate_record_facts(
                top_records,
                expected_source,
                expected_facts,
            )
            if top5_fact_pass:
                fact_top5_count += 1

            print_record_fact_diagnostics(
                "Fact-bearing evidence in final top-5",
                top5_fact_pass,
                top5_fact_diagnostics,
            )

            # ------------------------------------------------
            # Pass canonical records directly
            # ------------------------------------------------

            context_hits = top_records

            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            answer = generate_answer(
                question,
                context_hits,
            )

            if answer is None:
                answer = ""

            if not isinstance(
                answer,
                str,
            ):
                answer = str(
                    answer
                )

            answer = answer.strip()
            abstained = is_abstention_answer(answer)

            print()
            print("Answer:")
            print(answer)

            # ------------------------------------------------
            # Non-empty
            # ------------------------------------------------

            nonempty = (
                len(answer) > 0
            )

            if nonempty:
                generation_nonempty += 1

            # ------------------------------------------------
            # Structured expected facts
            # ------------------------------------------------

            (
                content_pass,
                fact_failures,
            ) = evaluate_facts(
                answer,
                expected_facts,
            )

            if content_pass:
                generation_content += 1

            # ------------------------------------------------
            # Citation presence
            # ------------------------------------------------

            citation_pass = None
            citation_source_pass = None
            citation_page_context_pass = None
            if not abstained:
                citation_pass = citation_present(answer)
                citation_source_pass = correct_citation_source(
                    answer, expected_source
                )
                citation_page_context_pass = citation_page_context_valid(
                    answer,
                    expected_source,
                    top_records,
                )
            citation_results.append(
                {
                    "abstained": abstained,
                    "citation": citation_pass or False,
                    "citation_source": citation_source_pass or False,
                    "citation_page_context": (
                        citation_page_context_pass or False
                    ),
                }
            )

            # ------------------------------------------------
            # Print evaluation
            # ------------------------------------------------

            print()

            print(
                f"Non-empty: "
                f"{'PASS' if nonempty else 'FAIL'}"
            )

            print(
                f"Expected content: "
                f"{'PASS' if content_pass else 'FAIL'}"
            )

            print(
                f"Abstention: "
                f"{'YES' if abstained else 'NO'}"
            )

            if not content_pass:

                print(
                    "Failed facts:"
                )

                for failure in fact_failures:

                    print(
                        f"  Fact "
                        f"{failure['index']} "
                        f"({failure['type']}): "
                        + "; ".join(
                            failure["details"]
                        )
                    )

                    for note in describe_missing_numbers(
                        answer,
                        failure["details"],
                    ):
                        print(f"    {note}")

                print(
                    "Evidence index sent to the model "
                    "(diagnostic):"
                )
                print(
                    build_evidence_index(
                        question,
                        normalize_records(context_hits),
                    ).strip()
                    or "  (none: no table rows matched the question)"
                )

            print(
                f"Citation syntax present: "
                f"{'N/A (abstained)' if citation_pass is None else ('PASS' if citation_pass else 'FAIL')}"
            )

            print(
                f"Citation source correct: "
                f"{'N/A (abstained)' if citation_source_pass is None else ('PASS' if citation_source_pass else 'FAIL')}"
            )

            print(
                f"Citation page/context valid: "
                f"{'N/A (abstained)' if citation_page_context_pass is None else ('PASS' if citation_page_context_pass else 'FAIL')}"
            )

            if citation_page_context_pass is False:
                for citation_problem in citation_validation_problems(
                    answer,
                    expected_source,
                    top_records,
                ):
                    print(f"  - {citation_problem}")

            # ------------------------------------------------
            # Diagnostic context
            # ------------------------------------------------

            if (
                not nonempty
                or not content_pass
                or citation_pass is False
                or citation_source_pass is False
                or citation_page_context_pass is False
                or not candidate_fact_pass
                or not top5_fact_pass
            ):

                print_context(
                    top_records
                )

                failed_generation.append(
                    {
                        "id": question_id,
                        "nonempty": nonempty,
                        "content": content_pass,
                        "abstained": abstained,
                        "citation": citation_pass,
                        "citation_source": citation_source_pass,
                        "citation_page_context": citation_page_context_pass,
                        "fact_candidates": candidate_fact_pass,
                        "fact_top5": top5_fact_pass,
                        "fact_failures": fact_failures,
                    }
                )

        except Exception as exc:

            print()
            print(
                "Generation ERROR:"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            print_context(
                top_records
            )

            failed_generation.append(
                {
                    "id": question_id,
                    "nonempty": False,
                    "content": False,
                    "abstained": False,
                    "citation": False,
                    "citation_source": False,
                    "citation_page_context": False,
                    "fact_candidates": candidate_fact_pass,
                    "fact_top5": top5_fact_pass,
                    "fact_failures": [
                        {
                            "index": 0,
                            "type": "exception",
                            "details": [
                                f"{type(exc).__name__}: {exc}"
                            ],
                        }
                    ],
                }
            )

    # ========================================================
    # RETRIEVAL RESULTS
    # ========================================================

    total = len(
        QUESTIONS
    )
    citation_metrics = summarize_citation_metrics(citation_results)

    print()
    print()
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)
    print()

    print(
        f"Questions: {total}"
    )

    print(
        f"\nDocument Recall@1: "
        f"{retrieval_r1}/{total} = "
        f"{retrieval_r1 / total * 100:.2f}%"
    )

    print(
        f"Document Recall@3: "
        f"{retrieval_r3}/{total} = "
        f"{retrieval_r3 / total * 100:.2f}%"
    )

    print(
        f"Document Recall@5: "
        f"{retrieval_r5}/{total} = "
        f"{retrieval_r5 / total * 100:.2f}%"
    )

    print()
    print(
        "FACT-LEVEL RETRIEVAL"
    )
    print(
        f"Source PDFs available: "
        f"{source_available_count}/{total} = "
        f"{source_available_count / total * 100:.2f}%"
    )
    print(
        f"Expected facts found in hybrid candidates: "
        f"{fact_candidate_count}/{total} = "
        f"{fact_candidate_count / total * 100:.2f}%"
    )
    print(
        f"Expected facts found in final top-5 context: "
        f"{fact_top5_count}/{total} = "
        f"{fact_top5_count / total * 100:.2f}%"
    )

    # ========================================================
    # GENERATION RESULTS
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("GENERATION RESULTS")
    print("=" * 70)
    print()

    print(
        f"Non-empty answers: "
        f"{generation_nonempty}/{total} = "
        f"{generation_nonempty / total * 100:.2f}%"
    )

    print(
        f"Expected content: "
        f"{generation_content}/{total} = "
        f"{generation_content / total * 100:.2f}%"
    )

    print(
        f"Abstentions: "
        f"{citation_metrics['abstention_count']}/{total} = "
        f"{citation_metrics['abstention_count'] / total * 100:.2f}%"
    )
    print("Citation metric denominators exclude abstentions and generation errors.")

    print(
        f"Citation syntax present (non-abstained answers): "
        f"{format_rate(citation_metrics['citation_count'], citation_metrics['non_abstained_count'])}"
    )

    print(
        f"Citation source correct (non-abstained answers): "
        f"{format_rate(citation_metrics['citation_source_count'], citation_metrics['non_abstained_count'])}"
    )

    print(
        f"Citation page/context valid (non-abstained answers): "
        f"{format_rate(citation_metrics['citation_page_context_count'], citation_metrics['non_abstained_count'])}"
    )

    # ========================================================
    # FAILED RETRIEVAL
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FAILED RETRIEVAL QUESTIONS")
    print("=" * 70)

    if failed_retrieval:

        for question_id in failed_retrieval:

            print(
                f"- {question_id}"
            )

    else:

        print()
        print("None.")

    # ========================================================
    # FAILED GENERATION
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FAILED GENERATION QUESTIONS")
    print("=" * 70)

    if failed_generation:

        for failure in failed_generation:

            print()

            print(
                f"- {failure['id']}"
            )

            print(
                f"  Non-empty: "
                f"{'PASS' if failure['nonempty'] else 'FAIL'}"
            )

            print(
                f"  Expected content: "
                f"{'PASS' if failure['content'] else 'FAIL'}"
            )

            print(
                f"  Abstention: "
                f"{'YES' if failure['abstained'] else 'NO'}"
            )

            if failure[
                "fact_failures"
            ]:

                print(
                    "  Failed facts:"
                )

                for fact_failure in failure[
                    "fact_failures"
                ]:

                    print(
                        f"    Fact "
                        f"{fact_failure['index']} "
                        f"({fact_failure['type']}): "
                        + "; ".join(
                            fact_failure["details"]
                        )
                    )

            print(
                f"  Citation syntax present: "
                f"{'N/A (abstained)' if failure['citation'] is None else ('PASS' if failure['citation'] else 'FAIL')}"
            )

            print(
                f"  Citation source correct: "
                f"{'N/A (abstained)' if failure['citation_source'] is None else ('PASS' if failure['citation_source'] else 'FAIL')}"
            )

            print(
                f"  Citation page/context: "
                f"{'N/A (abstained)' if failure['citation_page_context'] is None else ('PASS' if failure['citation_page_context'] else 'FAIL')}"
            )

            print(
                f"  Fact in hybrid candidates: "
                f"{'PASS' if failure['fact_candidates'] else 'FAIL'}"
            )

            print(
                f"  Fact in final top-5: "
                f"{'PASS' if failure['fact_top5'] else 'FAIL'}"
            )
    else:

        print()
        print("None.")

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print()

    print(
        "Retrieval benchmark: COMPLETE"
    )

    print(
        "Generation benchmark: COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()