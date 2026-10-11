"""
Regression tests for strict citation validation.

Covers the five cases from the stability audit (all of which the old
"at least one valid citation" check wrongly passed), answers with several
citations where only one is invalid, and correct answer formats that must
keep passing.

The numeric-support layer checks that numbers appear on the cited page. It is
NOT semantic verification; the tests below document both what it catches and
what it deliberately does not.
"""

import unittest

from src.benchmark import citation_page_context_valid, citation_validation_problems
from src.citation_resolver import resolve_citations
from src.citation_validation import validate_citations

SRC = "ESP32-S3.pdf"
NOT_FOUND = "I could not find this information in the indexed datasheet."

CTX = [
    {   # cell-per-line table text, as PDF extraction produces it
        "source": SRC, "page": 64,
        "text": "Recommended Operating Conditions\nVDD3P3\n3.0\n3.3\n3.6\nV\nTA\n\u201340\n85\n\u00b0C",
    },
    {"source": SRC, "page": 12, "text": "Clock: 40 MHz crystal. Core uses 5 mA."},
    {"source": "MAX98357A.pdf", "page": 3, "text": "Supply 2.5 V to 5.5 V. Gain 9 dB."},
]


def ok(answer, **kw):
    return validate_citations(answer, CTX, expected_source=SRC, **kw)


class AuditCaseTests(unittest.TestCase):
    """The five cases the old check passed."""

    def test_baseline_valid_answer_passes(self):
        self.assertTrue(ok("VDD is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64].").valid)

    def test_case1_valid_plus_page_not_in_context_fails(self):
        r = ok("VDD is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64]. Also 5 mA [ESP32-S3.pdf, p. 999].")
        self.assertFalse(r.valid)
        self.assertEqual(len(r.identity), 1)
        self.assertIn("p. 999", r.identity[0])

    def test_case2_valid_plus_unresolved_c_reference_fails(self):
        r = ok("VDD is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64]. Also 5 mA [C9].")
        self.assertFalse(r.valid)
        self.assertTrue(any("unresolved citation reference [C9]" in p for p in r.identity))

    def test_case3_cited_page_in_context_but_unrelated_fails(self):
        r = ok("VDD is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 12].")
        self.assertFalse(r.valid)
        self.assertEqual(r.identity, [])
        self.assertEqual(len(r.support), 1)

    def test_case4_uncited_claim_with_citation_elsewhere_fails(self):
        r = ok("VDD is 3.0 V to 3.6 V. Crystal is 40 MHz [ESP32-S3.pdf, p. 12].")
        self.assertFalse(r.valid)
        self.assertEqual(len(r.coverage), 1)
        self.assertIn("uncited claim", r.coverage[0])

    def test_case5_no_citations_at_all_fails(self):
        r = ok("VDD is 3.0 V to 3.6 V.")
        self.assertFalse(r.valid)
        self.assertIn("answer contains no citations", r.coverage)


class BenchmarkWrapperTests(unittest.TestCase):
    """The benchmark's own function must apply the same strictness."""

    def check(self, answer):
        return citation_page_context_valid(answer, SRC, CTX)

    def test_all_five_audit_cases_now_rejected(self):
        for answer in (
            "VDD 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64]. Also 5 mA [ESP32-S3.pdf, p. 999].",
            "VDD 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64]. Also 5 mA [C9].",
            "VDD 3.0 V to 3.6 V [ESP32-S3.pdf, p. 12].",
            "VDD is 3.0 V to 3.6 V. Crystal is 40 MHz [ESP32-S3.pdf, p. 12].",
            "VDD is 3.0 V to 3.6 V.",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(self.check(answer))

    def test_correct_answer_still_passes(self):
        self.assertTrue(self.check("VDD is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64]."))

    def test_problems_are_reported_as_text(self):
        problems = citation_validation_problems(
            "VDD 3.0 V [ESP32-S3.pdf, p. 64]. Also 5 mA [ESP32-S3.pdf, p. 999].", SRC, CTX
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("not in the retrieved context", problems[0])

    def test_page_999_original_self_test_still_rejected(self):
        self.assertFalse(citation_page_context_valid(
            "[ESP32-S3.pdf, p. 999]", SRC, CTX))


class MultipleCitationsOneInvalidTests(unittest.TestCase):
    def test_adjacent_citations_second_page_invalid(self):
        r = ok("VDD is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64][ESP32-S3.pdf, p. 999].")
        self.assertFalse(r.valid)
        self.assertEqual(len(r.identity), 1)

    def test_grouped_reference_with_invalid_id_after_resolution(self):
        # "[C1, C9]" with only 3 context blocks: resolver leaves [C9] visible.
        resolved = resolve_citations("VDD is 3.0 V to 3.6 V [C1, C9].", CTX)
        self.assertIn("[C9]", resolved)
        r = ok(resolved)
        self.assertFalse(r.valid)
        self.assertTrue(any("[C9]" in p for p in r.identity))

    def test_only_the_invalid_citation_is_reported(self):
        r = ok(
            "Min 3.0 V [ESP32-S3.pdf, p. 64]. Max 3.6 V [ESP32-S3.pdf, p. 64]. "
            "Crystal 40 MHz [ESP32-S3.pdf, p. 77]."
        )
        self.assertEqual(r.coverage, [])
        self.assertEqual(r.support, [])
        self.assertEqual(len(r.identity), 1)
        self.assertIn("p. 77", r.identity[0])

    def test_unknown_source_is_rejected(self):
        r = ok("VDD is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64]. Gain 9 dB [OTHER.pdf, p. 3].")
        self.assertFalse(r.valid)
        self.assertTrue(any("source that is not in the retrieved context" in p for p in r.identity))

    def test_malformed_citation_is_rejected(self):
        self.assertFalse(ok("VDD is 3.0 V [ESP32-S3.pdf, page 64].").valid)
        self.assertFalse(ok("VDD is 3.0 V [ESP32-S3.pdf p. 64].").valid)

    def test_expected_source_must_be_cited(self):
        r = ok("Supply 2.5 V to 5.5 V [MAX98357A.pdf, p. 3].")
        self.assertFalse(r.valid)
        self.assertTrue(any("expected source" in p for p in r.identity))


class CorrectAnswersStillPassTests(unittest.TestCase):
    def assertValid(self, answer):
        r = ok(answer)
        self.assertTrue(r.valid, r.problems)

    def test_each_bullet_cited(self):
        self.assertValid(
            "The operating voltage is:\n"
            "- Minimum: 3.0 V [ESP32-S3.pdf, p. 64]\n"
            "- Typical: 3.3 V [ESP32-S3.pdf, p. 64]\n"
            "- Maximum: 3.6 V [ESP32-S3.pdf, p. 64]"
        )

    def test_label_after_format(self):
        self.assertValid("3.0 V (Min), 3.3 V (Typ), 3.6 V (Max) [ESP32-S3.pdf, p. 64]")

    def test_citation_after_sentence_period(self):
        self.assertValid("VDD is 3.0 V to 3.6 V. [ESP32-S3.pdf, p. 64]")

    def test_decimals_and_page_numbers_do_not_split_sentences(self):
        self.assertValid("VDD is 3.0 V to 3.6 V, typically 3.3 V [ESP32-S3.pdf, p. 64].")

    def test_negative_and_range_temperatures(self):
        self.assertValid("Ambient range is -40 to 85 \u00b0C [ESP32-S3.pdf, p. 64].")

    def test_equivalent_unit_scaling(self):
        self.assertValid("Minimum is 3000 mV [ESP32-S3.pdf, p. 64].")

    def test_two_documents_both_in_context(self):
        self.assertValid(
            "VDD is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64]. "
            "The amplifier supply is 2.5 V to 5.5 V [MAX98357A.pdf, p. 3]."
        )

    def test_lead_in_line_without_measurement_needs_no_citation(self):
        self.assertValid("Here are the values.\nVDD is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64].")

    def test_end_to_end_with_real_resolver(self):
        resolved = resolve_citations("VDD is 3.0 V to 3.6 V [C1, C2].", CTX)
        self.assertValid(resolved)

    def test_not_found_message_is_allowed_only_when_configured(self):
        self.assertTrue(validate_citations(NOT_FOUND, CTX, not_found_message=NOT_FOUND).valid)
        self.assertFalse(validate_citations(NOT_FOUND, CTX).valid)

    def test_empty_answer_fails(self):
        self.assertFalse(ok("").valid)
        self.assertFalse(ok(None).valid)


class SupportLayerLimitsTests(unittest.TestCase):
    """What the numeric-support layer catches, and what it cannot."""

    def test_number_absent_from_cited_page_is_caught(self):
        r = ok("Maximum is 4.2 V [ESP32-S3.pdf, p. 64].")
        self.assertEqual(len(r.support), 1)
        self.assertIn("4.2", r.support[0])

    def test_wrong_attribution_of_a_present_number_is_NOT_caught(self):
        # 85 does appear on page 64, so numeric presence cannot tell that it is
        # a temperature and not a voltage. This is a documented limitation.
        self.assertTrue(ok("The maximum supply voltage is 85 V [ESP32-S3.pdf, p. 64].").valid)

    def test_claims_without_a_number_and_unit_are_not_judged(self):
        self.assertTrue(ok("The chip has a dual-core CPU [ESP32-S3.pdf, p. 64].").valid)


if __name__ == "__main__":
    unittest.main()