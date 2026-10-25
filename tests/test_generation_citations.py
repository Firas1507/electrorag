import unittest
from unittest.mock import patch

from src import generator
from src.benchmark import QUESTIONS, check_contains_any, evaluate_fact
from src.citation_validation import validate_citations


NOT_FOUND = generator.NOT_FOUND_MESSAGE
CONTEXT = [
    {
        "chunk_id": "ESP32-S3.pdf::p64::c0",
        "source": "ESP32-S3.pdf",
        "page": 64,
        "text": "Recommended operating voltage is 3.0 V to 3.6 V.",
    },
]


class ProductionCitationValidationTests(unittest.TestCase):
    def generate(self, model_answer):
        with patch(
            "src.generator.stream_answer",
            return_value=[model_answer],
        ):
            return generator.generate_answer(
                "What is the operating voltage?",
                CONTEXT,
            )

    def test_valid_citation_and_supported_measurement_are_preserved(self):
        answer = self.generate(
            "Operating voltage is 3.0 V to 3.6 V [C1]."
        )

        self.assertEqual(
            answer,
            "Operating voltage is 3.0 V to 3.6 V [ESP32-S3.pdf, p. 64].",
        )

    def test_nonexistent_context_id_abstains(self):
        self.assertEqual(
            self.generate("Operating voltage is 3.0 V [C2]."),
            NOT_FOUND,
        )

    def test_malformed_context_citation_abstains(self):
        self.assertEqual(
            self.generate("Operating voltage is 3.0 V (C1)."),
            NOT_FOUND,
        )

    def test_malformed_pdf_citation_abstains(self):
        self.assertEqual(
            self.generate(
                "Operating voltage is 3.0 V [ESP32-S3.pdf, page 64]."
            ),
            NOT_FOUND,
        )

    def test_citation_to_source_not_in_context_abstains(self):
        self.assertEqual(
            self.generate(
                "Operating voltage is 3.0 V [Other.pdf, p. 64]."
            ),
            NOT_FOUND,
        )

    def test_unsupported_numeric_claim_abstains(self):
        self.assertEqual(
            self.generate("Operating voltage is 5.0 V [C1]."),
            NOT_FOUND,
        )

    def test_unsupported_non_numeric_claim_abstains(self):
        self.assertEqual(
            self.generate("Wi-Fi is supported [C1]."),
            NOT_FOUND,
        )

    def test_validator_rejects_source_page_not_in_retrieved_context(self):
        report = validate_citations(
            "Operating voltage is 3.0 V [ESP32-S3.pdf, p. 63].",
            CONTEXT,
        )

        self.assertFalse(report.valid)
        self.assertTrue(any("page that is not in the retrieved context" in item
                            for item in report.identity))


class CitationPlacementRegressionTests(unittest.TestCase):
    def generate(self, answer, context):
        with patch("src.generator.stream_answer", return_value=[answer]):
            return generator.generate_answer("Answer from the datasheet.", context)

    def test_prompt_requires_inline_citations_and_no_uncited_lead_in(self):
        prompt = generator.build_prompt("Question?", CONTEXT)

        self.assertIn(
            "Put the citation on the same sentence or bullet",
            prompt,
        )
        self.assertIn(
            "Do not add an uncited factual introduction or summary",
            prompt,
        )

    def test_q9_style_uncited_introduction_is_rejected_but_claims_pass(self):
        context = [
            {
                "source": "ESP32-S3.pdf",
                "page": 67,
                "text": (
                    "Bluetooth LE @ 21.0 dBm 335 mA. "
                    "WAITI (Dual core in idle state) 13.2 mA."
                ),
            }
        ]
        rejected = (
            "Current varies by mode and RF conditions. "
            "Bluetooth LE @ 21.0 dBm: 335 mA [C1]."
        )
        accepted = (
            "Bluetooth LE @ 21.0 dBm: 335 mA [C1]. "
            "WAITI in modem-sleep mode: 13.2 mA [C1]."
        )

        self.assertEqual(self.generate(rejected, context), NOT_FOUND)
        self.assertIn("[ESP32-S3.pdf, p. 67]", self.generate(accepted, context))

    def test_q14_style_detached_citations_are_rejected_but_inline_passes(self):
        context = [
            {
                "source": "MAX98357A.pdf",
                "page": 7,
                "text": "RL = 8 Ω, LL = 68 µH. RL = 4 Ω, LL = 33 µH.",
            }
        ]
        detached = (
            "Supported loads:\n- 4 Ω + 33 µH\n- 8 Ω + 68 µH\n\n[C1]"
        )
        inline = (
            "Supported loads:\n"
            "- 4 Ω + 33 µH [C1]\n"
            "- 8 Ω + 68 µH [C1]"
        )

        self.assertEqual(self.generate(detached, context), NOT_FOUND)
        self.assertNotEqual(self.generate(inline, context), NOT_FOUND)

    def test_q19_style_one_trailing_citation_does_not_cover_prior_claims(self):
        context = [
            {
                "source": "TPS63020.pdf",
                "page": 4,
                "text": (
                    "EN is an input. EN = 1 enabled, EN = 0 disabled. "
                    "The EN pin must not be left open."
                ),
            }
        ]
        detached = (
            "The EN pin enables or disables the converter. "
            "When EN is high (1), it is enabled; when low (0), it is disabled. "
            "It must not be left open. [C1]"
        )
        inline = (
            "The EN pin enables or disables the converter [C1]. "
            "When EN is high (1), it is enabled [C1]. "
            "When EN is low (0), it is disabled [C1]. "
            "It must not be left open [C1]."
        )

        self.assertEqual(self.generate(detached, context), NOT_FOUND)
        self.assertNotEqual(self.generate(inline, context), NOT_FOUND)


class CitationRepairRegressionTests(unittest.TestCase):
    def test_legitimate_abstention_is_not_retried(self):
        with patch(
            "src.generator.stream_answer",
            return_value=[NOT_FOUND],
        ) as stream:
            answer = generator.generate_answer(
                "What information is available?",
                CONTEXT,
            )

        self.assertEqual(answer, NOT_FOUND)
        self.assertEqual(stream.call_count, 1)

    def test_tps63020_uncited_en_claim_is_repaired_and_validated(self):
        context = [
            {
                "source": "TPS63020.pdf",
                "page": 4,
                "text": (
                    "EN 12 I Enable input (1 enabled, 0 disabled), "
                    "must not be left open."
                ),
            }
        ]
        drafts = [
            "The EN pin is an input that enables or disables the converter. "
            "When EN is high (1), the converter is enabled; when low (0), "
            "it is disabled.",
            "EN is an enable input: 1 enables and 0 disables the device [C1]. "
            "The EN pin must not be left open [C1].",
        ]

        with patch(
            "src.generator.stream_answer",
            side_effect=[[draft] for draft in drafts],
        ) as stream:
            answer = generator.generate_answer(
                "How does the EN pin control the TPS63020?",
                context,
            )

        self.assertEqual(stream.call_count, 2)
        repair_prompt = stream.call_args_list[1].args[0]
        self.assertIn("before the sentence-ending punctuation", repair_prompt)
        self.assertIn("not after it", repair_prompt)
        self.assertIn("uncited claim", repair_prompt)
        self.assertIn("[TPS63020.pdf, p. 4]", answer)
        self.assertTrue(
            validate_citations(
                answer,
                context,
                expected_source="TPS63020.pdf",
            ).valid
        )

    def test_esp32_uncited_intro_is_repaired_and_validated(self):
        context = [
            {
                "source": "ESP32-S3.pdf",
                "page": 67,
                "text": (
                    "Table 5-8. Current Consumption for Bluetooth LE in Active Mode\n"
                    "TX\nBluetooth LE @ 21.0 dBm\n335\n"
                    "RX\nBluetooth LE\n93\nPeak (mA)"
                ),
            }
        ]
        drafts = [
            "The ESP32-S3 current consumption during operation varies by mode.",
            "Bluetooth LE TX at 21.0 dBm has a peak current of 335 mA [C1]. "
            "Bluetooth LE RX has a peak current of 93 mA [C1].",
        ]

        with patch(
            "src.generator.stream_answer",
            side_effect=[[draft] for draft in drafts],
        ) as stream:
            answer = generator.generate_answer(
                "How much current does the ESP32-S3 require during operation?",
                context,
            )

        self.assertEqual(stream.call_count, 2)
        self.assertIn("[ESP32-S3.pdf, p. 67]", answer)
        self.assertTrue(
            validate_citations(
                answer,
                context,
                expected_source="ESP32-S3.pdf",
            ).valid
        )

    def test_answer_that_still_fails_after_repair_abstains(self):
        with patch(
            "src.generator.stream_answer",
            side_effect=[
                ["Unsupported feature [C1]."],
                ["Still unsupported [C1]."],
            ],
        ):
            answer = generator.generate_answer(
                "What is supported?",
                CONTEXT,
            )

        self.assertEqual(answer, NOT_FOUND)


class BenchmarkContainsAnyTests(unittest.TestCase):
    def test_gain_benchmark_requires_every_expected_setting(self):
        question = next(item for item in QUESTIONS if item["id"] == "max98357_05")
        expected = question["expected_facts"][0]

        self.assertFalse(evaluate_fact("Gain settings are available.", expected)[0])
        self.assertTrue(
            evaluate_fact(
                "Available settings are 15 dB, 12 dB, 9 dB, 6 dB, and 3 dB.",
                expected,
            )[0]
        )

    def test_gain_settings_in_db_column_count_as_table_evidence(self):
        question = next(item for item in QUESTIONS if item["id"] == "max98357_05")
        expected = question["expected_facts"][0]
        table = (
            "Table 8. Gain Selection\n"
            "GAIN_SLOT\n"
            "I2S/LJ GAIN (dB)\n"
            "Connect to GND through 100k ohm resistor\n15\n"
            "Connect to GND\n12\n"
            "Unconnected\n9\n"
            "Connect to VDD\n6\n"
            "Connect to VDD through 100k ohm resistor\n3\n"
            "FREQUENCY (MHz)\n"
            "900\n800\n"
        )

        self.assertTrue(evaluate_fact(table, expected)[0])

    def test_table_cells_without_matching_unit_header_do_not_pass(self):
        question = next(item for item in QUESTIONS if item["id"] == "max98357_05")
        expected = question["expected_facts"][0]
        table = (
            "Table 8. Gain Selection\n"
            "I2S/LJ GAIN (V)\n"
            "Connect to GND\n15\n"
            "Connect to GND through 100k ohm resistor\n12\n"
            "Unconnected\n9\n"
            "Connect to VDD\n6\n"
            "Connect to VDD through 100k ohm resistor\n3\n"
        )

        self.assertFalse(evaluate_fact(table, expected)[0])

    def test_incomplete_gain_table_does_not_pass(self):
        question = next(item for item in QUESTIONS if item["id"] == "max98357_05")
        expected = question["expected_facts"][0]
        table = (
            "Table 8. Gain Selection\n"
            "I2S/LJ GAIN (dB)\n"
            "Connect to GND\n15\n"
            "Connect to GND\n12\n"
            "Unconnected\n9\n"
        )

        self.assertFalse(evaluate_fact(table, expected)[0])

    def test_single_character_unit_is_not_accepted_as_textual_evidence(self):
        passed, _ = check_contains_any(
            "A feature is present.",
            {"values": ["current", "mA", "A"]},
        )

        self.assertFalse(passed)

    def test_multi_character_expected_term_matches_as_a_token(self):
        passed, _ = check_contains_any(
            "Typical current is 240 mA.",
            {"values": ["current", "mA"]},
        )

        self.assertTrue(passed)

    def test_substring_inside_unrelated_word_does_not_match(self):
        passed, _ = check_contains_any(
            "The amplifier gains stability.",
            {"values": ["gain"]},
        )

        self.assertFalse(passed)

    def test_esp32_current_fact_does_not_pass_on_bare_letter_a(self):
        passed, _ = evaluate_fact(
            "A feature is present.",
            {
                "type": "contains_any",
                "values": ["current", "mA"],
            },
        )

        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
