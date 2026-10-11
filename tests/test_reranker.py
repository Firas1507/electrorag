import unittest
from unittest.mock import patch

from src.reranker import _specification_relevance, rerank


class FixedScoreModel:
    def predict(self, pairs, show_progress_bar=False):
        if show_progress_bar:
            raise AssertionError("Progress bars should be disabled in tests.")
        return [float(len(pairs) - index) for index, _ in enumerate(pairs)]


class VoltageRerankerTests(unittest.TestCase):
    question = "What voltage can VDD_SPI use on the ESP32-S3?"

    def test_vdd_spi_evidence_is_kept_in_the_final_top_five(self):
        candidates = [
            {
                "chunk_id": f"distractor-{index}",
                "source": "ESP32-S3.pdf",
                "page": index + 1,
                "text": (
                    "Recommended operating conditions: VDD supply rail voltage "
                    f"at subsystem {index}, with separate control circuitry."
                ),
            }
            for index in range(5)
        ]
        candidates.append(
            {
                "chunk_id": "vdd-spi-evidence",
                "source": "ESP32-S3.pdf",
                "page": 34,
                "text": (
                    "VDD_SPI voltage can be 3.3 V or 1.8 V depending "
                    "on configuration."
                ),
            }
        )

        with patch(
            "src.reranker._diversify",
            side_effect=lambda ranked, intent: ranked if intent == "voltage" else [],
        ):
            ranked = rerank(FixedScoreModel(), self.question, candidates)

        self.assertIn(
            "vdd-spi-evidence",
            [record["chunk_id"] for record in ranked[:5]],
        )

    def test_named_specific_rail_is_not_penalized(self):
        evidence = "VDD_SPI voltage depends on the selected power source."

        self.assertAlmostEqual(
            _specification_relevance("voltage", evidence, self.question),
            0.5,
        )

    def test_unrequested_specific_rail_still_receives_penalty(self):
        question = "What is the supply voltage of the ESP32-S3?"
        evidence = "VDD3P3_CPU voltage is from 3.0 to 3.6 V."

        self.assertAlmostEqual(
            _specification_relevance("voltage", evidence, question),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
