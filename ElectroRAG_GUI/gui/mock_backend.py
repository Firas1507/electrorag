"""Mock backend with FABRICATED data, for UI development only."""
from __future__ import annotations

import time

from gui import config
from gui.models import Answer, Citation, DocumentInfo, IndexStats, IndexSummary

_OFF_TOPIC = ("recipe", "weather", "stock price", "football", "movie", "joke")


class MockBackend:
    mode = "mock"

    def ask(self, question: str) -> Answer:
        time.sleep(0.9)  # simulate retrieval + generation so loading states are visible
        q = question.lower()

        if any(word in q for word in _OFF_TOPIC):
            return Answer(
                question,
                "unsupported",
                message="This question is outside the scope of the indexed electronics datasheets.",
            )

        if "voltage" in q or "supply" in q:
            citations = (
                Citation("MOCK_opamp_datasheet.pdf", 4, "[MOCK] Absolute maximum ratings table. Fabricated demo text.", 0.912, "mock-001"),
                Citation("MOCK_opamp_datasheet.pdf", 7, "[MOCK] Recommended operating conditions. Fabricated demo text.", 0.846, "mock-002"),
            )
            text = "**[MOCK ANSWER]** This is demo output showing how an answer with two citations is rendered."
        else:
            citations = (
                Citation("MOCK_ldo_regulator.pdf", 12, "[MOCK] Application information section. Fabricated demo text.", 0.781, "mock-003"),
            )
            text = "**[MOCK ANSWER]** This is demo output showing how an answer with one citation is rendered."

        return Answer(question, "answered", text, citations, model=f"{config.OLLAMA_MODEL} (mock)")

    def index_summary(self) -> IndexSummary:
        docs = (
            DocumentInfo("MOCK_opamp_datasheet.pdf", 142, 18, 18, 1.24, "2025-03-14", "Mock Op-Amp"),
            DocumentInfo("MOCK_ldo_regulator.pdf", 96, 24, 24, 0.87, "2025-04-02", "Mock LDO Regulator"),
            DocumentInfo("MOCK_mcu_reference.pdf", 1204, 410, 410, 9.6, "2025-05-21", "Mock MCU Reference Manual"),
            DocumentInfo("MOCK_mosfet_driver.pdf", 58, 12, 12, 0.52, "2025-06-30", None),
        )
        stats = IndexStats(True, len(docs), sum(d.chunk_count for d in docs), "mock_collection",
                           "Mock mode: this library is fabricated demo data.")
        return IndexSummary(stats, docs)
