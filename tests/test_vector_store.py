import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import chromadb
import pymupdf
from chromadb.api.client import SharedSystemClient

from src import pdf_loader, vector_store
from src.chunker import chunk_documents


class VectorStoreSynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "chroma"
        self.pdf_directory = self.root / "datasheets"
        self.pdf_directory.mkdir()
        self.client = chromadb.PersistentClient(path=str(self.database_path))
        self.collection = self.client.get_or_create_collection(
            name="synchronization-test"
        )

    def tearDown(self):
        self.collection = None
        self.client = None
        SharedSystemClient.clear_system_cache()
        gc.collect()
        self.temporary_directory.cleanup()

    def write_pdf(self, path: Path, repeat_count: int) -> None:
        document = pymupdf.open()
        page = document.new_page()
        text = "datasheet measurement value condition specification. " * repeat_count
        result = page.insert_textbox(
            page.rect + (-36, -36, 36, 36),
            text,
            fontsize=10,
        )
        self.assertGreaterEqual(result, 0)
        document.save(path)
        document.close()

    def chunks_for_pdf(self, path: Path):
        pages = pdf_loader.load_pdf(path, strict=True)
        return chunk_documents(pages)

    @staticmethod
    def fake_embeddings(chunks):
        return [
            [
                float(len(chunk["text"])),
                float(sum(map(ord, chunk["text"])) % 997),
                float(index),
            ]
            for index, chunk in enumerate(chunks)
        ]

    def index_pdf(self, path: Path):
        chunks = self.chunks_for_pdf(path)
        vector_store.synchronize_chunks(
            self.collection,
            chunks,
            self.fake_embeddings(chunks),
        )
        return chunks

    def records(self):
        result = self.collection.get(include=["documents", "metadatas"])
        return {
            chunk_id: (document, metadata)
            for chunk_id, document, metadata in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        }

    def test_initial_indexing_inserts_every_chunk(self):
        pdf_path = self.pdf_directory / "Power.pdf"
        self.write_pdf(pdf_path, 80)

        chunks = self.index_pdf(pdf_path)

        records = self.records()
        self.assertEqual(len(chunks), len(records))
        self.assertEqual(set(records), {chunk["chunk_id"] for chunk in chunks})
        self.assertTrue(
            all(metadata["source"] == "Power.pdf" for _, metadata in records.values())
        )

    def test_reindexing_unchanged_content_is_idempotent(self):
        pdf_path = self.pdf_directory / "Power.pdf"
        self.write_pdf(pdf_path, 80)
        self.index_pdf(pdf_path)
        before = self.records()

        self.index_pdf(pdf_path)

        self.assertEqual(self.records(), before)
        self.assertEqual(self.collection.count(), len(before))

    def test_updated_pdf_with_fewer_chunks_prunes_only_its_obsolete_chunks(self):
        pdf_path = self.pdf_directory / "Power.pdf"
        other_pdf_path = self.pdf_directory / "Controller.pdf"
        self.write_pdf(pdf_path, 80)
        self.write_pdf(other_pdf_path, 40)
        original_chunks = self.index_pdf(pdf_path)
        other_chunks = self.index_pdf(other_pdf_path)
        before = self.records()
        other_records = {
            chunk_id: record
            for chunk_id, record in before.items()
            if record[1]["source"] == "Controller.pdf"
        }
        original_records = {
            chunk_id: record
            for chunk_id, record in before.items()
            if record[1]["source"] == "Power.pdf"
        }

        self.write_pdf(pdf_path, 8)
        updated_chunks = self.index_pdf(pdf_path)

        records = self.records()
        self.assertLess(len(updated_chunks), len(original_chunks))
        self.assertNotEqual(
            original_records[updated_chunks[0]["chunk_id"]][0],
            records[updated_chunks[0]["chunk_id"]][0],
        )
        self.assertEqual(
            {chunk_id for chunk_id, (_, metadata) in records.items()
             if metadata["source"] == "Power.pdf"},
            {chunk["chunk_id"] for chunk in updated_chunks},
        )
        self.assertTrue(
            all(records[chunk["chunk_id"]][0] == chunk["text"] for chunk in updated_chunks)
        )
        self.assertEqual(
            {chunk_id: records[chunk_id] for chunk_id in other_records},
            other_records,
        )
        self.assertTrue(other_chunks)
        self.assertEqual(self.collection.count(), len(updated_chunks) + len(other_chunks))

    def test_explicit_source_removal_preserves_other_documents(self):
        pdf_path = self.pdf_directory / "Power.pdf"
        other_pdf_path = self.pdf_directory / "Controller.pdf"
        self.write_pdf(pdf_path, 50)
        self.write_pdf(other_pdf_path, 40)
        removed_chunks = self.index_pdf(pdf_path)
        other_chunks = self.index_pdf(other_pdf_path)

        removed_count = vector_store.remove_document(self.collection, "Power.pdf")

        records = self.records()
        self.assertEqual(removed_count, len(removed_chunks))
        self.assertEqual(self.collection.count(), len(other_chunks))
        self.assertEqual(
            {metadata["source"] for _, metadata in records.values()},
            {"Controller.pdf"},
        )
        self.assertEqual(
            {chunk_id for chunk_id, _ in records.items()},
            {chunk["chunk_id"] for chunk in other_chunks},
        )

    def test_strict_extraction_failure_preserves_existing_records(self):
        pdf_path = self.pdf_directory / "Power.pdf"
        self.write_pdf(pdf_path, 40)
        self.index_pdf(pdf_path)
        before = self.records()
        pdf_path.write_bytes(b"not a valid PDF")

        with patch(
            "src.embeddings.load_all_pdfs",
            side_effect=lambda strict=False: pdf_loader.load_all_pdfs(
                self.pdf_directory,
                strict=strict,
            ),
        ), patch("src.vector_store.get_collection") as get_collection:
            with self.assertRaisesRegex(RuntimeError, "Could not extract Power.pdf"):
                vector_store.main()

        get_collection.assert_not_called()
        self.assertEqual(self.records(), before)

    def test_embedding_failure_preserves_existing_records(self):
        pdf_path = self.pdf_directory / "Power.pdf"
        self.write_pdf(pdf_path, 40)
        existing_chunks = self.index_pdf(pdf_path)
        before = self.records()
        model = Mock()

        with patch(
            "src.vector_store.get_all_chunks",
            return_value=existing_chunks,
        ), patch("src.vector_store.load_model", return_value=model), patch(
            "src.vector_store.embed_texts",
            side_effect=RuntimeError("embedding failed"),
        ), patch("src.vector_store.get_collection") as get_collection:
            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                vector_store.main()

        get_collection.assert_not_called()
        self.assertEqual(self.records(), before)

    def test_missing_source_is_not_implicitly_deleted(self):
        pdf_path = self.pdf_directory / "Power.pdf"
        self.write_pdf(pdf_path, 40)
        chunks = self.index_pdf(pdf_path)

        vector_store.synchronize_chunks(
            self.collection,
            [{"source": "Controller.pdf", "chunk_id": "Controller.pdf::p1::c0",
              "text": "controller evidence"}],
            [[1.0, 2.0, 3.0]],
        )

        records = self.records()
        self.assertTrue(
            all(chunk["chunk_id"] in records for chunk in chunks)
        )


if __name__ == "__main__":
    unittest.main()
