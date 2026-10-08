"""Unit tests for Phase 1: PDF Ingestion and Chunking.

Tests cover:
- Text cleaning and normalization
- Chunking algorithm (overlap, word boundaries, edge cases)
- PDF extraction from file path and in-memory bytes using PyMuPDF
- Metadata and citation generation
- Error handling for invalid/empty/blank PDFs

Can be run directly via:
    python tests/test_ingestion.py
or via pytest (if installed).
"""

from pathlib import Path
import unittest
import tempfile
import fitz  # PyMuPDF

from ingestion import (
    DocumentChunk,
    clean_text,
    chunk_text,
    extract_text_from_pdf,
    ingest_pdf,
)


def create_sample_pdf(file_path: Path, pages_content: list) -> Path:
    """Helper to create a sample PDF with specified pages of text using PyMuPDF."""
    doc = fitz.open()
    for content in pages_content:
        page = doc.new_page(width=595, height=842)  # A4 standard size
        # Insert text at top-left margin (x=50, y=72)
        page.insert_text(fitz.Point(50, 72), content, fontsize=11)
    doc.save(str(file_path))
    doc.close()
    return file_path


class TestIngestionPhase1(unittest.TestCase):
    """Test suite for Phase 1: Ingestion & Chunking."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Text Cleaning Tests
    # -------------------------------------------------------------------------

    def test_clean_text_basic(self):
        raw = "Hello   world!  \t  This is   a test.\r\nNext line."
        expected = "Hello world! This is a test.\nNext line."
        self.assertEqual(clean_text(raw), expected)

    def test_clean_text_removes_null_bytes_and_form_feeds(self):
        raw = "Section 1\x00\x0cSection 2"
        cleaned = clean_text(raw)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x0c", cleaned)
        self.assertIn("Section 1", cleaned)
        self.assertIn("Section 2", cleaned)

    def test_clean_text_collapses_multiple_newlines(self):
        raw = "Paragraph 1\n\n\n\n\nParagraph 2"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Paragraph 1\n\nParagraph 2")

    def test_clean_text_empty_and_whitespace(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   \n\t   "), "")

    # -------------------------------------------------------------------------
    # 2. Chunking Tests
    # -------------------------------------------------------------------------

    def test_chunk_text_short_text(self):
        short = "This is a short text under fifty characters."
        chunks = chunk_text(short, chunk_size=100, chunk_overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short)

    def test_chunk_text_splits_longer_text(self):
        text = (
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
            "omicron pi rho sigma tau upsilon phi chi psi omega. "
            "Artificial Intelligence and Retrieval-Augmented Generation enhance LLMs "
            "by feeding factual, grounded context from verified source documents."
        )
        chunks = chunk_text(text, chunk_size=80, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)

        combined = " ".join(chunks)
        self.assertIn("Artificial Intelligence", combined)
        self.assertIn("omega", combined)

    def test_chunk_text_preserves_overlap(self):
        text = "The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow."
        chunk_size = 40
        chunk_overlap = 15
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.assertGreaterEqual(len(chunks), 2)
        # Verify adjacent chunks share some overlapping words
        for i in range(len(chunks) - 1):
            words_chunk1 = set(chunks[i].split())
            words_chunk2 = set(chunks[i + 1].split())
            shared_words = words_chunk1.intersection(words_chunk2)
            self.assertGreater(
                len(shared_words), 0, f"Chunk {i} and {i+1} share no overlap"
            )

    def test_chunk_text_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("Some text", chunk_size=100, chunk_overlap=100)

    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    # -------------------------------------------------------------------------
    # 3. PDF Extraction and Ingestion Tests
    # -------------------------------------------------------------------------

    def test_pdf_ingestion_from_file(self):
        pdf_path = self.tmp_path / "annual_report.pdf"
        pages = [
            "Annual Report 2024. Company revenue increased by 15% to $120 million. Operating margin improved.",
            "Risks and Challenges. Inflationary pressure and supply chain constraints may impact Q3 projections.",
        ]
        create_sample_pdf(pdf_path, pages)

        chunks = ingest_pdf(pdf_path)

        self.assertGreaterEqual(len(chunks), 2)
        # Check first chunk metadata
        c1 = chunks[0]
        self.assertEqual(c1.doc_name, "annual_report.pdf")
        self.assertEqual(c1.page_number, 1)
        self.assertIn("revenue increased", c1.text)
        self.assertEqual(c1.citation, "[annual_report.pdf, Page 1]")
        self.assertEqual(c1.metadata["doc_name"], "annual_report.pdf")
        self.assertEqual(c1.metadata["page_number"], 1)
        self.assertIn("annual_report.pdf_p1_c0", c1.chunk_id)

        # Check second page chunk
        c2 = [c for c in chunks if c.page_number == 2][0]
        self.assertEqual(c2.page_number, 2)
        self.assertIn("Inflationary pressure", c2.text)
        self.assertEqual(c2.citation, "[annual_report.pdf, Page 2]")

    def test_pdf_ingestion_from_bytes(self):
        pdf_path = self.tmp_path / "in_memory.pdf"
        create_sample_pdf(pdf_path, ["Single page document content testing bytes buffer ingestion."])

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        chunks = ingest_pdf(pdf_bytes, doc_name="custom_upload.pdf")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].doc_name, "custom_upload.pdf")
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].citation, "[custom_upload.pdf, Page 1]")
        self.assertIn("Single page document", chunks[0].text)

    # -------------------------------------------------------------------------
    # 4. Error Handling Tests
    # -------------------------------------------------------------------------

    def test_missing_pdf_file_raises_error(self):
        missing_path = Path("non_existent_file_12345.pdf")
        with self.assertRaises(FileNotFoundError):
            ingest_pdf(missing_path)

    def test_empty_pdf_file_raises_error(self):
        empty_file = self.tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")

        with self.assertRaises(ValueError):
            ingest_pdf(empty_file)

    def test_blank_pdf_without_text_raises_error(self):
        blank_pdf = self.tmp_path / "blank.pdf"
        doc = fitz.open()
        doc.new_page()  # Blank page without text
        doc.save(str(blank_pdf))
        doc.close()

        with self.assertRaises(ValueError):
            ingest_pdf(blank_pdf)


if __name__ == "__main__":
    unittest.main()
