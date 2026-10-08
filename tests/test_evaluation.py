"""Unit tests for Phase 8 & 9: Evaluation metrics, dataset loading, and grading."""

import unittest
from pathlib import Path

from vector_store import RetrievedChunk, VectorStore
from evaluation import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_hit_rate_at_k,
    calculate_faithfulness,
    calculate_answer_correctness,
    compute_overall_grade,
    load_evaluation_dataset,
    generate_sample_benchmark_pdf,
    run_evaluation_benchmark,
)
from ingestion import ingest_pdf


class TestEvaluation(unittest.TestCase):
    """Test suite for RAG evaluation metrics and benchmark execution."""

    def setUp(self):
        # Synthetic retrieved chunks for testing metrics deterministically
        self.retrieved_chunks = [
            RetrievedChunk(
                chunk_id="doc_p1_c0",
                text="Acme revenue reached $120M in 2024.",
                doc_name="report.pdf",
                page_number=1,
                chunk_index=0,
                score=0.91,
                citation="[report.pdf, Page 1]",
            ),
            RetrievedChunk(
                chunk_id="doc_p2_c0",
                text="Operating margin was 22%. R&D was $28M.",
                doc_name="report.pdf",
                page_number=2,
                chunk_index=0,
                score=0.84,
                citation="[report.pdf, Page 2]",
            ),
            RetrievedChunk(
                chunk_id="doc_p3_c0",
                text="Cloud infrastructure is on Google Cloud.",
                doc_name="report.pdf",
                page_number=3,
                chunk_index=0,
                score=0.72,
                citation="[report.pdf, Page 3]",
            ),
        ]

    # -------------------------------------------------------------------------
    # 1. Retrieval Metrics
    # -------------------------------------------------------------------------

    def test_precision_at_k(self):
        # Target relevant page is 1 only. Top 2 retrieved are pages 1 and 2.
        # Precision@2 = 1 / 2 = 0.5
        p = calculate_precision_at_k(self.retrieved_chunks, relevant_pages=[1], k=2)
        self.assertEqual(p, 0.5)

        # If relevant pages are [1, 2], both top 2 are hits -> 2/2 = 1.0
        p_all = calculate_precision_at_k(self.retrieved_chunks, relevant_pages=[1, 2], k=2)
        self.assertEqual(p_all, 1.0)

    def test_recall_at_k(self):
        # Expected relevant pages: [1, 4]. In top 3 chunks, page 1 is found, page 4 is not.
        # Recall@3 = 1 / 2 = 0.5
        r = calculate_recall_at_k(self.retrieved_chunks, relevant_pages=[1, 4], k=3)
        self.assertEqual(r, 0.5)

    def test_hit_rate_at_k(self):
        # Hit rate@2 for page 1 is 1.0
        hr = calculate_hit_rate_at_k(self.retrieved_chunks, relevant_pages=[1], k=2)
        self.assertEqual(hr, 1.0)

        # Hit rate@2 for page 4 (not in top 2) is 0.0
        hr_miss = calculate_hit_rate_at_k(self.retrieved_chunks, relevant_pages=[4], k=2)
        self.assertEqual(hr_miss, 0.0)

    # -------------------------------------------------------------------------
    # 2. Generation & Faithfulness Metrics
    # -------------------------------------------------------------------------

    def test_faithfulness_not_present_insufficient_evidence(self):
        # If question is out of domain and model says insufficient evidence -> 1.0
        ans = "I don't have sufficient evidence in the uploaded documents to answer this question."
        f = calculate_faithfulness(ans, self.retrieved_chunks, is_not_present=True)
        self.assertEqual(f, 1.0)

    def test_faithfulness_grounded_answer(self):
        ans = "The company generated revenue of $120M in 2024 and operating margin was 22%."
        f = calculate_faithfulness(ans, self.retrieved_chunks, is_not_present=False)
        self.assertGreaterEqual(f, 0.7)

    def test_correctness(self):
        ans = "The company revenue was $120 million, a 15% increase."
        gt = "Revenue was $120 million, increasing 15%."
        kws = ["120 million", "15%"]
        score = calculate_answer_correctness(ans, gt, kws, is_not_present=False)
        self.assertGreater(score, 0.8)

    # -------------------------------------------------------------------------
    # 3. Overall Grade Computation
    # -------------------------------------------------------------------------

    def test_grading_scale(self):
        self.assertEqual(compute_overall_grade(96.0), "A+")
        self.assertEqual(compute_overall_grade(91.5), "A")
        self.assertEqual(compute_overall_grade(83.0), "B")
        self.assertEqual(compute_overall_grade(74.0), "C")
        self.assertEqual(compute_overall_grade(65.0), "D")
        self.assertEqual(compute_overall_grade(52.0), "F")

    # -------------------------------------------------------------------------
    # 4. Evaluation Dataset Loading & Benchmark Run
    # -------------------------------------------------------------------------

    def test_load_dataset(self):
        data = load_evaluation_dataset()
        self.assertGreaterEqual(len(data), 20)
        first = data[0]
        self.assertIn("question", first)
        self.assertIn("ground_truth", first)
        self.assertIn("relevant_pages", first)

    def test_sample_pdf_generation_and_indexing(self):
        pdf_path = generate_sample_benchmark_pdf()
        self.assertTrue(pdf_path.exists())

        # Ingest and index
        chunks = ingest_pdf(pdf_path)
        self.assertGreaterEqual(len(chunks), 4)

        store = VectorStore(collection_name="test_eval_bench", in_memory=True)
        store.index_chunks(chunks)

        # Run benchmark on first 3 questions to verify end-to-end execution
        results = run_evaluation_benchmark(store, limit=3)
        self.assertIn("summary", results)
        self.assertIn("baseline", results["summary"])
        self.assertIn("agentic", results["summary"])
        self.assertIn("overall_score", results["summary"]["agentic"])
        self.assertIn("grade", results["summary"]["agentic"])


if __name__ == "__main__":
    unittest.main()
