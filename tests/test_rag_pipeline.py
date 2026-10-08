"""Unit tests for RAG pipeline components: retrieval, query analyzer, agent workflow, and citations."""

import unittest
from pathlib import Path
import tempfile

from ingestion import DocumentChunk
from vector_store import VectorStore
from retrieval import retrieve_chunks, retrieve_multi_query, format_context_for_prompt
from query_analyzer import analyze_query, _heuristic_classify_query
from agent import run_baseline_rag, run_agentic_rag
from generation import extract_citations, strip_citations, INSUFFICIENT_EVIDENCE_PHRASE


class TestRAGPipeline(unittest.TestCase):
    """Test suite for retrieval, query analysis, citations, and agent workflow."""

    def setUp(self):
        # Create an in-memory vector store for clean, fast testing
        self.store = VectorStore(
            collection_name="test_pipeline_collection",
            in_memory=True,
        )

        # Index standard corporate sample chunks
        self.sample_chunks = [
            DocumentChunk(
                chunk_id="report_p1_c0",
                text="The company's total revenue in 2024 was $120 million, representing a 15% increase year-over-year. Elena Rostova serves as CEO.",
                doc_name="report_2024.pdf",
                page_number=1,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="report_p1_c1",
                text="Operating margin expanded to 22%. North American software sales surged 28%, driving corporate profitability.",
                doc_name="report_2024.pdf",
                page_number=1,
                chunk_index=1,
            ),
            DocumentChunk(
                chunk_id="report_p2_c0",
                text="European supply chain congestion delayed deliveries, causing European regional margins to contract by 6%.",
                doc_name="report_2024.pdf",
                page_number=2,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="report_p4_c0",
                text="Identified enterprise risks include foreign currency exchange volatility and semiconductor shortages.",
                doc_name="report_2024.pdf",
                page_number=4,
                chunk_index=0,
            ),
        ]
        self.store.index_chunks(self.sample_chunks)

    # -------------------------------------------------------------------------
    # 1. Retrieval & Multi-Query Tests
    # -------------------------------------------------------------------------

    def test_single_query_retrieval(self):
        hits = retrieve_chunks("What was the 2024 revenue?", self.store, top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].page_number, 1)
        self.assertIn("120 million", hits[0].text)

    def test_multi_query_retrieval_deduplication(self):
        sub_qs = [
            "What was the 2024 revenue?",
            "Who is the CEO Elena Rostova?",
        ]
        # Both sub-questions might match chunk report_p1_c0
        hits = retrieve_multi_query(sub_qs, self.store, top_k_per_query=2)

        # Ensure no duplicate chunk IDs
        chunk_ids = [c.chunk_id for c in hits]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertTrue(any("report_p1_c0" in cid for cid in chunk_ids))

    def test_format_context_for_prompt(self):
        hits = retrieve_chunks("Operating margin", self.store, top_k=1)
        context = format_context_for_prompt(hits)
        self.assertIn("Source [report_2024.pdf, Page 1]", context)
        self.assertIn("Operating margin expanded", context)

    # -------------------------------------------------------------------------
    # 2. Query Analyzer Tests
    # -------------------------------------------------------------------------

    def test_classify_simple_query(self):
        res = _heuristic_classify_query("What was the company's revenue in 2024?")
        self.assertEqual(res.classification, "SIMPLE")

    def test_classify_complex_query(self):
        res = _heuristic_classify_query(
            "Why did revenue decline and what impact did this have on profitability?"
        )
        self.assertEqual(res.classification, "COMPLEX")
        self.assertGreater(len(res.sub_questions), 1)

    def test_classify_ambiguous_query(self):
        res = _heuristic_classify_query("What are the risks?")
        self.assertEqual(res.classification, "AMBIGUOUS")
        self.assertIsNotNone(res.rewritten_query)
        self.assertIn("risks", res.rewritten_query.lower())

        # Bare acronym keyword test
        res_rag = _heuristic_classify_query("rag")
        self.assertEqual(res_rag.classification, "AMBIGUOUS")
        self.assertEqual(res_rag.rewritten_query, "What is RAG?")

        # Bare concept keyword test
        res_trans = _heuristic_classify_query("transformer")
        self.assertEqual(res_trans.classification, "AMBIGUOUS")
        self.assertEqual(res_trans.rewritten_query, "What is a transformer?")

    # -------------------------------------------------------------------------
    # 3. Citation Extraction Tests
    # -------------------------------------------------------------------------

    def test_citation_extraction(self):
        text = (
            "Total revenue reached $120 million [annual_report.pdf, Page 1]. "
            "However, European profitability dropped [annual_report.pdf, Page 2]."
        )
        citations = extract_citations(text)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0], "[annual_report.pdf, Page 1]")
        self.assertEqual(citations[1], "[annual_report.pdf, Page 2]")

    def test_strip_citations(self):
        """Verify strip_citations cleanly strips source brackets from answer text."""
        raw1 = "Total revenue reached $120 million [annual_report.pdf, Page 1]."
        self.assertEqual(strip_citations(raw1), "Total revenue reached $120 million.")

        raw2 = "European profitability contracted [report.pdf, p. 2]. Net margin dropped [report.pdf]."
        self.assertEqual(strip_citations(raw2), "European profitability contracted. Net margin dropped.")

        raw3 = "The system achieved 99.99% uptime.\nSource: [report.pdf, Page 3]"
        self.assertEqual(strip_citations(raw3), "The system achieved 99.99% uptime.")

    def test_grounded_answer_excludes_source_from_text(self):
        """Verify generated grounded answer text has NO inline source brackets, while citations list has them."""
        # Baseline RAG
        b_res = run_baseline_rag("What was the 2024 revenue?", self.store)
        self.assertNotIn(".pdf", b_res.answer)
        self.assertNotIn("[", b_res.answer)
        self.assertIn("120 million", b_res.answer)
        self.assertGreater(len(b_res.citations), 0)
        self.assertIn("report_2024.pdf", b_res.citations[0])

        # Agentic RAG
        a_res = run_agentic_rag("What was the 2024 revenue?", self.store)
        self.assertNotIn(".pdf", a_res["answer"])
        self.assertNotIn("[", a_res["answer"])
        self.assertIn("120 million", a_res["answer"])
        self.assertGreater(len(a_res["citations"]), 0)

    # -------------------------------------------------------------------------
    # 4. Agent Workflow Tests (LangGraph)
    # -------------------------------------------------------------------------

    def test_agentic_workflow_execution(self):
        # Runs agentic RAG with in-memory store
        result = run_agentic_rag(
            query="Why did European profitability decline?",
            vector_store=self.store,
            api_key=None,  # Tests graceful generation & trace collection
        )

        self.assertIn("classification", result)
        self.assertIn("trace", result)
        self.assertGreater(len(result["trace"]), 0)

        # Check trace recorded nodes
        trace_steps = [t["step"] for t in result["trace"]]
        self.assertIn("Query Analysis", trace_steps)
        self.assertIn("Retrieval", trace_steps)
        self.assertIn("Evidence Check", trace_steps)

        # Chunks were retrieved
        self.assertGreater(len(result["retrieved_chunks"]), 0)

    def test_out_of_domain_unanswerable_query(self):
        """Verify out-of-domain queries strictly refuse to answer and do not hallucinate."""
        # Baseline RAG
        b_res = run_baseline_rag("who is the president of india", self.store)
        self.assertEqual(b_res.answer, INSUFFICIENT_EVIDENCE_PHRASE)
        self.assertFalse(b_res.is_sufficient)
        self.assertEqual(b_res.citations, [])

        # Agentic RAG
        a_res = run_agentic_rag("who is the president of india", self.store)
        self.assertEqual(a_res["answer"], INSUFFICIENT_EVIDENCE_PHRASE)
        self.assertFalse(a_res["evidence_sufficient"])
        self.assertEqual(a_res["citations"], [])
        self.assertTrue(a_res["agentic_activated"])  # Activated via retry loop


if __name__ == "__main__":
    unittest.main()
