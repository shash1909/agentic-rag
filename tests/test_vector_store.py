"""Unit tests for Phase 2: Embeddings and Qdrant Vector Store.

Tests cover:
- Embedding model loading and 384-dimensional vector output
- Normalization of embeddings
- In-memory Qdrant collection initialization
- Indexing DocumentChunk objects with metadata payloads
- Semantic similarity search (Top-K) and score ordering
- Semantic relevance matching (e.g. revenue query finds financial chunk)
- Document filtering and collection statistics
"""

import unittest
import numpy as np

from ingestion import DocumentChunk
from embeddings import EmbeddingManager, get_embedding, get_embeddings
from vector_store import VectorStore, RetrievedChunk


class TestPhase2EmbeddingsAndVectorStore(unittest.TestCase):
    """Test suite for Phase 2: Embeddings & Qdrant."""

    @classmethod
    def setUpClass(cls):
        # Initialize embedding manager once for the test suite
        cls.embedding_manager = EmbeddingManager()

    def setUp(self):
        # Use an in-memory vector store for each test so tests are fast and isolated
        self.store = VectorStore(
            collection_name="test_collection",
            in_memory=True,
        )

    # -------------------------------------------------------------------------
    # 1. Embedding Tests
    # -------------------------------------------------------------------------

    def test_embedding_dimension(self):
        dim = self.embedding_manager.dimension
        self.assertEqual(dim, 384)

    def test_single_embedding_structure(self):
        vector = get_embedding("Artificial intelligence in finance")
        self.assertEqual(len(vector), 384)
        # Vector should be normalized (L2 norm ≈ 1.0)
        norm = np.linalg.norm(vector)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_batch_embeddings(self):
        texts = [
            "Company revenue grew by 15 percent.",
            "Supply chain risks remain elevated.",
            "Operating margins improved across all divisions.",
        ]
        vectors = get_embeddings(texts)
        self.assertEqual(len(vectors), 3)
        for vec in vectors:
            self.assertEqual(len(vec), 384)

    # -------------------------------------------------------------------------
    # 2. Qdrant Vector Store Tests
    # -------------------------------------------------------------------------

    def test_index_and_retrieve_semantic_match(self):
        chunks = [
            DocumentChunk(
                chunk_id="fin_report_p1_c0",
                text="The company generated annual revenue of $120 million, up 15% year-over-year with strong operating profit.",
                doc_name="finance_report.pdf",
                page_number=1,
                chunk_index=0,
                metadata={"category": "finance"},
            ),
            DocumentChunk(
                chunk_id="tech_guide_p1_c0",
                text="The server deployment utilizes Kubernetes clusters with horizontal pod autoscaling and Redis caching.",
                doc_name="tech_guide.pdf",
                page_number=1,
                chunk_index=0,
                metadata={"category": "tech"},
            ),
            DocumentChunk(
                chunk_id="hr_policy_p2_c0",
                text="Employees are entitled to 25 days of annual paid leave, medical insurance, and 401k matching.",
                doc_name="hr_policy.pdf",
                page_number=2,
                chunk_index=0,
                metadata={"category": "hr"},
            ),
        ]

        indexed_count = self.store.index_chunks(chunks)
        self.assertEqual(indexed_count, 3)

        # Check stats
        stats = self.store.get_collection_stats()
        self.assertEqual(stats["points_count"], 3)

        # Search for financial query
        query_vec = get_embedding("What was the annual revenue and profit?")
        results = self.store.search(query_vec, top_k=2)

        self.assertGreaterEqual(len(results), 1)
        top_hit = results[0]

        # Top hit must be the finance report chunk
        self.assertEqual(top_hit.chunk_id, "fin_report_p1_c0")
        self.assertEqual(top_hit.doc_name, "finance_report.pdf")
        self.assertEqual(top_hit.page_number, 1)
        self.assertEqual(top_hit.citation, "[finance_report.pdf, Page 1]")
        self.assertGreater(top_hit.score, 0.5)

    def test_document_filter(self):
        chunks = [
            DocumentChunk(
                chunk_id="docA_p1_c0",
                text="Cloud security policies and firewall configurations.",
                doc_name="doc_a.pdf",
                page_number=1,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="docB_p1_c0",
                text="Cloud security checklist and compliance rules.",
                doc_name="doc_b.pdf",
                page_number=1,
                chunk_index=0,
            ),
        ]
        self.store.index_chunks(chunks)

        query_vec = get_embedding("security checklist")
        results = self.store.search(query_vec, top_k=5, doc_filter="doc_b.pdf")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].doc_name, "doc_b.pdf")

    def test_get_indexed_documents(self):
        chunks = [
            DocumentChunk(
                chunk_id="doc1_p1_c0",
                text="Text from doc 1",
                doc_name="doc1.pdf",
                page_number=1,
                chunk_index=0,
            ),
            DocumentChunk(
                chunk_id="doc2_p1_c0",
                text="Text from doc 2",
                doc_name="doc2.pdf",
                page_number=1,
                chunk_index=0,
            ),
        ]
        self.store.index_chunks(chunks)

        docs = self.store.get_indexed_documents()
        self.assertEqual(docs, ["doc1.pdf", "doc2.pdf"])

    def test_clear_collection(self):
        chunks = [
            DocumentChunk(
                chunk_id="temp_p1_c0",
                text="Temporary content",
                doc_name="temp.pdf",
                page_number=1,
                chunk_index=0,
            )
        ]
        self.store.index_chunks(chunks)
        self.assertEqual(self.store.get_collection_stats()["points_count"], 1)

        self.store.clear()
        self.assertEqual(self.store.get_collection_stats()["points_count"], 0)


if __name__ == "__main__":
    unittest.main()
