"""Retrieval module for Adaptive Agentic RAG.

Handles semantic vector retrieval, multi-query aggregation (for agentic sub-questions),
chunk deduplication, and context formatting for LLM generation.
"""

from typing import List, Optional, Dict
import logging

from config import TOP_K
from vector_store import VectorStore, RetrievedChunk
from embeddings import get_embedding

logger = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    vector_store: VectorStore,
    top_k: int = TOP_K,
    doc_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """Retrieve the top-K semantically relevant chunks for a single query.

    Args:
        query: User question or rewritten search query.
        vector_store: VectorStore instance containing indexed documents.
        top_k: Number of most similar chunks to return.
        doc_filter: Optional document name filter.

    Returns:
        List of RetrievedChunk instances sorted by descending score.
    """
    if not query or not query.strip():
        return []

    # 1. Embed user query into 384-dimensional dense vector
    query_vector = get_embedding(query.strip())

    # 2. Perform Cosine Top-K search in Qdrant
    results = vector_store.search(
        query_vector=query_vector,
        top_k=top_k,
        doc_filter=doc_filter,
    )

    return results


def retrieve_multi_query(
    queries: List[str],
    vector_store: VectorStore,
    top_k_per_query: int = 3,
    doc_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    """Retrieve and deduplicate chunks across multiple sub-queries.

    Used by Agentic RAG when decomposing complex questions into multiple sub-questions.

    Args:
        queries: List of sub-questions or query variations.
        vector_store: VectorStore instance.
        top_k_per_query: Max chunks to retrieve for each sub-question.
        doc_filter: Optional document name filter.

    Returns:
        Deduplicated list of RetrievedChunk objects sorted by highest similarity score.
    """
    if not queries:
        return []

    unique_chunks: Dict[str, RetrievedChunk] = {}

    for q in queries:
        if not q or not q.strip():
            continue
        results = retrieve_chunks(
            query=q,
            vector_store=vector_store,
            top_k=top_k_per_query,
            doc_filter=doc_filter,
        )
        for chunk in results:
            # If chunk already retrieved by another sub-question, keep the higher similarity score
            if chunk.chunk_id not in unique_chunks or chunk.score > unique_chunks[chunk.chunk_id].score:
                unique_chunks[chunk.chunk_id] = chunk

    # Sort all aggregated unique chunks by descending similarity score
    sorted_results = sorted(
        unique_chunks.values(),
        key=lambda c: c.score,
        reverse=True,
    )

    return sorted_results


def format_context_for_prompt(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a clean, structured context string for the LLM.

    Each passage includes its citation header so the model can cite exact sources.

    Args:
        chunks: List of RetrievedChunk instances.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "No relevant passages found in the uploaded documents."

    formatted_passages = []
    for idx, chunk in enumerate(chunks, 1):
        header = f"--- Source [{chunk.doc_name}, Page {chunk.page_number}] (Chunk: {chunk.chunk_id}) ---"
        formatted_passages.append(f"{header}\n{chunk.text.strip()}")

    return "\n\n".join(formatted_passages)
