"""Vector store module for Adaptive Agentic RAG using Qdrant.

Provides:
1. Local Qdrant vector database (in-memory or persistent disk).
2. Process-wide client caching and thread-safe locking to prevent lock contention.
3. Collection creation with Cosine similarity distance metric.
4. Indexing of DocumentChunk objects with embeddings and rich metadata payloads.
5. Top-K semantic similarity search returning structured RetrievedChunk objects.
6. Inspection helpers (collection stats, list of indexed documents, reset).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import uuid
import logging
import threading

# Disable SQLite check_same_thread for Streamlit multi-threaded script runner
try:
    from qdrant_client.local.persistence import CollectionPersistence
    CollectionPersistence.CHECK_SAME_THREAD = False
except Exception:
    pass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_STORAGE_PATH,
    TOP_K,
)
from ingestion import DocumentChunk
from embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

# Process-wide lock to serialize access to the local Qdrant SQLite storage
_qdrant_lock = threading.Lock()


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved from the vector store with similarity scoring."""
    chunk_id: str
    text: str
    doc_name: str
    page_number: int
    chunk_index: int
    score: float
    citation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieved chunk to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "citation": self.citation,
            "metadata": self.metadata,
        }


class VectorStore:
    """Thread-safe wrapper around local Qdrant for chunk indexing and semantic search."""

    # Cache open clients by path so multiple VectorStore instances in the same process
    # do not compete for the file lock (.qdrant_storage/.lock)
    _shared_clients: Dict[str, QdrantClient] = {}

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION_NAME,
        storage_path: Optional[Union[str, Path]] = QDRANT_STORAGE_PATH,
        in_memory: bool = False,
    ):
        # Ensure SQLite thread-safety flag is set
        try:
            from qdrant_client.local.persistence import CollectionPersistence
            CollectionPersistence.CHECK_SAME_THREAD = False
        except Exception:
            pass

        self.collection_name = collection_name
        self.embedding_manager = EmbeddingManager()
        self.vector_size = self.embedding_manager.dimension
        self.in_memory = in_memory

        with _qdrant_lock:
            if in_memory:
                self.client = QdrantClient(":memory:")
            else:
                path_str = str(storage_path)
                Path(path_str).mkdir(parents=True, exist_ok=True)

                if path_str in self._shared_clients:
                    self.client = self._shared_clients[path_str]
                else:
                    try:
                        self.client = QdrantClient(path=path_str)
                        self._shared_clients[path_str] = self.client
                    except RuntimeError as e:
                        if "already accessed by another instance" in str(e).lower():
                            logger.warning(
                                f"Storage folder '{path_str}' is locked by another process. "
                                "Falling back to in-memory Qdrant instance."
                            )
                            self.client = QdrantClient(":memory:")
                            self.in_memory = True
                        else:
                            raise

            self._ensure_collection_internal()

    def _ensure_collection_internal(self) -> None:
        """Internal helper to create collection (assumes _qdrant_lock is held)."""
        collections_response = self.client.get_collections()
        existing = [c.name for c in collections_response.collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(
                f"Created Qdrant collection '{self.collection_name}' with size {self.vector_size} (Cosine)."
            )

    def _ensure_collection(self) -> None:
        """Create collection if it does not already exist with lock."""
        with _qdrant_lock:
            self._ensure_collection_internal()

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> int:
        """Embed and upsert DocumentChunk instances into Qdrant in a thread-safe manner."""
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_manager.embed_texts(texts, batch_size=batch_size)

        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "citation": chunk.citation,
                "metadata": chunk.metadata,
            }

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        with _qdrant_lock:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = TOP_K,
        doc_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Perform thread-safe semantic similarity search against indexed chunks."""
        query_filter = None
        if doc_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_name",
                        match=qmodels.MatchValue(value=doc_filter),
                    )
                ]
            )

        with _qdrant_lock:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                )
                hits = response.points
            else:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                )

        retrieved: List[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=payload.get("chunk_id", ""),
                    text=payload.get("text", ""),
                    doc_name=payload.get("doc_name", ""),
                    page_number=payload.get("page_number", 1),
                    chunk_index=payload.get("chunk_index", 0),
                    score=float(hit.score),
                    citation=payload.get("citation", f"[{payload.get('doc_name')}, Page {payload.get('page_number')}]"),
                    metadata=payload.get("metadata", {}),
                )
            )

        return retrieved

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return high-level statistics about the Qdrant collection."""
        with _qdrant_lock:
            info = self.client.get_collection(self.collection_name)
            points_count = info.points_count or 0
            status_val = info.status.value if hasattr(info.status, "value") else str(info.status)

        return {
            "collection_name": self.collection_name,
            "status": status_val,
            "points_count": points_count,
            "vector_size": self.vector_size,
        }

    def get_indexed_documents(self) -> List[str]:
        """Return a sorted list of unique document names currently stored in the index."""
        with _qdrant_lock:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=["doc_name"],
                with_vectors=False,
            )

        doc_names = set()
        for record in records:
            if record.payload and "doc_name" in record.payload:
                doc_names.add(record.payload["doc_name"])

        return sorted(list(doc_names))

    def clear(self) -> None:
        """Delete all points in the collection and recreate a fresh empty collection."""
        with _qdrant_lock:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_internal()
