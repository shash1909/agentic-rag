"""Embeddings module for Adaptive Agentic RAG.

This module is responsible for:
1. Loading a lightweight, high-performance Sentence Transformers model (BAAI/bge-small-en-v1.5).
2. Generating dense 384-dimensional semantic vectors locally on CPU or GPU.
3. Providing singleton caching so the model loads once and is reused across queries and ingestion.
"""

from typing import List, Optional
import numpy as np

from config import EMBEDDING_MODEL_NAME


class EmbeddingManager:
    """Manages local Sentence Transformer embedding generation.

    Uses a lazy-loading singleton pattern so the model is only loaded into memory
    when first needed, preventing repeated initialization overhead.
    """

    _instance: Optional["EmbeddingManager"] = None
    _model = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "EmbeddingManager":
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance.model_name = model_name
        return cls._instance

    @property
    def model(self):
        """Lazy-load the underlying SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # Prefer local cached files first for instant offline loading
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the generated embedding vectors (384 for bge-small)."""
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        """Generate a dense vector embedding for a single string.

        Args:
            text: Text string to embed.

        Returns:
            List of floats representing the semantic embedding vector.
        """
        if not text or not text.strip():
            # Return zero vector if text is empty
            return [0.0] * self.dimension

        # Normalize embeddings so cosine similarity equals dot product
        vector = self.model.encode(
            text.strip(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate vector embeddings for a batch of text strings.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts to process in parallel batches.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() if t and t.strip() else "" for t in texts]

        vectors = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]


# -----------------------------------------------------------------------------
# Convenience helper functions
# -----------------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """Helper function to embed a single text string."""
    manager = EmbeddingManager()
    return manager.embed_text(text)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Helper function to embed a batch of text strings."""
    manager = EmbeddingManager()
    return manager.embed_texts(texts)
