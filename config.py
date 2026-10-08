"""Configuration settings for Adaptive Agentic RAG.

All configurable settings are stored here so that every component has a single,
transparent source of truth.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Document Ingestion & Chunking Settings
# -----------------------------------------------------------------------------
# Chunk size in characters. ~700 characters is roughly 110-140 words,
# providing sufficient context for complete definitions and Q&A pairs.
CHUNK_SIZE = 700

# Overlap between consecutive chunks in characters.
# Ensures sentences or ideas spanning boundaries are not split and lost.
CHUNK_OVERLAP = 120

# -----------------------------------------------------------------------------
# Embeddings & Vector Database Settings
# -----------------------------------------------------------------------------
# Lightweight, high-performing local embedding model from Hugging Face
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Qdrant collection name
QDRANT_COLLECTION_NAME = "rag_documents"

# Local storage path for Qdrant (in-memory or local disk)
QDRANT_STORAGE_PATH = BASE_DIR / ".qdrant_storage"

# Default number of top matching chunks to retrieve
TOP_K = 5

# -----------------------------------------------------------------------------
# LLM Settings (Supports Free Ollama Qwen, Gemini, and Built-in Local)
# -----------------------------------------------------------------------------
# Default provider: "ollama" (free local Qwen), "builtin" (zero-setup), or "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama settings (Free & local, no API keys needed)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5")

# Google Gemini API model available on free tier
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Custom / OpenAI-compatible endpoint key (if provided)
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY", "052d70ce2cc949978d33978089daf4c7.tyfDXksDRABbqbhI9ROvT4YO")

# -----------------------------------------------------------------------------
# Evaluation Settings
# -----------------------------------------------------------------------------
EVAL_DATASET_PATH = DATA_DIR / "evaluation.json"
