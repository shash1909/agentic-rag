"""PDF ingestion and chunking module for Adaptive Agentic RAG.

This module is responsible for Phase 1 of the RAG pipeline:
1. Extracting text from PDF files using PyMuPDF (fitz).
2. Cleaning raw extracted text (normalizing spaces, removing artifacts).
3. Splitting text into manageable, overlapping chunks while preserving word boundaries.
4. Preserving essential metadata on every chunk (document name, page number, chunk ID)
   so citations can point directly to the source.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import io
import re
import fitz  # PyMuPDF

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class DocumentChunk:
    """Represents a single chunk of text extracted from a document.

    Attributes:
        chunk_id: Unique identifier for the chunk, e.g., 'report.pdf_p1_c0'.
        text: The actual chunk text.
        doc_name: Name of the source document.
        page_number: 1-indexed page number where this chunk originates.
        chunk_index: 0-indexed position of this chunk within that page.
        metadata: Dictionary containing chunk attributes for easy retrieval and citations.
    """
    chunk_id: str
    text: str
    doc_name: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to a standard dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @property
    def citation(self) -> str:
        """Standard human-readable citation for this chunk."""
        return f"[{self.doc_name}, Page {self.page_number}]"


def clean_text(text: str) -> str:
    """Clean and normalize raw extracted PDF text.

    - Removes null bytes and form-feed characters.
    - Replaces consecutive whitespace characters while preserving paragraph breaks.
    - Strips leading and trailing whitespace.

    Args:
        text: Raw string extracted from PDF.

    Returns:
        Cleaned, human-readable string.
    """
    if not text:
        return ""

    # Remove null bytes and page breaks
    cleaned = text.replace("\x00", "").replace("\x0c", "\n")

    # Normalize carriage returns and tabs
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    # Collapse multiple spaces/tabs into a single space (per line)
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)

    # Collapse 3 or more consecutive newlines into 2 (preserving clean paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks respecting word boundaries.

    Why chunking is needed:
    - LLMs have context window limits and perform better with targeted context.
    - Embeddings represent dense semantic meaning of a specific passage.
    - Overlap ensures that ideas spanning chunk boundaries are not truncated.

    Args:
        text: Input string to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    # If text is already smaller than the target chunk size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break on clean sentence/paragraph boundaries rather than mid-sentence
        if end < len(text):
            sentence_boundary = -1
            search_start = start + (chunk_size // 2)
            search_end = min(end + 80, len(text))
            for sep in [".\n", ". ", "!\n", "! ", "?\n", "? ", "\n\n", "\n"]:
                pos = text.rfind(sep, search_start, search_end)
                if pos != -1:
                    pos_end = pos + len(sep.rstrip(" "))
                    if pos_end > sentence_boundary and pos_end <= search_end:
                        sentence_boundary = pos_end

            if sentence_boundary != -1:
                end = sentence_boundary
            else:
                boundary = text.rfind(" ", search_start, end)
                if boundary != -1:
                    end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Calculate next start with desired overlap
        next_start = end - chunk_overlap

        # Ensure strict forward progress
        if next_start <= start:
            next_start = start + step

        start = next_start

    return chunks


def extract_text_from_pdf(
    pdf_source: Union[str, Path, bytes, io.BytesIO],
    doc_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract page-by-page text from a PDF file using PyMuPDF.

    Args:
        pdf_source: File path (str/Path) or in-memory bytes/BytesIO (e.g. from Streamlit upload).
        doc_name: Optional human-readable name of document. If omitted, derived from path or assigned default.

    Returns:
        List of dicts: [{"page_number": int, "text": str, "doc_name": str}]

    Raises:
        FileNotFoundError: If a file path is provided but does not exist.
        ValueError: If PDF is empty, unreadable, or contains no extractable text.
    """
    # Determine document name and open PyMuPDF document
    doc: Optional[fitz.Document] = None

    try:
        if isinstance(pdf_source, (str, Path)):
            path_obj = Path(pdf_source)
            if not path_obj.exists():
                raise FileNotFoundError(f"PDF file not found: {path_obj}")
            if path_obj.stat().st_size == 0:
                raise ValueError(f"PDF file is empty (0 bytes): {path_obj.name}")
            doc_name = doc_name or path_obj.name
            doc = fitz.open(str(path_obj))
        elif isinstance(pdf_source, bytes):
            if len(pdf_source) == 0:
                raise ValueError("PDF byte buffer is empty (0 bytes).")
            doc_name = doc_name or "uploaded_document.pdf"
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        elif isinstance(pdf_source, io.BytesIO):
            data = pdf_source.getvalue()
            if len(data) == 0:
                raise ValueError("PDF byte stream is empty (0 bytes).")
            doc_name = doc_name or getattr(pdf_source, "name", "uploaded_document.pdf")
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            raise TypeError(f"Unsupported PDF source type: {type(pdf_source)}")

        if doc.page_count == 0:
            raise ValueError(f"PDF document '{doc_name}' has 0 pages.")

        pages_data: List[Dict[str, Any]] = []
        total_readable_chars = 0

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            page_text = page.get_text("text") or ""
            cleaned_page_text = clean_text(page_text)
            total_readable_chars += len(cleaned_page_text)

            pages_data.append({
                "page_number": page_idx + 1,  # 1-indexed for humans
                "text": cleaned_page_text,
                "doc_name": doc_name,
            })

        if total_readable_chars == 0:
            raise ValueError(
                f"The PDF '{doc_name}' contains no readable text. "
                "It may be blank or contain scanned images without an OCR text layer."
            )

        return pages_data

    except Exception as e:
        # Re-raise explicit ValueErrors/FileNotFoundErrors
        if isinstance(e, (ValueError, FileNotFoundError, TypeError)):
            raise
        # Wrap underlying PyMuPDF C-level errors in beginner-friendly Python ValueError
        raise ValueError(f"Failed to process PDF '{doc_name}': {str(e)}") from e

    finally:
        if doc is not None:
            doc.close()


def ingest_pdf(
    pdf_source: Union[str, Path, bytes, io.BytesIO],
    doc_name: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """Complete Phase 1 ingestion pipeline for a single PDF.

    Steps:
    1. Extract page-by-page text using PyMuPDF.
    2. Split each page's text into clean overlapping chunks.
    3. Attach complete citation metadata (doc_name, page_number, chunk_id).

    Args:
        pdf_source: File path or raw bytes of the PDF.
        doc_name: Optional display name for the document.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of DocumentChunk instances ready for embedding and indexing.
    """
    pages_data = extract_text_from_pdf(pdf_source, doc_name=doc_name)
    all_chunks: List[DocumentChunk] = []

    for page_info in pages_data:
        page_num = page_info["page_number"]
        page_text = page_info["text"]
        document_name = page_info["doc_name"]

        if not page_text:
            continue

        raw_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_idx, chunk_content in enumerate(raw_chunks):
            # Clean safe filename for chunk ID
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", document_name)
            chunk_id = f"{safe_name}_p{page_num}_c{chunk_idx}"

            chunk_obj = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_content,
                doc_name=document_name,
                page_number=page_num,
                chunk_index=chunk_idx,
                metadata={
                    "doc_name": document_name,
                    "page_number": page_num,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "char_count": len(chunk_content),
                    "word_count": len(chunk_content.split()),
                },
            )
            all_chunks.append(chunk_obj)

    if not all_chunks:
        raise ValueError(
            f"No valid text chunks could be produced from document '{pages_data[0]['doc_name']}'."
        )

    return all_chunks
