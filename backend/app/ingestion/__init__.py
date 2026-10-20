"""FastRAG Document Ingestion and Chunking Package.

Provides document loaders, text normalization, boundary-aware chunking,
validation rules, and atomic pipeline orchestration.
"""

from app.ingestion.chunking import Chunker, generate_chunk_id
from app.ingestion.loaders import (
    DocumentLoadingError,
    load_directory,
    load_documents,
    load_jsonl,
    load_jsonl_stream,
    load_text_file,
)
from app.ingestion.models import Chunk, Document, IngestionSummary
from app.ingestion.normalization import normalize_text
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.validation import (
    IngestionValidationError,
    validate_chunk,
    validate_chunks_batch,
    validate_document,
    validate_documents_batch,
)

__all__ = [
    "Chunk",
    "Chunker",
    "Document",
    "DocumentLoadingError",
    "IngestionPipeline",
    "IngestionSummary",
    "IngestionValidationError",
    "generate_chunk_id",
    "load_directory",
    "load_documents",
    "load_jsonl",
    "load_jsonl_stream",
    "load_text_file",
    "normalize_text",
    "validate_chunk",
    "validate_chunks_batch",
    "validate_document",
    "validate_documents_batch",
]
