"""FastRAG Configurable Document Chunking.

Implements deterministic, boundary-aware text chunking with configurable window size,
overlap, character offset tracking, and content-addressed chunk identifiers.
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.ingestion.models import Chunk, Document


def generate_chunk_id(
    document_id: str,
    chunk_index: int,
    chunk_size: int,
    chunk_overlap: int,
    chunk_text: str,
) -> str:
    """Generate a reproducible, content-addressed chunk identifier.

    Format:
        {document_id}#c{chunk_index:04d}_{sha256[:12]}

    Guarantees:
    - Stable ordering and human-readable document prefix.
    - Deterministic across repeated runs.
    - Sensitivity to chunk text and configuration changes.
    """
    fingerprint_raw = f"{document_id}:{chunk_index}:{chunk_size}:{chunk_overlap}:{chunk_text}"
    content_hash = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}#c{chunk_index:04d}_{content_hash}"


class Chunker:
    """Configurable, deterministic text chunker with semantic boundary awareness."""

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 150

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        preserve_paragraphs: bool = True,
    ) -> None:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
            raise TypeError(f"chunk_size must be an integer, got {type(chunk_size).__name__}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be strictly positive (>= 1), got {chunk_size}")

        if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
            raise TypeError(f"chunk_overlap must be an integer, got {type(chunk_overlap).__name__}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative (>= 0), got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_paragraphs = preserve_paragraphs

    def _find_breakpoint(self, text: str, start: int, target_end: int) -> int:
        """Find the optimal semantic boundary between start + chunk_size//2 and target_end.

        Evaluates boundary delimiters in descending priority order:
        1. Paragraph boundary: '\n\n'
        2. Line boundary: '\n'
        3. Sentence boundary: '. ', '? ', '! '
        4. Word boundary: ' '
        Falls back to target_end if no clean delimiter is found in the search window.
        """
        search_start = start + (self.chunk_size // 2)
        if search_start >= target_end:
            return target_end

        window = text[search_start:target_end]

        delimiters = [
            ("\n\n", 2),
            ("\n", 1),
            (". ", 2),
            ("? ", 2),
            ("! ", 2),
            (" ", 1),
        ]

        if not self.preserve_paragraphs:
            # Drop paragraph/line priority when paragraph preservation is disabled
            delimiters = [(". ", 2), ("? ", 2), ("! ", 2), (" ", 1)]

        for delim, delim_len in delimiters:
            pos = window.rfind(delim)
            if pos != -1:
                # Return absolute position including the delimiter
                return search_start + pos + delim_len

        return target_end

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a Document into deterministic, boundary-aware Chunk records.

        Args:
            document: Validated Document record.

        Returns:
            Ordered list of Chunk objects referencing the parent document.
        """
        text = document.text
        if not text or document.is_empty:
            return []

        doc_len = len(text)

        # Fast path: text fits within a single chunk window
        if doc_len <= self.chunk_size:
            chunk_text = text.strip()
            l_strip = len(text) - len(text.lstrip())
            r_strip = len(text) - len(text.rstrip())
            start_off = l_strip
            end_off = doc_len - r_strip

            if not chunk_text:
                return []

            chunk_id = generate_chunk_id(
                document_id=document.document_id,
                chunk_index=0,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                chunk_text=chunk_text,
            )
            return [
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=0,
                    text=chunk_text,
                    source=document.source,
                    start_offset=start_off,
                    end_offset=end_off,
                    metadata=dict(document.metadata),
                )
            ]

        # Multi-chunk sliding window path
        chunks: list[Chunk] = []
        current_start = 0
        chunk_idx = 0

        while current_start < doc_len:
            target_end = min(current_start + self.chunk_size, doc_len)

            if target_end < doc_len:
                actual_end = self._find_breakpoint(text, current_start, target_end)
            else:
                actual_end = doc_len

            raw_slice = text[current_start:actual_end]
            chunk_text = raw_slice.strip()

            if chunk_text:
                l_offset = len(raw_slice) - len(raw_slice.lstrip())
                r_offset = len(raw_slice) - len(raw_slice.rstrip())

                exact_start = current_start + l_offset
                exact_end = actual_end - r_offset

                chunk_id = generate_chunk_id(
                    document_id=document.document_id,
                    chunk_index=chunk_idx,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    chunk_text=chunk_text,
                )

                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    source=document.source,
                    start_offset=exact_start,
                    end_offset=exact_end,
                    metadata=dict(document.metadata),
                )
                chunks.append(chunk)
                chunk_idx += 1

            if actual_end >= doc_len:
                break

            # Calculate next window start with overlap
            next_start = actual_end - self.chunk_overlap
            # Guarantee strictly forward progress to prevent infinite loops
            current_start = max(next_start, current_start + 1)

        return chunks
