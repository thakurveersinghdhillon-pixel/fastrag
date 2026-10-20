"""FastRAG Document Loaders.

Loads documents from local JSONL corpora and plain-text (.txt) files
with schema validation and memory-efficient streaming.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Iterable

from app.ingestion.models import Document


class DocumentLoadingError(Exception):
    """Raised when an unrecoverable error occurs while loading documents from storage."""
    pass


def load_jsonl_stream(file_path: Path | str) -> Generator[Document, None, None]:
    """Yield Document objects from a JSONL file one line at a time.

    Supports both 'doc_id' and 'document_id' keys in the JSON records.

    Raises:
        FileNotFoundError: If the file does not exist.
        DocumentLoadingError: If a line contains invalid JSON or lacks required fields.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DocumentLoadingError(
                    f"{path} line {line_num}: invalid JSON syntax ({exc})"
                ) from exc

            if not isinstance(record, dict):
                raise DocumentLoadingError(
                    f"{path} line {line_num}: expected JSON object, found {type(record).__name__}"
                )

            try:
                yield Document.from_dict(record)
            except Exception as exc:
                raise DocumentLoadingError(
                    f"{path} line {line_num}: invalid document record ({exc})"
                ) from exc


def load_jsonl(file_path: Path | str) -> list[Document]:
    """Load all documents from a JSONL file into a list."""
    return list(load_jsonl_stream(file_path))


def load_text_file(
    file_path: Path | str,
    document_id: str | None = None,
    title: str | None = None,
    source: str | None = None,
    metadata: dict | None = None,
) -> Document:
    """Load a plain text (.txt) file into a Document record.

    If document_id is omitted, defaults to the file stem (e.g., 'notes' for 'notes.txt').
    If title is omitted, defaults to document_id.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Text file not found at: {path}")

    text_content = path.read_text(encoding="utf-8")
    doc_id = document_id or path.stem
    doc_title = title or path.stem
    doc_source = source or str(path)

    return Document(
        document_id=doc_id,
        text=text_content,
        title=doc_title,
        source=doc_source,
        metadata=metadata or {},
        raw_text=text_content,
    )


def load_directory(directory_path: Path | str, recursive: bool = True) -> list[Document]:
    """Discover and load all supported documents (.jsonl, .txt) from a directory."""
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    documents: list[Document] = []
    pattern = "**/*" if recursive else "*"

    for file_path in sorted(dir_path.glob(pattern)):
        if file_path.is_file():
            if file_path.suffix.lower() == ".jsonl":
                documents.extend(load_jsonl(file_path))
            elif file_path.suffix.lower() == ".txt":
                documents.append(load_text_file(file_path))

    return documents


def load_documents(source: Path | str) -> list[Document]:
    """Convenience loader dispatching to JSONL, TXT, or directory loader based on path type."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir():
        return load_directory(path)
    elif path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    elif path.suffix.lower() == ".txt":
        return [load_text_file(path)]
    else:
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Supported formats: .jsonl, .txt"
        )
