"""FastRAG Conservative Text Normalization.

Preserves semantic content, casing, punctuation, technical terminology, and code identifiers
while sanitizing encoding anomalies, null bytes, line-ending discrepancies, and erratic spacing.
"""

from __future__ import annotations

import re
import unicodedata


# Precompile regex patterns for performance
_NULL_CHAR_RE = re.compile(r"\x00+")
_LINE_ENDING_RE = re.compile(r"\r\n|\r")
_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n\r]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str, preserve_paragraphs: bool = True) -> str:
    """Conservatively normalize text without destroying technical identifiers, casing, or punctuation.

    Transformation rules:
    1. Unicode Normalization: Uses standard Unicode NFC (Normalization Form C) to ensure
       canonical composition without modifying accented glyphs or emoji characters.
    2. Null Character Stripping: Strips binary null bytes (`\\x00`) that cause C-string truncation.
    3. Line Ending Normalization: Converts Windows (`\\r\\n`) and classic Mac (`\\r`) line breaks to Unix standard (`\\n`).
    4. Horizontal Whitespace Consolidation: Replaces runs of tabs and spaces within a single line with a single space.
    5. Vertical Whitespace Regulation:
       - If `preserve_paragraphs=True`, collapses runs of 3+ newlines into standard double-newlines (`\\n\\n`),
         preserving distinct paragraph structure.
       - If `preserve_paragraphs=False`, collapses all newline clusters into single spaces.
    6. Boundary Trimming: Strips leading and trailing whitespace from the document bounds.

    Guarantees:
    - Never lowercases or stems words.
    - Never strips punctuation, markdown syntax, mathematical notation, URLs, or code symbols.
    """
    if not text:
        return ""

    # 1. Unicode canonical composition
    normalized = unicodedata.normalize("NFC", text)

    # 2. Remove null bytes
    normalized = _NULL_CHAR_RE.sub("", normalized)

    # 3. Standardize line endings to \n
    normalized = _LINE_ENDING_RE.sub("\n", normalized)

    if preserve_paragraphs:
        # 4. Collapse runs of tabs/spaces within lines, while keeping \n intact
        lines = [
            _HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip()
            for line in normalized.split("\n")
        ]
        rejoined = "\n".join(lines)
        # 5. Collapse excessive line breaks (3 or more) to clean double-newline paragraph separation
        normalized = _MULTIPLE_NEWLINES_RE.sub("\n\n", rejoined)
    else:
        # Collapse all whitespace, including newlines, into single spaces
        normalized = re.sub(r"\s+", " ", normalized)

    # 6. Trim document bounds
    return normalized.strip()
