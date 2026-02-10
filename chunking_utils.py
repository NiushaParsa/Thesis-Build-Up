"""
Text Chunking Utilities
=======================
Uses a **HuggingFace tokenizer** (default: ``gpt2``) to split text into
non-overlapping chunks at various token-level granularities.

Every chunk carries:
* ``content``     – exact slice of the original text
* ``token_count`` – number of tokens the tokenizer produced
* ``span_start``  – character offset where the chunk starts
* ``span_end``    – character offset where the chunk ends
"""

from __future__ import annotations

from typing import Dict, List

from transformers import AutoTokenizer

from config import TOKENIZER_NAME

_tokenizer = None


def get_tokenizer() -> AutoTokenizer:
    """Lazy-load a **fast** HuggingFace tokenizer (needed for offset mapping)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
        if not _tokenizer.is_fast:
            raise RuntimeError(
                f"A fast tokenizer is required for offset mapping, but "
                f"'{TOKENIZER_NAME}' only has a slow implementation."
            )
    return _tokenizer


# ── Chunking ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int) -> List[Dict]:
    """
    Split *text* into **non-overlapping** chunks of *chunk_size* tokens.

    The last chunk may contain fewer tokens than *chunk_size*.

    Parameters
    ----------
    text : str
        Full document text.
    chunk_size : int
        Target number of tokens per chunk.

    Returns
    -------
    list[dict]
        Each dict has keys ``content``, ``token_count``,
        ``span_start``, ``span_end``.
    """
    if not text or not text.strip():
        return []

    tokenizer = get_tokenizer()

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )

    token_ids: list[int] = encoding["input_ids"]
    offsets: list[tuple[int, int]] = encoding["offset_mapping"]

    if not token_ids:
        return []

    chunks: List[Dict] = []
    for i in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[i : i + chunk_size]
        chunk_offsets = offsets[i : i + chunk_size]

        span_start = chunk_offsets[0][0]
        span_end = chunk_offsets[-1][1]
        content = text[span_start:span_end]

        chunks.append(
            {
                "content": content,
                "token_count": len(chunk_ids),
                "span_start": span_start,
                "span_end": span_end,
            }
        )

    return chunks


# ── Full-text builder ────────────────────────────────────
def build_full_text(paper: dict) -> str:
    """
    Concatenate title + abstract + every section of a QASPER paper
    into a single flat string.
    """
    parts: list[str] = []

    if paper.get("title"):
        parts.append(paper["title"])

    if paper.get("abstract"):
        parts.append(paper["abstract"])

    full_text = paper.get("full_text", {})
    section_names = full_text.get("section_name", [])
    paragraphs_list = full_text.get("paragraphs", [])

    for section_name, paragraphs in zip(section_names, paragraphs_list):
        if section_name:
            parts.append(section_name)
        if paragraphs:
            parts.extend(p for p in paragraphs if p and p.strip())

    return "\n\n".join(parts)


def count_tokens(text: str) -> int:
    """Return the token count for *text* (no special tokens)."""
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))
