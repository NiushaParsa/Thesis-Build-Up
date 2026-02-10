"""
OpenAI Embedding Utilities
==========================
Batch-capable embedding helper that calls the OpenAI API directly
(Weaviate auto-embedding is intentionally not used).

Features
--------
* Automatic batching (controlled by ``EMBEDDING_BATCH_SIZE``)
* Exponential-backoff retries on transient errors
* Empty-string safety
"""

from __future__ import annotations

import time
import logging
from typing import List, Optional

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """Lazy-initialise a singleton OpenAI client."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-PASTE"):
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Please set a valid key in your .env file."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def get_embeddings(texts: List[str], max_retries: int = 5) -> List[List[float]]:
    """
    Return embedding vectors for *texts* using the configured OpenAI model.

    Parameters
    ----------
    texts : list[str]
        Strings to embed.
    max_retries : int
        Per-batch retry budget with exponential back-off.

    Returns
    -------
    list[list[float]]
        One embedding vector per input text, same order.
    """
    if not texts:
        return []

    client = get_openai_client()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        # OpenAI rejects empty strings
        batch = [t if t.strip() else " " for t in batch]

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=OPENAI_EMBEDDING_MODEL,
                    input=batch,
                )
                # Guarantee ordering
                sorted_data = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend(item.embedding for item in sorted_data)
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt, 60)
                    logger.warning(
                        "Embedding API error (attempt %d/%d): %s  – retrying in %ds",
                        attempt + 1, max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Failed after %d attempts", max_retries)
                    raise

    return all_embeddings


def get_single_embedding(text: str) -> List[float]:
    """Convenience wrapper – embed a single string."""
    return get_embeddings([text])[0]
