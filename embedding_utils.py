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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None

# Mutable so prepare_dataset.py can override via CLI flag
_embedding_max_workers: int = 8


def set_embedding_max_workers(n: int) -> None:
    """Override the number of concurrent embedding batch workers."""
    global _embedding_max_workers
    _embedding_max_workers = n


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


def _embed_single_batch(
    batch_index: int,
    batch: List[str],
    max_retries: int,
) -> Tuple[int, List[List[float]]]:
    """
    Embed one batch and return ``(batch_index, embeddings)`` so results
    can be reassembled in order.
    """
    client = get_openai_client()
    # OpenAI rejects empty strings
    safe_batch = [t if t.strip() else " " for t in batch]

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=safe_batch,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return batch_index, [item.embedding for item in sorted_data]
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 60)
                logger.warning(
                    "Embedding API error (batch %d, attempt %d/%d): %s – retrying in %ds",
                    batch_index, attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("Batch %d failed after %d attempts", batch_index, max_retries)
                raise


def get_embeddings(texts: List[str], max_retries: int = 5) -> List[List[float]]:
    """
    Return embedding vectors for *texts* using the configured OpenAI model.

    Batches are dispatched to a thread pool (``EMBEDDING_MAX_WORKERS``)
    so multiple API calls run concurrently, dramatically reducing wall-
    clock time for large inputs.

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

    # Split into batches
    batches: List[Tuple[int, List[str]]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batches.append((len(batches), texts[i : i + EMBEDDING_BATCH_SIZE]))

    # Single batch → skip thread-pool overhead
    if len(batches) == 1:
        _, embeddings = _embed_single_batch(0, batches[0][1], max_retries)
        return embeddings

    # Multiple batches → dispatch concurrently
    results: dict[int, List[List[float]]] = {}
    workers = min(_embedding_max_workers, len(batches))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_embed_single_batch, idx, batch, max_retries): idx
            for idx, batch in batches
        }
        for future in as_completed(futures):
            batch_idx, embeddings = future.result()
            results[batch_idx] = embeddings

    # Reassemble in original order
    all_embeddings: List[List[float]] = []
    for idx in range(len(batches)):
        all_embeddings.extend(results[idx])

    return all_embeddings


def get_single_embedding(text: str) -> List[float]:
    """Convenience wrapper – embed a single string."""
    return get_embeddings([text])[0]
