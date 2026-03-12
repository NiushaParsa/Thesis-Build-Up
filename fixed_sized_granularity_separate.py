"""
Baseline Method – Fixed-Sized Granularity (Separate)
=====================================================
For each question, retrieval is performed **independently** at each
granularity level within the **same document**.

Algorithm per question:
1. Read the question vector from ``PaperQuestion``.
2. For each granularity level (1 … 5):
   a. Search ``PaperChunk`` with the question vector, filtered by
      ``document_id`` and ``granularity_level``.
   b. Collect top-K chunks (text, score, id, idx).
3. Fetch all evidence from ``PaperEvidence`` for this question.
4. Compute token-level F1 between the joined top-K texts and the
   joined evidence ground truth.

The function yields one evaluation record per (question, granularity).
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid as _uuid
from typing import Dict, Generator, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from config import CHUNK_SIZES
from metrics import count_tokens, token_f1

logger = logging.getLogger(__name__)

METHOD_NAME = "fixed-sized granularity - separate"
NAMESPACE_DNS = _uuid.NAMESPACE_DNS


def _make_eval_id(question_id: str, granularity_value: int) -> str:
    """Deterministic UUID-5 for an evaluation record."""
    seed = f"{METHOD_NAME}|{question_id}|{granularity_value}"
    return str(_uuid.uuid5(NAMESPACE_DNS, seed))


# ── Evidence fetcher ─────────────────────────────────────
def _fetch_evidence(
    client: QdrantClient,
    question_point_id: str,
) -> str:
    """Return joined evidence text for a question (empty string if none)."""
    evidence_texts: List[str] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name="PaperEvidence",
            scroll_filter=Filter(must=[
                FieldCondition(key="question_id", match=MatchValue(value=question_point_id)),
            ]),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            ev = (point.payload or {}).get("evidence_text", "")
            if ev and ev.strip():
                evidence_texts.append(ev.strip())
        if next_offset is None:
            break
        offset = next_offset

    return "\n".join(evidence_texts)


# ── Single-question evaluator ────────────────────────────
def evaluate_question(
    client: QdrantClient,
    question_point_id: str,
    question_vector: List[float],
    document_id: str,
    question_text: str,
    split: str,
    top_k: int = 5,
    granularity_levels: Optional[List[int]] = None,
    store_retrieved_text: bool = False,
) -> Generator[Dict, None, None]:
    """Yield one evaluation record per granularity level for a single question.

    Parameters
    ----------
    client : QdrantClient
    question_point_id : str
        UUID of the question point in PaperQuestion.
    question_vector : list[float]
        The question's embedding vector.
    document_id : str
        Paper ID used to filter chunks.
    question_text : str
    split : str
        Dataset split (train / validation / test).
    top_k : int
        Number of chunks to retrieve per granularity.
    granularity_levels : list[int] | None
        1-indexed levels (default: derived from CHUNK_SIZES).
    store_retrieved_text : bool
        If True, include the full joined text in the record.
    """
    if granularity_levels is None:
        granularity_levels = list(range(1, len(CHUNK_SIZES) + 1))

    # 1. Fetch evidence ground-truth (shared across granularities)
    evidence_text = _fetch_evidence(client, question_point_id)
    if not evidence_text:
        logger.debug(
            "No evidence found for question %s (doc %s) – skipping.",
            question_point_id, document_id,
        )
        return  # generator yields nothing

    evidence_tok_count = count_tokens(evidence_text)
    evidence_hash = hashlib.md5(evidence_text.encode()).hexdigest()

    # 2. For each granularity, search + evaluate
    for level in granularity_levels:
        t_start = time.perf_counter()

        response = client.query_points(
            collection_name="PaperChunk",
            query=question_vector,
            query_filter=Filter(must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(key="granularity_level", match=MatchValue(value=level)),
            ]),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        search_results = response.points

        t_end = time.perf_counter()
        retrieval_time_ms = round((t_end - t_start) * 1000, 2)

        # Collect results (already ranked by Qdrant score descending)
        topk_chunk_ids: List[str] = []
        topk_chunk_indices: List[int] = []
        topk_scores: List[float] = []
        topk_texts: List[str] = []

        for hit in search_results:
            topk_chunk_ids.append(str(hit.id))
            topk_chunk_indices.append((hit.payload or {}).get("chunk_idx", -1))
            topk_scores.append(round(hit.score, 6))
            topk_texts.append((hit.payload or {}).get("content", ""))

        retrieved_k = len(topk_chunk_ids)

        # Join all retrieved chunks into one text block
        retrieved_text = "\n".join(topk_texts)
        retrieved_tok_count = count_tokens(retrieved_text)

        # Token-level F1
        f1_value = token_f1(retrieved_text, evidence_text) if retrieved_k > 0 else 0.0

        # Aggregate scores
        avg_score = round(sum(topk_scores) / len(topk_scores), 6) if topk_scores else 0.0
        best_score = round(max(topk_scores), 6) if topk_scores else 0.0

        # Granularity token size (for readability)
        gran_tokens = CHUNK_SIZES[level - 1] if level <= len(CHUNK_SIZES) else None

        record: Dict = {
            "eval_id":                    _make_eval_id(question_point_id, level),
            "method_name":                METHOD_NAME,
            "question_id":                question_point_id,
            "document_id":                document_id,
            "split":                      split,
            "granularity_level":          level,
            "granularity_tokens":         gran_tokens,
            "k_requested":               top_k,
            "retrieved_k":               retrieved_k,
            "retrieval_time_ms":         retrieval_time_ms,
            "evidence_hash":             evidence_hash,
            "evidence_token_count":      evidence_tok_count,
            "retrieved_joined_token_count": retrieved_tok_count,
            "topk_chunk_ids":            topk_chunk_ids,
            "topk_chunk_indices":        topk_chunk_indices,
            "topk_scores":               topk_scores,
            "f1_joined_topk":            round(f1_value, 6),
            "avg_score_topk":            avg_score,
            "best_score_topk":           best_score,
        }

        if store_retrieved_text:
            record["retrieved_text"] = retrieved_text
            record["evidence_text"] = evidence_text

        yield record
