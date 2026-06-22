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
3. Fetch unique evidence text and stored vectors from ``PaperEvidence``.
4. Compute per-chunk query similarity, evidence similarities, and F1.
5. Compute precision/recall/F1 for joined top-K text against joined evidence.

The function yields one evaluation record per (question, granularity).
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid as _uuid
from typing import Any, Dict, Generator, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from config import CHUNK_SIZES
from metrics import count_tokens, token_f1, token_precision_recall_f1

logger = logging.getLogger(__name__)

METHOD_NAME = "fixed-sized granularity - separate"
NAMESPACE_DNS = _uuid.NAMESPACE_DNS


def _make_eval_id(question_id: str, granularity_value: int) -> str:
    """Deterministic UUID-5 for an evaluation record."""
    seed = f"{METHOD_NAME}|{question_id}|{granularity_value}"
    return str(_uuid.uuid5(NAMESPACE_DNS, seed))


# ── Evidence fetcher ─────────────────────────────────────
def _dense_vector(vector: Any, point_id: str) -> List[float]:
    """Return a dense vector from Qdrant's unnamed/named representation."""
    if isinstance(vector, list):
        return vector
    if isinstance(vector, dict) and len(vector) == 1:
        value = next(iter(vector.values()))
        if isinstance(value, list):
            return value
    raise ValueError(f"Point {point_id} has no usable dense vector")


def cosine_similarity(left: List[float], right: List[float]) -> float:
    """Return cosine similarity, using 0.0 when either vector has zero norm."""
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} != {len(right)}")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fetch_evidence(
    client: QdrantClient,
    question_point_id: str,
) -> List[Dict]:
    """Return unique non-empty evidence records with their stored vectors.

    Exact duplicates after surrounding-whitespace removal are evaluated once.
    Sorting by point ID makes selection deterministic if duplicate text somehow
    exists under multiple IDs; the smallest ID supplies the retained vector.
    """
    raw_evidence = []
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
            with_vectors=True,
        )
        for point in results:
            ev = (point.payload or {}).get("evidence_text", "")
            if ev and ev.strip():
                point_id = str(point.id)
                raw_evidence.append(
                    {
                        "evidence_id": point_id,
                        "text": ev.strip(),
                        "vector": _dense_vector(point.vector, point_id),
                    }
                )
        if next_offset is None:
            break
        offset = next_offset

    unique_evidence = []
    seen_text = set()
    for evidence in sorted(raw_evidence, key=lambda item: item["evidence_id"]):
        if evidence["text"] in seen_text:
            continue
        seen_text.add(evidence["text"])
        unique_evidence.append(evidence)
    return unique_evidence


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
    evidence_records = _fetch_evidence(client, question_point_id)
    if not evidence_records:
        logger.debug(
            "No evidence found for question %s (doc %s) – skipping.",
            question_point_id, document_id,
        )
        return  # generator yields nothing

    evidence_text = "\n".join(item["text"] for item in evidence_records)
    evidence_ids = [item["evidence_id"] for item in evidence_records]
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
            with_vectors=True,
        )
        search_results = response.points

        t_end = time.perf_counter()
        retrieval_time_ms = round((t_end - t_start) * 1000, 2)

        # Collect results (already ranked by Qdrant score descending)
        topk_chunk_ids: List[str] = []
        topk_chunk_indices: List[int] = []
        topk_scores: List[float] = []
        topk_texts: List[str] = []
        retrieved_chunks: List[Dict] = []
        all_evidence_similarities: List[float] = []

        for rank, hit in enumerate(search_results, start=1):
            payload = hit.payload or {}
            chunk_id = str(hit.id)
            chunk_idx = payload.get("chunk_idx", -1)
            chunk_text = payload.get("content", "")
            query_similarity = round(hit.score, 6)
            chunk_vector = _dense_vector(hit.vector, chunk_id)
            chunk_token_count = payload.get("chunk_size")
            if chunk_token_count is None:
                chunk_token_count = count_tokens(chunk_text)

            evidence_similarities = []
            evidence_f1_scores = []
            chunk_similarity_values = []
            chunk_f1_values = []
            for evidence in evidence_records:
                raw_similarity = cosine_similarity(chunk_vector, evidence["vector"])
                similarity = round(raw_similarity, 6)
                f1 = round(token_f1(chunk_text, evidence["text"]), 6)
                evidence_similarities.append(
                    {
                        "evidence_id": evidence["evidence_id"],
                        "cosine_similarity": similarity,
                    }
                )
                evidence_f1_scores.append(
                    {"evidence_id": evidence["evidence_id"], "token_f1": f1}
                )
                chunk_similarity_values.append(raw_similarity)
                chunk_f1_values.append(f1)
                all_evidence_similarities.append(raw_similarity)

            chunk_record = {
                "chunk_id": chunk_id,
                "chunk_idx": chunk_idx,
                "granularity_level": level,
                "granularity_tokens": CHUNK_SIZES[level - 1] if level <= len(CHUNK_SIZES) else None,
                "chunk_token_count": chunk_token_count,
                "rank": rank,
                "query_similarity": query_similarity,
                "evidence_cosine_similarities": evidence_similarities,
                "max_evidence_similarity": round(max(chunk_similarity_values), 6),
                "mean_evidence_similarity": round(_mean(chunk_similarity_values), 6),
                "evidence_token_f1_scores": evidence_f1_scores,
                "max_chunk_f1": round(max(chunk_f1_values), 6),
            }
            if store_retrieved_text:
                chunk_record["text"] = chunk_text
            retrieved_chunks.append(chunk_record)
            topk_chunk_ids.append(chunk_id)
            topk_chunk_indices.append(chunk_idx)
            topk_scores.append(query_similarity)
            topk_texts.append(chunk_text)

        retrieved_k = len(topk_chunk_ids)

        # Join all retrieved chunks into one text block
        retrieved_text = "\n".join(topk_texts)
        retrieved_tok_count = count_tokens(retrieved_text)

        # Set-level token precision / recall / F1
        precision, recall, f1_value = token_precision_recall_f1(retrieved_text, evidence_text)

        # Aggregate scores
        avg_score = round(sum(topk_scores) / len(topk_scores), 6) if topk_scores else 0.0
        best_score = round(max(topk_scores), 6) if topk_scores else 0.0
        mean_evidence_similarity = round(_mean(all_evidence_similarities), 6)
        best_evidence_similarity = (
            round(max(all_evidence_similarities), 6) if all_evidence_similarities else 0.0
        )

        # Granularity token size (for readability)
        gran_tokens = CHUNK_SIZES[level - 1] if level <= len(CHUNK_SIZES) else None

        record: Dict = {
            "schema_version":             2,
            "eval_id":                    _make_eval_id(question_point_id, level),
            "method_name":                METHOD_NAME,
            "question_id":                question_point_id,
            "document_id":                document_id,
            "split":                      split,
            "granularity_level":          level,
            "granularity_tokens":         gran_tokens,
            "k_requested":               top_k,
            "retrieved_k":               retrieved_k,
            "returned_k":                retrieved_k,
            "retrieval_time_ms":         retrieval_time_ms,
            "retrieval_latency_ms":       retrieval_time_ms,
            "evidence_hash":             evidence_hash,
            "evidence_token_count":      evidence_tok_count,
            "unique_evidence_count":      len(evidence_records),
            "unique_evidence_ids":        evidence_ids,
            "joined_unique_evidence_token_count": evidence_tok_count,
            "retrieved_joined_token_count": retrieved_tok_count,
            "joined_retrieved_text_token_count": retrieved_tok_count,
            "topk_chunk_ids":            topk_chunk_ids,
            "topk_chunk_indices":        topk_chunk_indices,
            "topk_scores":               topk_scores,
            "retrieved_chunks":          retrieved_chunks,
            "set_level_precision":       round(precision, 6),
            "set_level_recall":          round(recall, 6),
            "set_level_f1":              round(f1_value, 6),
            "f1_joined_topk":            round(f1_value, 6),
            "mean_query_similarity_topk": avg_score,
            "best_query_similarity_topk": best_score,
            "avg_score_topk":            avg_score,
            "best_score_topk":           best_score,
            "mean_evidence_similarity_topk": mean_evidence_similarity,
            "best_evidence_similarity_topk": best_evidence_similarity,
        }

        if store_retrieved_text:
            record["retrieved_text"] = retrieved_text
            record["evidence_text"] = evidence_text

        yield record
