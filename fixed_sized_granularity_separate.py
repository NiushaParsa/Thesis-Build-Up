"""Fixed-size separate-granularity retrieval and schema-v2 evaluation."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import (
    CHUNK_SIZES,
    EMBEDDING_DIM,
    OPENAI_EMBEDDING_MODEL,
    PAPER_CHUNK_COLLECTION,
    PAPER_EVIDENCE_COLLECTION,
    PAPER_QUESTION_COLLECTION,
    RETRIEVAL_EVALUATION_COLLECTION,
    RETRIEVAL_TOP_K,
    ROUTER_DATASET_COLLECTION,
    ROUTER_LABEL_TIE_EPSILON,
    TOKENIZER_NAME,
)
from embedding_utils import get_embeddings
from evaluation_utils import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_VERSION,
    METHOD_NAME,
    NORMALIZATION_VERSION,
    build_evaluation_config,
    evaluation_config_hash as compute_evaluation_config_hash,
    make_evaluation_id,
    new_evaluation_run_id,
)
from metrics import count_tokens, normalize_text, token_precision_recall_f1


logger = logging.getLogger(__name__)


def _make_eval_id(question_id: str, granularity_value: int, config_hash: str) -> str:
    """Compatibility wrapper for the configuration-aware evaluation ID."""
    return make_evaluation_id(METHOD_NAME, question_id, granularity_value, config_hash)


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


def _validate_dimension(vector: List[float], expected: int, point_id: str) -> None:
    if len(vector) != expected:
        raise ValueError(
            f"Vector dimension mismatch for {point_id}: {len(vector)} != {expected}"
        )


def _fetch_evidence(
    client: QdrantClient,
    question_point_id: str,
    *,
    evidence_collection: str = PAPER_EVIDENCE_COLLECTION,
    embedding_dimension: int = EMBEDDING_DIM,
    embedding_fn=None,
) -> Dict:
    """Fetch, normalize-deduplicate, and vectorize evidence deterministically."""
    raw_points = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=evidence_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="question_id", match=MatchValue(value=question_point_id)
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        raw_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset

    raw_count = len(raw_points)
    valid = []
    for point in sorted(raw_points, key=lambda item: str(item.id)):
        text = (point.payload or {}).get("evidence_text", "")
        if isinstance(text, str) and text.strip():
            valid.append(
                {
                    "evidence_id": str(point.id),
                    "text": text.strip(),
                    "stored_vector": point.vector,
                }
            )

    unique = []
    seen_normalized = set()
    for evidence in valid:
        normalized = normalize_text(evidence["text"])
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        unique.append(evidence)

    missing_vector_indices = []
    for index, evidence in enumerate(unique):
        try:
            vector = _dense_vector(evidence.pop("stored_vector"), evidence["evidence_id"])
        except ValueError:
            vector = None
            evidence.pop("stored_vector", None)
            missing_vector_indices.append(index)
        if vector is not None:
            _validate_dimension(vector, embedding_dimension, evidence["evidence_id"])
            evidence["vector"] = vector

    if missing_vector_indices:
        embed = embedding_fn or get_embeddings
        texts = [unique[index]["text"] for index in missing_vector_indices]
        try:
            generated = embed(texts)
        except Exception as exc:
            raise RuntimeError(
                f"Could not obtain compatible vectors for {len(texts)} evidence passages: {exc}"
            ) from exc
        if len(generated) != len(texts):
            raise ValueError(
                f"Embedding service returned {len(generated)} vectors for {len(texts)} passages"
            )
        for index, vector in zip(missing_vector_indices, generated):
            _validate_dimension(vector, embedding_dimension, unique[index]["evidence_id"])
            unique[index]["vector"] = vector
            unique[index]["vector_source"] = "embedding_fallback"

    for evidence in unique:
        evidence.setdefault("vector_source", "stored")

    return {
        "raw_evidence_count": raw_count,
        "valid_evidence_count": len(valid),
        "unique_evidence_count": len(unique),
        "records": unique,
    }


def evaluate_question(
    client: QdrantClient,
    question_point_id: str,
    question_vector: List[float],
    document_id: str,
    question_text: str,
    split: str,
    top_k: int = RETRIEVAL_TOP_K,
    granularity_levels: Optional[List[int]] = None,
    store_retrieved_text: bool = False,
    *,
    chunk_sizes: Optional[List[int]] = None,
    chunk_collection: str = PAPER_CHUNK_COLLECTION,
    question_collection: str = PAPER_QUESTION_COLLECTION,
    evidence_collection: str = PAPER_EVIDENCE_COLLECTION,
    embedding_model: str = OPENAI_EMBEDDING_MODEL,
    embedding_dimension: int = EMBEDDING_DIM,
    tokenizer_name: str = TOKENIZER_NAME,
    evaluation_run_id: Optional[str] = None,
    evaluation_config_hash: Optional[str] = None,
    evaluation_timestamp: Optional[str] = None,
    evidence_embedding_fn=None,
) -> Generator[Dict, None, None]:
    """Yield one complete schema-v2 record per requested granularity level."""
    chunk_sizes = list(chunk_sizes or CHUNK_SIZES)
    if granularity_levels is None:
        granularity_levels = list(range(1, len(chunk_sizes) + 1))
    if top_k < 1:
        raise ValueError("top_k must be positive")
    _validate_dimension(question_vector, embedding_dimension, question_point_id)

    if evaluation_config_hash is None:
        config = build_evaluation_config(
            method=METHOD_NAME,
            top_k=top_k,
            chunk_sizes=chunk_sizes,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            tokenizer=tokenizer_name,
            chunk_collection=chunk_collection,
            question_collection=question_collection,
            evidence_collection=evidence_collection,
            evaluation_collection=RETRIEVAL_EVALUATION_COLLECTION,
            router_collection=ROUTER_DATASET_COLLECTION,
            router_label_tie_epsilon=ROUTER_LABEL_TIE_EPSILON,
            store_text=store_retrieved_text,
        )
        evaluation_config_hash = compute_evaluation_config_hash(config)
    evaluation_run_id = evaluation_run_id or new_evaluation_run_id()
    evaluation_timestamp = evaluation_timestamp or datetime.now(timezone.utc).isoformat()

    evidence_summary = _fetch_evidence(
        client,
        question_point_id,
        evidence_collection=evidence_collection,
        embedding_dimension=embedding_dimension,
        embedding_fn=evidence_embedding_fn,
    )
    evidence_records = evidence_summary["records"]
    if not evidence_records:
        logger.debug(
            "No valid evidence found for question %s (doc %s) – skipping.",
            question_point_id,
            document_id,
        )
        return

    evidence_text = "\n".join(item["text"] for item in evidence_records)
    evidence_ids = [item["evidence_id"] for item in evidence_records]
    evidence_vector_sources = [item["vector_source"] for item in evidence_records]
    evidence_token_count = count_tokens(evidence_text)
    evidence_hash = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()

    for level in granularity_levels:
        if level < 1 or level > len(chunk_sizes):
            raise ValueError(f"Unknown granularity level: {level}")
        granularity_tokens = chunk_sizes[level - 1]
        started = time.perf_counter()
        response = client.query_points(
            collection_name=chunk_collection,
            query=question_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    ),
                    FieldCondition(
                        key="granularity_level", match=MatchValue(value=level)
                    ),
                ]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        retrieval_latency_ms = round((time.perf_counter() - started) * 1000, 2)

        retrieved_chunks = []
        topk_chunk_ids = []
        topk_chunk_indices = []
        topk_chunk_ranks = []
        topk_chunk_spans = []
        topk_chunk_token_counts = []
        topk_scores = []
        topk_texts = []
        all_evidence_similarities_raw = []
        max_evidence_per_chunk_raw = []
        max_f1_per_chunk_raw = []

        for rank, hit in enumerate(response.points, start=1):
            payload = hit.payload or {}
            chunk_id = str(hit.id)
            chunk_text = payload.get("content", "")
            chunk_vector = _dense_vector(hit.vector, chunk_id)
            _validate_dimension(chunk_vector, embedding_dimension, chunk_id)
            chunk_token_count = payload.get("chunk_size")
            if chunk_token_count is None:
                chunk_token_count = count_tokens(chunk_text)
            query_chunk_similarity = round(float(hit.score), 6)

            evidence_similarity_rows = []
            evidence_f1_rows = []
            raw_similarities = []
            raw_f1_values = []
            for evidence in evidence_records:
                similarity = cosine_similarity(chunk_vector, evidence["vector"])
                precision, recall, f1 = token_precision_recall_f1(
                    chunk_text, evidence["text"]
                )
                raw_similarities.append(similarity)
                raw_f1_values.append(f1)
                evidence_similarity_rows.append(
                    {
                        "evidence_id": evidence["evidence_id"],
                        "cosine_similarity": round(similarity, 6),
                    }
                )
                evidence_f1_rows.append(
                    {
                        "evidence_id": evidence["evidence_id"],
                        "precision": round(precision, 6),
                        "recall": round(recall, 6),
                        "token_f1": round(f1, 6),
                    }
                )

            best_f1_index = max(range(len(raw_f1_values)), key=raw_f1_values.__getitem__)
            best_f1_row = evidence_f1_rows[best_f1_index]
            all_evidence_similarities_raw.extend(raw_similarities)
            max_evidence_per_chunk_raw.append(max(raw_similarities))
            max_f1_per_chunk_raw.append(max(raw_f1_values))
            span = {
                "span_start": payload.get("span_start"),
                "span_end": payload.get("span_end"),
            }
            chunk_record = {
                "chunk_id": chunk_id,
                "chunk_idx": payload.get("chunk_idx", -1),
                "granularity_level": level,
                "granularity_tokens": granularity_tokens,
                "chunk_token_count": chunk_token_count,
                "rank": rank,
                "span_start": span["span_start"],
                "span_end": span["span_end"],
                "query_chunk_similarity": query_chunk_similarity,
                "query_similarity": query_chunk_similarity,
                "evidence_cosine_similarities": evidence_similarity_rows,
                "max_evidence_similarity": round(max(raw_similarities), 6),
                "mean_evidence_similarity": round(_mean(raw_similarities), 6),
                "evidence_token_f1_scores": evidence_f1_rows,
                "max_chunk_f1": round(max(raw_f1_values), 6),
                "mean_chunk_f1": round(_mean(raw_f1_values), 6),
                "precision_at_max_chunk_f1": best_f1_row["precision"],
                "recall_at_max_chunk_f1": best_f1_row["recall"],
                "best_f1_evidence_id": best_f1_row["evidence_id"],
            }
            if store_retrieved_text:
                chunk_record["text"] = chunk_text
            retrieved_chunks.append(chunk_record)
            topk_chunk_ids.append(chunk_id)
            topk_chunk_indices.append(chunk_record["chunk_idx"])
            topk_chunk_ranks.append(rank)
            topk_chunk_spans.append(span)
            topk_chunk_token_counts.append(chunk_token_count)
            topk_scores.append(query_chunk_similarity)
            topk_texts.append(chunk_text)

        returned_k = len(retrieved_chunks)
        retrieved_text = "\n".join(topk_texts)
        retrieved_token_count = count_tokens(retrieved_text)
        precision, recall, f1 = token_precision_recall_f1(retrieved_text, evidence_text)

        query_values = [chunk["query_chunk_similarity"] for chunk in retrieved_chunks]
        mean_query = round(_mean(query_values), 6)
        best_query = round(max(query_values), 6) if query_values else 0.0
        mean_evidence = round(_mean(all_evidence_similarities_raw), 6)
        best_evidence = (
            round(max(all_evidence_similarities_raw), 6)
            if all_evidence_similarities_raw
            else 0.0
        )
        mean_max_evidence = round(_mean(max_evidence_per_chunk_raw), 6)
        best_chunk_f1 = (
            round(max(max_f1_per_chunk_raw), 6) if max_f1_per_chunk_raw else 0.0
        )
        mean_chunk_f1 = round(_mean(max_f1_per_chunk_raw), 6)

        record = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "method_name": METHOD_NAME,
            "eval_id": _make_eval_id(
                question_point_id, level, evaluation_config_hash
            ),
            "evaluation_run_id": evaluation_run_id,
            "evaluation_config_hash": evaluation_config_hash,
            "question_id": question_point_id,
            "document_id": document_id,
            "split": split,
            "granularity_level": level,
            "granularity_tokens": granularity_tokens,
            "k_requested": top_k,
            "retrieved_k": returned_k,
            "returned_k": returned_k,
            "retrieval_time_ms": retrieval_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            **{key: evidence_summary[key] for key in (
                "raw_evidence_count",
                "valid_evidence_count",
                "unique_evidence_count",
            )},
            "unique_evidence_ids": evidence_ids,
            "evidence_vector_sources": evidence_vector_sources,
            "evidence_hash": evidence_hash,
            "evidence_token_count": evidence_token_count,
            "joined_unique_evidence_token_count": evidence_token_count,
            "retrieved_joined_token_count": retrieved_token_count,
            "joined_retrieved_text_token_count": retrieved_token_count,
            "topk_chunk_ids": topk_chunk_ids,
            "topk_chunk_indices": topk_chunk_indices,
            "topk_chunk_ranks": topk_chunk_ranks,
            "topk_chunk_spans": topk_chunk_spans,
            "topk_chunk_token_counts": topk_chunk_token_counts,
            "topk_scores": topk_scores,
            "retrieved_chunks": retrieved_chunks,
            "precision_joined_topk": round(precision, 6),
            "recall_joined_topk": round(recall, 6),
            "f1_joined_topk": round(f1, 6),
            "set_level_precision": round(precision, 6),
            "set_level_recall": round(recall, 6),
            "set_level_f1": round(f1, 6),
            "best_query_similarity_topk": best_query,
            "mean_query_similarity_topk": mean_query,
            "best_score_topk": best_query,
            "avg_score_topk": mean_query,
            "best_evidence_similarity_topk": best_evidence,
            "mean_evidence_similarity_topk": mean_evidence,
            "mean_max_evidence_similarity_topk": mean_max_evidence,
            "best_chunk_f1_topk": best_chunk_f1,
            "mean_chunk_f1_topk": mean_chunk_f1,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "tokenizer_identity": tokenizer_name,
            "metric_version": METRIC_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "timestamp": evaluation_timestamp,
        }
        if store_retrieved_text:
            record["retrieved_text"] = retrieved_text
            record["evidence_text"] = evidence_text
        yield record
