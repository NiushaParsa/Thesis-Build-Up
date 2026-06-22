"""Mixed-granularity retrieval strategies with schema-v2 metrics."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import (
    CHUNK_SIZES,
    EMBEDDING_DIM,
    MIXED_DEDUP_CANDIDATE_MULTIPLIER,
    MIXED_DEDUP_OVERLAP_THRESHOLD,
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
from evaluation_utils import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_VERSION,
    NORMALIZATION_VERSION,
    build_evaluation_config,
    evaluation_config_hash as compute_evaluation_config_hash,
    make_evaluation_id,
    new_evaluation_run_id,
)
from fixed_sized_granularity_separate import (
    _dense_vector,
    _fetch_evidence,
    _mean,
    _validate_dimension,
    cosine_similarity,
)
from metrics import count_tokens, token_precision_recall_f1


MIXED_RAW_METHOD = "mixed-raw"
MIXED_DEDUPLICATED_METHOD = "mixed-deduplicated"
MIXED_FILTER_BEHAVIOR = "same-document-all-granularity-levels"
OVERLAP_DEFINITION = "intersection-length/minimum-span-length"


def span_overlap_ratio(left: Tuple[int, int], right: Tuple[int, int]) -> float:
    """Return overlap relative to the shorter valid character span."""
    left_start, left_end = left
    right_start, right_end = right
    if (
        not isinstance(left_start, int)
        or not isinstance(left_end, int)
        or not isinstance(right_start, int)
        or not isinstance(right_end, int)
        or left_start < 0
        or right_start < 0
        or left_end <= left_start
        or right_end <= right_start
    ):
        return 0.0
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start))
    shorter = min(left_end - left_start, right_end - right_start)
    return intersection / shorter if shorter else 0.0


def deduplicate_ranked_hits(
    hits: List[Any], top_k: int, overlap_threshold: float
) -> Tuple[List[Tuple[Any, int]], List[dict], int]:
    """Greedily retain score-ranked hits whose spans do not strongly overlap."""
    selected: List[Tuple[Any, int]] = []
    suppressed = []
    candidates_examined = 0
    for candidate_rank, hit in enumerate(hits, start=1):
        candidates_examined += 1
        payload = hit.payload or {}
        candidate_span = (payload.get("span_start"), payload.get("span_end"))
        duplicate = None
        for retained_hit, retained_candidate_rank in selected:
            retained_payload = retained_hit.payload or {}
            ratio = span_overlap_ratio(
                candidate_span,
                (retained_payload.get("span_start"), retained_payload.get("span_end")),
            )
            if ratio >= overlap_threshold:
                duplicate = (retained_hit, retained_candidate_rank, ratio)
                break
        if duplicate is None:
            selected.append((hit, candidate_rank))
            if len(selected) == top_k:
                break
            continue
        retained_hit, retained_candidate_rank, ratio = duplicate
        suppressed.append(
            {
                "chunk_id": str(hit.id),
                "candidate_rank": candidate_rank,
                "granularity_level": payload.get("granularity_level"),
                "span_start": candidate_span[0],
                "span_end": candidate_span[1],
                "overlaps_chunk_id": str(retained_hit.id),
                "overlaps_candidate_rank": retained_candidate_rank,
                "overlap_ratio": round(ratio, 6),
            }
        )
    return selected, suppressed, candidates_examined


def _build_record(
    *,
    ranked_hits: List[Tuple[Any, int]],
    method_name: str,
    question_point_id: str,
    document_id: str,
    split: str,
    top_k: int,
    chunk_sizes: List[int],
    retrieval_latency_ms: float,
    evidence_summary: Dict,
    evaluation_run_id: str,
    evaluation_config_hash: str,
    evaluation_timestamp: str,
    embedding_model: str,
    embedding_dimension: int,
    tokenizer_name: str,
    store_retrieved_text: bool,
    strategy_fields: dict,
) -> dict:
    """Build the same per-chunk and aggregate metric schema as fixed-separate."""
    evidence_records = evidence_summary["records"]
    evidence_text = "\n".join(item["text"] for item in evidence_records)
    evidence_ids = [item["evidence_id"] for item in evidence_records]
    evidence_vector_sources = [item["vector_source"] for item in evidence_records]
    evidence_token_count = count_tokens(evidence_text)
    evidence_hash = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()

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

    for rank, (hit, candidate_rank) in enumerate(ranked_hits, start=1):
        payload = hit.payload or {}
        chunk_id = str(hit.id)
        chunk_text = payload.get("content", "")
        chunk_vector = _dense_vector(hit.vector, chunk_id)
        _validate_dimension(chunk_vector, embedding_dimension, chunk_id)
        level = payload.get("granularity_level")
        if not isinstance(level, int) or level < 1 or level > len(chunk_sizes):
            raise ValueError(f"Chunk {chunk_id} has invalid granularity level: {level}")
        granularity_tokens = chunk_sizes[level - 1]
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
            "candidate_rank": candidate_rank,
            **span,
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

    composition = []
    counts = Counter(chunk["granularity_level"] for chunk in retrieved_chunks)
    for level, tokens in enumerate(chunk_sizes, start=1):
        chunks = [
            chunk for chunk in retrieved_chunks if chunk["granularity_level"] == level
        ]
        composition.append(
            {
                "granularity_level": level,
                "granularity_tokens": tokens,
                "count": counts[level],
                "ranks": [chunk["rank"] for chunk in chunks],
                "candidate_ranks": [chunk["candidate_rank"] for chunk in chunks],
            }
        )

    record = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "method_name": method_name,
        "eval_id": make_evaluation_id(
            method_name, question_point_id, "all", evaluation_config_hash
        ),
        "evaluation_run_id": evaluation_run_id,
        "evaluation_config_hash": evaluation_config_hash,
        "question_id": question_point_id,
        "document_id": document_id,
        "split": split,
        "granularity_scope": "all",
        "k_requested": top_k,
        "retrieved_k": returned_k,
        "returned_k": returned_k,
        "retrieval_time_ms": retrieval_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        **{
            key: evidence_summary[key]
            for key in (
                "raw_evidence_count",
                "valid_evidence_count",
                "unique_evidence_count",
            )
        },
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
        "topk_granularity_levels": [
            chunk["granularity_level"] for chunk in retrieved_chunks
        ],
        "topk_granularity_tokens": [
            chunk["granularity_tokens"] for chunk in retrieved_chunks
        ],
        "granularity_composition": composition,
        "granularity_counts": {
            str(item["granularity_tokens"]): item["count"] for item in composition
        },
        "granularity_ranks": {
            str(item["granularity_tokens"]): item["ranks"] for item in composition
        },
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
        **strategy_fields,
    }
    if store_retrieved_text:
        record["retrieved_text"] = retrieved_text
        record["evidence_text"] = evidence_text
    return record


def evaluate_mixed_question(
    client: QdrantClient,
    question_point_id: str,
    question_vector: List[float],
    document_id: str,
    question_text: str,
    split: str,
    *,
    variant: str,
    top_k: int = RETRIEVAL_TOP_K,
    store_retrieved_text: bool = False,
    chunk_sizes: Optional[List[int]] = None,
    chunk_collection: str = PAPER_CHUNK_COLLECTION,
    question_collection: str = PAPER_QUESTION_COLLECTION,
    evidence_collection: str = PAPER_EVIDENCE_COLLECTION,
    evaluation_collection: str = RETRIEVAL_EVALUATION_COLLECTION,
    router_collection: str = ROUTER_DATASET_COLLECTION,
    embedding_model: str = OPENAI_EMBEDDING_MODEL,
    embedding_dimension: int = EMBEDDING_DIM,
    tokenizer_name: str = TOKENIZER_NAME,
    overlap_threshold: float = MIXED_DEDUP_OVERLAP_THRESHOLD,
    candidate_multiplier: int = MIXED_DEDUP_CANDIDATE_MULTIPLIER,
    evaluation_run_id: Optional[str] = None,
    evaluation_config_hash: Optional[str] = None,
    evaluation_timestamp: Optional[str] = None,
    evidence_embedding_fn=None,
) -> Optional[dict]:
    """Evaluate one question using an explicitly named mixed strategy."""
    if variant not in {MIXED_RAW_METHOD, MIXED_DEDUPLICATED_METHOD}:
        raise ValueError(f"Unknown mixed-granularity variant: {variant}")
    if top_k < 1 or candidate_multiplier < 1:
        raise ValueError("top_k and candidate_multiplier must be positive")
    if not 0.0 < overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be in (0, 1]")
    chunk_sizes = list(chunk_sizes or CHUNK_SIZES)
    _validate_dimension(question_vector, embedding_dimension, question_point_id)

    strategy_settings = {"variant": variant}
    if variant == MIXED_DEDUPLICATED_METHOD:
        strategy_settings.update(
            {
                "overlap_threshold": overlap_threshold,
                "overlap_definition": OVERLAP_DEFINITION,
                "candidate_multiplier": candidate_multiplier,
            }
        )
    if evaluation_config_hash is None:
        config = build_evaluation_config(
            method=variant,
            top_k=top_k,
            chunk_sizes=chunk_sizes,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            tokenizer=tokenizer_name,
            chunk_collection=chunk_collection,
            question_collection=question_collection,
            evidence_collection=evidence_collection,
            evaluation_collection=evaluation_collection,
            router_collection=router_collection,
            router_label_tie_epsilon=ROUTER_LABEL_TIE_EPSILON,
            store_text=store_retrieved_text,
            filter_behavior=MIXED_FILTER_BEHAVIOR,
            strategy_settings=strategy_settings,
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
    if not evidence_summary["records"]:
        return None

    candidate_limit = (
        top_k
        if variant == MIXED_RAW_METHOD
        else max(top_k, top_k * candidate_multiplier)
    )
    started = time.perf_counter()
    response = client.query_points(
        collection_name=chunk_collection,
        query=question_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]
        ),
        limit=candidate_limit,
        with_payload=True,
        with_vectors=True,
    )
    retrieval_latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if variant == MIXED_RAW_METHOD:
        selected = [(hit, rank) for rank, hit in enumerate(response.points[:top_k], 1)]
        suppressed = []
        candidates_examined = len(selected)
    else:
        selected, suppressed, candidates_examined = deduplicate_ranked_hits(
            response.points, top_k, overlap_threshold
        )

    strategy_fields = {
        "mixed_variant": variant,
        "candidate_limit": candidate_limit,
        "candidates_retrieved": len(response.points),
        "candidates_examined": candidates_examined,
        "deduplication_enabled": variant == MIXED_DEDUPLICATED_METHOD,
        "deduplication_overlap_threshold": (
            overlap_threshold if variant == MIXED_DEDUPLICATED_METHOD else None
        ),
        "deduplication_overlap_definition": (
            OVERLAP_DEFINITION if variant == MIXED_DEDUPLICATED_METHOD else None
        ),
        "suppressed_chunk_count": len(suppressed),
        "suppressed_chunks": suppressed,
    }
    return _build_record(
        ranked_hits=selected,
        method_name=variant,
        question_point_id=question_point_id,
        document_id=document_id,
        split=split,
        top_k=top_k,
        chunk_sizes=chunk_sizes,
        retrieval_latency_ms=retrieval_latency_ms,
        evidence_summary=evidence_summary,
        evaluation_run_id=evaluation_run_id,
        evaluation_config_hash=evaluation_config_hash,
        evaluation_timestamp=evaluation_timestamp,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        tokenizer_name=tokenizer_name,
        store_retrieved_text=store_retrieved_text,
        strategy_fields=strategy_fields,
    )
