"""Shared configuration, deterministic IDs, oracle labels, and Qdrant batching."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client.models import PointStruct


logger = logging.getLogger(__name__)

EVALUATION_SCHEMA_VERSION = 2
METRIC_VERSION = "qasper-token-prf-v2"
NORMALIZATION_VERSION = "lowercase-remove-punctuation-collapse-whitespace-v1"
ROUTER_LABEL_VERSION = "oracle-f1-evidence-smaller-v1"
METHOD_NAME = "fixed-sized granularity - separate"
FILTER_BEHAVIOR = "same-document-and-exact-granularity-level"


def build_evaluation_config(
    *,
    method: str,
    top_k: int,
    chunk_sizes: List[int],
    embedding_model: str,
    embedding_dimension: int,
    tokenizer: str,
    chunk_collection: str,
    question_collection: str,
    evidence_collection: str,
    evaluation_collection: str,
    router_collection: str,
    router_label_tie_epsilon: float,
    store_text: bool,
    filter_behavior: str = FILTER_BEHAVIOR,
    strategy_settings: Optional[dict] = None,
    schema_version: int = EVALUATION_SCHEMA_VERSION,
    metric_version: str = METRIC_VERSION,
    normalization_version: str = NORMALIZATION_VERSION,
) -> dict:
    """Return the canonical set of result-affecting evaluation settings."""
    config = {
        "method": method,
        "top_k": top_k,
        "chunk_sizes": list(chunk_sizes),
        "granularity_level_to_tokens": {
            str(level): tokens for level, tokens in enumerate(chunk_sizes, start=1)
        },
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "tokenizer": tokenizer,
        "metric_version": metric_version,
        "normalization_version": normalization_version,
        "chunk_collection": chunk_collection,
        "question_collection": question_collection,
        "evidence_collection": evidence_collection,
        "evaluation_collection": evaluation_collection,
        "router_collection": router_collection,
        "filter_behavior": filter_behavior,
        "router_label_tie_epsilon": router_label_tie_epsilon,
        "router_label_version": ROUTER_LABEL_VERSION,
        "schema_version": schema_version,
        "store_text": bool(store_text),
    }
    if strategy_settings:
        config["strategy_settings"] = strategy_settings
    return config


def evaluation_config_hash(config: dict) -> str:
    """Hash canonical JSON so equal configurations produce equal IDs."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_evaluation_id(
    method: str,
    question_id: str,
    granularity_value: Any,
    config_hash: str,
) -> str:
    seed = f"{method}|{question_id}|{granularity_value}|{config_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def make_router_id(question_id: str, config_hash: str) -> str:
    seed = f"{ROUTER_LABEL_VERSION}|{question_id}|{config_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def new_evaluation_run_id() -> str:
    return str(uuid.uuid4())


def _smallest_chunk_record(records: Iterable[dict]) -> dict:
    return min(records, key=lambda record: (record["granularity_tokens"], record["granularity_level"]))


def select_oracle_labels(records: List[dict], tie_epsilon: float) -> dict:
    """Select analytical labels and the deterministic primary router target."""
    if not records:
        raise ValueError("Cannot select an oracle label without granularity records")

    best_f1_value = max(record["f1_joined_topk"] for record in records)
    exact_best_f1 = [record for record in records if record["f1_joined_topk"] == best_f1_value]
    best_f1_record = _smallest_chunk_record(exact_best_f1)

    best_similarity_value = max(
        record["mean_max_evidence_similarity_topk"] for record in records
    )
    exact_best_similarity = [
        record
        for record in records
        if record["mean_max_evidence_similarity_topk"] == best_similarity_value
    ]
    best_similarity_record = _smallest_chunk_record(exact_best_similarity)

    f1_candidates = [
        record
        for record in records
        if best_f1_value - record["f1_joined_topk"] <= tie_epsilon
    ]
    if len(f1_candidates) == 1:
        target = f1_candidates[0]
        reason = "highest_joined_f1"
    else:
        candidate_best_similarity = max(
            record["mean_max_evidence_similarity_topk"] for record in f1_candidates
        )
        similarity_candidates = [
            record
            for record in f1_candidates
            if candidate_best_similarity - record["mean_max_evidence_similarity_topk"]
            <= tie_epsilon
        ]
        if len(similarity_candidates) == 1:
            target = similarity_candidates[0]
            reason = "f1_tie_broken_by_evidence_similarity"
        else:
            target = _smallest_chunk_record(similarity_candidates)
            reason = "f1_and_similarity_tie_broken_by_smaller_chunk"

    return {
        "best_granularity_by_f1": best_f1_record["granularity_level"],
        "best_granularity_by_evidence_similarity": best_similarity_record[
            "granularity_level"
        ],
        "router_target_granularity": target["granularity_level"],
        "label_tie_break_reason": reason,
    }


def build_router_record(
    *,
    question: dict,
    records: List[dict],
    expected_levels: List[int],
    tie_epsilon: float,
    evaluation_run_id: str,
    config_hash: str,
    embedding_model: str,
    embedding_dimension: int,
) -> Tuple[Optional[dict], Optional[str]]:
    """Build one router payload only when every expected level is valid."""
    levels = [record.get("granularity_level") for record in records]
    if len(levels) != len(set(levels)):
        return None, "duplicate_granularity_records"
    if set(levels) != set(expected_levels):
        missing = sorted(set(expected_levels) - set(levels))
        extra = sorted(set(levels) - set(expected_levels))
        return None, f"incomplete_granularities:missing={missing},extra={extra}"

    required_metrics = (
        "f1_joined_topk",
        "mean_max_evidence_similarity_topk",
        "best_query_similarity_topk",
        "mean_query_similarity_topk",
    )
    for record in records:
        if not isinstance(record.get("returned_k"), int) or record["returned_k"] < 1:
            return None, f"no_retrieved_chunks:{record.get('granularity_level')}"
        for field in required_metrics:
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                return None, f"invalid_metric:{record.get('granularity_level')}:{field}"

    ordered = sorted(records, key=lambda record: record["granularity_level"])
    labels = select_oracle_labels(ordered, tie_epsilon)
    per_granularity = [
        {
            "granularity_level": record["granularity_level"],
            "granularity_tokens": record["granularity_tokens"],
            "f1_joined_topk": record["f1_joined_topk"],
            "precision_joined_topk": record["precision_joined_topk"],
            "recall_joined_topk": record["recall_joined_topk"],
            "mean_max_evidence_similarity_topk": record[
                "mean_max_evidence_similarity_topk"
            ],
            "best_evidence_similarity_topk": record[
                "best_evidence_similarity_topk"
            ],
            "best_query_similarity_topk": record["best_query_similarity_topk"],
            "mean_query_similarity_topk": record["mean_query_similarity_topk"],
            "best_chunk_f1_topk": record["best_chunk_f1_topk"],
            "mean_chunk_f1_topk": record["mean_chunk_f1_topk"],
            "returned_k": record["returned_k"],
        }
        for record in ordered
    ]
    payload = {
        "schema_version": 1,
        "router_record_id": make_router_id(question["point_id"], config_hash),
        "question_id": question["point_id"],
        "document_id": question["document_id"],
        "split": question["split"],
        "question_text": question["question_text"],
        "per_granularity_metrics": per_granularity,
        **labels,
        "label_version": ROUTER_LABEL_VERSION,
        "evaluation_run_id": evaluation_run_id,
        "evaluation_config_hash": config_hash,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
    }
    return payload, None


class BufferedQdrantUpserter:
    """Best-effort batching that records failures without stopping JSONL output."""

    def __init__(self, client, collection_name: str, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.client = client
        self.collection_name = collection_name
        self.batch_size = batch_size
        self._points: List[PointStruct] = []
        self.upserted = 0
        self.errors: List[str] = []
        self.disabled = False

    def add(self, *, point_id: str, payload: dict, vector) -> None:
        if self.disabled:
            return
        self._points.append(PointStruct(id=point_id, payload=payload, vector=vector))
        if len(self._points) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._points or self.disabled:
            return
        points = self._points
        self._points = []
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            self.upserted += len(points)
        except Exception as exc:  # JSONL must survive persistence failures
            message = f"{self.collection_name} persistence failed: {exc}"
            logger.exception(message)
            self.errors.append(message)
            self.disabled = True
