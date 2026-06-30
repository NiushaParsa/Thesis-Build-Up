"""Router-selected single-granularity retrieval with schema-v2 metrics."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
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
from evaluation_utils import (
    ROUTER_LABEL_VERSION,
    build_evaluation_config,
    evaluation_config_hash as compute_evaluation_config_hash,
    make_evaluation_id,
    make_router_id,
    new_evaluation_run_id,
)
from fixed_sized_granularity_separate import (
    _fetch_evidence,
    _query_points_with_retries,
    _validate_dimension,
)
from granularity_router import predict_with_artifact
from mixed_granularity import _build_record


ROUTER_SELECTED_CLI_METHOD = "router-selected"
ROUTER_SELECTED_METHOD = "router-selected granularity"
ROUTER_SELECTED_FILTER_BEHAVIOR = "same-document-and-router-predicted-granularity-level"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RouterPredictor:
    """Thin inference wrapper around the persisted granularity-router artifact."""

    def __init__(
        self,
        *,
        artifact: dict,
        model_path: str,
        model_hash: str,
        model_choice: str = "primary",
    ) -> None:
        self.artifact = artifact
        self.model_path = model_path
        self.model_hash = model_hash
        self.model_choice = model_choice
        self.class_tokens = [int(tokens) for tokens in artifact["class_tokens"]]
        self.embedding_dimension = int(artifact["embedding_dimension"])
        self.model_version = artifact.get("artifact_version")
        self.selected_model_type = (
            artifact.get("selected_model_type")
            if model_choice == "primary"
            else model_choice
        )
        self.oracle_evaluation_config_hash = artifact.get(
            "oracle_evaluation_config_hash"
        )

    @classmethod
    def from_path(cls, path: Path, model_choice: str = "primary") -> "RouterPredictor":
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            artifact=artifact,
            model_path=str(path),
            model_hash=file_sha256(path),
            model_choice=model_choice,
        )

    def predict(self, question_vector: List[float]) -> dict:
        features = np.asarray([question_vector], dtype=np.float32)
        predictions, probabilities = predict_with_artifact(
            self.artifact, features, self.model_choice
        )
        prediction_index = int(predictions[0])
        probability = probabilities[0]
        predicted_tokens = int(self.class_tokens[prediction_index])
        return {
            "predicted_granularity_tokens": predicted_tokens,
            "prediction_confidence": float(probability[prediction_index]),
            "class_probabilities": {
                str(tokens): float(probability[index])
                for index, tokens in enumerate(self.class_tokens)
            },
            "prediction_index": prediction_index,
        }


def build_router_selected_config(
    *,
    top_k: int,
    chunk_sizes: List[int],
    router_predictor: RouterPredictor,
    oracle_evaluation_config_hash: Optional[str],
    store_text: bool,
    chunk_collection: str = PAPER_CHUNK_COLLECTION,
    question_collection: str = PAPER_QUESTION_COLLECTION,
    evidence_collection: str = PAPER_EVIDENCE_COLLECTION,
    evaluation_collection: str = RETRIEVAL_EVALUATION_COLLECTION,
    router_collection: str = ROUTER_DATASET_COLLECTION,
    embedding_model: str = OPENAI_EMBEDDING_MODEL,
    embedding_dimension: int = EMBEDDING_DIM,
    tokenizer_name: str = TOKENIZER_NAME,
    tie_epsilon: float = ROUTER_LABEL_TIE_EPSILON,
) -> dict:
    """Return the canonical routed-retrieval result-affecting configuration."""
    return build_evaluation_config(
        method=ROUTER_SELECTED_METHOD,
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
        router_label_tie_epsilon=tie_epsilon,
        store_text=store_text,
        filter_behavior=ROUTER_SELECTED_FILTER_BEHAVIOR,
        strategy_settings={
            "router_model_path": router_predictor.model_path,
            "router_model_hash": router_predictor.model_hash,
            "router_model_version": router_predictor.model_version,
            "router_model_choice": router_predictor.model_choice,
            "router_selected_model_type": router_predictor.selected_model_type,
            "oracle_evaluation_config_hash": oracle_evaluation_config_hash,
            "oracle_label_version": ROUTER_LABEL_VERSION,
        },
    )


def _fetch_oracle_payload(
    client: QdrantClient,
    *,
    question_id: str,
    router_collection: str,
    oracle_evaluation_config_hash: Optional[str],
) -> Tuple[Optional[dict], str]:
    if not oracle_evaluation_config_hash:
        return None, "not_requested"

    point_id = make_router_id(question_id, oracle_evaluation_config_hash)
    try:
        points = client.retrieve(
            collection_name=router_collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if points:
            return points[0].payload or {}, "found_by_id"
    except Exception:
        # Older tests/fakes may not implement retrieve; fall back to a filtered scroll.
        pass

    results, _ = client.scroll(
        collection_name=router_collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="question_id", match=MatchValue(value=question_id)),
                FieldCondition(
                    key="evaluation_config_hash",
                    match=MatchValue(value=oracle_evaluation_config_hash),
                ),
            ]
        ),
        limit=2,
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        return None, "missing"
    if len(results) > 1:
        return None, "duplicate"
    return results[0].payload or {}, "found_by_filter"


def _oracle_fields(
    *,
    oracle_payload: Optional[dict],
    oracle_lookup_status: str,
    predicted_level: int,
    selected_f1: float,
    selected_mean_evidence_similarity: float,
) -> dict:
    fields = {
        "oracle_lookup_status": oracle_lookup_status,
        "oracle_target_granularity": None,
        "oracle_best_granularity_by_f1": None,
        "oracle_best_granularity_by_evidence_similarity": None,
        "router_oracle_match": None,
        "oracle_best_f1": None,
        "oracle_best_evidence_similarity": None,
        "regret_f1": None,
        "regret_evidence_similarity": None,
    }
    if not oracle_payload:
        return fields

    per_granularity = oracle_payload.get("per_granularity_metrics") or []
    f1_values = [
        metric.get("f1_joined_topk")
        for metric in per_granularity
        if isinstance(metric.get("f1_joined_topk"), (int, float))
    ]
    similarity_values = [
        metric.get("mean_max_evidence_similarity_topk")
        for metric in per_granularity
        if isinstance(metric.get("mean_max_evidence_similarity_topk"), (int, float))
    ]
    oracle_best_f1 = max(f1_values) if f1_values else None
    oracle_best_similarity = max(similarity_values) if similarity_values else None
    oracle_target = oracle_payload.get("router_target_granularity")
    fields.update(
        {
            "oracle_target_granularity": oracle_target,
            "oracle_best_granularity_by_f1": oracle_payload.get(
                "best_granularity_by_f1"
            ),
            "oracle_best_granularity_by_evidence_similarity": oracle_payload.get(
                "best_granularity_by_evidence_similarity"
            ),
            "router_oracle_match": (
                bool(int(oracle_target) == predicted_level)
                if oracle_target is not None
                else None
            ),
            "oracle_best_f1": (
                round(float(oracle_best_f1), 6)
                if oracle_best_f1 is not None
                else None
            ),
            "oracle_best_evidence_similarity": (
                round(float(oracle_best_similarity), 6)
                if oracle_best_similarity is not None
                else None
            ),
            "regret_f1": (
                round(float(oracle_best_f1) - selected_f1, 6)
                if oracle_best_f1 is not None
                else None
            ),
            "regret_evidence_similarity": (
                round(
                    float(oracle_best_similarity) - selected_mean_evidence_similarity,
                    6,
                )
                if oracle_best_similarity is not None
                else None
            ),
        }
    )
    return fields


def evaluate_router_selected_question(
    client: QdrantClient,
    question_point_id: str,
    question_vector: List[float],
    document_id: str,
    question_text: str,
    split: str,
    *,
    router_predictor: RouterPredictor,
    oracle_evaluation_config_hash: Optional[str],
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
    evaluation_run_id: Optional[str] = None,
    evaluation_config_hash: Optional[str] = None,
    evaluation_timestamp: Optional[str] = None,
    evidence_embedding_fn=None,
) -> Tuple[Optional[dict], Optional[str]]:
    """Evaluate one question after selecting exactly one granularity via router."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    chunk_sizes = list(chunk_sizes or CHUNK_SIZES)
    _validate_dimension(question_vector, embedding_dimension, question_point_id)
    if router_predictor.embedding_dimension != embedding_dimension:
        raise ValueError(
            f"Router embedding dimension {router_predictor.embedding_dimension} "
            f"does not match evaluation dimension {embedding_dimension}"
        )

    total_started = time.perf_counter()
    router_started = time.perf_counter()
    prediction = router_predictor.predict(question_vector)
    router_latency_ms = round((time.perf_counter() - router_started) * 1000, 2)
    predicted_tokens = prediction["predicted_granularity_tokens"]
    if predicted_tokens not in chunk_sizes:
        raise ValueError(
            f"Router predicted token size {predicted_tokens}, not in {chunk_sizes}"
        )
    predicted_level = chunk_sizes.index(predicted_tokens) + 1

    evidence_summary = _fetch_evidence(
        client,
        question_point_id,
        evidence_collection=evidence_collection,
        embedding_dimension=embedding_dimension,
        embedding_fn=evidence_embedding_fn,
    )
    if not evidence_summary["records"]:
        return None, "no_valid_evidence"

    if evaluation_config_hash is None:
        config = build_router_selected_config(
            top_k=top_k,
            chunk_sizes=chunk_sizes,
            router_predictor=router_predictor,
            oracle_evaluation_config_hash=oracle_evaluation_config_hash,
            store_text=store_retrieved_text,
            chunk_collection=chunk_collection,
            question_collection=question_collection,
            evidence_collection=evidence_collection,
            evaluation_collection=evaluation_collection,
            router_collection=router_collection,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            tokenizer_name=tokenizer_name,
        )
        evaluation_config_hash = compute_evaluation_config_hash(config)
    evaluation_run_id = evaluation_run_id or new_evaluation_run_id()

    retrieval_started = time.perf_counter()
    response = _query_points_with_retries(
        client,
        collection_name=chunk_collection,
        query=question_vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                FieldCondition(
                    key="granularity_level", match=MatchValue(value=predicted_level)
                ),
            ]
        ),
        limit=top_k,
        with_payload=True,
        with_vectors=True,
    )
    retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 2)
    ranked_hits = [(hit, rank) for rank, hit in enumerate(response.points, start=1)]
    record = _build_record(
        ranked_hits=ranked_hits,
        method_name=ROUTER_SELECTED_METHOD,
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
        strategy_fields={
            "router_model_path": router_predictor.model_path,
            "router_model_hash": router_predictor.model_hash,
            "router_model_version": router_predictor.model_version,
            "router_model_choice": router_predictor.model_choice,
            "router_selected_model_type": router_predictor.selected_model_type,
            "predicted_granularity_level": predicted_level,
            "predicted_granularity_tokens": predicted_tokens,
            "prediction_confidence": round(
                prediction["prediction_confidence"], 6
            ),
            "class_probabilities": {
                key: round(value, 6)
                for key, value in prediction["class_probabilities"].items()
            },
            "router_latency_ms": router_latency_ms,
            "oracle_evaluation_config_hash": oracle_evaluation_config_hash,
        },
    )
    record["eval_id"] = make_evaluation_id(
        ROUTER_SELECTED_METHOD,
        question_point_id,
        predicted_level,
        evaluation_config_hash,
    )
    record["granularity_scope"] = "router-selected"
    record["granularity_level"] = predicted_level
    record["granularity_tokens"] = predicted_tokens

    oracle_payload, oracle_status = _fetch_oracle_payload(
        client,
        question_id=question_point_id,
        router_collection=router_collection,
        oracle_evaluation_config_hash=oracle_evaluation_config_hash,
    )
    record.update(
        _oracle_fields(
            oracle_payload=oracle_payload,
            oracle_lookup_status=oracle_status,
            predicted_level=predicted_level,
            selected_f1=record["f1_joined_topk"],
            selected_mean_evidence_similarity=record[
                "mean_max_evidence_similarity_topk"
            ],
        )
    )
    record["total_latency_ms"] = round((time.perf_counter() - total_started) * 1000, 2)
    return record, None
