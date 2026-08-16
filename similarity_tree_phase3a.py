#!/usr/bin/env python
"""Phase 3A multiscale question-to-chunk similarity-tree router.

This experiment is deliberately read-only with respect to Qdrant.  It derives
features only from the question embedding and chunks belonging to the source
paper, trains on the preserved evidence-length Oracle labels, and evaluates a
locked hierarchy-aware linear classifier on the preserved validation split.
Ground-truth evidence, answers, evidence embeddings, retrieval F1, and Oracle
lengths are never feature inputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import (
    CHUNK_SIZES,
    EMBEDDING_DIM,
    OPENAI_EMBEDDING_MODEL,
    PAPER_CHUNK_COLLECTION,
    PAPER_QUESTION_COLLECTION,
    QDRANT_API_KEY,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    QDRANT_URL,
    TOKENIZER_NAME,
)
from granularity_router import classification_metrics


PHASE = "Phase 3A"
FORMULATION_VERSION = "similarity-tree-router-evidence-length-oracle-v1"
CLASS_TOKENS = (10, 20, 40, 80, 160)
ORACLE_VERSION = "oracle-evidence-length-gpt2-smaller-midpoint-v1"
SOURCE_ORACLE_CONFIG_HASH = (
    "9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8"
)
DEFAULT_ORACLE_ROOT = Path(
    "outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/similarity_tree_phase3a_evidence_length_oracle"
)
DEFAULT_REPORT_ROOT = Path(
    "reports/similarity_tree_phase3a_evidence_length_oracle"
)
SEED = 42
FOLDS = 5
SOFTMAX_TEMPERATURE = 0.05
FEATURE_SCHEMA_VERSION = "phase3a-similarity-tree-features-v1"
PRIMARY_MODEL = "tree_logistic_regression"
TOP_K = 5
MODEL_SELECTION_METRIC = "macro_f1"
LEAKAGE_TERMS = (
    "evidence",
    "oracle",
    "answer",
    "retrieval_f1",
    "joined_f1",
    "chunk_f1",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Malformed JSONL at {path}:{line_number}") from error
    return rows


def qdrant_client(*, prefer_grpc: bool = True) -> QdrantClient:
    """Connect to the existing service; no method in this module mutates it."""
    api_key = QDRANT_API_KEY or None
    if QDRANT_URL:
        return QdrantClient(
            url=QDRANT_URL,
            api_key=api_key,
            prefer_grpc=False,
            timeout=300,
            check_compatibility=False,
        )
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_HTTP_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        api_key=api_key,
        prefer_grpc=prefer_grpc,
        timeout=300,
        check_compatibility=False,
    )


def dense_vector(vector: Any, identity: str) -> np.ndarray:
    if isinstance(vector, list):
        values = vector
    elif isinstance(vector, dict) and len(vector) == 1:
        values = next(iter(vector.values()))
    else:
        raise RuntimeError(f"{identity} has no usable dense vector")
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (EMBEDDING_DIM,) or not np.isfinite(result).all():
        raise RuntimeError(f"{identity} has an invalid vector shape or values")
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm == 0.0:
        raise RuntimeError(f"{identity} has a zero or non-finite vector norm")
    return result / norm


def collection_snapshot(client: QdrantClient) -> dict[str, dict]:
    snapshot: dict[str, dict] = {}
    for item in client.get_collections().collections:
        info = client.get_collection(item.name)
        params = info.config.params.vectors
        snapshot[item.name] = {
            "status": str(info.status.value if hasattr(info.status, "value") else info.status),
            "points_count": int(info.points_count or 0),
            "indexed_vectors_count": int(info.indexed_vectors_count or 0),
            "vector_size": int(params.size) if hasattr(params, "size") else None,
            "distance": str(params.distance.value) if hasattr(params, "distance") else None,
        }
    return snapshot


def load_oracle(path: Path, expected_split: str) -> list[dict]:
    rows = read_jsonl(path)
    expected_count = {"train": 2245, "validation": 924}[expected_split]
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {expected_split} Oracle rows, found {len(rows)}"
        )
    seen: set[str] = set()
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in seen:
            raise RuntimeError(f"Missing or duplicate Oracle question ID: {question_id!r}")
        seen.add(question_id)
        if row.get("split") != expected_split:
            raise RuntimeError(f"Oracle split mismatch for {question_id}")
        if int(row.get("oracle_label", -1)) not in CLASS_TOKENS:
            raise RuntimeError(f"Invalid Oracle label for {question_id}")
        if row.get("label_version") != ORACLE_VERSION:
            raise RuntimeError(f"Oracle version mismatch for {question_id}")
        if row.get("source_router_evaluation_config_hash") != SOURCE_ORACLE_CONFIG_HASH:
            raise RuntimeError(f"Oracle configuration mismatch for {question_id}")
    return rows


def validate_split_isolation(train: Sequence[dict], validation: Sequence[dict]) -> None:
    train_questions = {str(row["question_id"]) for row in train}
    validation_questions = {str(row["question_id"]) for row in validation}
    train_documents = {str(row["document_id"]) for row in train}
    validation_documents = {str(row["document_id"]) for row in validation}
    if train_questions & validation_questions:
        raise RuntimeError("Question leakage exists between train and validation")
    if train_documents & validation_documents:
        raise RuntimeError("Paper leakage exists between train and validation")


def retrieve_questions(client: QdrantClient, rows: Sequence[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    ids = [str(row["question_id"]) for row in rows]
    for start in range(0, len(ids), 128):
        requested = ids[start : start + 128]
        points = client.retrieve(
            collection_name=PAPER_QUESTION_COLLECTION,
            ids=requested,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            point_id = str(point.id)
            payload = point.payload or {}
            result[point_id] = {
                "vector": dense_vector(point.vector, f"question {point_id}"),
                "document_id": str(payload.get("document_id") or ""),
                "split": str(payload.get("split") or ""),
                "question_text": str(payload.get("question_text") or ""),
            }
    missing = sorted(set(ids) - set(result))
    if missing:
        raise RuntimeError(f"PaperQuestion is missing {len(missing)} Oracle IDs")
    for row in rows:
        question_id = str(row["question_id"])
        stored = result[question_id]
        if stored["document_id"] != str(row["document_id"]):
            raise RuntimeError(f"Document mismatch for question {question_id}")
        if stored["split"] != str(row["split"]):
            raise RuntimeError(f"Split mismatch for question {question_id}")
        if stored["question_text"] != str(row["question_text"]):
            raise RuntimeError(f"Question-text mismatch for question {question_id}")
    return result


def score_document_level(
    client: QdrantClient,
    *,
    document_id: str,
    level: int,
    question_matrix: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Return all cosine scores ordered by chunk index for one paper/level."""
    score_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    expected_total: int | None = None
    observed_spans: list[tuple[int, int, int]] = []
    offset = None
    while True:
        for attempt in range(1, 4):
            try:
                points, next_offset = client.scroll(
                    collection_name=PAPER_CHUNK_COLLECTION,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id", match=MatchValue(value=document_id)
                            ),
                            FieldCondition(
                                key="granularity_level", match=MatchValue(value=level)
                            ),
                        ]
                    ),
                    limit=256,
                    offset=offset,
                    with_payload=[
                        "document_id",
                        "granularity_level",
                        "chunk_idx",
                        "total_chunks",
                        "chunk_size",
                        "span_start",
                        "span_end",
                    ],
                    with_vectors=True,
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)
        if points:
            vectors = np.stack(
                [dense_vector(point.vector, f"chunk {point.id}") for point in points]
            )
            scores = question_matrix @ vectors.T
            indices = []
            for point in points:
                payload = point.payload or {}
                if int(payload.get("granularity_level", -1)) != level:
                    raise RuntimeError(f"Qdrant filter violation in document {document_id}")
                index = int(payload.get("chunk_idx", -1))
                total = int(payload.get("total_chunks", -1))
                if expected_total is None:
                    expected_total = total
                elif expected_total != total:
                    raise RuntimeError(f"Inconsistent total_chunks in document {document_id}")
                indices.append(index)
                observed_spans.append(
                    (index, int(payload.get("span_start", -1)), int(payload.get("span_end", -1)))
                )
            score_parts.append(scores.astype(np.float32, copy=False))
            index_parts.append(np.asarray(indices, dtype=np.int64))
        if next_offset is None:
            break
        offset = next_offset
    if not score_parts or expected_total is None:
        raise RuntimeError(f"No chunks for document={document_id}, level={level}")
    indices = np.concatenate(index_parts)
    scores = np.concatenate(score_parts, axis=1)
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    scores = scores[:, order]
    if indices.tolist() != list(range(expected_total)):
        raise RuntimeError(f"Non-contiguous chunk indices for document={document_id}, level={level}")
    spans = sorted(observed_spans)
    if any(start < 0 or end <= start for _, start, end in spans):
        raise RuntimeError(f"Invalid chunk spans for document={document_id}, level={level}")
    return scores, {
        "count": expected_total,
        "first_span_start": spans[0][1],
        "last_span_end": spans[-1][2],
    }


def _quantile(values: np.ndarray, value: float) -> float:
    return float(np.quantile(values, value))


def level_statistics(scores: Sequence[float], token_size: int) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid score distribution for level {token_size}")
    descending = np.sort(values)[::-1]
    maximum = float(descending[0])
    shifted = (values - maximum) / SOFTMAX_TEMPERATURE
    weights = np.exp(shifted)
    probabilities = weights / weights.sum()
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum())
    normalized_entropy = entropy / math.log(len(values)) if len(values) > 1 else 0.0
    effective_fraction = math.exp(entropy) / len(values)
    prefix = f"level_{token_size}_"
    return {
        prefix + "log_count": math.log1p(len(values)),
        prefix + "max": maximum,
        prefix + "mean": float(values.mean()),
        prefix + "std": float(values.std()),
        prefix + "q50": _quantile(values, 0.50),
        prefix + "q75": _quantile(values, 0.75),
        prefix + "q90": _quantile(values, 0.90),
        prefix + "q95": _quantile(values, 0.95),
        prefix + "top2_mean": float(descending[: min(2, len(values))].mean()),
        prefix + "top5_mean": float(descending[: min(5, len(values))].mean()),
        prefix + "top10_mean": float(descending[: min(10, len(values))].mean()),
        prefix + "margin_top1_top2": (
            maximum - float(descending[1]) if len(descending) > 1 else 0.0
        ),
        prefix + "max_mean_gap": maximum - float(values.mean()),
        prefix + "near_max_002_fraction": float(np.mean(values >= maximum - 0.02)),
        prefix + "near_max_005_fraction": float(np.mean(values >= maximum - 0.05)),
        prefix + "softmax_entropy_norm_t005": normalized_entropy,
        prefix + "softmax_effective_fraction_t005": effective_fraction,
    }


def _describe(values: Sequence[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            prefix + "mean": 0.0,
            prefix + "std": 0.0,
            prefix + "q10": 0.0,
            prefix + "q50": 0.0,
            prefix + "q90": 0.0,
            prefix + "max": 0.0,
        }
    return {
        prefix + "mean": float(array.mean()),
        prefix + "std": float(array.std()),
        prefix + "q10": _quantile(array, 0.10),
        prefix + "q50": _quantile(array, 0.50),
        prefix + "q90": _quantile(array, 0.90),
        prefix + "max": float(array.max()),
    }


def tree_edge_statistics(
    child_scores: Sequence[float],
    parent_scores: Sequence[float],
    child_tokens: int,
    parent_tokens: int,
) -> dict[str, float]:
    child = np.asarray(child_scores, dtype=np.float64)
    parent = np.asarray(parent_scores, dtype=np.float64)
    if len(parent) != math.ceil(len(child) / 2):
        raise ValueError(
            f"Broken hierarchy {parent_tokens}->{child_tokens}: {len(parent)} vs {len(child)}"
        )
    max_deltas: list[float] = []
    mean_deltas: list[float] = []
    sibling_gaps: list[float] = []
    child_maxima: list[float] = []
    near_equal_002 = 0
    near_equal_005 = 0
    two_child_nodes = 0
    for parent_index, parent_score in enumerate(parent):
        children = child[parent_index * 2 : parent_index * 2 + 2]
        child_max = float(children.max())
        child_maxima.append(child_max)
        max_deltas.append(child_max - float(parent_score))
        mean_deltas.append(float(children.mean()) - float(parent_score))
        if len(children) == 2:
            gap = abs(float(children[0]) - float(children[1]))
            sibling_gaps.append(gap)
            near_equal_002 += int(gap <= 0.02)
            near_equal_005 += int(gap <= 0.05)
            two_child_nodes += 1
    prefix = f"edge_{parent_tokens}_to_{child_tokens}_"
    features = {}
    features.update(_describe(max_deltas, prefix + "child_max_minus_parent_"))
    features.update(_describe(mean_deltas, prefix + "child_mean_minus_parent_"))
    features.update(_describe(sibling_gaps, prefix + "sibling_abs_gap_"))
    features[prefix + "near_equal_002_fraction"] = (
        near_equal_002 / two_child_nodes if two_child_nodes else 0.0
    )
    features[prefix + "near_equal_005_fraction"] = (
        near_equal_005 / two_child_nodes if two_child_nodes else 0.0
    )
    if len(parent) > 1 and np.std(parent) > 0 and np.std(child_maxima) > 0:
        correlation = float(np.corrcoef(parent, child_maxima)[0, 1])
    else:
        correlation = 0.0
    features[prefix + "parent_child_max_correlation"] = correlation
    best_child_ancestor = int(np.argmax(child)) // 2
    features[prefix + "argmax_alignment"] = float(best_child_ancestor == int(np.argmax(parent)))
    return features


def extract_features(scores_by_tokens: dict[int, Sequence[float]]) -> tuple[dict, dict]:
    """Create deployable level and hierarchy features from cosine scores only."""
    if set(scores_by_tokens) != set(CLASS_TOKENS):
        raise ValueError("All five score levels are required")
    level_features: dict[str, float] = {}
    for tokens in CLASS_TOKENS:
        level_features.update(level_statistics(scores_by_tokens[tokens], tokens))
    tree_features = dict(level_features)
    for child_tokens, parent_tokens in zip(CLASS_TOKENS[:-1], CLASS_TOKENS[1:]):
        tree_features.update(
            tree_edge_statistics(
                scores_by_tokens[child_tokens],
                scores_by_tokens[parent_tokens],
                child_tokens,
                parent_tokens,
            )
        )
    assert_no_leakage_feature_names(level_features)
    assert_no_leakage_feature_names(tree_features)
    return level_features, tree_features


def assert_no_leakage_feature_names(features: dict[str, Any] | Sequence[str]) -> None:
    names = features.keys() if isinstance(features, dict) else features
    bad = [name for name in names if any(term in name.lower() for term in LEAKAGE_TERMS)]
    if bad:
        raise RuntimeError(f"Forbidden leakage-prone feature names: {bad}")


def hierarchy_counts_are_valid(counts: dict[int, int]) -> bool:
    if set(counts) != set(CLASS_TOKENS):
        return False
    return all(
        counts[parent] == math.ceil(counts[child] / 2)
        for child, parent in zip(CLASS_TOKENS[:-1], CLASS_TOKENS[1:])
    )


def feature_row(
    oracle: dict,
    scores_by_tokens: dict[int, Sequence[float]],
    hierarchy_metadata: dict[int, dict],
) -> dict:
    counts = {tokens: len(scores_by_tokens[tokens]) for tokens in CLASS_TOKENS}
    if not hierarchy_counts_are_valid(counts):
        raise RuntimeError(f"Broken hierarchy for question {oracle['question_id']}: {counts}")
    level_features, tree_features = extract_features(scores_by_tokens)
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "question_id": str(oracle["question_id"]),
        "document_id": str(oracle["document_id"]),
        "split": str(oracle["split"]),
        "question_text": str(oracle["question_text"]),
        "oracle_label": int(oracle["oracle_label"]),
        "oracle_label_version": str(oracle["label_version"]),
        "chunk_counts": {str(key): value for key, value in counts.items()},
        "hierarchy_metadata": {str(key): value for key, value in hierarchy_metadata.items()},
        "level_features": level_features,
        "tree_features": tree_features,
        "scores_by_tokens": {
            str(tokens): [round(float(score), 8) for score in scores_by_tokens[tokens]]
            for tokens in CLASS_TOKENS
        },
    }


def _load_existing_recovery(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    by_id: dict[str, dict] = {}
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in by_id:
            raise RuntimeError(f"Invalid recovery JSONL identity: {question_id!r}")
        by_id[question_id] = row
    return by_id


def extract_split(
    client: QdrantClient,
    *,
    oracle_rows: Sequence[dict],
    question_points: dict[str, dict],
    split: str,
    output_root: Path,
) -> dict:
    recovery_path = output_root / "recovery" / f"{split}_features.jsonl"
    existing = _load_existing_recovery(recovery_path)
    expected_ids = {str(row["question_id"]) for row in oracle_rows}
    if not set(existing).issubset(expected_ids):
        raise RuntimeError(f"Recovery file contains unknown {split} question IDs")
    oracle_by_document: dict[str, list[dict]] = defaultdict(list)
    for row in oracle_rows:
        if str(row["question_id"]) not in existing:
            oracle_by_document[str(row["document_id"])].append(row)
    started = time.perf_counter()
    document_count = len(oracle_by_document)
    for document_number, document_id in enumerate(sorted(oracle_by_document), start=1):
        rows = sorted(oracle_by_document[document_id], key=lambda row: str(row["question_id"]))
        matrix = np.stack([question_points[str(row["question_id"])]["vector"] for row in rows])
        score_matrices: dict[int, np.ndarray] = {}
        hierarchy_metadata: dict[int, dict] = {}
        for level, tokens in enumerate(CLASS_TOKENS, start=1):
            scores, metadata = score_document_level(
                client,
                document_id=document_id,
                level=level,
                question_matrix=matrix,
            )
            score_matrices[tokens] = scores
            hierarchy_metadata[tokens] = metadata
        counts = {tokens: score_matrices[tokens].shape[1] for tokens in CLASS_TOKENS}
        if not hierarchy_counts_are_valid(counts):
            raise RuntimeError(f"Chunk hierarchy is invalid for document {document_id}: {counts}")
        for row_index, oracle in enumerate(rows):
            scores_by_tokens = {
                tokens: score_matrices[tokens][row_index] for tokens in CLASS_TOKENS
            }
            completed = feature_row(oracle, scores_by_tokens, hierarchy_metadata)
            append_jsonl(recovery_path, completed)
            existing[str(oracle["question_id"])] = completed
        if document_number % 10 == 0 or document_number == document_count:
            print(
                json.dumps(
                    {
                        "event": "phase3a_extraction_progress",
                        "split": split,
                        "documents_completed_this_invocation": document_number,
                        "documents_remaining_this_invocation": document_count - document_number,
                        "questions_complete": len(existing),
                        "questions_expected": len(oracle_rows),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    if set(existing) != expected_ids:
        raise RuntimeError(f"Incomplete {split} extraction: {len(existing)}/{len(expected_ids)}")
    ordered = [existing[str(row["question_id"])] for row in oracle_rows]
    full_path = output_root / "features" / f"{split}_similarity_trees.jsonl.gz"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = full_path.with_suffix(full_path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
        for row in ordered:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(full_path)
    slim_rows = [
        {key: value for key, value in row.items() if key != "scores_by_tokens"}
        for row in ordered
    ]
    slim_path = output_root / "features" / f"{split}_features.jsonl"
    atomic_jsonl(slim_path, slim_rows)
    return {
        "split": split,
        "questions": len(ordered),
        "documents": len({str(row["document_id"]) for row in oracle_rows}),
        "raw_similarity_tree_path": str(full_path),
        "feature_path": str(slim_path),
        "raw_similarity_tree_sha256": sha256_file(full_path),
        "feature_sha256": sha256_file(slim_path),
        "invocation_wall_seconds": time.perf_counter() - started,
        "resumed_records": len(ordered) - sum(len(value) for value in oracle_by_document.values()),
    }


def load_similarity_trees(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Malformed compressed JSONL at line {line_number}") from error
    return rows


def feature_matrix(rows: Sequence[dict], feature_key: str) -> tuple[np.ndarray, list[str]]:
    if not rows:
        raise ValueError("Cannot make a feature matrix from zero rows")
    names = sorted(rows[0][feature_key])
    assert_no_leakage_feature_names(names)
    for row in rows:
        if sorted(row[feature_key]) != names:
            raise RuntimeError("Inconsistent feature schema")
    values = np.asarray([[row[feature_key][name] for name in names] for row in rows], dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Feature matrix contains non-finite values")
    return values, names


def target_array(rows: Sequence[dict]) -> np.ndarray:
    mapping = {tokens: index for index, tokens in enumerate(CLASS_TOKENS)}
    return np.asarray([mapping[int(row["oracle_label"])] for row in rows], dtype=np.int64)


def grouped_stratified_folds(rows: Sequence[dict], fold_count: int, seed: int) -> np.ndarray:
    if fold_count < 2:
        raise ValueError("At least two folds are required")
    label_to_index = {tokens: index for index, tokens in enumerate(CLASS_TOKENS)}
    groups: dict[str, np.ndarray] = {}
    for row in rows:
        document_id = str(row["document_id"])
        if document_id not in groups:
            groups[document_id] = np.zeros(len(CLASS_TOKENS), dtype=np.int64)
        groups[document_id][label_to_index[int(row["oracle_label"])]] += 1
    if len(groups) < fold_count:
        raise ValueError("There are fewer papers than folds")
    rng = random.Random(seed)
    tie = {document: rng.random() for document in groups}
    ordered = sorted(
        groups,
        key=lambda document: (
            -int(groups[document].max()),
            -int(groups[document].sum()),
            tie[document],
            document,
        ),
    )
    total = sum(groups.values(), np.zeros(len(CLASS_TOKENS), dtype=np.int64))
    target = total / fold_count
    target_size = sum(total) / fold_count
    fold_counts = np.zeros((fold_count, len(CLASS_TOKENS)), dtype=np.int64)
    fold_sizes = np.zeros(fold_count, dtype=np.int64)
    assignment: dict[str, int] = {}
    for document in ordered:
        vector = groups[document]
        candidates = []
        for fold in range(fold_count):
            trial_counts = fold_counts.copy()
            trial_sizes = fold_sizes.copy()
            trial_counts[fold] += vector
            trial_sizes[fold] += int(vector.sum())
            class_cost = float((((trial_counts - target) / (target + 1.0)) ** 2).sum())
            size_cost = float((((trial_sizes - target_size) / (target_size + 1.0)) ** 2).sum())
            candidates.append((class_cost + 0.2 * size_cost, int(fold_sizes[fold]), fold))
        chosen = min(candidates)[2]
        assignment[document] = chosen
        fold_counts[chosen] += vector
        fold_sizes[chosen] += int(vector.sum())
    folds = np.asarray([assignment[str(row["document_id"])] for row in rows], dtype=np.int64)
    for document in groups:
        observed = {int(folds[index]) for index, row in enumerate(rows) if str(row["document_id"]) == document}
        if len(observed) != 1:
            raise RuntimeError(f"Paper {document} crosses grouped folds")
    if set(folds.tolist()) != set(range(fold_count)):
        raise RuntimeError("Grouped fold allocator produced an empty fold")
    return folds


def fit_standardizer(features: np.ndarray) -> dict[str, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return {"mean": mean, "scale": scale}


def transform(features: np.ndarray, standardizer: dict[str, np.ndarray]) -> np.ndarray:
    return ((features - standardizer["mean"]) / standardizer["scale"]).astype(np.float32)


class LinearClassifier(torch.nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.classifier = torch.nn.Linear(feature_count, len(CLASS_TOKENS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_linear_classifier(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> tuple[LinearClassifier, list[float]]:
    set_seed(seed)
    model = LinearClassifier(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    x = torch.from_numpy(features)
    y = torch.from_numpy(targets)
    losses = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model, losses


def predict_linear(model: LinearClassifier, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(features)), dim=1).numpy()
    return probabilities.argmax(axis=1), probabilities


def quadratic_weighted_kappa(targets: np.ndarray, predictions: np.ndarray) -> float:
    count = len(CLASS_TOKENS)
    observed = np.zeros((count, count), dtype=np.float64)
    for target, prediction in zip(targets, predictions):
        observed[int(target), int(prediction)] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(1, observed.sum())
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / ((count - 1) ** 2), (count, count))
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else 0.0


def extended_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> dict:
    if probabilities is None:
        probabilities = np.eye(len(CLASS_TOKENS), dtype=np.float32)[predictions]
        top2_available = False
    else:
        top2_available = True
    result = classification_metrics(targets, predictions, probabilities, CLASS_TOKENS)
    if not top2_available:
        result["top_2_accuracy"] = None
        result["top_2_accuracy_status"] = "unavailable_without_comparable_five_class_scores"
    else:
        result["top_2_accuracy_status"] = "available"
    distances = np.abs(targets - predictions)
    token_values = np.asarray(CLASS_TOKENS)
    result.update(
        {
            "mean_absolute_class_distance": float(distances.mean()),
            "within_one_level_accuracy": float(np.mean(distances <= 1)),
            "mean_absolute_token_distance": float(
                np.abs(token_values[targets] - token_values[predictions]).mean()
            ),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(targets, predictions),
            "correct_count": int(np.sum(targets == predictions)),
            "example_count": int(len(targets)),
            "predicted_distribution": {
                str(tokens): int(np.sum(predictions == index))
                for index, tokens in enumerate(CLASS_TOKENS)
            },
        }
    )
    return result


def candidate_key(metrics: dict, candidate: dict) -> tuple:
    return (
        float(metrics[MODEL_SELECTION_METRIC]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        -float(candidate["weight_decay"]),
        -float(candidate["learning_rate"]),
    )


def cross_validate_linear(
    features: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    learning_rates: Sequence[float],
    weight_decays: Sequence[float],
    epochs: int,
    seed: int,
) -> dict:
    candidates = []
    fold_count = int(folds.max()) + 1
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            probabilities = np.zeros((len(targets), len(CLASS_TOKENS)), dtype=np.float32)
            fold_rows = []
            for fold in range(fold_count):
                train_mask = folds != fold
                test_mask = folds == fold
                standardizer = fit_standardizer(features[train_mask])
                model, losses = fit_linear_classifier(
                    transform(features[train_mask], standardizer),
                    targets[train_mask],
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    epochs=epochs,
                    seed=seed + fold,
                )
                _, fold_probabilities = predict_linear(model, transform(features[test_mask], standardizer))
                probabilities[test_mask] = fold_probabilities
                fold_predictions = fold_probabilities.argmax(axis=1)
                fold_rows.append(
                    {
                        "fold": fold,
                        "train_examples": int(train_mask.sum()),
                        "held_out_examples": int(test_mask.sum()),
                        "final_train_loss": losses[-1],
                        "metrics": extended_metrics(targets[test_mask], fold_predictions, fold_probabilities),
                    }
                )
            predictions = probabilities.argmax(axis=1)
            metrics = extended_metrics(targets, predictions, probabilities)
            candidate = {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "oof_metrics": metrics,
                "fold_metrics": fold_rows,
            }
            candidates.append(candidate)
    selected = max(candidates, key=lambda item: candidate_key(item["oof_metrics"], item))
    return {"selected": selected, "candidates": candidates}


def fit_full_model(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str],
    selection: dict,
    *,
    seed: int,
) -> tuple[dict, LinearClassifier]:
    standardizer = fit_standardizer(features)
    model, losses = fit_linear_classifier(
        transform(features, standardizer),
        targets,
        learning_rate=float(selection["learning_rate"]),
        weight_decay=float(selection["weight_decay"]),
        epochs=int(selection["epochs"]),
        seed=seed,
    )
    artifact = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "class_tokens": list(CLASS_TOKENS),
        "feature_names": list(feature_names),
        "standardizer_mean": standardizer["mean"],
        "standardizer_scale": standardizer["scale"],
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "selected_hyperparameters": selection,
        "random_seed": seed,
        "final_training_loss": losses[-1],
        "created_at": utc_now(),
    }
    return artifact, model


def predict_artifact(artifact: dict, rows: Sequence[dict], feature_key: str) -> tuple[np.ndarray, np.ndarray]:
    features, names = feature_matrix(rows, feature_key)
    if names != artifact["feature_names"]:
        raise RuntimeError("Feature-name mismatch during model reload")
    standardizer = {
        "mean": np.asarray(artifact["standardizer_mean"], dtype=np.float32),
        "scale": np.asarray(artifact["standardizer_scale"], dtype=np.float32),
    }
    model = LinearClassifier(len(names))
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    return predict_linear(model, transform(features, standardizer))


def _level_stat(row: dict, tokens: int, name: str) -> float:
    return float(row["level_features"][f"level_{tokens}_{name}"])


def fixed_heuristic(rows: Sequence[dict], statistic: str) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(
        [[_level_stat(row, tokens, statistic) for tokens in CLASS_TOKENS] for row in rows],
        dtype=np.float64,
    )
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return scores.argmax(axis=1), probabilities.astype(np.float32)


def penalized_top5_predictions(rows: Sequence[dict], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(
        [
            [
                _level_stat(row, tokens, "top5_mean")
                - alpha * _level_stat(row, tokens, "log_count")
                for tokens in CLASS_TOKENS
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return scores.argmax(axis=1), probabilities.astype(np.float32)


def leaf_breadth_predictions(rows: Sequence[dict], delta: float) -> tuple[np.ndarray, None]:
    predictions = []
    for row in rows:
        scores = np.asarray(row["scores_by_tokens"]["10"], dtype=np.float64)
        selected = np.flatnonzero(scores >= float(scores.max()) - delta)
        span_tokens = max(10, int(selected[-1] - selected[0] + 1) * 10)
        class_index = min(
            range(len(CLASS_TOKENS)),
            key=lambda index: (abs(CLASS_TOKENS[index] - span_tokens), CLASS_TOKENS[index]),
        )
        predictions.append(class_index)
    return np.asarray(predictions, dtype=np.int64), None


def tune_parameterized_heuristic(
    rows: Sequence[dict],
    targets: np.ndarray,
    folds: np.ndarray,
    parameters: Sequence[float],
    predictor: Callable[[Sequence[dict], float], tuple[np.ndarray, np.ndarray | None]],
) -> dict:
    oof_predictions = np.zeros(len(rows), dtype=np.int64)
    fold_selections = []
    for fold in range(int(folds.max()) + 1):
        train_indices = np.flatnonzero(folds != fold)
        held_indices = np.flatnonzero(folds == fold)
        train_rows = [rows[index] for index in train_indices]
        held_rows = [rows[index] for index in held_indices]
        choices = []
        for parameter in parameters:
            predictions, probabilities = predictor(train_rows, parameter)
            choices.append(
                {
                    "parameter": parameter,
                    "metrics": extended_metrics(targets[train_indices], predictions, probabilities),
                }
            )
        selected = max(
            choices,
            key=lambda item: (
                item["metrics"][MODEL_SELECTION_METRIC],
                item["metrics"]["balanced_accuracy"],
                item["metrics"]["accuracy"],
                -item["parameter"],
            ),
        )
        held_predictions, _ = predictor(held_rows, float(selected["parameter"]))
        oof_predictions[held_indices] = held_predictions
        fold_selections.append({"fold": fold, "selected_parameter": selected["parameter"]})
    all_choices = []
    for parameter in parameters:
        predictions, probabilities = predictor(rows, parameter)
        all_choices.append(
            {
                "parameter": parameter,
                "metrics": extended_metrics(targets, predictions, probabilities),
            }
        )
    final = max(
        all_choices,
        key=lambda item: (
            item["metrics"][MODEL_SELECTION_METRIC],
            item["metrics"]["balanced_accuracy"],
            item["metrics"]["accuracy"],
            -item["parameter"],
        ),
    )
    return {
        "grouped_oof_metrics": extended_metrics(targets, oof_predictions, None),
        "fold_selections": fold_selections,
        "final_parameter_selected_on_all_train": final["parameter"],
        "all_train_candidates": all_choices,
    }


def majority_reference(targets: np.ndarray, class_index: int) -> dict:
    predictions = np.full(len(targets), class_index, dtype=np.int64)
    probabilities = np.zeros((len(targets), len(CLASS_TOKENS)), dtype=np.float32)
    probabilities[:, class_index] = 1.0
    return extended_metrics(targets, predictions, probabilities)


def write_confusion_csv(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["oracle\\predicted", *CLASS_TOKENS])
        for tokens, values in zip(CLASS_TOKENS, metrics["confusion_matrix"]):
            writer.writerow([tokens, *values])


def write_histogram_svg(path: Path, oracle: dict, predicted: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 800, 480
    left, top, plot_width, plot_height = 70, 45, 680, 340
    maximum = max(max(oracle.values()), max(predicted.values()), 1)
    bars = []
    labels = []
    for index, tokens in enumerate(CLASS_TOKENS):
        center = left + (index + 0.5) * plot_width / len(CLASS_TOKENS)
        for offset, values, color in [(-18, oracle, "#4C78A8"), (18, predicted, "#F58518")]:
            value = values[str(tokens)]
            bar_height = plot_height * value / maximum
            bars.append(
                f'<rect x="{center + offset - 14:.1f}" y="{top + plot_height - bar_height:.1f}" '
                f'width="28" height="{bar_height:.1f}" fill="{color}"/>'
            )
            labels.append(
                f'<text x="{center + offset:.1f}" y="{top + plot_height - bar_height - 5:.1f}" '
                f'text-anchor="middle" font-size="11">{value}</text>'
            )
        labels.append(
            f'<text x="{center:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-size="13">{tokens}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="400" y="24" text-anchor="middle" font-size="18">Phase 3A Oracle vs predicted distribution</text>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>
{''.join(bars)}{''.join(labels)}
<rect x="260" y="440" width="14" height="14" fill="#4C78A8"/><text x="280" y="452" font-size="13">Oracle</text>
<rect x="390" y="440" width="14" height="14" fill="#F58518"/><text x="410" y="452" font-size="13">Predicted</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def audit_command(args: argparse.Namespace) -> int:
    if tuple(CHUNK_SIZES) != CLASS_TOKENS:
        raise RuntimeError(f"Configured chunk sizes changed: {CHUNK_SIZES}")
    train_path = args.oracle_root / "train_oracle.jsonl"
    validation_path = args.oracle_root / "validation_oracle.jsonl"
    train = load_oracle(train_path, "train")
    validation = load_oracle(validation_path, "validation")
    validate_split_isolation(train, validation)
    client = qdrant_client(prefer_grpc=False)
    try:
        snapshot = collection_snapshot(client)
        required = {PAPER_CHUNK_COLLECTION, PAPER_QUESTION_COLLECTION}
        if not required.issubset(snapshot):
            raise RuntimeError(f"Missing required Qdrant collections: {sorted(required-set(snapshot))}")
        for name in required:
            if snapshot[name]["status"] != "green":
                raise RuntimeError(f"Qdrant collection {name} is not green")
            if snapshot[name]["vector_size"] != EMBEDDING_DIM:
                raise RuntimeError(f"Qdrant collection {name} has wrong vector size")
            if snapshot[name]["distance"] != "Cosine":
                raise RuntimeError(f"Qdrant collection {name} is not cosine")
        sample = train[0]
        questions = retrieve_questions(client, [sample])
        matrix = questions[str(sample["question_id"])]["vector"][None, :]
        sample_counts = {}
        sample_scores = {}
        sample_metadata = {}
        for level, tokens in enumerate(CLASS_TOKENS, start=1):
            scores, metadata = score_document_level(
                client,
                document_id=str(sample["document_id"]),
                level=level,
                question_matrix=matrix,
            )
            sample_counts[tokens] = int(scores.shape[1])
            sample_scores[tokens] = scores[0]
            sample_metadata[tokens] = metadata
        if not hierarchy_counts_are_valid(sample_counts):
            raise RuntimeError(f"Sample chunk hierarchy failed: {sample_counts}")
        qdrant_top = client.query_points(
            collection_name=PAPER_CHUNK_COLLECTION,
            query=matrix[0].tolist(),
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=str(sample["document_id"]))
                    ),
                    FieldCondition(key="granularity_level", match=MatchValue(value=1)),
                ]
            ),
            limit=10,
            with_payload=False,
            with_vectors=False,
        )
        manual_top = sorted((float(value) for value in sample_scores[10]), reverse=True)[:10]
        qdrant_top_scores = [float(point.score) for point in qdrant_top.points]
        maximum_score_difference = max(
            abs(left - right) for left, right in zip(manual_top, qdrant_top_scores)
        )
        if maximum_score_difference > 1e-5:
            raise RuntimeError(
                "Manual cosine scores do not reproduce Qdrant ranking: "
                f"max difference={maximum_score_difference}"
            )
        level_features, tree_features = extract_features(sample_scores)
    finally:
        client.close()
    result = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "audited_at": utc_now(),
        "read_only_qdrant": True,
        "forbidden_feature_inputs": list(LEAKAGE_TERMS),
        "qdrant_endpoint": QDRANT_URL or f"{QDRANT_HOST}:{QDRANT_HTTP_PORT} (gRPC {QDRANT_GRPC_PORT})",
        "collection_snapshot": snapshot,
        "oracle": {
            "train_path": str(train_path),
            "train_sha256": sha256_file(train_path),
            "train_examples": len(train),
            "train_documents": len({row["document_id"] for row in train}),
            "validation_path": str(validation_path),
            "validation_sha256": sha256_file(validation_path),
            "validation_examples": len(validation),
            "validation_documents": len({row["document_id"] for row in validation}),
            "label_version": ORACLE_VERSION,
            "source_config_hash": SOURCE_ORACLE_CONFIG_HASH,
        },
        "sample_hierarchy": {
            "question_id": sample["question_id"],
            "document_id": sample["document_id"],
            "counts": {str(key): value for key, value in sample_counts.items()},
            "metadata": {str(key): value for key, value in sample_metadata.items()},
            "valid": True,
        },
        "manual_cosine_vs_qdrant_top10": {
            "granularity_tokens": 10,
            "maximum_absolute_score_difference": maximum_score_difference,
            "tolerance": 1e-5,
            "matches": True,
        },
        "feature_schema": {
            "version": FEATURE_SCHEMA_VERSION,
            "level_feature_count": len(level_features),
            "tree_feature_count": len(tree_features),
            "level_feature_names": sorted(level_features),
            "tree_feature_names": sorted(tree_features),
        },
        "environment": {
            "python": subprocess.check_output([os.sys.executable, "--version"], text=True).strip(),
            "python_executable": os.sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    atomic_json(args.output_root / "integrity" / "preflight_audit.json", result)
    atomic_json(
        args.output_root / "configuration" / "experiment.json",
        {
            "phase": PHASE,
            "formulation_version": FORMULATION_VERSION,
            "created_at": utc_now(),
            "classes": list(CLASS_TOKENS),
            "feature_source": "full same-paper question-to-chunk cosine score trees",
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
            "chunk_collection": PAPER_CHUNK_COLLECTION,
            "question_collection": PAPER_QUESTION_COLLECTION,
            "qdrant_access": "read_only",
            "oracle_version": ORACLE_VERSION,
            "oracle_config_hash": SOURCE_ORACLE_CONFIG_HASH,
            "primary_model": PRIMARY_MODEL,
            "training_split": "preserved train only",
            "hyperparameter_selection": f"{FOLDS}-fold paper-grouped cross-validation on train only",
            "final_evaluation_split": "preserved validation",
            "selection_metric": MODEL_SELECTION_METRIC,
            "linear_learning_rates": [0.03, 0.01, 0.003],
            "linear_weight_decays": [0.0, 0.001, 0.01],
            "linear_epochs": 300,
            "random_seed": SEED,
            "softmax_temperature_feature": SOFTMAX_TEMPERATURE,
            "retrieval": {"top_k": TOP_K, "paper_restricted": True},
            "forbidden_features": [
                "ground-truth evidence",
                "evidence length",
                "answers",
                "evidence embeddings",
                "evidence-to-chunk similarity",
                "retrieval F1",
                "Oracle label as a feature",
            ],
        },
    )
    print(json.dumps(result, indent=2))
    return 0


def extract_command(args: argparse.Namespace) -> int:
    train = load_oracle(args.oracle_root / "train_oracle.jsonl", "train")
    validation = load_oracle(args.oracle_root / "validation_oracle.jsonl", "validation")
    validate_split_isolation(train, validation)
    preflight_path = args.output_root / "integrity" / "preflight_audit.json"
    if not preflight_path.exists():
        raise RuntimeError("Run the Phase 3A audit before extraction")
    # REST is intentionally preferred for this long-running, full-vector
    # scroll.  The local gRPC path can hit a server deadline after many
    # thousands of pages; REST with the 300-second client timeout is slower
    # per request but reliably resumable.
    client = qdrant_client(prefer_grpc=False)
    try:
        before = collection_snapshot(client)
        questions = retrieve_questions(client, [*train, *validation])
        summaries = [
            extract_split(
                client,
                oracle_rows=rows,
                question_points=questions,
                split=split,
                output_root=args.output_root,
            )
            for split, rows in [("train", train), ("validation", validation)]
        ]
        after = collection_snapshot(client)
    finally:
        client.close()
    if before != after:
        raise RuntimeError("Qdrant collection snapshot changed during read-only extraction")
    summary = {
        "phase": PHASE,
        "completed_at": utc_now(),
        "splits": summaries,
        "qdrant_snapshot_before": before,
        "qdrant_snapshot_after": after,
        "collections_unchanged": True,
    }
    atomic_json(args.output_root / "features" / "extraction_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _json_safe_artifact(artifact: dict) -> dict:
    return {
        key: (
            value.tolist()
            if isinstance(value, np.ndarray)
            else {name: tensor.tolist() for name, tensor in value.items()}
            if key == "model_state_dict"
            else value
        )
        for key, value in artifact.items()
    }


def train_evaluate_command(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    train_path = args.output_root / "features" / "train_similarity_trees.jsonl.gz"
    validation_path = args.output_root / "features" / "validation_similarity_trees.jsonl.gz"
    train_rows = load_similarity_trees(train_path)
    validation_rows = load_similarity_trees(validation_path)
    if len(train_rows) != 2245 or len(validation_rows) != 924:
        raise RuntimeError("Phase 3A extracted feature counts are incomplete")
    validate_split_isolation(train_rows, validation_rows)
    train_targets = target_array(train_rows)
    validation_targets = target_array(validation_rows)
    folds = grouped_stratified_folds(train_rows, FOLDS, SEED)
    fold_manifest = [
        {
            "fold": fold,
            "examples": int(np.sum(folds == fold)),
            "documents": len(
                {str(row["document_id"]) for index, row in enumerate(train_rows) if folds[index] == fold}
            ),
            "class_distribution": {
                str(tokens): int(np.sum(train_targets[folds == fold] == class_index))
                for class_index, tokens in enumerate(CLASS_TOKENS)
            },
        }
        for fold in range(FOLDS)
    ]
    atomic_json(args.output_root / "cross_validation" / "paper_grouped_folds.json", fold_manifest)

    fixed_heuristics = {}
    for name, statistic in [("max_similarity", "max"), ("top5_mean_similarity", "top5_mean")]:
        train_predictions, train_probabilities = fixed_heuristic(train_rows, statistic)
        validation_predictions, validation_probabilities = fixed_heuristic(validation_rows, statistic)
        fixed_heuristics[name] = {
            "train_metrics": extended_metrics(train_targets, train_predictions, train_probabilities),
            "validation_metrics": extended_metrics(
                validation_targets, validation_predictions, validation_probabilities
            ),
        }
    penalized = tune_parameterized_heuristic(
        train_rows,
        train_targets,
        folds,
        [0.0, 0.0025, 0.005, 0.01, 0.02, 0.04],
        penalized_top5_predictions,
    )
    penalized_predictions, penalized_probabilities = penalized_top5_predictions(
        validation_rows, float(penalized["final_parameter_selected_on_all_train"])
    )
    penalized["validation_metrics"] = extended_metrics(
        validation_targets, penalized_predictions, penalized_probabilities
    )
    breadth = tune_parameterized_heuristic(
        train_rows,
        train_targets,
        folds,
        [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15],
        leaf_breadth_predictions,
    )
    breadth_predictions, _ = leaf_breadth_predictions(
        validation_rows, float(breadth["final_parameter_selected_on_all_train"])
    )
    breadth["validation_metrics"] = extended_metrics(validation_targets, breadth_predictions, None)
    heuristic_summary = {
        "fixed": fixed_heuristics,
        "penalized_top5": penalized,
        "leaf_relevance_breadth": breadth,
    }
    atomic_json(args.output_root / "heuristics" / "metrics.json", heuristic_summary)

    model_results = {}
    model_artifacts = {}
    for model_name, feature_key in [
        ("level_aggregate_logistic_regression", "level_features"),
        (PRIMARY_MODEL, "tree_features"),
    ]:
        train_features, feature_names = feature_matrix(train_rows, feature_key)
        cv = cross_validate_linear(
            train_features,
            train_targets,
            folds,
            learning_rates=[0.03, 0.01, 0.003],
            weight_decays=[0.0, 0.001, 0.01],
            epochs=300,
            seed=SEED,
        )
        artifact, _ = fit_full_model(
            train_features,
            train_targets,
            feature_names,
            cv["selected"],
            seed=SEED,
        )
        predictions, probabilities = predict_artifact(artifact, validation_rows, feature_key)
        metrics = extended_metrics(validation_targets, predictions, probabilities)
        model_path = args.output_root / "models" / f"{model_name}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, model_path)
        json_path = args.output_root / "models" / f"{model_name}.json"
        atomic_json(json_path, _json_safe_artifact(artifact))
        atomic_json(args.output_root / "cross_validation" / f"{model_name}.json", cv)
        model_results[model_name] = {
            "feature_key": feature_key,
            "feature_count": len(feature_names),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "selected_hyperparameters": cv["selected"],
            "validation_metrics": metrics,
        }
        model_artifacts[model_name] = artifact

    primary_artifact = model_artifacts[PRIMARY_MODEL]
    primary_predictions, primary_probabilities = predict_artifact(
        primary_artifact, validation_rows, "tree_features"
    )
    primary_metrics = extended_metrics(validation_targets, primary_predictions, primary_probabilities)
    train_majority_index = int(np.bincount(train_targets, minlength=5).argmax())
    validation_majority_index = int(np.bincount(validation_targets, minlength=5).argmax())
    references = {
        "train_prior_majority": {
            "class": CLASS_TOKENS[train_majority_index],
            "selection_status": "deployable_train_only_reference",
            "validation_metrics": majority_reference(validation_targets, train_majority_index),
        },
        "validation_oracle_majority": {
            "class": CLASS_TOKENS[validation_majority_index],
            "selection_status": "descriptive_non_deployable_validation_label_reference",
            "validation_metrics": majority_reference(validation_targets, validation_majority_index),
        },
    }
    predictions_path = args.output_root / "validation" / "predictions.jsonl"
    prediction_rows = []
    for row, target, prediction, probability in zip(
        validation_rows, validation_targets, primary_predictions, primary_probabilities
    ):
        ranking = np.argsort(-probability, kind="stable")
        prediction_rows.append(
            {
                "phase": PHASE,
                "formulation_version": FORMULATION_VERSION,
                "question_id": row["question_id"],
                "document_id": row["document_id"],
                "split": row["split"],
                "question_text": row["question_text"],
                "oracle_label": CLASS_TOKENS[int(target)],
                "predicted_label": CLASS_TOKENS[int(prediction)],
                "predicted_class_id": int(prediction),
                "prediction_status": "valid_five_logit_argmax",
                "class_probabilities_by_label": {
                    str(tokens): float(probability[index])
                    for index, tokens in enumerate(CLASS_TOKENS)
                },
                "ranked_predictions": [CLASS_TOKENS[int(index)] for index in ranking],
                "top_2_predictions": [CLASS_TOKENS[int(index)] for index in ranking[:2]],
                "model": PRIMARY_MODEL,
            }
        )
    atomic_jsonl(predictions_path, prediction_rows)
    classification_dir = args.output_root / "classification"
    atomic_json(classification_dir / "metrics.json", primary_metrics)
    write_confusion_csv(classification_dir / "confusion_matrix.csv", primary_metrics)
    write_histogram_svg(
        classification_dir / "predicted_vs_oracle.svg",
        primary_metrics["class_distribution"],
        primary_metrics["predicted_distribution"],
    )
    summary = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "completed_at": utc_now(),
        "primary_model": PRIMARY_MODEL,
        "model_selection_metric": MODEL_SELECTION_METRIC,
        "dataset": {
            "train_examples": len(train_rows),
            "train_documents": len({row["document_id"] for row in train_rows}),
            "validation_examples": len(validation_rows),
            "validation_documents": len({row["document_id"] for row in validation_rows}),
            "train_distribution": {
                str(tokens): int(np.sum(train_targets == index))
                for index, tokens in enumerate(CLASS_TOKENS)
            },
            "validation_distribution": {
                str(tokens): int(np.sum(validation_targets == index))
                for index, tokens in enumerate(CLASS_TOKENS)
            },
        },
        "feature_input": "full same-paper question-to-chunk cosine score hierarchy only",
        "oracle_input_to_features": False,
        "evidence_input_to_features": False,
        "answers_input_to_features": False,
        "retrieval_f1_input_to_features": False,
        "references": references,
        "heuristics": heuristic_summary,
        "models": model_results,
        "primary_validation_metrics": primary_metrics,
        "primary_prediction_path": str(predictions_path),
        "training_and_validation_wall_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_root / "classification_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def retrieve_command(args: argparse.Namespace) -> int:
    import qwen_phase1 as phase1

    predictions_path = args.output_root / "validation" / "predictions.jsonl"
    predictions = read_jsonl(predictions_path)
    if len(predictions) != 924:
        raise RuntimeError("Expected 924 Phase 3A validation predictions")
    result_path = args.output_root / "retrieval" / "results.jsonl"
    existing = {str(row["question_id"]): row for row in read_jsonl(result_path)}
    if not set(existing).issubset({str(row["question_id"]) for row in predictions}):
        raise RuntimeError("Retrieval recovery contains an unknown question")
    before_client = qdrant_client(prefer_grpc=False)
    try:
        before = collection_snapshot(before_client)
    finally:
        before_client.close()
    client = phase1.qdrant_client()
    started = time.perf_counter()
    try:
        for index, prediction in enumerate(predictions, start=1):
            question_id = str(prediction["question_id"])
            if question_id in existing:
                continue
            points = client.retrieve(
                collection_name=phase1.PAPER_QUESTION_COLLECTION,
                ids=[question_id],
                with_payload=True,
                with_vectors=True,
            )
            if len(points) != 1:
                raise RuntimeError(f"Question lookup failed during retrieval: {question_id}")
            predicted_tokens = int(prediction["predicted_label"])
            level = CLASS_TOKENS.index(predicted_tokens) + 1
            records = list(
                phase1.evaluate_question(
                    client=client,
                    question_point_id=question_id,
                    question_vector=points[0].vector,
                    document_id=str(prediction["document_id"]),
                    question_text=str(prediction["question_text"]),
                    split="validation",
                    top_k=TOP_K,
                    granularity_levels=[level],
                    store_retrieved_text=False,
                    chunk_sizes=list(CLASS_TOKENS),
                    embedding_model=phase1.OPENAI_EMBEDDING_MODEL,
                    embedding_dimension=phase1.EMBEDDING_DIM,
                    tokenizer_name=phase1.TOKENIZER_NAME,
                    evaluation_run_id="phase3a-similarity-tree-primary",
                    evaluation_config_hash=SOURCE_ORACLE_CONFIG_HASH,
                )
            )
            if len(records) != 1:
                raise RuntimeError(f"Expected one retrieval record for {question_id}")
            record = dict(records[0])
            record.update(
                {
                    "method_name": "phase3a-similarity-tree-routed-granularity",
                    "phase": PHASE,
                    "formulation_version": FORMULATION_VERSION,
                    "router_model": PRIMARY_MODEL,
                    "predicted_granularity_tokens": predicted_tokens,
                    "predicted_granularity_level": level,
                    "oracle_evidence_length_label": int(prediction["oracle_label"]),
                    "paper_restricted": True,
                    "top_k": TOP_K,
                }
            )
            append_jsonl(result_path, record)
            existing[question_id] = record
            if index % 50 == 0 or index == len(predictions):
                print(
                    json.dumps(
                        {
                            "event": "phase3a_retrieval_progress",
                            "complete": len(existing),
                            "expected": len(predictions),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        client.close()
    ordered = [existing[str(row["question_id"])] for row in predictions]
    atomic_jsonl(result_path, ordered)
    after_client = qdrant_client(prefer_grpc=False)
    try:
        after = collection_snapshot(after_client)
    finally:
        after_client.close()
    if before != after:
        raise RuntimeError("Qdrant collections changed during Phase 3A retrieval")
    values = [float(row["f1_joined_topk"]) for row in ordered]
    summary = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "router_model": PRIMARY_MODEL,
        "evaluated_examples": len(ordered),
        "valid_prediction_retrievals": len(ordered),
        "retrieval_coverage": len(ordered) / len(predictions),
        "mean_joined_retrieval_f1": statistics.fmean(values),
        "median_joined_retrieval_f1": statistics.median(values),
        "coverage_adjusted_full_set_mean_joined_retrieval_f1": sum(values) / len(predictions),
        "top_k": TOP_K,
        "paper_restricted": True,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "similarity": "cosine",
        "retrieval_wall_seconds_this_invocation": time.perf_counter() - started,
        "qdrant_collections_unchanged": before == after,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
    }
    atomic_json(args.output_root / "retrieval" / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def finalize_command(args: argparse.Namespace) -> int:
    classification = json.loads((args.output_root / "classification_summary.json").read_text(encoding="utf-8"))
    retrieval = json.loads((args.output_root / "retrieval" / "summary.json").read_text(encoding="utf-8"))
    extraction = json.loads((args.output_root / "features" / "extraction_summary.json").read_text(encoding="utf-8"))
    preflight = json.loads((args.output_root / "integrity" / "preflight_audit.json").read_text(encoding="utf-8"))
    client = qdrant_client(prefer_grpc=False)
    try:
        final_snapshot = collection_snapshot(client)
    finally:
        client.close()
    if final_snapshot != preflight["collection_snapshot"]:
        raise RuntimeError("Qdrant snapshot differs from Phase 3A preflight")
    artifacts = []
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and "recovery" not in path.parts:
            artifacts.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    final = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "completed_at": utc_now(),
        "status": "complete",
        "primary_model": PRIMARY_MODEL,
        "classification": classification["primary_validation_metrics"],
        "references": classification["references"],
        "model_comparison": classification["models"],
        "heuristics": classification["heuristics"],
        "retrieval": retrieval,
        "dataset": classification["dataset"],
        "features": {
            "source": classification["feature_input"],
            "extraction": extraction["splits"],
            "evidence_used": False,
            "answers_used": False,
            "retrieval_f1_used": False,
        },
        "integrity": {
            "qdrant_read_only": True,
            "qdrant_snapshot_unchanged": True,
            "oracle_train_sha256": preflight["oracle"]["train_sha256"],
            "oracle_validation_sha256": preflight["oracle"]["validation_sha256"],
        },
        "artifacts": artifacts,
    }
    atomic_json(args.output_root / "final_summary.json", final)
    post_snapshot = {
        "verified_at": utc_now(),
        "qdrant_snapshot": final_snapshot,
        "matches_preflight": True,
        "artifact_count_before_final_summary": len(artifacts),
    }
    atomic_json(args.output_root / "integrity" / "final_audit.json", post_snapshot)
    print(json.dumps(final, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit").set_defaults(func=audit_command)
    subparsers.add_parser("extract").set_defaults(func=extract_command)
    subparsers.add_parser("train-evaluate").set_defaults(func=train_evaluate_command)
    subparsers.add_parser("retrieve").set_defaults(func=retrieve_command)
    subparsers.add_parser("finalize").set_defaults(func=finalize_command)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
