#!/usr/bin/env python
"""Validate evaluation and router JSONL artifacts, optionally against Qdrant."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from config import (
    CHUNK_SIZES,
    EMBEDDING_DIM,
    EVALUATION_OUTPUT_DIR,
    RETRIEVAL_EVALUATION_COLLECTION,
    ROUTER_DATASET_COLLECTION,
)
from evaluation_utils import METHOD_NAME, make_evaluation_id, make_router_id
from qdrant_schema import get_qdrant_client


ALLOWED_SPLITS = {"train", "validation", "test"}


def _latest(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No files match {directory / pattern}")
    return matches[-1]


def _read_jsonl(path: Path, errors: list) -> list:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
    return records


def _check_finite(value, path: str, errors: list) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path}: non-finite value {value}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _check_finite(child, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite(child, f"{path}[{index}]", errors)


def validate_evaluation_records(records: list, expected_levels: list, errors: list) -> dict:
    ids = set()
    groups = defaultdict(dict)
    for index, record in enumerate(records):
        prefix = f"evaluation[{index}]"
        eval_id = record.get("eval_id")
        if eval_id in ids:
            errors.append(f"{prefix}: duplicate eval_id {eval_id}")
        ids.add(eval_id)
        try:
            granularity_value = (
                record.get("granularity_level")
                if record.get("method_name") == METHOD_NAME
                else record.get("granularity_scope", "all")
            )
            expected_id = make_evaluation_id(
                record.get("method_name", METHOD_NAME),
                record.get("question_id", ""),
                granularity_value,
                record.get("evaluation_config_hash", ""),
            )
            if eval_id != expected_id:
                errors.append(f"{prefix}: deterministic eval_id mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix}: cannot validate deterministic eval_id: {exc}")

        returned = record.get("returned_k")
        aligned_fields = (
            "topk_chunk_ids",
            "topk_chunk_indices",
            "topk_chunk_ranks",
            "topk_chunk_spans",
            "topk_chunk_token_counts",
            "topk_scores",
            "retrieved_chunks",
        )
        for field in aligned_fields:
            value = record.get(field)
            if not isinstance(value, list) or not isinstance(returned, int) or len(value) != returned:
                errors.append(f"{prefix}.{field}: not aligned with returned_k={returned}")
        ranks = record.get("topk_chunk_ranks", [])
        if ranks != list(range(1, (returned or 0) + 1)):
            errors.append(f"{prefix}: ranks are not consecutive retrieval order")
        unique_evidence_count = record.get("unique_evidence_count")
        for chunk_index, chunk in enumerate(record.get("retrieved_chunks", [])):
            if chunk.get("rank") != chunk_index + 1:
                errors.append(f"{prefix}.retrieved_chunks[{chunk_index}]: rank mismatch")
            for field in ("evidence_cosine_similarities", "evidence_token_f1_scores"):
                if len(chunk.get(field, [])) != unique_evidence_count:
                    errors.append(
                        f"{prefix}.retrieved_chunks[{chunk_index}].{field}: evidence alignment mismatch"
                    )
        if record.get("method_name") != METHOD_NAME:
            composition = record.get("granularity_composition", [])
            composition_levels = [item.get("granularity_level") for item in composition]
            if composition_levels != expected_levels:
                errors.append(f"{prefix}: invalid mixed granularity composition levels")
            if sum(item.get("count", 0) for item in composition) != returned:
                errors.append(f"{prefix}: mixed composition count does not equal returned_k")
            for field in ("topk_granularity_levels", "topk_granularity_tokens"):
                if len(record.get(field, [])) != returned:
                    errors.append(f"{prefix}.{field}: not aligned with returned_k")
        split = record.get("split")
        if split not in ALLOWED_SPLITS:
            errors.append(f"{prefix}: invalid split {split!r}")
        _check_finite(record, prefix, errors)
        key = (record.get("question_id"), record.get("evaluation_config_hash"))
        level = record.get("granularity_level", record.get("granularity_scope"))
        if level in groups[key]:
            errors.append(f"{prefix}: duplicate question/config/granularity")
        groups[key][level] = record
    return {"ids": ids, "groups": groups}


def validate_router_records(
    records: list,
    evaluation_groups: dict,
    expected_levels: list,
    errors: list,
) -> dict:
    ids = set()
    for index, record in enumerate(records):
        prefix = f"router[{index}]"
        record_id = record.get("router_record_id")
        if record_id in ids:
            errors.append(f"{prefix}: duplicate router_record_id {record_id}")
        ids.add(record_id)
        expected_id = make_router_id(
            record.get("question_id", ""), record.get("evaluation_config_hash", "")
        )
        if record_id != expected_id:
            errors.append(f"{prefix}: deterministic router_record_id mismatch")
        metrics = record.get("per_granularity_metrics", [])
        levels = [metric.get("granularity_level") for metric in metrics]
        if sorted(levels) != sorted(expected_levels):
            errors.append(f"{prefix}: configured granularities missing or duplicated: {levels}")
        available = set(levels)
        for field in (
            "best_granularity_by_f1",
            "best_granularity_by_evidence_similarity",
            "router_target_granularity",
        ):
            if record.get(field) not in available:
                errors.append(f"{prefix}.{field}: references unavailable granularity")
        split = record.get("split")
        if split not in ALLOWED_SPLITS:
            errors.append(f"{prefix}: invalid split {split!r}")
        key = (record.get("question_id"), record.get("evaluation_config_hash"))
        evaluation_records = evaluation_groups.get(key)
        if evaluation_records is None:
            errors.append(f"{prefix}: no matching evaluation records")
        else:
            evaluation_splits = {item.get("split") for item in evaluation_records.values()}
            if evaluation_splits != {split}:
                errors.append(f"{prefix}: split differs from evaluation records")
        _check_finite(record, prefix, errors)
    return {"ids": ids}


def validate_qdrant_router_vectors(
    router_collection: str,
    evaluation_collection: str,
    errors: list,
    check_router: bool = True,
) -> dict:
    client = get_qdrant_client()
    scanned = 0
    try:
        existing = {item.name for item in client.get_collections().collections}
        collections = [evaluation_collection]
        if check_router:
            collections.append(router_collection)
        for collection in collections:
            if collection not in existing:
                errors.append(f"Qdrant collection missing: {collection}")
        if not check_router or router_collection not in existing:
            return {"router_vectors_scanned": 0}
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=router_collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                scanned += 1
                vector = point.vector
                if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM:
                    errors.append(
                        f"router point {point.id}: vector dimension is not {EMBEDDING_DIM}"
                    )
                elif any(not math.isfinite(value) for value in vector):
                    errors.append(f"router point {point.id}: non-finite vector value")
                label = (point.payload or {}).get("router_target_granularity")
                levels = {
                    row.get("granularity_level")
                    for row in (point.payload or {}).get("per_granularity_metrics", [])
                }
                if label not in levels:
                    errors.append(f"router point {point.id}: unavailable router label")
            if next_offset is None:
                break
            offset = next_offset
    finally:
        client.close()
    return {"router_vectors_scanned": scanned}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate evaluation/router artifacts")
    parser.add_argument("--evaluation-jsonl", type=Path, default=None)
    parser.add_argument("--router-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(EVALUATION_OUTPUT_DIR))
    parser.add_argument("--check-qdrant", action="store_true")
    parser.add_argument("--evaluation-collection", default=RETRIEVAL_EVALUATION_COLLECTION)
    parser.add_argument("--router-collection", default=ROUTER_DATASET_COLLECTION)
    args = parser.parse_args()

    errors = []
    try:
        evaluation_path = args.evaluation_jsonl or _latest(
            args.output_dir, "RetrievalEval*.jsonl"
        )
    except FileNotFoundError as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)]}, indent=2))
        return 1

    evaluations = _read_jsonl(evaluation_path, errors)
    fixed_evaluations = any(
        record.get("method_name") == METHOD_NAME for record in evaluations
    )
    router_path = args.router_jsonl
    if router_path is None and fixed_evaluations:
        try:
            router_path = _latest(args.output_dir, "RouterDataset_*.jsonl")
        except FileNotFoundError as exc:
            errors.append(str(exc))
    routers = _read_jsonl(router_path, errors) if router_path else []
    expected_levels = list(range(1, len(CHUNK_SIZES) + 1))
    evaluation_report = validate_evaluation_records(evaluations, expected_levels, errors)
    validate_router_records(
        routers, evaluation_report["groups"], expected_levels, errors
    )
    qdrant_report = {}
    if args.check_qdrant:
        qdrant_report = validate_qdrant_router_vectors(
            args.router_collection,
            args.evaluation_collection,
            errors,
            check_router=fixed_evaluations or bool(routers),
        )
    report = {
        "valid": not errors,
        "evaluation_jsonl": str(evaluation_path),
        "router_jsonl": str(router_path) if router_path else None,
        "evaluation_records": len(evaluations),
        "router_records": len(routers),
        "configured_granularity_levels": expected_levels,
        **qdrant_report,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
