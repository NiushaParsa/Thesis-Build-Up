#!/usr/bin/env python
"""Read-only integrity checks for the QASPER checkpoint and Qdrant data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from config import CHUNK_SIZES
from qdrant_schema import COLLECTIONS, EMBEDDING_DIM, get_qdrant_client


EXPECTED_STAGES = [*CHUNK_SIZES, "questions", "evidence"]
EVALUATION_REQUIRED_FIELDS = {
    "eval_id",
    "method_name",
    "question_id",
    "document_id",
    "split",
    "granularity_level",
    "granularity_tokens",
    "k_requested",
    "retrieved_k",
    "retrieval_time_ms",
    "evidence_hash",
    "evidence_token_count",
    "retrieved_joined_token_count",
    "topk_chunk_ids",
    "topk_chunk_indices",
    "topk_scores",
    "f1_joined_topk",
    "avg_score_topk",
    "best_score_topk",
}


def deterministic_uuid(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def _example(bucket: list, value: Any, limit: int) -> None:
    if len(bucket) < limit:
        bucket.append(value)


def load_checkpoint_report(path: Path, example_limit: int) -> Tuple[dict, dict]:
    report = {
        "path": str(path),
        "documents": 0,
        "stage_counts": {},
        "incomplete_documents": [],
        "invalid_entries": [],
    }
    if not path.exists():
        report["error"] = "checkpoint file does not exist"
        return {}, report

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report["error"] = str(exc)
        return {}, report

    if not isinstance(raw, dict):
        report["error"] = "checkpoint root must be an object"
        return {}, report

    counts = Counter()
    for document_id, entry in sorted(raw.items()):
        if not isinstance(entry, dict) or not isinstance(entry.get("done"), list):
            _example(
                report["invalid_entries"],
                {"document_id": document_id, "value": entry},
                example_limit,
            )
            continue
        done = entry["done"]
        counts.update(done)
        missing = [stage for stage in EXPECTED_STAGES if stage not in done]
        if missing:
            report["incomplete_documents"].append(
                {
                    "document_id": document_id,
                    "split": entry.get("split", "unknown"),
                    "missing_stages": missing,
                }
            )

    report["documents"] = len(raw)
    report["stage_counts"] = {str(stage): counts[stage] for stage in EXPECTED_STAGES}
    return raw, report


def scroll_points(
    client,
    collection: str,
    batch_size: int,
    with_payload,
    with_vectors: bool,
    max_points: Optional[int] = None,
) -> Iterator[Any]:
    offset = None
    yielded = 0
    while True:
        remaining = None if max_points is None else max_points - yielded
        if remaining is not None and remaining <= 0:
            break
        limit = batch_size if remaining is None else min(batch_size, remaining)
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        for point in points:
            yield point
            yielded += 1
        if next_offset is None or not points:
            break
        offset = next_offset


def _collection_vector_size(info) -> Optional[int]:
    vectors = info.config.params.vectors
    if hasattr(vectors, "size"):
        return vectors.size
    if isinstance(vectors, dict) and len(vectors) == 1:
        value = next(iter(vectors.values()))
        return getattr(value, "size", None)
    return None


def _plain_vector(vector) -> Optional[list]:
    if isinstance(vector, list):
        return vector
    if isinstance(vector, dict) and len(vector) == 1:
        value = next(iter(vector.values()))
        return value if isinstance(value, list) else None
    return None


def validate_vectors(
    client,
    collection: str,
    batch_size: int,
    max_points: Optional[int],
    example_limit: int,
) -> dict:
    invalid_examples = []
    invalid_count = 0
    scanned = 0
    for point in scroll_points(
        client,
        collection,
        batch_size,
        with_payload=False,
        with_vectors=True,
        max_points=max_points,
    ):
        scanned += 1
        vector = _plain_vector(point.vector)
        reason = None
        if vector is None:
            reason = "missing or unsupported vector representation"
        elif len(vector) != EMBEDDING_DIM:
            reason = f"dimension {len(vector)} != {EMBEDDING_DIM}"
        elif any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
            reason = "contains a non-numeric or non-finite value"
        elif not any(value != 0 for value in vector):
            reason = "zero vector"
        if reason:
            invalid_count += 1
            _example(
                invalid_examples,
                {"point_id": str(point.id), "reason": reason},
                example_limit,
            )
    return {
        "scanned": scanned,
        "scan_is_complete": max_points is None,
        "invalid_vectors": invalid_count,
        "examples": invalid_examples,
    }


def validate_chunks(
    client,
    checkpoint: dict,
    batch_size: int,
    max_points: Optional[int],
    example_limit: int,
) -> dict:
    groups: Dict[Tuple[str, int], dict] = {}
    duplicate_keys = 0
    duplicate_examples = []
    id_mismatches = 0
    id_mismatch_examples = []
    malformed = 0
    malformed_examples = []
    scanned = 0

    fields = [
        "document_id",
        "chunk_idx",
        "total_chunks",
        "chunk_size",
        "granularity_level",
        "span_start",
        "span_end",
    ]
    for point in scroll_points(
        client,
        "PaperChunk",
        batch_size,
        with_payload=fields,
        with_vectors=False,
        max_points=max_points,
    ):
        scanned += 1
        payload = point.payload or {}
        try:
            document_id = str(payload["document_id"])
            level = int(payload["granularity_level"])
            chunk_idx = int(payload["chunk_idx"])
            total_chunks = int(payload["total_chunks"])
        except (KeyError, TypeError, ValueError) as exc:
            malformed += 1
            _example(
                malformed_examples,
                {"point_id": str(point.id), "reason": str(exc)},
                example_limit,
            )
            continue

        key = (document_id, level)
        group = groups.setdefault(key, {"indices": set(), "totals": set()})
        if chunk_idx in group["indices"]:
            duplicate_keys += 1
            _example(
                duplicate_examples,
                {"document_id": document_id, "level": level, "chunk_idx": chunk_idx},
                example_limit,
            )
        group["indices"].add(chunk_idx)
        group["totals"].add(total_chunks)

        expected_id = deterministic_uuid(f"{document_id}_g{level}_c{chunk_idx}")
        if str(point.id) != expected_id:
            id_mismatches += 1
            _example(
                id_mismatch_examples,
                {"point_id": str(point.id), "expected_id": expected_id},
                example_limit,
            )

    missing_group_count = 0
    missing_groups = []
    invalid_groups = []
    missing_indices_count = 0
    complete_scan = max_points is None
    if complete_scan:
        for document_id, entry in checkpoint.items():
            if not isinstance(entry, dict):
                continue
            done = entry.get("done", [])
            for level, chunk_size in enumerate(CHUNK_SIZES, start=1):
                if chunk_size in done and (document_id, level) not in groups:
                    missing_group_count += 1
                    _example(
                        missing_groups,
                        {"document_id": document_id, "level": level, "chunk_size": chunk_size},
                        example_limit,
                    )

        for (document_id, level), group in groups.items():
            totals = group["totals"]
            indices = group["indices"]
            issue = {}
            if len(totals) != 1:
                issue["declared_totals"] = sorted(totals)
            else:
                total = next(iter(totals))
                missing = sorted(set(range(max(total, 0))) - indices)
                out_of_range = sorted(index for index in indices if index < 0 or index >= total)
                missing_indices_count += len(missing)
                if missing:
                    issue["missing_indices"] = missing[:example_limit]
                    issue["missing_index_count"] = len(missing)
                if out_of_range:
                    issue["out_of_range_indices"] = out_of_range[:example_limit]
            if issue:
                issue.update({"document_id": document_id, "level": level})
                _example(invalid_groups, issue, example_limit)

    return {
        "scanned": scanned,
        "scan_is_complete": complete_scan,
        "document_granularity_groups": len(groups),
        "missing_checkpointed_groups": missing_group_count,
        "missing_checkpointed_group_examples": missing_groups,
        "missing_chunk_indices": missing_indices_count,
        "invalid_group_examples": invalid_groups,
        "duplicate_deterministic_keys": duplicate_keys,
        "duplicate_examples": duplicate_examples,
        "deterministic_id_mismatches": id_mismatches,
        "id_mismatch_examples": id_mismatch_examples,
        "malformed_payloads": malformed,
        "malformed_examples": malformed_examples,
        "groups": groups,
    }


def validate_questions(
    client,
    batch_size: int,
    max_points: Optional[int],
    example_limit: int,
) -> dict:
    ids = set()
    documents = Counter()
    semantic_keys = set()
    duplicate_keys = 0
    duplicate_examples = []
    malformed = 0
    malformed_examples = []

    for point in scroll_points(
        client,
        "PaperQuestion",
        batch_size,
        with_payload=["document_id", "question_text", "split"],
        with_vectors=False,
        max_points=max_points,
    ):
        ids.add(str(point.id))
        payload = point.payload or {}
        document_id = payload.get("document_id")
        question_text = payload.get("question_text")
        if not document_id or not isinstance(question_text, str) or not question_text.strip():
            malformed += 1
            _example(malformed_examples, {"point_id": str(point.id)}, example_limit)
            continue
        documents[str(document_id)] += 1
        key = (str(document_id), question_text)
        if key in semantic_keys:
            duplicate_keys += 1
            _example(
                duplicate_examples,
                {"point_id": str(point.id), "document_id": str(document_id)},
                example_limit,
            )
        semantic_keys.add(key)

    return {
        "scanned": len(ids),
        "scan_is_complete": max_points is None,
        "ids": ids,
        "documents": documents,
        "duplicate_semantic_keys": duplicate_keys,
        "duplicate_examples": duplicate_examples,
        "malformed_payloads": malformed,
        "malformed_examples": malformed_examples,
        "deterministic_id_note": (
            "Question IDs cannot be recomputed from Qdrant payloads because "
            "original_question_id is not stored there."
        ),
    }


def validate_evidence(
    client,
    batch_size: int,
    max_points: Optional[int],
    example_limit: int,
) -> dict:
    ids = set()
    question_ids = set()
    documents = Counter()
    semantic_keys = set()
    duplicate_keys = 0
    duplicate_examples = []
    id_mismatches = 0
    id_mismatch_examples = []
    negative_offsets = 0
    negative_offset_examples = []
    malformed = 0
    malformed_examples = []
    scanned = 0

    fields = [
        "question_id",
        "document_id",
        "evidence_text",
        "span_start",
        "span_end",
    ]
    for point in scroll_points(
        client,
        "PaperEvidence",
        batch_size,
        with_payload=fields,
        with_vectors=False,
        max_points=max_points,
    ):
        scanned += 1
        ids.add(str(point.id))
        payload = point.payload or {}
        question_id = payload.get("question_id")
        document_id = payload.get("document_id")
        evidence_text = payload.get("evidence_text")
        if not question_id or not document_id or not isinstance(evidence_text, str) or not evidence_text.strip():
            malformed += 1
            _example(malformed_examples, {"point_id": str(point.id)}, example_limit)
            continue
        question_id = str(question_id)
        document_id = str(document_id)
        question_ids.add(question_id)
        documents[document_id] += 1

        evidence_hash = hashlib.md5(evidence_text.encode()).hexdigest()[:12]
        expected_id = deterministic_uuid(f"{question_id}_{evidence_hash}")
        if str(point.id) != expected_id:
            id_mismatches += 1
            _example(
                id_mismatch_examples,
                {"point_id": str(point.id), "expected_id": expected_id},
                example_limit,
            )
        key = (question_id, evidence_hash)
        if key in semantic_keys:
            duplicate_keys += 1
            _example(duplicate_examples, {"point_id": str(point.id), "key": key}, example_limit)
        semantic_keys.add(key)

        if payload.get("span_start") == -1 or payload.get("span_end") == -1:
            negative_offsets += 1
            _example(
                negative_offset_examples,
                {
                    "point_id": str(point.id),
                    "document_id": document_id,
                    "question_id": question_id,
                },
                example_limit,
            )

    return {
        "scanned": scanned,
        "scan_is_complete": max_points is None,
        "ids": ids,
        "question_ids": question_ids,
        "documents": documents,
        "duplicate_deterministic_keys": duplicate_keys,
        "duplicate_examples": duplicate_examples,
        "deterministic_id_mismatches": id_mismatches,
        "id_mismatch_examples": id_mismatch_examples,
        "evidence_offsets_equal_to_minus_one": negative_offsets,
        "negative_offset_examples": negative_offset_examples,
        "malformed_payloads": malformed,
        "malformed_examples": malformed_examples,
    }


def validate_qasper_expectations(
    question_ids: set,
    evidence_ids: set,
    example_limit: int,
) -> dict:
    """Compare stored IDs with IDs derived from the source QASPER records."""
    from datasets import load_dataset

    dataset = load_dataset("allenai/qasper")
    expected_questions = {}
    expected_evidence = {}
    zero_answerable_documents = []

    for split_name, split_data in dataset.items():
        for paper in split_data:
            document_id = paper["id"]
            qas = paper.get("qas", {})
            answerable_count = 0
            for question_text, original_question_id, answers_data in zip(
                qas.get("question", []),
                qas.get("question_id", []),
                qas.get("answers", []),
            ):
                evidence_texts = {
                    evidence.strip()
                    for annotation in answers_data.get("answer", [])
                    for evidence in annotation.get("highlighted_evidence", [])
                    if evidence and evidence.strip()
                }
                if not evidence_texts:
                    continue
                answerable_count += 1
                question_id = deterministic_uuid(f"{document_id}_{original_question_id}")
                expected_questions[question_id] = {
                    "document_id": document_id,
                    "split": split_name,
                    "question_text": question_text,
                }
                for evidence_text in evidence_texts:
                    evidence_hash = hashlib.md5(evidence_text.encode()).hexdigest()[:12]
                    evidence_id = deterministic_uuid(f"{question_id}_{evidence_hash}")
                    expected_evidence[evidence_id] = {
                        "document_id": document_id,
                        "question_id": question_id,
                    }
            if answerable_count == 0:
                zero_answerable_documents.append(document_id)

    missing_question_ids = sorted(set(expected_questions) - question_ids)
    extra_question_ids = sorted(question_ids - set(expected_questions))
    missing_evidence_ids = sorted(set(expected_evidence) - evidence_ids)
    extra_evidence_ids = sorted(evidence_ids - set(expected_evidence))
    return {
        "dataset": "allenai/qasper",
        "expected_questions": len(expected_questions),
        "stored_questions": len(question_ids),
        "missing_questions": len(missing_question_ids),
        "missing_question_examples": [
            {"point_id": point_id, **expected_questions[point_id]}
            for point_id in missing_question_ids[:example_limit]
        ],
        "unexpected_questions": len(extra_question_ids),
        "unexpected_question_examples": extra_question_ids[:example_limit],
        "expected_evidence": len(expected_evidence),
        "stored_evidence": len(evidence_ids),
        "missing_evidence": len(missing_evidence_ids),
        "missing_evidence_examples": [
            {"point_id": point_id, **expected_evidence[point_id]}
            for point_id in missing_evidence_ids[:example_limit]
        ],
        "unexpected_evidence": len(extra_evidence_ids),
        "unexpected_evidence_examples": extra_evidence_ids[:example_limit],
        "documents_with_no_answerable_questions": len(zero_answerable_documents),
        "zero_answerable_document_examples": zero_answerable_documents[:example_limit],
    }


def validate_evaluation_outputs(path: Path, example_limit: int) -> dict:
    files = []
    for output_path in sorted(path.glob("*.jsonl")) if path.exists() else []:
        records = 0
        invalid_json = 0
        missing_fields = 0
        duplicate_eval_ids = 0
        examples = []
        eval_ids = set()
        with output_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    invalid_json += 1
                    _example(examples, {"line": line_number, "error": str(exc)}, example_limit)
                    continue
                records += 1
                missing = sorted(EVALUATION_REQUIRED_FIELDS - set(record))
                if missing:
                    missing_fields += 1
                    _example(examples, {"line": line_number, "missing_fields": missing}, example_limit)
                eval_id = record.get("eval_id")
                if eval_id in eval_ids:
                    duplicate_eval_ids += 1
                    _example(examples, {"line": line_number, "duplicate_eval_id": eval_id}, example_limit)
                eval_ids.add(eval_id)
        files.append(
            {
                "path": str(output_path),
                "records": records,
                "invalid_json_lines": invalid_json,
                "records_missing_fields": missing_fields,
                "duplicate_eval_ids": duplicate_eval_ids,
                "examples": examples,
            }
        )
    return {"files": files}


def _strip_internal(report: dict) -> dict:
    public = dict(report)
    for key in ("groups", "ids", "documents", "question_ids"):
        public.pop(key, None)
    return public


def run_validation(args) -> Tuple[dict, bool]:
    checkpoint, checkpoint_report = load_checkpoint_report(args.checkpoint, args.example_limit)
    report = {
        "mode": "read-only",
        "checkpoint": checkpoint_report,
        "evaluation_outputs": validate_evaluation_outputs(args.outputs_dir, args.example_limit),
        "qdrant": {},
    }
    has_problems = bool(checkpoint_report.get("error") or checkpoint_report["incomplete_documents"])

    client = get_qdrant_client()
    try:
        existing = {item.name for item in client.get_collections().collections}
        missing_collections = sorted(set(COLLECTIONS) - existing)
        report["qdrant"]["missing_collections"] = missing_collections
        if missing_collections:
            return report, True

        collection_counts = {}
        schema_dimensions = {}
        for collection in COLLECTIONS:
            info = client.get_collection(collection)
            collection_counts[collection] = info.points_count
            schema_dimensions[collection] = _collection_vector_size(info)
            if schema_dimensions[collection] != EMBEDDING_DIM:
                has_problems = True
        report["qdrant"]["collection_counts"] = collection_counts
        report["qdrant"]["schema_vector_dimensions"] = schema_dimensions

        max_points = args.max_points or None
        chunk_report = validate_chunks(
            client, checkpoint, args.batch_size, max_points, args.example_limit
        )
        question_report = validate_questions(
            client, args.batch_size, max_points, args.example_limit
        )
        evidence_report = validate_evidence(
            client, args.batch_size, max_points, args.example_limit
        )

        relationship_report = {
            "questions_without_evidence": 0,
            "question_without_evidence_examples": [],
            "evidence_for_missing_questions": 0,
            "missing_question_examples": [],
            "checkpointed_question_docs_with_no_questions": [],
            "checkpointed_question_docs_with_no_questions_count": 0,
            "checkpointed_evidence_docs_with_no_evidence": [],
            "checkpointed_evidence_docs_with_no_evidence_count": 0,
            "scan_is_complete": max_points is None,
        }
        if max_points is None:
            question_ids = question_report["ids"]
            evidence_question_ids = evidence_report["question_ids"]
            no_evidence = sorted(question_ids - evidence_question_ids)
            missing_questions = sorted(evidence_question_ids - question_ids)
            relationship_report["questions_without_evidence"] = len(no_evidence)
            relationship_report["question_without_evidence_examples"] = no_evidence[: args.example_limit]
            relationship_report["evidence_for_missing_questions"] = len(missing_questions)
            relationship_report["missing_question_examples"] = missing_questions[: args.example_limit]

            for document_id, entry in checkpoint.items():
                if not isinstance(entry, dict):
                    continue
                done = entry.get("done", [])
                if "questions" in done and question_report["documents"][document_id] == 0:
                    relationship_report["checkpointed_question_docs_with_no_questions_count"] += 1
                    _example(
                        relationship_report["checkpointed_question_docs_with_no_questions"],
                        document_id,
                        args.example_limit,
                    )
                if "evidence" in done and evidence_report["documents"][document_id] == 0:
                    relationship_report["checkpointed_evidence_docs_with_no_evidence_count"] += 1
                    _example(
                        relationship_report["checkpointed_evidence_docs_with_no_evidence"],
                        document_id,
                        args.example_limit,
                    )

        vector_limit = None if args.full_vector_scan else args.vector_sample
        vector_reports = {
            collection: validate_vectors(
                client,
                collection,
                args.batch_size,
                vector_limit,
                args.example_limit,
            )
            for collection in COLLECTIONS
        }

        report["qdrant"].update(
            {
                "chunks": _strip_internal(chunk_report),
                "questions": _strip_internal(question_report),
                "evidence": _strip_internal(evidence_report),
                "relationships": relationship_report,
                "vectors": vector_reports,
            }
        )

        qasper_report = None
        if args.qasper_check:
            if max_points is not None:
                raise ValueError("--qasper-check requires a complete payload scan; omit --max-points")
            qasper_report = validate_qasper_expectations(
                question_report["ids"],
                evidence_report["ids"],
                args.example_limit,
            )
            report["qasper_expectations"] = qasper_report

        problem_counts = [
            chunk_report["missing_checkpointed_groups"],
            chunk_report["missing_chunk_indices"],
            chunk_report["duplicate_deterministic_keys"],
            chunk_report["deterministic_id_mismatches"],
            chunk_report["malformed_payloads"],
            question_report["malformed_payloads"],
            evidence_report["duplicate_deterministic_keys"],
            evidence_report["deterministic_id_mismatches"],
            evidence_report["evidence_offsets_equal_to_minus_one"],
            evidence_report["malformed_payloads"],
            relationship_report["questions_without_evidence"],
            relationship_report["evidence_for_missing_questions"],
            *(item["invalid_vectors"] for item in vector_reports.values()),
        ]
        if qasper_report:
            problem_counts.extend(
                [
                    qasper_report["missing_questions"],
                    qasper_report["unexpected_questions"],
                    qasper_report["missing_evidence"],
                    qasper_report["unexpected_evidence"],
                ]
            )
        has_problems = has_problems or any(problem_counts)
    finally:
        client.close()

    for item in report["evaluation_outputs"]["files"]:
        if item["invalid_json_lines"] or item["records_missing_fields"] or item["duplicate_eval_ids"]:
            has_problems = True
    return report, has_problems


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate QASPER checkpoint, Qdrant collections, vectors, and evaluation JSONL without modifying data."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.json"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Limit payload checks per collection for a smoke test; 0 scans all points.",
    )
    parser.add_argument(
        "--vector-sample",
        type=int,
        default=1000,
        help="Vectors checked per collection unless --full-vector-scan is used.",
    )
    parser.add_argument(
        "--full-vector-scan",
        action="store_true",
        help="Check every vector. This transfers all stored vectors and can be very slow.",
    )
    parser.add_argument(
        "--qasper-check",
        action="store_true",
        help=(
            "Load allenai/qasper and compare every expected deterministic question/evidence ID. "
            "May require Hugging Face network access."
        ),
    )
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optionally write the report to a new JSON file; Qdrant remains read-only.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Return exit code 0 even when integrity problems are reported.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_points < 0 or args.vector_sample < 1 or args.example_limit < 1:
        parser.error("batch size, vector sample, and example limit must be positive; max points cannot be negative")
    if args.qasper_check and args.max_points:
        parser.error("--qasper-check cannot be combined with --max-points")
    return args


def main() -> int:
    args = parse_args()
    report, has_problems = run_validation(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if args.no_fail or not has_problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
