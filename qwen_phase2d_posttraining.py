#!/usr/bin/env python
"""Read-only retrieval and comparison utilities for the Phase 2D classifier.

This module never trains a model and never writes to Qdrant. It validates the
canonical predictions produced by :mod:`qwen_phase2d_sequence_classifier`,
resumes the unchanged same-paper top-k=5 retrieval evaluation, and writes only
inside the isolated Phase 2D output/comparison roots.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import qwen_phase2 as phase2
import qwen_phase2b as phase2b
import qwen_phase2c_sequence_classifier as phase2c
import qwen_phase2d_sequence_classifier as phase2d


PHASE1_SUMMARY = phase2.PHASE1_ROOT / "final_summary.json"
PHASE2_SUMMARY = phase2.DEFAULT_OUTPUT_ROOT / "final_summary.json"
PHASE2B_UNWEIGHTED_SUMMARY = (
    phase2b.DEFAULT_OUTPUT_ROOTS[phase2b.VARIANT_UNWEIGHTED]
    / "final_summary.json"
)
PHASE2B_CLASSBALANCED_SUMMARY = (
    phase2b.DEFAULT_OUTPUT_ROOTS[phase2b.VARIANT_CLASSBALANCED]
    / "final_summary.json"
)
PHASE2C_SUMMARY = phase2c.DEFAULT_OUTPUT_ROOT / "final_summary.json"
PHASE2D_SUMMARY = phase2d.DEFAULT_OUTPUT_ROOT / "final_summary.json"
DEFAULT_COMPARISON_OUTPUT = Path(
    "outputs/qwen_phase2d_comparison_evidence_length_oracle/"
    "six_way_comparison.json"
)
DEFAULT_COMPARISON_ROOT = DEFAULT_COMPARISON_OUTPUT.parent

TOP_K = 5
METHOD_NAME = "qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-router"
# These identities freeze the already established downstream evaluation. They
# deliberately match Phase 1/2/2B and must not drift with the new classifier.
RETRIEVAL_CONFIG_HASH = (
    "9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8"
)
RETRIEVAL_SCHEMA_VERSION = 2
RETRIEVAL_METRIC_VERSION = "qasper-token-prf-v2"
RETRIEVAL_NORMALIZATION_VERSION = (
    "lowercase-remove-punctuation-collapse-whitespace-v1"
)
PHASE2C_INSTRUCTION_SHA256 = (
    "9e879535647c2bfcd3627d0d65f84c36a1bf442ed95bb5b07029c878ca990de7"
)
PHASE2D_INSTRUCTION_SHA256 = (
    "b3237368922abe709e2bd2d756fb9f25d39e7f5670e5c4cb15daaa3a2d1cf2e5"
)
TRAIN_ORACLE_SHA256 = (
    "64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88"
)
VALIDATION_ORACLE_SHA256 = (
    "ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d"
)
EXPECTED_ORACLE_DISTRIBUTION = {
    "10": 13,
    "20": 81,
    "40": 178,
    "80": 232,
    "160": 420,
}
FROZEN_RETRIEVAL_IDENTITY = {
    "top_k": TOP_K,
    "paper_restricted": True,
    "embedding_model": "text-embedding-3-small",
    "embedding_dimension": 1536,
    "tokenizer": "gpt2",
    "metric": "f1_joined_topk",
    "evaluation_config_hash": RETRIEVAL_CONFIG_HASH,
    "schema_version": RETRIEVAL_SCHEMA_VERSION,
    "metric_version": RETRIEVAL_METRIC_VERSION,
    "normalization_version": RETRIEVAL_NORMALIZATION_VERSION,
}

RETRIEVAL_IDENTITY_LEGACY_MINIMAL = "legacy_topk_paper_only"
RETRIEVAL_IDENTITY_LEGACY_PARTIAL = "legacy_partial"
RETRIEVAL_IDENTITY_FULL = "full"

_TRAINING_PROTOCOL_ALLOWED_DIFFERENCES = {
    "formulation_version",
    "instruction",
    "instruction_sha256",
    "run_id",
    "output_root",
    "repository_commit",
    "training_script_sha256",
    "experiment_fingerprint",
    "resume_contract_sha256",
    "created_at",
}
_DATASET_PROTOCOL_ALLOWED_DIFFERENCES = {
    "sequence_length",
    "experiment_fingerprint",
    "created_at",
    "verified_at",
}


@dataclass(frozen=True)
class RetrievalContext:
    output_root: Path
    run_id: str
    checkpoint_id: str
    experiment_fingerprint: str
    predictions: tuple[dict[str, Any], ...]
    final_summary: dict[str, Any]


def utc_now() -> str:
    return phase2.utc_now()


def _phase2d_formulation_version() -> str:
    value = str(getattr(phase2d, "FORMULATION_VERSION", ""))
    if not value:
        raise RuntimeError("Phase 2D exposes no formulation version")
    return value


def _chunk_sizes() -> tuple[int, ...]:
    values = tuple(int(value) for value in phase2d.CHUNK_SIZES)
    if values != (10, 20, 40, 80, 160):
        raise RuntimeError(f"Unexpected Phase 2D class order: {values}")
    return values


def resolve_output_root(output_root: Path | None) -> Path:
    """Resolve the isolated Phase 2D root and reject all prior Qwen roots."""

    root = Path(output_root or phase2d.DEFAULT_OUTPUT_ROOT)
    forbidden = {
        phase2.PHASE1_ROOT.resolve(),
        phase2.DEFAULT_OUTPUT_ROOT.resolve(),
        *(
            path.resolve()
            for path in phase2b.DEFAULT_OUTPUT_ROOTS.values()
        ),
        phase2c.DEFAULT_OUTPUT_ROOT.resolve(),
    }
    resolved = root.resolve()
    if any(
        resolved == prior
        or prior in resolved.parents
        or resolved in prior.parents
        for prior in forbidden
    ):
        raise RuntimeError("Phase 2D post-training output must use an isolated root")
    marker_path = root / "configuration" / "experiment.json"
    if not marker_path.is_file():
        raise RuntimeError(
            "Phase 2D post-training output has no configuration marker: "
            f"{marker_path}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_marker = {
        "phase": "Phase 2D",
        "formulation_version": _phase2d_formulation_version(),
        "model_id": phase2d.MODEL_ID,
        "model_revision": phase2d.MODEL_REVISION,
        "architecture": "AutoModelForSequenceClassification",
        "instruction": phase2d.SUPERVISOR_INSTRUCTION,
        "instruction_sha256": PHASE2D_INSTRUCTION_SHA256,
        "input_template": "{instruction}\\n\\nQuestion: {original_question_text}",
        "id2label": {
            str(index): str(label)
            for index, label in phase2d.ID_TO_LABEL.items()
        },
        "label2id": {
            str(label): index
            for label, index in phase2d.LABEL_TO_ID.items()
        },
        "objective": "uniform_five_class_cross_entropy",
    }
    mismatches = {
        key: {"expected": value, "actual": marker.get(key)}
        for key, value in expected_marker.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Phase 2D output-root configuration marker mismatch: {mismatches}"
        )
    return root


def resolve_comparison_output(
    output: Path,
    source_paths: Sequence[Path] = (),
) -> Path:
    """Restrict comparison writes to the dedicated Phase 2D comparison tree."""

    path = Path(output)
    resolved = path.resolve()
    root = DEFAULT_COMPARISON_ROOT.resolve()
    if root not in resolved.parents or resolved == root:
        raise RuntimeError(
            "Phase 2D comparison output must be a file inside the dedicated "
            f"comparison root: {DEFAULT_COMPARISON_ROOT}"
        )
    if path.suffix.lower() != ".json":
        raise RuntimeError("Phase 2D comparison output must be a JSON file")
    if path.exists() and path.is_dir():
        raise RuntimeError("Phase 2D comparison output cannot be a directory")
    protected_sources = {Path(value).resolve() for value in source_paths}
    if resolved in protected_sources:
        raise RuntimeError("Phase 2D comparison output cannot overwrite a source")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"JSONL artifact is empty: {path}")
    return rows


def _optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return read_jsonl(path)


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{message}: expected {expected!r}, got {actual!r}")


def _prediction_tokens(row: Mapping[str, Any]) -> int:
    """Return the explicitly persisted chunk-size prediction.

    Phase 2D is a five-logit classifier, so there is no parser and no fallback
    class. Supporting the two explicit field names makes the posttrainer robust
    to the final artifact's human-readable naming without inferring a default.
    """

    present = [
        key
        for key in ("predicted_label", "parsed_prediction")
        if row.get(key) is not None
    ]
    if not present:
        raise RuntimeError(
            f"Classifier prediction has no explicit chunk label: {row.get('question_id')}"
        )
    values = {int(row[key]) for key in present}
    if len(values) != 1:
        raise RuntimeError(
            f"Classifier prediction label fields disagree: {row.get('question_id')}"
        )
    value = next(iter(values))
    if value not in _chunk_sizes():
        raise RuntimeError(
            f"Classifier prediction is not one of the five classes: "
            f"{row.get('question_id')}"
        )
    return value


def _prediction_class_id(row: Mapping[str, Any]) -> int:
    if row.get("predicted_class_id") is None:
        raise RuntimeError(
            f"Classifier prediction has no explicit class ID: {row.get('question_id')}"
        )
    class_id = int(row["predicted_class_id"])
    chunks = _chunk_sizes()
    if class_id < 0 or class_id >= len(chunks):
        raise RuntimeError(
            f"Classifier class ID is outside 0..4: {row.get('question_id')}"
        )
    if chunks[class_id] != _prediction_tokens(row):
        raise RuntimeError(
            f"Classifier ID-to-chunk mapping mismatch: {row.get('question_id')}"
        )
    return class_id


def _summary_fingerprint(final: Mapping[str, Any]) -> str:
    training = final.get("training") or {}
    return str(
        final.get("experiment_fingerprint")
        or training.get("experiment_fingerprint")
        or ""
    )


def load_retrieval_context(
    output_root: Path,
    run_id: str,
    *,
    expected_examples: int = phase2.EXPECTED_COUNTS["validation"],
) -> RetrievalContext:
    """Validate all canonical classifier predictions and run provenance."""

    output_root = resolve_output_root(output_root)
    final_path = output_root / "final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    _require_equal(final.get("run_id"), run_id, "Final-summary run mismatch")
    _require_equal(final.get("model_id"), phase2d.MODEL_ID, "Model ID mismatch")
    _require_equal(
        final.get("model_revision"),
        phase2d.MODEL_REVISION,
        "Model revision mismatch",
    )
    if final.get("status") not in {
        "classification_complete_retrieval_pending",
        "complete",
    }:
        raise RuntimeError(f"Phase 2D classification is incomplete: {final.get('status')}")
    _require_equal(
        int(final.get("evaluated_examples", -1)),
        expected_examples,
        "Final-summary example count mismatch",
    )
    if final.get("formulation_version") is not None:
        _require_equal(
            final.get("formulation_version"),
            _phase2d_formulation_version(),
            "Final-summary formulation mismatch",
        )

    checkpoint_id = str(final.get("selected_checkpoint_id", ""))
    if not checkpoint_id:
        raise RuntimeError("Final summary has no selected checkpoint ID")
    fingerprint = _summary_fingerprint(final)
    if not fingerprint:
        raise RuntimeError("Final summary has no Phase 2D experiment fingerprint")

    prediction_path = output_root / "validation" / "predictions.jsonl"
    predictions = read_jsonl(prediction_path)
    _require_equal(
        len(predictions), expected_examples, "Canonical prediction count mismatch"
    )
    ids = [str(row.get("question_id", "")) for row in predictions]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("Canonical Phase 2D predictions have missing or duplicate IDs")

    valid_status = str(getattr(phase2d, "PREDICTION_STATUS", ""))
    for row in predictions:
        question_id = row["question_id"]
        if not row.get("document_id") or not row.get("question_text"):
            raise RuntimeError(f"Prediction lacks source identity: {question_id}")
        _require_equal(
            row.get("selected_checkpoint_id"),
            checkpoint_id,
            f"Prediction checkpoint mismatch for {question_id}",
        )
        _require_equal(
            row.get("experiment_fingerprint"),
            fingerprint,
            f"Prediction fingerprint mismatch for {question_id}",
        )
        if row.get("formulation_version") is not None:
            _require_equal(
                row.get("formulation_version"),
                _phase2d_formulation_version(),
                f"Prediction formulation mismatch for {question_id}",
            )
        if valid_status and row.get("prediction_status") is not None:
            _require_equal(
                row.get("prediction_status"),
                valid_status,
                f"Prediction status mismatch for {question_id}",
            )
        if any(
            row.get(key) is True
            for key in ("default_applied", "used_default", "fallback_applied")
        ):
            raise RuntimeError(f"A default/fallback class was used: {question_id}")
        _prediction_class_id(row)
        if int(row.get("oracle_label", -1)) not in _chunk_sizes():
            raise RuntimeError(f"Prediction has an invalid Oracle label: {question_id}")
        if not row.get("selected_checkpoint_path"):
            raise RuntimeError(f"Prediction has no checkpoint path: {question_id}")

    classification = final.get("classification") or {}
    invalid_count = final.get(
        "invalid_output_count",
        final.get("invalid_outputs", classification.get("invalid_output_count")),
    )
    if invalid_count is not None:
        _require_equal(int(invalid_count), 0, "Phase 2D invalid prediction count")
    valid_count = final.get(
        "valid_output_count",
        final.get("valid_outputs", classification.get("valid_output_count")),
    )
    if valid_count is not None:
        _require_equal(
            int(valid_count), expected_examples, "Phase 2D valid prediction count"
        )
    if final.get("valid_output_rate") is not None:
        _require_equal(float(final["valid_output_rate"]), 1.0, "Valid-output rate")

    return RetrievalContext(
        output_root=output_root,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        experiment_fingerprint=fingerprint,
        predictions=tuple(predictions),
        final_summary=final,
    )


def retrieval_run_id(run_id: str) -> str:
    return f"{run_id}-retrieval-top5-paper"


def validate_retrieval_record(
    record: Mapping[str, Any],
    prediction: Mapping[str, Any],
    context: RetrievalContext,
    phase1_module: Any,
) -> None:
    """Reject stale, cross-run, or differently configured retrieval rows."""

    predicted_tokens = _prediction_tokens(prediction)
    predicted_class_id = _prediction_class_id(prediction)
    predicted_level = predicted_class_id + 1
    expected = {
        "method_name": METHOD_NAME,
        "phase2d_run_id": context.run_id,
        "formulation_version": _phase2d_formulation_version(),
        "experiment_fingerprint": context.experiment_fingerprint,
        "evaluation_run_id": retrieval_run_id(context.run_id),
        "question_id": prediction["question_id"],
        "document_id": prediction["document_id"],
        "split": "validation",
        "granularity_tokens": predicted_tokens,
        "granularity_level": predicted_level,
        "predicted_class_id": predicted_class_id,
        "predicted_granularity_tokens": predicted_tokens,
        "predicted_granularity_level": predicted_level,
        "evidence_length_oracle": int(prediction["oracle_label"]),
        "oracle_label_version": phase2d.ORACLE_VERSION,
        "selected_checkpoint_id": context.checkpoint_id,
        "selected_checkpoint_path": prediction["selected_checkpoint_path"],
        "model_id": phase2d.MODEL_ID,
        "model_revision": phase2d.MODEL_REVISION,
        "k_requested": TOP_K,
        "top_k": TOP_K,
        "paper_restricted": True,
        "embedding_model": phase1_module.OPENAI_EMBEDDING_MODEL,
        "embedding_dimension": phase1_module.EMBEDDING_DIM,
        "tokenizer_identity": phase1_module.TOKENIZER_NAME,
        "evaluation_config_hash": RETRIEVAL_CONFIG_HASH,
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "metric_version": RETRIEVAL_METRIC_VERSION,
        "normalization_version": RETRIEVAL_NORMALIZATION_VERSION,
    }
    mismatches = {
        key: {"expected": value, "actual": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Stale or incompatible Phase 2D retrieval record for "
            f"{prediction['question_id']}: {mismatches}"
        )
    if "phase2d_retrieval_wall_seconds" not in record:
        raise RuntimeError(
            f"Retrieval record has no durable runtime: {prediction['question_id']}"
        )


def build_retrieval_summary(
    context: RetrievalContext,
    records: Sequence[Mapping[str, Any]],
    *,
    segment_wall_seconds: float,
    runtime_segments: Sequence[Mapping[str, Any]],
    uninterrupted: bool,
    phase1_module: Any,
) -> dict[str, Any]:
    """Build a complete summary using only validated durable records."""

    values = [float(row["f1_joined_topk"]) for row in records]
    total = len(context.predictions)
    valid = len(values)
    cumulative_question_wall = float(
        sum(float(row["phase2d_retrieval_wall_seconds"]) for row in records)
    )
    uninterrupted_wall = float(segment_wall_seconds) if uninterrupted else None
    reported_wall = (
        uninterrupted_wall
        if uninterrupted_wall is not None
        else cumulative_question_wall
    )
    return {
        "status": "complete",
        "run_id": context.run_id,
        "phase": "Phase 2D",
        "method_name": METHOD_NAME,
        "formulation_version": _phase2d_formulation_version(),
        "experiment_fingerprint": context.experiment_fingerprint,
        "evaluation_run_id": retrieval_run_id(context.run_id),
        "selected_checkpoint_id": context.checkpoint_id,
        "evaluated_examples": total,
        "valid_prediction_retrievals": valid,
        "invalid_predictions_without_retrieval": total - valid,
        "retrieval_coverage": valid / total if total else 0.0,
        "valid_only_mean_joined_retrieval_f1": (
            float(statistics.fmean(values)) if values else None
        ),
        "valid_only_median_joined_retrieval_f1": (
            float(statistics.median(values)) if values else None
        ),
        "coverage_adjusted_full_set_mean_joined_retrieval_f1": (
            float(sum(values) / total) if total else None
        ),
        "full_set_note": (
            "Every canonical Phase 2D classifier output is an explicit argmax "
            "over the five class logits. No default or parser fallback is used; "
            "an invalid prediction would be excluded rather than mapped."
        ),
        "top_k": TOP_K,
        "paper_restricted": True,
        "embedding_model": phase1_module.OPENAI_EMBEDDING_MODEL,
        "embedding_dimension": phase1_module.EMBEDDING_DIM,
        "tokenizer": phase1_module.TOKENIZER_NAME,
        "metric": "f1_joined_topk",
        "evaluation_config_hash": RETRIEVAL_CONFIG_HASH,
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "metric_version": RETRIEVAL_METRIC_VERSION,
        "normalization_version": RETRIEVAL_NORMALIZATION_VERSION,
        "runtime_segments": len(runtime_segments),
        "current_segment_wall_seconds": float(segment_wall_seconds),
        "completed_invocation_wall_seconds": float(
            sum(float(row["wall_seconds"]) for row in runtime_segments)
        ),
        "cumulative_durable_question_processing_seconds": cumulative_question_wall,
        "complete_uninterrupted_run_wall_seconds": uninterrupted_wall,
        "reported_retrieval_wall_seconds": reported_wall,
        "reported_retrieval_wall_basis": (
            "complete_uninterrupted_invocation"
            if uninterrupted_wall is not None
            else "durable_sum_of_per_question_processing_times_after_resume"
        ),
        "runtime_note": (
            "Each completed question stores an fsynced processing duration. "
            "After a resume, the durable per-question sum is reported instead "
            "of fabricating an uninterrupted wall time."
        ),
        "completed_at": utc_now(),
    }


def _known_pre_retrieval_wall(final: Mapping[str, Any]) -> float | None:
    training = final.get("training") or {}
    runtime = final.get("runtime") or {}
    training_wall = training.get("elapsed_seconds")
    if training_wall is None:
        training_wall = runtime.get("training_wall_seconds")
    load = runtime.get("model_load_seconds")
    inference = runtime.get("isolated_inference_wall_seconds")
    if inference is None:
        inference = runtime.get("final_validation_inference_wall_seconds")
    values = (training_wall, load, inference)
    if all(isinstance(value, (int, float)) for value in values):
        return sum(float(value) for value in values)
    return None


def update_final_summary(
    context: RetrievalContext, summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach retrieval results without replacing classifier provenance."""

    final_path = context.output_root / "final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    _require_equal(final.get("run_id"), context.run_id, "Final update run mismatch")
    final["status"] = "complete"
    if "experiment_status" in final:
        final["experiment_status"] = "complete"
    final["retrieval"] = dict(summary)
    runtime = final.setdefault("runtime", {})
    runtime["retrieval_wall_seconds"] = summary["reported_retrieval_wall_seconds"]
    pre_retrieval = _known_pre_retrieval_wall(final)
    if pre_retrieval is not None:
        runtime["known_training_plus_final_validation_wall_seconds"] = pre_retrieval
        runtime[
            "known_training_final_validation_and_retrieval_wall_seconds"
        ] = pre_retrieval + float(summary["reported_retrieval_wall_seconds"])
    artifacts = final.setdefault("artifacts", {})
    artifacts["retrieval_results"] = str(
        context.output_root / "retrieval" / "results.jsonl"
    )
    artifacts["retrieval_summary"] = str(
        context.output_root / "retrieval" / "summary.json"
    )
    artifacts["retrieval_runtime_segments"] = str(
        context.output_root / "retrieval" / "runtime_segments.jsonl"
    )
    final["completed_at"] = utc_now()
    phase2.atomic_json(final_path, final)
    return final


def _validate_completed_summary(
    summary: Mapping[str, Any],
    context: RetrievalContext,
    records: Sequence[Mapping[str, Any]],
) -> None:
    record_count = len(records)
    expected = {
        "status": "complete",
        "run_id": context.run_id,
        "method_name": METHOD_NAME,
        "formulation_version": _phase2d_formulation_version(),
        "experiment_fingerprint": context.experiment_fingerprint,
        "selected_checkpoint_id": context.checkpoint_id,
        "evaluated_examples": len(context.predictions),
        "valid_prediction_retrievals": record_count,
        "top_k": TOP_K,
        "paper_restricted": True,
    }
    for key, value in expected.items():
        _require_equal(summary.get(key), value, f"Retrieval-summary {key} mismatch")
    values = [float(row["f1_joined_topk"]) for row in records]
    numeric_expected = {
        "retrieval_coverage": record_count / len(context.predictions),
        "valid_only_mean_joined_retrieval_f1": float(statistics.fmean(values)),
        "valid_only_median_joined_retrieval_f1": float(statistics.median(values)),
        "coverage_adjusted_full_set_mean_joined_retrieval_f1": float(
            sum(values) / len(context.predictions)
        ),
    }
    for key, value in numeric_expected.items():
        actual = summary.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), value, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"Retrieval-summary {key} does not recompute from results"
            )


def evaluate_retrieval(
    output_root: Path,
    run_id: str,
    *,
    phase1_module: Any | None = None,
    expected_examples: int = phase2.EXPECTED_COUNTS["validation"],
) -> dict[str, Any]:
    """Run or resume unchanged read-only retrieval for one Phase 2D run."""

    if phase1_module is None:
        import qwen_phase1 as phase1_module

    context = load_retrieval_context(
        output_root, run_id, expected_examples=expected_examples
    )
    result_path = context.output_root / "retrieval" / "results.jsonl"
    existing = _optional_jsonl(result_path)
    existing_ids = [str(row.get("question_id", "")) for row in existing]
    if any(not value for value in existing_ids) or len(existing_ids) != len(
        set(existing_ids)
    ):
        raise RuntimeError("Phase 2D retrieval JSONL has missing or duplicate IDs")
    prediction_by_id = {row["question_id"]: row for row in context.predictions}
    if not set(existing_ids).issubset(prediction_by_id):
        raise RuntimeError("Retrieval JSONL contains an unknown prediction ID")
    by_id = {row["question_id"]: row for row in existing}
    for question_id, record in by_id.items():
        validate_retrieval_record(
            record, prediction_by_id[question_id], context, phase1_module
        )

    summary_path = context.output_root / "retrieval" / "summary.json"
    if len(by_id) == len(context.predictions) and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ordered = [by_id[row["question_id"]] for row in context.predictions]
        _validate_completed_summary(summary, context, ordered)
        update_final_summary(context, summary)
        return summary

    started = time.perf_counter()
    initial_count = len(by_id)
    client = (
        phase1_module.qdrant_client()
        if initial_count < len(context.predictions)
        else None
    )
    completed_this_segment = 0
    for prediction in context.predictions:
        question_id = prediction["question_id"]
        if question_id in by_id:
            continue
        if client is None:
            raise RuntimeError("Qdrant client was not initialized")
        question_started = time.perf_counter()
        points = client.retrieve(
            collection_name=phase1_module.PAPER_QUESTION_COLLECTION,
            ids=[question_id],
            with_payload=True,
            with_vectors=True,
        )
        if len(points) != 1:
            raise RuntimeError(f"Question point lookup failed: {question_id}")
        predicted_tokens = _prediction_tokens(prediction)
        predicted_class_id = _prediction_class_id(prediction)
        predicted_level = predicted_class_id + 1
        generated = list(
            phase1_module.evaluate_question(
                client=client,
                question_point_id=question_id,
                question_vector=points[0].vector,
                document_id=prediction["document_id"],
                question_text=prediction["question_text"],
                split="validation",
                top_k=TOP_K,
                granularity_levels=[predicted_level],
                store_retrieved_text=False,
                chunk_sizes=list(_chunk_sizes()),
                embedding_model=phase1_module.OPENAI_EMBEDDING_MODEL,
                embedding_dimension=phase1_module.EMBEDDING_DIM,
                tokenizer_name=phase1_module.TOKENIZER_NAME,
                evaluation_run_id=retrieval_run_id(context.run_id),
                evaluation_config_hash=RETRIEVAL_CONFIG_HASH,
            )
        )
        if len(generated) != 1:
            raise RuntimeError(f"Expected one retrieval result: {question_id}")
        record = dict(generated[0])
        record.update(
            {
                "method_name": METHOD_NAME,
                "phase2d_run_id": context.run_id,
                "formulation_version": _phase2d_formulation_version(),
                "experiment_fingerprint": context.experiment_fingerprint,
                "predicted_class_id": predicted_class_id,
                "predicted_granularity_tokens": predicted_tokens,
                "predicted_granularity_level": predicted_level,
                "classifier_prediction_status": prediction.get(
                    "prediction_status", "valid_classifier_argmax"
                ),
                "evidence_length_oracle": int(prediction["oracle_label"]),
                "oracle_label_version": phase2d.ORACLE_VERSION,
                "selected_checkpoint_id": context.checkpoint_id,
                "selected_checkpoint_path": prediction["selected_checkpoint_path"],
                "model_id": phase2d.MODEL_ID,
                "model_revision": phase2d.MODEL_REVISION,
                "top_k": TOP_K,
                "paper_restricted": True,
                "phase2d_retrieval_wall_seconds": (
                    time.perf_counter() - question_started
                ),
            }
        )
        validate_retrieval_record(record, prediction, context, phase1_module)
        phase2.append_jsonl(result_path, record)
        by_id[question_id] = record
        completed_this_segment += 1

    ordered = [by_id[row["question_id"]] for row in context.predictions]
    phase2.atomic_jsonl(result_path, ordered)
    segment_wall = time.perf_counter() - started
    segment_path = context.output_root / "retrieval" / "runtime_segments.jsonl"
    phase2.append_jsonl(
        segment_path,
        {
            "run_id": context.run_id,
            "formulation_version": _phase2d_formulation_version(),
            "experiment_fingerprint": context.experiment_fingerprint,
            "evaluation_run_id": retrieval_run_id(context.run_id),
            "records_before_segment": initial_count,
            "new_records": completed_this_segment,
            "records_after_segment": len(ordered),
            "wall_seconds": segment_wall,
            "completed_at": utc_now(),
        },
    )
    segments = read_jsonl(segment_path)
    for segment in segments:
        _require_equal(
            segment.get("run_id"), context.run_id, "Runtime-segment run mismatch"
        )
        _require_equal(
            segment.get("formulation_version"),
            _phase2d_formulation_version(),
            "Runtime-segment formulation mismatch",
        )
        _require_equal(
            segment.get("experiment_fingerprint"),
            context.experiment_fingerprint,
            "Runtime-segment fingerprint mismatch",
        )

    summary = build_retrieval_summary(
        context,
        ordered,
        segment_wall_seconds=segment_wall,
        runtime_segments=segments,
        uninterrupted=initial_count == 0,
        phase1_module=phase1_module,
    )
    phase2.atomic_json(summary_path, summary)
    update_final_summary(context, summary)
    return summary


def _metric(container: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in container:
            return container[name]
    return None


def _load_run_metadata(
    summary_path: Path,
    summary: Mapping[str, Any],
    filename: str,
) -> tuple[Path, dict[str, Any]]:
    """Load a run artifact locally without trusting stale remote path strings."""

    run_id = str(summary.get("run_id") or "")
    if not run_id:
        raise RuntimeError(f"Comparison summary has no run ID: {summary_path}")
    candidates = [summary_path.parent / "runs" / run_id / filename]
    artifact_value = (summary.get("artifacts") or {}).get(
        filename.removesuffix(".json")
    )
    if artifact_value:
        artifact = Path(str(artifact_value))
        candidates.extend((artifact, summary_path.parent / artifact))
    for candidate in candidates:
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
    raise RuntimeError(
        f"Cannot locate {filename} for comparison source {summary_path}; "
        f"checked {[str(value) for value in candidates]}"
    )


def _without_keys(value: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _require_mapping_values(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    description: str,
) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{description} mismatch: {mismatches}")


def audit_phase2c_phase2d_protocol(
    phase2c_summary_path: Path,
    phase2d_summary_path: Path,
) -> dict[str, Any]:
    """Prove from saved metadata that Phase 2D is a prompt-only ablation."""

    c_path = Path(phase2c_summary_path)
    d_path = Path(phase2d_summary_path)
    c_final = json.loads(c_path.read_text(encoding="utf-8"))
    d_final = json.loads(d_path.read_text(encoding="utf-8"))
    c_config_path, c_config = _load_run_metadata(
        c_path, c_final, "training_config.json"
    )
    d_config_path, d_config = _load_run_metadata(
        d_path, d_final, "training_config.json"
    )
    c_data_path, c_data = _load_run_metadata(
        c_path, c_final, "dataset_manifest.json"
    )
    d_data_path, d_data = _load_run_metadata(
        d_path, d_final, "dataset_manifest.json"
    )

    old_mapping = (
        "1 = very short context, 2 = short context, 3 = medium context, "
        "4 = long context, 5 = very long context"
    )
    new_mapping = (
        "1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, 4 = 80 tokens, "
        "5 = 160 tokens"
    )
    expected_phase2d_instruction = phase2c.SUPERVISOR_INSTRUCTION.replace(
        old_mapping, new_mapping
    )
    if expected_phase2d_instruction == phase2c.SUPERVISOR_INSTRUCTION:
        raise RuntimeError("Phase 2C prompt mapping was not found exactly once")
    if expected_phase2d_instruction != phase2d.SUPERVISOR_INSTRUCTION:
        raise RuntimeError("Phase 2D instruction is not the exact prompt-only edit")
    expected_prompts = {
        "phase2c_instruction": phase2c.SUPERVISOR_INSTRUCTION,
        "phase2c_instruction_sha256": PHASE2C_INSTRUCTION_SHA256,
        "phase2d_instruction": phase2d.SUPERVISOR_INSTRUCTION,
        "phase2d_instruction_sha256": PHASE2D_INSTRUCTION_SHA256,
    }
    actual_prompts = {
        "phase2c_instruction": c_config.get("instruction"),
        "phase2c_instruction_sha256": c_config.get("instruction_sha256"),
        "phase2d_instruction": d_config.get("instruction"),
        "phase2d_instruction_sha256": d_config.get("instruction_sha256"),
    }
    _require_mapping_values(
        actual_prompts, expected_prompts, "Phase 2C/2D instruction provenance"
    )
    if phase2c.text_sha256(phase2c.SUPERVISOR_INSTRUCTION) != (
        PHASE2C_INSTRUCTION_SHA256
    ):
        raise RuntimeError("Frozen Phase 2C instruction constant hash drifted")
    if phase2d.text_sha256(phase2d.SUPERVISOR_INSTRUCTION) != (
        PHASE2D_INSTRUCTION_SHA256
    ):
        raise RuntimeError("Frozen Phase 2D instruction constant hash drifted")

    expected_c_config = vars(phase2c.TrainingConfig())
    expected_d_config = vars(phase2d.TrainingConfig())
    _require_mapping_values(
        c_config, expected_c_config, "Saved Phase 2C training configuration"
    )
    _require_mapping_values(
        d_config, expected_d_config, "Saved Phase 2D training configuration"
    )
    normalized_c_config = _without_keys(
        c_config, _TRAINING_PROTOCOL_ALLOWED_DIFFERENCES
    )
    normalized_d_config = _without_keys(
        d_config, _TRAINING_PROTOCOL_ALLOWED_DIFFERENCES
    )
    if normalized_c_config != normalized_d_config:
        differing = sorted(
            key
            for key in set(normalized_c_config) | set(normalized_d_config)
            if normalized_c_config.get(key) != normalized_d_config.get(key)
        )
        raise RuntimeError(
            "Phase 2C/2D non-prompt training protocol differs at "
            f"{differing}"
        )

    expected_train_distribution = {
        str(key): value
        for key, value in phase2.EXPECTED_DISTRIBUTIONS["train"].items()
    }
    expected_validation_distribution = {
        str(key): value
        for key, value in phase2.EXPECTED_DISTRIBUTIONS["validation"].items()
    }
    expected_data = {
        "train_examples": phase2.EXPECTED_COUNTS["train"],
        "validation_examples": phase2.EXPECTED_COUNTS["validation"],
        "train_documents": 845,
        "validation_documents": 277,
        "train_distribution": expected_train_distribution,
        "validation_distribution": expected_validation_distribution,
        "train_oracle_sha256": TRAIN_ORACLE_SHA256,
        "validation_oracle_sha256": VALIDATION_ORACLE_SHA256,
        "active_train_examples": phase2.EXPECTED_COUNTS["train"],
        "active_validation_examples": phase2.EXPECTED_COUNTS["validation"],
        "active_train_distribution": expected_train_distribution,
        "active_validation_distribution": expected_validation_distribution,
        "model_inputs": [
            "fixed_supervisor_instruction",
            "original_question_text",
        ],
    }
    _require_mapping_values(c_data, expected_data, "Saved Phase 2C dataset")
    _require_mapping_values(d_data, expected_data, "Saved Phase 2D dataset")
    normalized_c_data = _without_keys(
        c_data, _DATASET_PROTOCOL_ALLOWED_DIFFERENCES
    )
    normalized_d_data = _without_keys(
        d_data, _DATASET_PROTOCOL_ALLOWED_DIFFERENCES
    )
    if normalized_c_data != normalized_d_data:
        differing = sorted(
            key
            for key in set(normalized_c_data) | set(normalized_d_data)
            if normalized_c_data.get(key) != normalized_d_data.get(key)
        )
        raise RuntimeError(
            f"Phase 2C/2D frozen dataset protocol differs at {differing}"
        )

    expected_final = {
        "model_id": phase2d.MODEL_ID,
        "model_revision": phase2d.MODEL_REVISION,
        "evaluated_examples": phase2.EXPECTED_COUNTS["validation"],
        "oracle_distribution": EXPECTED_ORACLE_DISTRIBUTION,
        "id2label": {
            str(index): label for index, label in phase2d.ID_TO_LABEL.items()
        },
    }
    _require_mapping_values(c_final, expected_final, "Phase 2C final metadata")
    _require_mapping_values(d_final, expected_final, "Phase 2D final metadata")
    _require_equal(
        c_config.get("run_id"), c_final.get("run_id"), "Phase 2C run provenance"
    )
    _require_equal(
        d_config.get("run_id"), d_final.get("run_id"), "Phase 2D run provenance"
    )

    return {
        "status": "passed",
        "relationship": "prompt_only_single_seed_ablation",
        "phase2c_instruction_sha256": PHASE2C_INSTRUCTION_SHA256,
        "phase2d_instruction_sha256": PHASE2D_INSTRUCTION_SHA256,
        "only_semantic_change": {
            "phase2c_mapping": old_mapping,
            "phase2d_mapping": new_mapping,
        },
        "allowed_training_metadata_differences": sorted(
            _TRAINING_PROTOCOL_ALLOWED_DIFFERENCES
        ),
        "allowed_dataset_consequences": sorted(
            _DATASET_PROTOCOL_ALLOWED_DIFFERENCES
        ),
        "frozen_model_id": phase2d.MODEL_ID,
        "frozen_model_revision": phase2d.MODEL_REVISION,
        "train_oracle_sha256": TRAIN_ORACLE_SHA256,
        "validation_oracle_sha256": VALIDATION_ORACLE_SHA256,
        "phase2c": {
            "summary": str(c_path),
            "training_config": str(c_config_path),
            "training_config_sha256": phase2.sha256_file(c_config_path),
            "dataset_manifest": str(c_data_path),
            "dataset_manifest_sha256": phase2.sha256_file(c_data_path),
        },
        "phase2d": {
            "summary": str(d_path),
            "training_config": str(d_config_path),
            "training_config_sha256": phase2.sha256_file(d_config_path),
            "dataset_manifest": str(d_data_path),
            "dataset_manifest_sha256": phase2.sha256_file(d_data_path),
        },
    }


def validate_retrieval_identity(
    retrieval: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Validate frozen retrieval fields with explicit legacy-schema handling."""

    if mode not in {
        RETRIEVAL_IDENTITY_LEGACY_MINIMAL,
        RETRIEVAL_IDENTITY_LEGACY_PARTIAL,
        RETRIEVAL_IDENTITY_FULL,
    }:
        raise ValueError(f"Unknown retrieval-identity mode: {mode}")
    actual = {
        "top_k": retrieval.get("top_k"),
        "paper_restricted": retrieval.get("paper_restricted"),
        "embedding_model": retrieval.get("embedding_model"),
        "embedding_dimension": retrieval.get("embedding_dimension"),
        "tokenizer": _metric(retrieval, "tokenizer", "tokenizer_identity"),
        "metric": retrieval.get("metric"),
        "evaluation_config_hash": retrieval.get("evaluation_config_hash"),
        "schema_version": retrieval.get("schema_version"),
        "metric_version": retrieval.get("metric_version"),
        "normalization_version": retrieval.get("normalization_version"),
    }
    required = {"top_k", "paper_restricted"}
    if mode == RETRIEVAL_IDENTITY_LEGACY_PARTIAL:
        required.update({"embedding_model", "tokenizer", "metric"})
    elif mode == RETRIEVAL_IDENTITY_FULL:
        required.update(FROZEN_RETRIEVAL_IDENTITY)
    missing_required = sorted(key for key in required if actual.get(key) is None)
    if missing_required:
        raise RuntimeError(
            f"Retrieval identity is missing required fields: {missing_required}"
        )
    mismatches = {
        key: {"expected": FROZEN_RETRIEVAL_IDENTITY[key], "actual": value}
        for key, value in actual.items()
        if value is not None and value != FROZEN_RETRIEVAL_IDENTITY[key]
    }
    if mismatches:
        raise RuntimeError(f"Frozen retrieval identity mismatch: {mismatches}")
    missing_legacy = sorted(
        key for key, value in actual.items() if value is None
    )
    status = (
        "complete_frozen_identity"
        if not missing_legacy
        else "accepted_explicit_legacy_schema"
    )
    return {
        "status": status,
        "mode": mode,
        "validated": {
            key: value for key, value in actual.items() if value is not None
        },
        "legacy_missing_fields": missing_legacy,
    }


def normalized_comparison_row(
    name: str,
    path: Path,
    *,
    expected_variant: str | None = None,
    expected_formulation: str | None = None,
    retrieval_identity_mode: str = RETRIEVAL_IDENTITY_FULL,
) -> dict[str, Any]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError(f"Comparison source is incomplete: {path}")
    if expected_variant is not None and summary.get("variant") != expected_variant:
        raise RuntimeError(f"Comparison source variant mismatch: {path}")
    if expected_formulation is not None and (
        summary.get("formulation_version") != expected_formulation
    ):
        raise RuntimeError(f"Comparison source formulation mismatch: {path}")
    classification = summary.get("classification") or {}
    retrieval = summary.get("retrieval") or {}
    if not classification or not retrieval:
        raise RuntimeError(f"Comparison source lacks metrics: {path}")
    retrieval_identity = validate_retrieval_identity(
        retrieval, retrieval_identity_mode
    )
    runtime = summary.get("runtime") or {}
    total_runtime = _metric(
        runtime,
        "known_training_final_validation_and_retrieval_wall_seconds",
        "known_inference_plus_retrieval_wall_seconds",
    )
    return {
        "name": name,
        "source": str(path),
        "status": summary["status"],
        "phase": summary.get("phase"),
        "variant": summary.get("variant"),
        "formulation_version": summary.get("formulation_version"),
        "run_id": summary.get("run_id"),
        "model_id": summary.get("model_id"),
        "model_revision": summary.get("model_revision"),
        "evaluated_examples": int(summary["evaluated_examples"]),
        "oracle_distribution": summary["oracle_distribution"],
        "predicted_distribution": summary["predicted_distribution"],
        "valid_output_rate": summary.get("valid_output_rate"),
        "accuracy": classification.get("accuracy"),
        "macro_f1": classification.get("macro_f1"),
        "weighted_f1": classification.get("weighted_f1"),
        "balanced_accuracy": classification.get("balanced_accuracy"),
        "top_2_accuracy": classification.get("top_2_accuracy"),
        "top_2_accuracy_status": classification.get("top_2_accuracy_status"),
        "retrieval_coverage": _metric(retrieval, "retrieval_coverage", "coverage"),
        "mean_joined_retrieval_f1": _metric(
            retrieval,
            "valid_only_mean_joined_retrieval_f1",
            "valid_only_mean_joined_f1",
        ),
        "median_joined_retrieval_f1": _metric(
            retrieval,
            "valid_only_median_joined_retrieval_f1",
            "valid_only_median_joined_f1",
        ),
        "top_k": retrieval.get("top_k"),
        "paper_restricted": retrieval.get("paper_restricted"),
        "retrieval_identity": retrieval_identity,
        "known_total_runtime_seconds": total_runtime,
    }


def compare_summaries(
    phase1_summary: Path,
    phase2_summary: Path,
    phase2b_unweighted_summary: Path,
    phase2b_classbalanced_summary: Path,
    phase2c_summary: Path,
    phase2d_summary: Path,
) -> dict[str, Any]:
    """Compare all six new-Oracle Qwen runs without claiming causal identity."""

    rows = [
        normalized_comparison_row(
            "phase1_zero_shot",
            phase1_summary,
            retrieval_identity_mode=RETRIEVAL_IDENTITY_LEGACY_MINIMAL,
        ),
        normalized_comparison_row(
            "phase2_numeric_sft",
            phase2_summary,
            retrieval_identity_mode=RETRIEVAL_IDENTITY_LEGACY_PARTIAL,
        ),
        normalized_comparison_row(
            "phase2b_alias_unweighted",
            phase2b_unweighted_summary,
            expected_variant=phase2b.VARIANT_UNWEIGHTED,
            retrieval_identity_mode=RETRIEVAL_IDENTITY_FULL,
        ),
        normalized_comparison_row(
            "phase2b_alias_classbalanced",
            phase2b_classbalanced_summary,
            expected_variant=phase2b.VARIANT_CLASSBALANCED,
            retrieval_identity_mode=RETRIEVAL_IDENTITY_FULL,
        ),
        normalized_comparison_row(
            "phase2c_base_sequence_classifier",
            phase2c_summary,
            expected_formulation=phase2c.FORMULATION_VERSION,
            retrieval_identity_mode=RETRIEVAL_IDENTITY_FULL,
        ),
        normalized_comparison_row(
            "phase2d_base_sequence_classifier_token_count_prompt",
            phase2d_summary,
            expected_formulation=_phase2d_formulation_version(),
            retrieval_identity_mode=RETRIEVAL_IDENTITY_FULL,
        ),
    ]
    counts = {row["evaluated_examples"] for row in rows}
    distributions = {
        json.dumps(row["oracle_distribution"], sort_keys=True) for row in rows
    }
    if len(counts) != 1 or len(distributions) != 1:
        raise RuntimeError("Comparison summaries do not use the same validation set")
    if next(iter(counts)) != phase2.EXPECTED_COUNTS["validation"]:
        raise RuntimeError("Comparison does not contain all 924 validation examples")
    if rows[0]["oracle_distribution"] != EXPECTED_ORACLE_DISTRIBUTION:
        raise RuntimeError("Comparison Oracle distribution is not the frozen split")
    for row in rows:
        if row["top_k"] != TOP_K or row["paper_restricted"] is not True:
            raise RuntimeError("Comparison retrieval configurations are not aligned")
    prompt_only_protocol_audit = audit_phase2c_phase2d_protocol(
        phase2c_summary, phase2d_summary
    )

    def winners(metric: str) -> list[str]:
        available = [row for row in rows if row[metric] is not None]
        best = max(float(row[metric]) for row in available)
        return [
            row["name"]
            for row in available
            if math.isclose(float(row[metric]), best)
        ]

    return {
        "status": "complete",
        "comparison": (
            "Qwen Phases 1, 2, 2B-A, 2B-B, 2C, and 2D under the "
            "evidence-length Oracle"
        ),
        "evaluated_examples": next(iter(counts)),
        "oracle_distribution": rows[0]["oracle_distribution"],
        "prompt_only_protocol_audit": prompt_only_protocol_audit,
        "retrieval_identity_handling": {
            row["name"]: row["retrieval_identity"] for row in rows
        },
        "comparability_note": (
            "All rows use the same preserved validation questions, evidence-length "
            "Oracle, five chunk classes, and downstream retrieval. Phase 2C and "
            "Phase 2D form a prompt-only single-seed ablation: Phase 2D replaces "
            "the qualitative 1--5 context descriptions with exact 10/20/40/80/160 "
            "token counts while preserving the Base checkpoint, classifier, data, "
            "optimization, seed, selection rule, and retrieval configuration. "
            "Earlier cross-phase differences remain multiply confounded."
        ),
        "rows": rows,
        "highest_metrics": {
            metric: winners(metric)
            for metric in (
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "balanced_accuracy",
                "mean_joined_retrieval_f1",
            )
        },
        "created_at": utc_now(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieval = subparsers.add_parser("evaluate-retrieval")
    retrieval.add_argument("--run-id", required=True)
    retrieval.add_argument("--output-root", type=Path)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--phase1-summary", type=Path, default=PHASE1_SUMMARY)
    comparison.add_argument("--phase2-summary", type=Path, default=PHASE2_SUMMARY)
    comparison.add_argument(
        "--phase2b-unweighted-summary",
        type=Path,
        default=PHASE2B_UNWEIGHTED_SUMMARY,
    )
    comparison.add_argument(
        "--phase2b-classbalanced-summary",
        type=Path,
        default=PHASE2B_CLASSBALANCED_SUMMARY,
    )
    comparison.add_argument(
        "--phase2c-summary", type=Path, default=PHASE2C_SUMMARY
    )
    comparison.add_argument(
        "--phase2d-summary", type=Path, default=PHASE2D_SUMMARY
    )
    comparison.add_argument(
        "--output", type=Path, default=DEFAULT_COMPARISON_OUTPUT
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate-retrieval":
        root = resolve_output_root(args.output_root)
        result = evaluate_retrieval(root, args.run_id)
    else:
        source_paths = (
            args.phase1_summary,
            args.phase2_summary,
            args.phase2b_unweighted_summary,
            args.phase2b_classbalanced_summary,
            args.phase2c_summary,
            args.phase2d_summary,
        )
        output = resolve_comparison_output(args.output, source_paths)
        result = compare_summaries(
            args.phase1_summary,
            args.phase2_summary,
            args.phase2b_unweighted_summary,
            args.phase2b_classbalanced_summary,
            args.phase2c_summary,
            args.phase2d_summary,
        )
        phase2.atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
