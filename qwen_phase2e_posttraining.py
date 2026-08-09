#!/usr/bin/env python
"""Read-only downstream retrieval for the locked Phase 2E grid winner.

The Phase 2E learning-rate winner is selected exclusively from the saved
classification metrics.  This module first audits that immutable selection,
binds the canonical predictions to the selected checkpoint, and only then
runs the unchanged Phase 1 same-paper, top-k=5 retrieval implementation.

No model training or Qdrant mutation is performed.  Writes are restricted to
the winning Phase 2E trial and its study-level selected-final summary; prior
phases and non-winning Phase 2E trials are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import qwen_phase2 as phase2
import qwen_phase2d_posttraining as phase2d_post
import qwen_phase2d_sequence_classifier as phase2d
import qwen_phase2e_sequence_classifier_lr_grid as phase2e


# Reuse the exact frozen downstream evaluation identity established in Phase 1
# and enforced in Phase 2D.  Aliasing these constants prevents a Phase 2E-only
# redefinition from silently changing retrieval behavior.
TOP_K = phase2d_post.TOP_K
RETRIEVAL_CONFIG_HASH = phase2d_post.RETRIEVAL_CONFIG_HASH
RETRIEVAL_SCHEMA_VERSION = phase2d_post.RETRIEVAL_SCHEMA_VERSION
RETRIEVAL_METRIC_VERSION = phase2d_post.RETRIEVAL_METRIC_VERSION
RETRIEVAL_NORMALIZATION_VERSION = phase2d_post.RETRIEVAL_NORMALIZATION_VERSION
FROZEN_RETRIEVAL_IDENTITY = dict(phase2d_post.FROZEN_RETRIEVAL_IDENTITY)

METHOD_NAME = (
    "qwen-phase2e-base-sequence-classifier-token-count-prompt-"
    "lr-grid-selected-router"
)


@dataclass(frozen=True)
class LockedWinner:
    study_root: Path
    variant: str
    run_id: str
    learning_rate: float
    checkpoint_id: str
    trial_root: Path
    selection: dict[str, Any]


@dataclass(frozen=True)
class RetrievalContext:
    winner: LockedWinner
    experiment_fingerprint: str
    predictions: tuple[dict[str, Any], ...]
    final_summary: dict[str, Any]


def utc_now() -> str:
    return phase2.utc_now()


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


def _chunk_sizes() -> tuple[int, ...]:
    values = tuple(int(value) for value in phase2d.CHUNK_SIZES)
    if values != (10, 20, 40, 80, 160):
        raise RuntimeError(f"Unexpected Phase 2E class order: {values}")
    return values


def _prediction_tokens(row: Mapping[str, Any]) -> int:
    present = [
        key
        for key in ("predicted_label", "parsed_prediction")
        if row.get(key) is not None
    ]
    if not present:
        raise RuntimeError(
            "Classifier prediction has no explicit chunk label: "
            f"{row.get('question_id')}"
        )
    values = {int(row[key]) for key in present}
    if len(values) != 1:
        raise RuntimeError(
            "Classifier prediction label fields disagree: "
            f"{row.get('question_id')}"
        )
    value = next(iter(values))
    if value not in _chunk_sizes():
        raise RuntimeError(
            "Classifier prediction is not one of the five classes: "
            f"{row.get('question_id')}"
        )
    return value


def _prediction_class_id(row: Mapping[str, Any]) -> int:
    if row.get("predicted_class_id") is None:
        raise RuntimeError(
            "Classifier prediction has no explicit class ID: "
            f"{row.get('question_id')}"
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


def _candidate_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant": row.get("variant"),
        "learning_rate": row.get("learning_rate"),
        "run_id": row.get("run_id"),
        "checkpoint_id": row.get("checkpoint_id"),
        "global_step": row.get("global_step"),
        "validation_loss": row.get("validation_loss"),
        "classification_metrics": row.get("classification_metrics"),
    }


def load_locked_winner(study_root: Path) -> LockedWinner:
    """Audit and return the classification-selected Phase 2E winner.

    Selection is recomputed only from the 15 classification candidates saved
    in the pre-retrieval lock.  No retrieval metric or artifact is consulted.
    """

    root = phase2e._validate_study_root(Path(study_root))
    grid_path = root / "configuration" / "grid_experiment.json"
    selection_path = root / "comparison" / "selected_trial.json"
    selected_final_path = root / "comparison" / "selected_final_summary.json"
    for path in (grid_path, selection_path, selected_final_path):
        if not path.is_file():
            raise RuntimeError(f"Phase 2E retrieval requires saved artifact: {path}")

    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_grid = {
        "phase": phase2e.PHASE,
        "study_id": phase2e.STUDY_ID,
        "formulation_version": phase2e.FORMULATION_VERSION,
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "retrieval_selection_rule": phase2e._stable_grid_protocol()[
            "retrieval_selection_rule"
        ],
    }
    for key, value in expected_grid.items():
        _require_equal(grid.get(key), value, f"Phase 2E grid marker {key} mismatch")

    expected_selection = {
        "status": "classification_winner_locked_before_retrieval",
        "phase": phase2e.PHASE,
        "study_id": phase2e.STUDY_ID,
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "selection_order": list(phase2e.SELECTION_ORDER),
        "candidate_count": 15,
        "trial_count": 3,
        "retrieval_was_not_used_for_selection": True,
    }
    for key, value in expected_selection.items():
        _require_equal(
            selection.get(key), value, f"Phase 2E selection-lock {key} mismatch"
        )
    if not selection.get("locked_at"):
        raise RuntimeError("Phase 2E selection lock has no lock timestamp")

    candidates = selection.get("all_epoch_candidates")
    if not isinstance(candidates, list) or len(candidates) != 15:
        raise RuntimeError("Phase 2E selection lock must contain 15 candidates")
    observed_keys: set[tuple[str, str]] = set()
    for variant in phase2e.VARIANT_ORDER:
        rows = [row for row in candidates if row.get("variant") == variant]
        if len(rows) != 5:
            raise RuntimeError(f"Phase 2E {variant} must have five candidates")
        steps = tuple(int(row.get("global_step", -1)) for row in rows)
        if steps != phase2e.EXPECTED_STEPS:
            raise RuntimeError(f"Phase 2E {variant} candidate steps drifted: {steps}")
        for row in rows:
            _require_equal(
                row.get("run_id"),
                phase2e.RUN_IDS[variant],
                f"Phase 2E {variant} candidate run mismatch",
            )
            if not math.isclose(
                float(row.get("learning_rate")),
                phase2e.LEARNING_RATES[variant],
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    f"Phase 2E {variant} candidate learning rate drifted"
                )
            key = (variant, str(row.get("checkpoint_id")))
            if not key[1] or key in observed_keys:
                raise RuntimeError("Phase 2E selection candidates are not unique")
            observed_keys.add(key)

    recomputed = phase2e.select_candidate_rows(candidates)
    winner_row = selection.get("winner")
    if not isinstance(winner_row, Mapping):
        raise RuntimeError("Phase 2E selection lock has no winner")
    if _candidate_identity(winner_row) != _candidate_identity(recomputed):
        raise RuntimeError(
            "Locked Phase 2E winner does not recompute from classification metrics"
        )
    variant = str(winner_row.get("variant"))
    if variant not in phase2e.VARIANT_ORDER:
        raise RuntimeError(f"Invalid Phase 2E winner variant: {variant}")
    run_id = phase2e.RUN_IDS[variant]
    learning_rate = phase2e.LEARNING_RATES[variant]
    checkpoint_id = str(winner_row.get("checkpoint_id") or "")
    if not checkpoint_id:
        raise RuntimeError("Phase 2E winner has no checkpoint ID")

    selected_final = json.loads(selected_final_path.read_text(encoding="utf-8"))
    expected_final = {
        "study_id": phase2e.STUDY_ID,
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "variant": variant,
        "learning_rate": learning_rate,
        "run_id": run_id,
        "selected_checkpoint_id": checkpoint_id,
        "retrieval_was_not_used_for_selection": True,
        "global_grid_winner": True,
    }
    for key, value in expected_final.items():
        _require_equal(
            selected_final.get(key),
            value,
            f"Phase 2E selected-final {key} mismatch",
        )
    if selected_final.get("status") not in {
        "selected_checkpoint_final_validation_complete",
        "complete",
    }:
        raise RuntimeError("Phase 2E selected checkpoint lacks final validation")

    winning_root = phase2e.trial_root(root, variant)
    for other in phase2e.VARIANT_ORDER:
        if other == variant:
            continue
        other_retrieval = phase2e.trial_root(root, other) / "retrieval"
        if any(
            (other_retrieval / name).exists()
            for name in ("results.jsonl", "summary.json", "runtime_segments.jsonl")
        ):
            raise RuntimeError(
                "Retrieval artifacts exist for non-winning Phase 2E trial: "
                f"{other}"
            )

    return LockedWinner(
        study_root=root,
        variant=variant,
        run_id=run_id,
        learning_rate=learning_rate,
        checkpoint_id=checkpoint_id,
        trial_root=winning_root,
        selection=selection,
    )


def load_retrieval_context(
    study_root: Path,
    *,
    expected_examples: int = phase2.EXPECTED_COUNTS["validation"],
) -> RetrievalContext:
    winner = load_locked_winner(study_root)
    marker_path = winner.trial_root / "configuration" / "experiment.json"
    if not marker_path.is_file():
        raise RuntimeError(f"Winning Phase 2E trial has no marker: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_marker = {
        "phase": phase2e.PHASE,
        "study_id": phase2e.STUDY_ID,
        "formulation_version": phase2e.FORMULATION_VERSION,
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "variant": winner.variant,
        "learning_rate": winner.learning_rate,
        "epochs": 5,
        "run_id": winner.run_id,
        "model_id": phase2d.MODEL_ID,
        "model_revision": phase2d.MODEL_REVISION,
        "architecture": "AutoModelForSequenceClassification",
        "instruction": phase2d.SUPERVISOR_INSTRUCTION,
        "instruction_sha256": phase2e.PROMPT_SHA256,
        "input_template": "{instruction}\\n\\nQuestion: {original_question_text}",
        "objective": "uniform_five_class_cross_entropy",
    }
    for key, value in expected_marker.items():
        _require_equal(
            marker.get(key), value, f"Winning Phase 2E marker {key} mismatch"
        )

    final_path = winner.trial_root / "final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    expected_final = {
        "run_id": winner.run_id,
        "model_id": phase2d.MODEL_ID,
        "model_revision": phase2d.MODEL_REVISION,
        "formulation_version": phase2e.FORMULATION_VERSION,
        "study_id": phase2e.STUDY_ID,
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "variant": winner.variant,
        "learning_rate": winner.learning_rate,
        "epochs": 5,
        "evaluated_examples": expected_examples,
        "selected_checkpoint_id": winner.checkpoint_id,
        "retrieval_was_not_used_for_selection": True,
        "global_grid_winner": True,
    }
    for key, value in expected_final.items():
        _require_equal(final.get(key), value, f"Phase 2E final-summary {key} mismatch")
    if final.get("status") not in {
        "classification_complete_retrieval_pending",
        "complete",
    }:
        raise RuntimeError(f"Phase 2E classification is incomplete: {final.get('status')}")

    fingerprint = _summary_fingerprint(final)
    if not fingerprint:
        raise RuntimeError("Phase 2E final summary has no experiment fingerprint")
    prediction_path = winner.trial_root / "validation" / "predictions.jsonl"
    predictions = read_jsonl(prediction_path)
    _require_equal(
        len(predictions), expected_examples, "Canonical prediction count mismatch"
    )
    ids = [str(row.get("question_id") or "") for row in predictions]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("Canonical Phase 2E predictions have missing/duplicate IDs")
    valid_status = str(getattr(phase2d, "PREDICTION_STATUS", ""))
    for row in predictions:
        question_id = row["question_id"]
        if not row.get("document_id") or not row.get("question_text"):
            raise RuntimeError(f"Prediction lacks source identity: {question_id}")
        expected = {
            "selected_checkpoint_id": winner.checkpoint_id,
            "experiment_fingerprint": fingerprint,
            "formulation_version": phase2e.FORMULATION_VERSION,
        }
        for key, value in expected.items():
            _require_equal(
                row.get(key), value, f"Prediction {key} mismatch for {question_id}"
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
            raise RuntimeError(f"Prediction has invalid Oracle label: {question_id}")
        if not row.get("selected_checkpoint_path"):
            raise RuntimeError(f"Prediction has no checkpoint path: {question_id}")

    classification = final.get("classification") or {}
    invalid_count = final.get(
        "invalid_output_count",
        final.get("invalid_outputs", classification.get("invalid_output_count")),
    )
    if invalid_count is not None:
        _require_equal(int(invalid_count), 0, "Phase 2E invalid prediction count")
    valid_count = final.get(
        "valid_output_count",
        final.get("valid_outputs", classification.get("valid_output_count")),
    )
    if valid_count is not None:
        _require_equal(int(valid_count), expected_examples, "Valid prediction count")
    if final.get("valid_output_rate") is not None:
        _require_equal(float(final["valid_output_rate"]), 1.0, "Valid-output rate")

    return RetrievalContext(
        winner=winner,
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
    predicted_tokens = _prediction_tokens(prediction)
    predicted_class_id = _prediction_class_id(prediction)
    predicted_level = predicted_class_id + 1
    winner = context.winner
    expected = {
        "method_name": METHOD_NAME,
        "phase2e_run_id": winner.run_id,
        "study_id": phase2e.STUDY_ID,
        "variant": winner.variant,
        "learning_rate": winner.learning_rate,
        "formulation_version": phase2e.FORMULATION_VERSION,
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "experiment_fingerprint": context.experiment_fingerprint,
        "evaluation_run_id": retrieval_run_id(winner.run_id),
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
        "selected_checkpoint_id": winner.checkpoint_id,
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
            "Stale or incompatible Phase 2E retrieval record for "
            f"{prediction['question_id']}: {mismatches}"
        )
    if "phase2e_retrieval_wall_seconds" not in record:
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
    values = [float(row["f1_joined_topk"]) for row in records]
    total = len(context.predictions)
    valid = len(values)
    cumulative_question_wall = float(
        sum(float(row["phase2e_retrieval_wall_seconds"]) for row in records)
    )
    uninterrupted_wall = float(segment_wall_seconds) if uninterrupted else None
    reported_wall = (
        uninterrupted_wall if uninterrupted_wall is not None else cumulative_question_wall
    )
    winner = context.winner
    return {
        "status": "complete",
        "phase": phase2e.PHASE,
        "method_name": METHOD_NAME,
        "study_id": phase2e.STUDY_ID,
        "phase2e_run_id": winner.run_id,
        "variant": winner.variant,
        "learning_rate": winner.learning_rate,
        "formulation_version": phase2e.FORMULATION_VERSION,
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "experiment_fingerprint": context.experiment_fingerprint,
        "evaluation_run_id": retrieval_run_id(winner.run_id),
        "selected_checkpoint_id": winner.checkpoint_id,
        "classification_winner_was_locked_before_retrieval": True,
        "retrieval_was_not_used_for_selection": True,
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
            "Every canonical Phase 2E classifier output is an explicit argmax "
            "over five logits. No default or parser fallback is used; an "
            "invalid prediction would be excluded rather than mapped."
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
            "After resume, the durable per-question sum is reported rather "
            "than fabricating an uninterrupted wall time."
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


def update_final_summaries(
    context: RetrievalContext, summary: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach retrieval only to the winning trial and global selected summary."""

    winner = context.winner
    final_path = winner.trial_root / "final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    _require_equal(final.get("run_id"), winner.run_id, "Final update run mismatch")
    _require_equal(
        final.get("selected_checkpoint_id"),
        winner.checkpoint_id,
        "Final update checkpoint mismatch",
    )
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
        winner.trial_root / "retrieval" / "results.jsonl"
    )
    artifacts["retrieval_summary"] = str(
        winner.trial_root / "retrieval" / "summary.json"
    )
    artifacts["retrieval_runtime_segments"] = str(
        winner.trial_root / "retrieval" / "runtime_segments.jsonl"
    )
    final["completed_at"] = utc_now()
    phase2.atomic_json(final_path, final)

    selected_path = winner.study_root / "comparison" / "selected_final_summary.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    expected = {
        "run_id": winner.run_id,
        "variant": winner.variant,
        "selected_checkpoint_id": winner.checkpoint_id,
        "grid_fingerprint": phase2e.grid_fingerprint(),
    }
    for key, value in expected.items():
        _require_equal(selected.get(key), value, f"Selected-final {key} mismatch")
    selected["status"] = "complete"
    selected["retrieval"] = dict(summary)
    selected.setdefault("artifacts", {}).update(
        {
            "retrieval_results": str(
                winner.trial_root / "retrieval" / "results.jsonl"
            ),
            "retrieval_summary": str(
                winner.trial_root / "retrieval" / "summary.json"
            ),
            "retrieval_runtime_segments": str(
                winner.trial_root / "retrieval" / "runtime_segments.jsonl"
            ),
        }
    )
    selected["completed_at"] = utc_now()
    phase2.atomic_json(selected_path, selected)
    return final, selected


def _validate_completed_summary(
    summary: Mapping[str, Any],
    context: RetrievalContext,
    records: Sequence[Mapping[str, Any]],
) -> None:
    winner = context.winner
    expected = {
        "status": "complete",
        "method_name": METHOD_NAME,
        "study_id": phase2e.STUDY_ID,
        "phase2e_run_id": winner.run_id,
        "variant": winner.variant,
        "learning_rate": winner.learning_rate,
        "formulation_version": phase2e.FORMULATION_VERSION,
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "experiment_fingerprint": context.experiment_fingerprint,
        "selected_checkpoint_id": winner.checkpoint_id,
        "classification_winner_was_locked_before_retrieval": True,
        "retrieval_was_not_used_for_selection": True,
        "evaluated_examples": len(context.predictions),
        "valid_prediction_retrievals": len(records),
        "top_k": TOP_K,
        "paper_restricted": True,
    }
    for key, value in expected.items():
        _require_equal(summary.get(key), value, f"Retrieval-summary {key} mismatch")
    values = [float(row["f1_joined_topk"]) for row in records]
    total = len(context.predictions)
    numeric_expected = {
        "retrieval_coverage": len(records) / total,
        "valid_only_mean_joined_retrieval_f1": float(statistics.fmean(values)),
        "valid_only_median_joined_retrieval_f1": float(statistics.median(values)),
        "coverage_adjusted_full_set_mean_joined_retrieval_f1": float(
            sum(values) / total
        ),
    }
    for key, value in numeric_expected.items():
        actual = summary.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), value, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(f"Retrieval-summary {key} does not recompute")


def evaluate_selected_retrieval(
    study_root: Path,
    *,
    phase1_module: Any | None = None,
    expected_examples: int = phase2.EXPECTED_COUNTS["validation"],
) -> dict[str, Any]:
    """Run/resume unchanged retrieval for the locked Phase 2E winner only."""

    if phase1_module is None:
        import qwen_phase1 as phase1_module

    context = load_retrieval_context(
        study_root, expected_examples=expected_examples
    )
    winner = context.winner
    result_path = winner.trial_root / "retrieval" / "results.jsonl"
    existing = _optional_jsonl(result_path)
    existing_ids = [str(row.get("question_id") or "") for row in existing]
    if any(not value for value in existing_ids) or len(existing_ids) != len(
        set(existing_ids)
    ):
        raise RuntimeError("Phase 2E retrieval JSONL has missing or duplicate IDs")
    prediction_by_id = {row["question_id"]: row for row in context.predictions}
    if not set(existing_ids).issubset(prediction_by_id):
        raise RuntimeError("Retrieval JSONL contains an unknown prediction ID")
    by_id = {row["question_id"]: row for row in existing}
    for question_id, record in by_id.items():
        validate_retrieval_record(
            record, prediction_by_id[question_id], context, phase1_module
        )

    summary_path = winner.trial_root / "retrieval" / "summary.json"
    if len(by_id) == len(context.predictions) and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ordered = [by_id[row["question_id"]] for row in context.predictions]
        _validate_completed_summary(summary, context, ordered)
        update_final_summaries(context, summary)
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
                evaluation_run_id=retrieval_run_id(winner.run_id),
                evaluation_config_hash=RETRIEVAL_CONFIG_HASH,
            )
        )
        if len(generated) != 1:
            raise RuntimeError(f"Expected one retrieval result: {question_id}")
        record = dict(generated[0])
        record.update(
            {
                "method_name": METHOD_NAME,
                "phase2e_run_id": winner.run_id,
                "study_id": phase2e.STUDY_ID,
                "variant": winner.variant,
                "learning_rate": winner.learning_rate,
                "formulation_version": phase2e.FORMULATION_VERSION,
                "grid_fingerprint": phase2e.grid_fingerprint(),
                "experiment_fingerprint": context.experiment_fingerprint,
                "predicted_class_id": predicted_class_id,
                "predicted_granularity_tokens": predicted_tokens,
                "predicted_granularity_level": predicted_level,
                "classifier_prediction_status": prediction.get(
                    "prediction_status", "valid_classifier_argmax"
                ),
                "evidence_length_oracle": int(prediction["oracle_label"]),
                "oracle_label_version": phase2d.ORACLE_VERSION,
                "selected_checkpoint_id": winner.checkpoint_id,
                "selected_checkpoint_path": prediction["selected_checkpoint_path"],
                "model_id": phase2d.MODEL_ID,
                "model_revision": phase2d.MODEL_REVISION,
                "top_k": TOP_K,
                "paper_restricted": True,
                "phase2e_retrieval_wall_seconds": (
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
    segment_path = winner.trial_root / "retrieval" / "runtime_segments.jsonl"
    phase2.append_jsonl(
        segment_path,
        {
            "phase2e_run_id": winner.run_id,
            "study_id": phase2e.STUDY_ID,
            "variant": winner.variant,
            "formulation_version": phase2e.FORMULATION_VERSION,
            "grid_fingerprint": phase2e.grid_fingerprint(),
            "experiment_fingerprint": context.experiment_fingerprint,
            "evaluation_run_id": retrieval_run_id(winner.run_id),
            "records_before_segment": initial_count,
            "new_records": completed_this_segment,
            "records_after_segment": len(ordered),
            "wall_seconds": segment_wall,
            "completed_at": utc_now(),
        },
    )
    segments = read_jsonl(segment_path)
    for segment in segments:
        expected = {
            "phase2e_run_id": winner.run_id,
            "study_id": phase2e.STUDY_ID,
            "variant": winner.variant,
            "formulation_version": phase2e.FORMULATION_VERSION,
            "grid_fingerprint": phase2e.grid_fingerprint(),
            "experiment_fingerprint": context.experiment_fingerprint,
        }
        for key, value in expected.items():
            _require_equal(
                segment.get(key), value, f"Runtime-segment {key} mismatch"
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
    update_final_summaries(context, summary)
    return summary


def _manifest_entries(path: Path) -> list[tuple[str, str]]:
    """Parse one GNU sha256sum manifest without trusting its paths."""

    if not path.is_file():
        raise RuntimeError(f"Required SHA-256 manifest is missing: {path}")
    entries: list[tuple[str, str]] = []
    observed: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or not relative
        ):
            raise RuntimeError(
                f"Malformed SHA-256 manifest line {line_number}: {path}"
            )
        if (
            "\\" in relative
            or relative.startswith("/")
            or any(ord(character) < 32 or ord(character) == 127 for character in relative)
            or any(part in {"", ".", ".."} for part in relative.split("/"))
        ):
            raise RuntimeError(f"Unsafe SHA-256 manifest path: {relative}")
        if relative in observed:
            raise RuntimeError(f"Duplicate SHA-256 manifest path: {relative}")
        observed.add(relative)
        entries.append((digest.lower(), relative))
    if not entries:
        raise RuntimeError(f"SHA-256 manifest is empty: {path}")
    return entries


def _manifest_target(base: Path, relative: str) -> Path:
    base = base.resolve()
    target = (base / Path(*relative.split("/"))).resolve()
    try:
        target.relative_to(base)
    except ValueError as error:
        raise RuntimeError(f"SHA-256 manifest path escapes its root: {relative}") from error
    return target


def _verify_exact_manifest(
    path: Path,
    base: Path,
    *,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    entries = _manifest_entries(path)
    if expected_count is not None and len(entries) != expected_count:
        raise RuntimeError(
            f"SHA-256 manifest count mismatch for {path}: "
            f"expected {expected_count}, got {len(entries)}"
        )
    verified: list[dict[str, Any]] = []
    for expected, relative in entries:
        target = _manifest_target(base, relative)
        if not target.is_file():
            raise RuntimeError(f"Manifest-backed file is missing: {relative}")
        actual = phase2.sha256_file(target)
        if actual != expected:
            raise RuntimeError(
                f"Manifest-backed file SHA-256 mismatch: {relative}"
            )
        verified.append(
            {
                "path": relative,
                "sha256": actual,
                "bytes": target.stat().st_size,
            }
        )
    return verified


def _read_transfer_inventory(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Phase 2E transfer inventory is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "variant",
        "run_id",
        "checkpoint_id",
        "relative_path",
        "archive_bytes",
        "archive_sha256",
        "chunk_count",
    }
    if len(rows) != 3 or any(set(row) != required for row in rows):
        raise RuntimeError("Transfer inventory must contain the exact three-trial schema")
    if [row["variant"] for row in rows] != list(phase2e.VARIANT_ORDER):
        raise RuntimeError("Transfer inventory variant order or identity drifted")
    return rows


def _verify_transfer_and_checkpoints(
    context: RetrievalContext,
) -> dict[str, Any]:
    root = context.winner.study_root
    transfer_root = root / "integrity" / "transfer_manifests"
    manifests = transfer_root / "manifests"
    expected_manifest_names = {
        "metadata_archive.sha256",
        "metadata_chunks.sha256",
        "metadata_files.sha256",
        "transfer_inventory.tsv",
    }
    for variant in phase2e.VARIANT_ORDER:
        expected_manifest_names.update(
            {
                f"{variant}_archive.sha256",
                f"{variant}_chunks.sha256",
                f"{variant}_selected_checkpoint_files.sha256",
            }
        )
    bundle = _verify_exact_manifest(
        transfer_root / "manifest_bundle.sha256",
        transfer_root,
        expected_count=len(expected_manifest_names),
    )
    bundled_names = {
        str(Path(row["path"]).relative_to("manifests")).replace("\\", "/")
        for row in bundle
        if row["path"].startswith("manifests/")
    }
    if bundled_names != expected_manifest_names or len(bundle) != len(bundled_names):
        raise RuntimeError("Transfer manifest bundle inventory drifted")

    verification_path = (
        root / "integrity" / "selected_checkpoints_transfer_verification.json"
    )
    if not verification_path.is_file():
        raise RuntimeError("Selected-checkpoint transfer verification is missing")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    expected_verification = {
        "status": "passed",
        "study": root.name,
        "metadata_and_all_three_selected_checkpoints_verified": True,
        "all_remote_and_local_hashes_match": True,
        "remote_originals_retained": True,
    }
    for key, value in expected_verification.items():
        _require_equal(
            verification.get(key), value, f"Transfer-verification {key} mismatch"
        )
    if not verification.get("verified_at"):
        raise RuntimeError("Transfer verification has no timestamp")

    inventory_path = manifests / "transfer_inventory.tsv"
    inventory = _read_transfer_inventory(inventory_path)
    recorded_variants = verification.get("variants")
    if not isinstance(recorded_variants, list) or recorded_variants != inventory:
        raise RuntimeError("Transfer-verification variants differ from inventory")

    checkpoint_audits: list[dict[str, Any]] = []
    total_checkpoint_files = 0
    for row in inventory:
        variant = row["variant"]
        run_id = phase2e.RUN_IDS[variant]
        if row["run_id"] != run_id:
            raise RuntimeError(f"Transfer run identity mismatch: {variant}")
        checkpoint_id = row["checkpoint_id"]
        if not checkpoint_id.startswith("step-") or len(checkpoint_id) != 11:
            raise RuntimeError(f"Unsafe transfer checkpoint ID: {checkpoint_id}")
        expected_relative = (
            f"{root.name}/trials/{variant}/runs/{run_id}/checkpoints/"
            f"{checkpoint_id}"
        )
        if row["relative_path"] != expected_relative:
            raise RuntimeError(f"Transfer checkpoint path mismatch: {variant}")
        if int(row["archive_bytes"]) <= 0 or int(row["chunk_count"]) <= 0:
            raise RuntimeError(f"Invalid transfer archive inventory: {variant}")

        archive_entries = _manifest_entries(manifests / f"{variant}_archive.sha256")
        expected_archive = f"archives/phase2e-{variant}-{checkpoint_id}.tar.zst"
        if archive_entries != [(row["archive_sha256"], expected_archive)]:
            raise RuntimeError(f"Transfer archive manifest mismatch: {variant}")
        chunk_entries = _manifest_entries(manifests / f"{variant}_chunks.sha256")
        expected_chunks = [
            (
                f"{variant}_chunks/phase2e-{variant}-{checkpoint_id}.tar.zst."
                f"part-{index:03d}"
            )
            for index in range(int(row["chunk_count"]))
        ]
        if [relative for _, relative in chunk_entries] != expected_chunks:
            raise RuntimeError(f"Transfer chunk manifest mismatch: {variant}")

        checkpoint_manifest = (
            manifests / f"{variant}_selected_checkpoint_files.sha256"
        )
        verified = _verify_exact_manifest(
            checkpoint_manifest,
            root.parent,
            expected_count=9,
        )
        prefix = f"{expected_relative}/"
        if any(not item["path"].startswith(prefix) for item in verified):
            raise RuntimeError(f"Checkpoint manifest path mismatch: {variant}")
        checkpoint_root = root.parent / Path(*expected_relative.split("/"))
        actual_files = {
            path.resolve()
            for path in checkpoint_root.rglob("*")
            if path.is_file()
        }
        manifested_files = {
            _manifest_target(root.parent, item["path"]) for item in verified
        }
        if actual_files != manifested_files:
            raise RuntimeError(f"Checkpoint file inventory drifted: {variant}")

        run_root = root / "trials" / variant / "runs" / run_id
        best = json.loads((run_root / "best_checkpoint.json").read_text("utf-8"))
        run_summary = json.loads((run_root / "summary.json").read_text("utf-8"))
        _require_equal(
            best.get("checkpoint_id"), checkpoint_id, f"Best checkpoint {variant}"
        )
        _require_equal(
            run_summary.get("selected_checkpoint_id"),
            checkpoint_id,
            f"Run-summary checkpoint {variant}",
        )
        if variant == context.winner.variant:
            _require_equal(
                checkpoint_id,
                context.winner.checkpoint_id,
                "Global winner transfer checkpoint",
            )
        total_checkpoint_files += len(verified)
        checkpoint_audits.append(
            {
                "variant": variant,
                "run_id": run_id,
                "checkpoint_id": checkpoint_id,
                "relative_path": expected_relative,
                "archive_bytes": int(row["archive_bytes"]),
                "archive_sha256": row["archive_sha256"],
                "chunk_count": int(row["chunk_count"]),
                "verified_file_count": len(verified),
            }
        )
    if total_checkpoint_files != 27:
        raise RuntimeError("Phase 2E must preserve exactly 27 checkpoint files")

    forbidden_directories = {
        "tensorboard",
        "checkpoint_archives",
        "archives",
        "metadata_chunks",
        *(f"{variant}_chunks" for variant in phase2e.VARIANT_ORDER),
    }
    staged = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if (
            path.is_symlink()
            or (path.is_dir() and path.name in forbidden_directories)
            or (
                path.is_file()
                and (path.name.endswith(".tar.zst") or ".tar.zst.part-" in path.name)
            )
        )
    ]
    if staged:
        raise RuntimeError(f"Forbidden transfer/TensorBoard payload in study: {staged}")

    return {
        "verification_record": str(verification_path),
        "verification_record_sha256": phase2.sha256_file(verification_path),
        "verification_status": verification["status"],
        "verification_timestamp": verification["verified_at"],
        "manifest_bundle": str(transfer_root / "manifest_bundle.sha256"),
        "manifest_bundle_sha256": phase2.sha256_file(
            transfer_root / "manifest_bundle.sha256"
        ),
        "manifest_files_verified": len(bundle),
        "checkpoint_files_verified": total_checkpoint_files,
        "checkpoints": checkpoint_audits,
        "forbidden_payload_count": 0,
    }


def _audit_metadata_after_retrieval(
    context: RetrievalContext,
    *,
    expected_metadata_files: int | None,
) -> dict[str, Any]:
    root = context.winner.study_root
    manifest_path = (
        root
        / "integrity"
        / "transfer_manifests"
        / "manifests"
        / "metadata_files.sha256"
    )
    entries = _manifest_entries(manifest_path)
    if expected_metadata_files is not None and len(entries) != expected_metadata_files:
        raise RuntimeError(
            "Transferred metadata manifest count mismatch: "
            f"expected {expected_metadata_files}, got {len(entries)}"
        )
    allowed = {
        f"{root.name}/comparison/selected_final_summary.json",
        (
            f"{root.name}/trials/{context.winner.variant}/"
            "final_summary.json"
        ),
    }
    matched = 0
    changes: list[dict[str, Any]] = []
    for expected, relative in entries:
        target = _manifest_target(root.parent, relative)
        if not target.is_file():
            raise RuntimeError(f"Transferred metadata file is missing: {relative}")
        actual = phase2.sha256_file(target)
        if actual == expected:
            matched += 1
            continue
        if relative not in allowed:
            raise RuntimeError(f"Unexpected post-transfer metadata drift: {relative}")
        changes.append(
            {
                "path": relative,
                "pre_retrieval_sha256": expected,
                "current_sha256": actual,
                "reason": "authorized_post_retrieval_summary_update",
            }
        )
    changed_paths = {item["path"] for item in changes}
    if changed_paths != allowed:
        raise RuntimeError(
            "Retrieval-complete metadata must change exactly the winning-trial "
            "and selected-final summaries"
        )
    return {
        "manifest": str(manifest_path),
        "manifest_entries": len(entries),
        "unchanged_entries_verified": matched,
        "authorized_changed_entries": len(changes),
        "authorized_changes": sorted(changes, key=lambda item: item["path"]),
        "all_other_metadata_matches_transfer_manifest": True,
    }


def audit_final_post_retrieval(
    study_root: Path,
    *,
    phase1_module: Any | None = None,
    expected_examples: int = phase2.EXPECTED_COUNTS["validation"],
    expected_metadata_files: int | None = 64,
) -> dict[str, Any]:
    """Audit the promoted Phase 2E study after retrieval, without rerunning it."""

    if phase1_module is None:
        import qwen_phase1 as phase1_module

    context = load_retrieval_context(
        study_root, expected_examples=expected_examples
    )
    winner = context.winner
    result_path = winner.trial_root / "retrieval" / "results.jsonl"
    summary_path = winner.trial_root / "retrieval" / "summary.json"
    if not result_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Phase 2E final audit requires complete retrieval artifacts")
    records = read_jsonl(result_path)
    if len(records) != expected_examples:
        raise RuntimeError(
            f"Retrieval result count mismatch: expected {expected_examples}, "
            f"got {len(records)}"
        )
    record_ids = [str(row.get("question_id") or "") for row in records]
    prediction_ids = [str(row["question_id"]) for row in context.predictions]
    if (
        any(not question_id for question_id in record_ids)
        or len(record_ids) != len(set(record_ids))
        or record_ids != prediction_ids
    ):
        raise RuntimeError(
            "Retrieval results must contain the canonical unique question IDs "
            "in preserved order"
        )
    for record, prediction in zip(records, context.predictions):
        validate_retrieval_record(record, prediction, context, phase1_module)
        value = record.get("f1_joined_topk")
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise RuntimeError(f"Invalid retrieval F1: {record['question_id']}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_completed_summary(summary, context, records)
    summary_identity = {
        "evaluation_config_hash": RETRIEVAL_CONFIG_HASH,
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "metric_version": RETRIEVAL_METRIC_VERSION,
        "normalization_version": RETRIEVAL_NORMALIZATION_VERSION,
        "metric": "f1_joined_topk",
        "top_k": TOP_K,
        "paper_restricted": True,
        "embedding_model": phase1_module.OPENAI_EMBEDDING_MODEL,
        "embedding_dimension": phase1_module.EMBEDDING_DIM,
        "tokenizer": phase1_module.TOKENIZER_NAME,
        "valid_prediction_retrievals": expected_examples,
        "invalid_predictions_without_retrieval": 0,
        "retrieval_coverage": 1.0,
    }
    for key, value in summary_identity.items():
        _require_equal(summary.get(key), value, f"Final retrieval {key} mismatch")

    selected_path = winner.study_root / "comparison" / "selected_final_summary.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if context.final_summary.get("status") != "complete" or selected.get("status") != "complete":
        raise RuntimeError("Retrieval-complete final summaries are not complete")
    _require_equal(
        context.final_summary.get("retrieval"), summary, "Winning final retrieval"
    )
    _require_equal(selected.get("retrieval"), summary, "Selected-final retrieval")
    _require_equal(
        selected.get("classification"),
        winner.selection["winner"]["classification_metrics"],
        "Selected-final classification",
    )

    transfer = _verify_transfer_and_checkpoints(context)
    metadata = _audit_metadata_after_retrieval(
        context, expected_metadata_files=expected_metadata_files
    )
    values = [float(row["f1_joined_topk"]) for row in records]
    audit_path = winner.study_root / "integrity" / "final_post_retrieval_audit.json"
    audit = {
        "status": "passed",
        "phase": phase2e.PHASE,
        "study_id": phase2e.STUDY_ID,
        "study_root": str(winner.study_root),
        "grid_fingerprint": phase2e.grid_fingerprint(),
        "selection": {
            "status": "classification_winner_lock_verified",
            "variant": winner.variant,
            "learning_rate": winner.learning_rate,
            "run_id": winner.run_id,
            "selected_checkpoint_id": winner.checkpoint_id,
            "candidate_count": len(winner.selection["all_epoch_candidates"]),
            "retrieval_was_not_used_for_selection": True,
        },
        "retrieval": {
            "status": "complete_and_recomputed",
            "results": str(result_path),
            "summary": str(summary_path),
            "records": len(records),
            "unique_question_ids": len(set(record_ids)),
            "retrieval_coverage": len(records) / expected_examples,
            "mean_joined_retrieval_f1": float(statistics.fmean(values)),
            "median_joined_retrieval_f1": float(statistics.median(values)),
            "top_k": TOP_K,
            "paper_restricted": True,
            "evaluation_config_hash": RETRIEVAL_CONFIG_HASH,
        },
        "transfer": transfer,
        "metadata": metadata,
        "experiment_rerun": False,
        "retrieval_rerun": False,
        "qdrant_mutation": False,
        "completed_at": utc_now(),
        "artifact": str(audit_path),
    }
    phase2.atomic_json(audit_path, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root", type=Path, default=phase2e.DEFAULT_STUDY_ROOT
    )
    parser.add_argument(
        "command", choices=("audit-selected", "retrieve-selected", "audit-final")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit-selected":
        winner = load_locked_winner(args.study_root)
        result = {
            "status": "classification_winner_lock_verified",
            "study_id": phase2e.STUDY_ID,
            "variant": winner.variant,
            "learning_rate": winner.learning_rate,
            "run_id": winner.run_id,
            "selected_checkpoint_id": winner.checkpoint_id,
            "retrieval_was_not_used_for_selection": True,
        }
    elif args.command == "retrieve-selected":
        result = evaluate_selected_retrieval(args.study_root)
    else:
        result = audit_final_post_retrieval(args.study_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
