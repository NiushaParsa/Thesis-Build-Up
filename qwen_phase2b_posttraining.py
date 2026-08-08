#!/usr/bin/env python
"""Post-training retrieval and comparison utilities for isolated Phase 2B runs.

This module never trains a model and never mutates Qdrant.  It reads canonical
Phase 2B predictions, runs the unchanged Phase 1 same-paper top-k=5 retrieval
path, writes only inside the selected Phase 2B output root, and can compare the
four Qwen summary artifacts that use the evidence-length Oracle.
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


PHASE1_SUMMARY = (
    phase2.PHASE1_ROOT / "final_summary.json"
)
PHASE2_SUMMARY = (
    phase2.DEFAULT_OUTPUT_ROOT / "final_summary.json"
)
ALIAS_UNWEIGHTED_SUMMARY = (
    phase2b.DEFAULT_OUTPUT_ROOTS[phase2b.VARIANT_UNWEIGHTED]
    / "final_summary.json"
)
ALIAS_CLASSBALANCED_SUMMARY = (
    phase2b.DEFAULT_OUTPUT_ROOTS[phase2b.VARIANT_CLASSBALANCED]
    / "final_summary.json"
)
TOP_K = 5
# Frozen retrieval identities are local on purpose.  The remote Phase 2
# training helper predates Phase 2's post-training additions and does not
# expose these names; Phase 2B post-training must not require modifying it.
RETRIEVAL_CONFIG_HASH = (
    "9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8"
)
RETRIEVAL_SCHEMA_VERSION = 2
RETRIEVAL_METRIC_VERSION = "qasper-token-prf-v2"
RETRIEVAL_NORMALIZATION_VERSION = (
    "lowercase-remove-punctuation-collapse-whitespace-v1"
)
METHOD_NAMES = {
    phase2b.VARIANT_UNWEIGHTED: (
        "qwen-phase2b-alias-unweighted-full-parameter-router"
    ),
    phase2b.VARIANT_CLASSBALANCED: (
        "qwen-phase2b-alias-classbalanced-full-parameter-router"
    ),
}


@dataclass(frozen=True)
class RetrievalContext:
    output_root: Path
    variant: str
    run_id: str
    checkpoint_id: str
    experiment_fingerprint: str
    predictions: tuple[dict[str, Any], ...]
    final_summary: dict[str, Any]


def utc_now() -> str:
    return phase2.utc_now()


def resolve_output_root(variant: str, output_root: Path | None) -> Path:
    """Resolve an isolated Phase 2B root and reject legacy experiment roots."""
    if variant not in phase2b.VARIANTS:
        raise ValueError(f"Unsupported Phase 2B variant: {variant}")
    root = Path(output_root or phase2b.DEFAULT_OUTPUT_ROOTS[variant])
    forbidden = {
        phase2.PHASE1_ROOT.resolve(),
        phase2.DEFAULT_OUTPUT_ROOT.resolve(),
    }
    if root.resolve() in forbidden:
        raise RuntimeError("Phase 2B post-training output must use an isolated root")
    return root


def _optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return read_jsonl(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a non-empty JSONL artifact without Phase 2 posttraining helpers."""

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"JSONL artifact is empty: {path}")
    return rows


def _require_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{message}: expected {expected!r}, got {actual!r}")


def load_retrieval_context(
    output_root: Path,
    variant: str,
    run_id: str,
    *,
    expected_examples: int = phase2.EXPECTED_COUNTS["validation"],
) -> RetrievalContext:
    """Validate canonical Phase 2B predictions and their run provenance."""
    output_root = resolve_output_root(variant, output_root)
    final_path = output_root / "final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    _require_equal(final.get("variant"), variant, "Final-summary variant mismatch")
    _require_equal(final.get("run_id"), run_id, "Final-summary run mismatch")
    if final.get("status") not in {
        "classification_complete_retrieval_pending",
        "complete",
    }:
        raise RuntimeError(f"Phase 2B classification is incomplete: {final.get('status')}")
    _require_equal(
        int(final.get("evaluated_examples", -1)),
        expected_examples,
        "Final-summary example count mismatch",
    )

    checkpoint_id = str(final.get("selected_checkpoint_id", ""))
    if not checkpoint_id:
        raise RuntimeError("Final summary has no selected checkpoint ID")
    training = final.get("training") or {}
    fingerprint = str(training.get("experiment_fingerprint", ""))
    if not fingerprint:
        raise RuntimeError("Final summary has no Phase 2B experiment fingerprint")

    prediction_path = output_root / "validation" / "predictions.jsonl"
    predictions = read_jsonl(prediction_path)
    _require_equal(
        len(predictions), expected_examples, "Canonical prediction count mismatch"
    )
    ids = [str(row.get("question_id", "")) for row in predictions]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("Canonical Phase 2B predictions have missing or duplicate IDs")

    for row in predictions:
        question_id = row["question_id"]
        _require_equal(
            row.get("phase2b_variant"),
            variant,
            f"Prediction variant mismatch for {question_id}",
        )
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
        prediction = row.get("parsed_prediction")
        if prediction is None or int(prediction) not in phase2b.CHUNK_SIZES:
            raise RuntimeError(
                f"Canonical Phase 2B prediction is not a mapped chunk class: {question_id}"
            )
        alias = int(row.get("predicted_alias", -1))
        if phase2b.ALIAS_TO_CHUNK.get(alias) != int(prediction):
            raise RuntimeError(f"Alias-to-chunk mismatch for {question_id}")
        if not row.get("selected_checkpoint_path"):
            raise RuntimeError(f"Prediction has no checkpoint path: {question_id}")

    return RetrievalContext(
        output_root=output_root,
        variant=variant,
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
    """Reject stale, cross-variant, or differently configured retrieval rows."""
    predicted_tokens = int(prediction["parsed_prediction"])
    predicted_level = phase2b.CHUNK_SIZES.index(predicted_tokens) + 1
    expected = {
        "method_name": METHOD_NAMES[context.variant],
        "phase2b_run_id": context.run_id,
        "phase2b_variant": context.variant,
        "formulation_version": phase2b.FORMULATION_VERSION,
        "experiment_fingerprint": context.experiment_fingerprint,
        "evaluation_run_id": retrieval_run_id(context.run_id),
        "question_id": prediction["question_id"],
        "document_id": prediction["document_id"],
        "split": "validation",
        "granularity_tokens": predicted_tokens,
        "granularity_level": predicted_level,
        "predicted_granularity_tokens": predicted_tokens,
        "predicted_granularity_level": predicted_level,
        "predicted_alias": int(prediction["predicted_alias"]),
        "evidence_length_oracle": int(prediction["oracle_label"]),
        "oracle_label_version": phase2.ORACLE_VERSION,
        "selected_checkpoint_id": context.checkpoint_id,
        "selected_checkpoint_path": prediction["selected_checkpoint_path"],
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
            f"Stale or incompatible Phase 2B retrieval record for "
            f"{prediction['question_id']}: {mismatches}"
        )
    if "phase2b_retrieval_wall_seconds" not in record:
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
    """Build the complete JSON-safe retrieval summary from durable records."""
    values = [float(row["f1_joined_topk"]) for row in records]
    total = len(context.predictions)
    valid = len(values)
    cumulative_question_wall = float(
        sum(float(row["phase2b_retrieval_wall_seconds"]) for row in records)
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
        "variant": context.variant,
        "method_name": METHOD_NAMES[context.variant],
        "formulation_version": phase2b.FORMULATION_VERSION,
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
            "Only canonical mapped Phase 2B predictions are retrieved. Invalid "
            "outputs would receive no default granularity and no retrieval record."
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
            "Every completed question stores an fsynced processing duration. "
            "After a resume, the durable per-question sum is reported instead "
            "of fabricating an uninterrupted wall time."
        ),
        "completed_at": utc_now(),
    }


def _known_pre_retrieval_wall(final: Mapping[str, Any]) -> float | None:
    training = (final.get("training") or {}).get("elapsed_seconds")
    runtime = final.get("runtime") or {}
    load = runtime.get("model_load_seconds")
    inference = runtime.get("isolated_inference_wall_seconds")
    if all(isinstance(value, (int, float)) for value in (training, load, inference)):
        return float(training) + float(load) + float(inference)
    return None


def update_final_summary(
    context: RetrievalContext, summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach retrieval results without replacing classification provenance."""
    final_path = context.output_root / "final_summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    _require_equal(final.get("variant"), context.variant, "Final update variant mismatch")
    _require_equal(final.get("run_id"), context.run_id, "Final update run mismatch")
    final["status"] = "complete"
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
        "variant": context.variant,
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
    variant: str,
    run_id: str,
    *,
    phase1_module: Any | None = None,
    expected_examples: int = phase2.EXPECTED_COUNTS["validation"],
) -> dict[str, Any]:
    """Run or resume unchanged read-only Qdrant retrieval for one Phase 2B run."""
    if phase1_module is None:
        import qwen_phase1 as phase1_module

    context = load_retrieval_context(
        output_root, variant, run_id, expected_examples=expected_examples
    )
    result_path = context.output_root / "retrieval" / "results.jsonl"
    existing = _optional_jsonl(result_path)
    existing_ids = [str(row.get("question_id", "")) for row in existing]
    if any(not value for value in existing_ids) or len(existing_ids) != len(
        set(existing_ids)
    ):
        raise RuntimeError("Phase 2B retrieval JSONL has missing or duplicate IDs")
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
        predicted_tokens = int(prediction["parsed_prediction"])
        predicted_level = phase2b.CHUNK_SIZES.index(predicted_tokens) + 1
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
                chunk_sizes=list(phase2b.CHUNK_SIZES),
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
                "method_name": METHOD_NAMES[context.variant],
                "phase2b_run_id": context.run_id,
                "phase2b_variant": context.variant,
                "formulation_version": phase2b.FORMULATION_VERSION,
                "experiment_fingerprint": context.experiment_fingerprint,
                "predicted_alias": int(prediction["predicted_alias"]),
                "predicted_granularity_tokens": predicted_tokens,
                "predicted_granularity_level": predicted_level,
                "qwen_raw_output": prediction["raw_qwen_output"],
                "qwen_prediction_status": prediction["prediction_status"],
                "evidence_length_oracle": int(prediction["oracle_label"]),
                "oracle_label_version": phase2.ORACLE_VERSION,
                "selected_checkpoint_id": context.checkpoint_id,
                "selected_checkpoint_path": prediction[
                    "selected_checkpoint_path"
                ],
                "top_k": TOP_K,
                "paper_restricted": True,
                "phase2b_retrieval_wall_seconds": (
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
            "variant": context.variant,
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
            segment.get("variant"), context.variant, "Runtime-segment variant mismatch"
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


def _metric(retrieval: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in retrieval:
            return retrieval[name]
    return None


def normalized_comparison_row(
    name: str, path: Path, *, expected_variant: str | None = None
) -> dict[str, Any]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError(f"Comparison source is incomplete: {path}")
    if expected_variant is not None and summary.get("variant") != expected_variant:
        raise RuntimeError(f"Comparison source variant mismatch: {path}")
    classification = summary.get("classification") or {}
    retrieval = summary.get("retrieval") or {}
    if not classification or not retrieval:
        raise RuntimeError(f"Comparison source lacks metrics: {path}")
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
        "variant": summary.get("variant"),
        "run_id": summary.get("run_id"),
        "model_id": summary.get("model_id"),
        "model_revision": summary.get("model_revision"),
        "evaluated_examples": int(summary["evaluated_examples"]),
        "oracle_distribution": summary["oracle_distribution"],
        "predicted_distribution": summary["predicted_distribution"],
        "valid_output_rate": float(summary["valid_output_rate"]),
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
        "known_total_runtime_seconds": total_runtime,
    }


def compare_summaries(
    phase1_summary: Path,
    phase2_summary: Path,
    alias_unweighted_summary: Path,
    alias_classbalanced_summary: Path,
) -> dict[str, Any]:
    """Normalize and compare the four directly new-Oracle Qwen summaries."""
    rows = [
        normalized_comparison_row("phase1_zero_shot", phase1_summary),
        normalized_comparison_row("phase2_numeric_sft", phase2_summary),
        normalized_comparison_row(
            "phase2b_alias_unweighted",
            alias_unweighted_summary,
            expected_variant=phase2b.VARIANT_UNWEIGHTED,
        ),
        normalized_comparison_row(
            "phase2b_alias_classbalanced",
            alias_classbalanced_summary,
            expected_variant=phase2b.VARIANT_CLASSBALANCED,
        ),
    ]
    counts = {row["evaluated_examples"] for row in rows}
    distributions = {
        json.dumps(row["oracle_distribution"], sort_keys=True) for row in rows
    }
    if len(counts) != 1 or len(distributions) != 1:
        raise RuntimeError("Comparison summaries do not use the same validation set")
    for row in rows:
        if row["top_k"] != TOP_K or row["paper_restricted"] is not True:
            raise RuntimeError("Comparison retrieval configurations are not aligned")

    def winners(metric: str) -> list[str]:
        available = [row for row in rows if row[metric] is not None]
        best = max(float(row[metric]) for row in available)
        return [row["name"] for row in available if math.isclose(float(row[metric]), best)]

    return {
        "status": "complete",
        "comparison": (
            "Qwen Phase 1, numeric-target Phase 2, and the two Phase 2B "
            "restricted-alias variants under the same evidence-length Oracle"
        ),
        "evaluated_examples": next(iter(counts)),
        "oracle_distribution": rows[0]["oracle_distribution"],
        "comparability_note": (
            "All four rows use the same preserved validation questions, Oracle, "
            "chunk classes, and downstream retrieval. Phase 2B deliberately "
            "changes target encoding, decision rule, and (for one variant) loss "
            "weighting, so it is a controlled method comparison rather than an "
            "identical-prompt replication."
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
    retrieval.add_argument("--variant", choices=phase2b.VARIANTS, required=True)
    retrieval.add_argument("--run-id", required=True)
    retrieval.add_argument("--output-root", type=Path)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--phase1-summary", type=Path, default=PHASE1_SUMMARY)
    comparison.add_argument("--phase2-summary", type=Path, default=PHASE2_SUMMARY)
    comparison.add_argument(
        "--alias-unweighted-summary",
        type=Path,
        default=ALIAS_UNWEIGHTED_SUMMARY,
    )
    comparison.add_argument(
        "--alias-classbalanced-summary",
        type=Path,
        default=ALIAS_CLASSBALANCED_SUMMARY,
    )
    comparison.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate-retrieval":
        root = resolve_output_root(args.variant, args.output_root)
        result = evaluate_retrieval(root, args.variant, args.run_id)
    else:
        result = compare_summaries(
            args.phase1_summary,
            args.phase2_summary,
            args.alias_unweighted_summary,
            args.alias_classbalanced_summary,
        )
        if args.output is not None:
            phase2.atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
