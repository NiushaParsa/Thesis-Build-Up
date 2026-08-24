#!/usr/bin/env python
"""Phase 4 retrieval-utility-aware routing by expected regret minimization.

The router reuses the clean Phase 3C-OOF representation: five paper-grouped
OOF Qwen logits for training (a frozen full-refit Qwen model for validation)
and 173 inference-safe similarity-tree features.  Gold evidence is used only
to construct the five joined-top-5 retrieval utilities for training.  Five
fixed XGBoost regressors estimate the conditional regret of each available
granularity and the deployed action is the one with minimum predicted regret.

There is deliberately no Phase 4 hyperparameter search.  The shallow tree
settings are inherited from the frozen Phase 3C-OOF fusion model.  Validation
utility records are opened only after validation predictions have been saved
and hashed, so they cannot affect model fitting or action selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import qwen_phase3c_oof as phase3c
import similarity_tree_phase3b as phase3b


PHASE = "Phase 4"
FORMULATION_VERSION = "retrieval-utility-expected-regret-xgboost-v1"
CLASS_TOKENS = phase3b.CLASS_TOKENS
SEED = phase3b.SEED
FOLDS = phase3b.FOLDS
TOP_K = phase3b.TOP_K
EVALUATION_CONFIG_HASH = phase3b.SOURCE_ORACLE_CONFIG_HASH
FEATURE_COUNT = phase3c.FUSION_FEATURE_COUNT
OUTPUT_ROOT = Path("outputs/qwen_phase4_expected_regret_retrieval_utility")
REPORT_ROOT = Path("reports/qwen_phase4_expected_regret_retrieval_utility")
DOC_PATH = Path("docs/QWEN_PHASE4_RESULTS.md")

TRAIN_UTILITY_PATHS = (
    Path("outputs/oracle_frozen/train/RouterDataset_20260622_202111.jsonl"),
    Path(
        "outputs/oracle_frozen/train_rerun_incomplete/"
        "RouterDataset_20260623_171254.jsonl"
    ),
)
VALIDATION_UTILITY_PATH = Path(
    "outputs/oracle_frozen/validation/RouterDataset_20260623_171712.jsonl"
)

FIXED_CANDIDATE = dict(phase3c.FIXED_CANDIDATE)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42

BASELINE_PREDICTIONS = {
    "phase2d": Path(
        "outputs/qwen_phase2d_sequence_classifier_token_count_prompt_"
        "evidence_length_oracle/validation/predictions.jsonl"
    ),
    "phase3b": Path(
        "outputs/similarity_tree_phase3b_xgboost_evidence_length_oracle/"
        "validation/predictions.jsonl"
    ),
    "original_phase3c": Path(
        "outputs/qwen_phase3c_fusion_evidence_length_oracle/"
        "validation/predictions.jsonl"
    ),
    "phase3c_oof": Path(
        "outputs/qwen_phase3c_oof_fusion_evidence_length_oracle/"
        "validation/predictions.jsonl"
    ),
}


def utc_now() -> str:
    return phase3c.utc_now()


def read_jsonl(path: Path) -> list[dict]:
    return phase3b.read_jsonl(path)


def atomic_json(path: Path, value: Any) -> None:
    phase3b.atomic_json(path, value)


def atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    phase3b.atomic_jsonl(path, rows)


def stable_hash(value: Any) -> str:
    return phase3c.stable_hash(value)


def source_hashes() -> dict[str, str]:
    paths = {
        "train_features": phase3b.feature_path(phase3b.SOURCE_FEATURE_ROOT, "train"),
        "validation_features": phase3b.feature_path(
            phase3b.SOURCE_FEATURE_ROOT, "validation"
        ),
        "train_oof_qwen_logits": (
            phase3c.DEFAULT_OUTPUT_ROOT / "qwen_features" / "train_oof_logits.npz"
        ),
        "validation_qwen_logits": (
            phase3c.DEFAULT_OUTPUT_ROOT / "qwen_features" / "validation_logits.npz"
        ),
        "train_utilities_main": TRAIN_UTILITY_PATHS[0],
        "train_utilities_recovery": TRAIN_UTILITY_PATHS[1],
        "validation_utilities": VALIDATION_UTILITY_PATH,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 4 inputs: {missing}")
    return {name: phase3b.sha256_file(path) for name, path in paths.items()}


def validate_utility_row(row: Mapping[str, Any], split: str) -> None:
    if str(row.get("split")) != split:
        raise RuntimeError(
            f"Utility split mismatch for {row.get('question_id')}: {row.get('split')}"
        )
    if str(row.get("evaluation_config_hash")) != EVALUATION_CONFIG_HASH:
        raise RuntimeError(
            f"Retrieval configuration changed for {row.get('question_id')}"
        )
    metrics = row.get("per_granularity_metrics")
    if not isinstance(metrics, list) or len(metrics) != len(CLASS_TOKENS):
        raise RuntimeError(f"Incomplete utility vector: {row.get('question_id')}")
    tokens = [int(item["granularity_tokens"]) for item in metrics]
    if tokens != list(CLASS_TOKENS):
        raise RuntimeError(
            f"Utility order changed for {row.get('question_id')}: {tokens}"
        )
    utilities = np.asarray(
        [float(item["f1_joined_topk"]) for item in metrics], dtype=np.float64
    )
    if not np.isfinite(utilities).all() or np.any(utilities < 0) or np.any(utilities > 1):
        raise RuntimeError(f"Invalid utility values: {row.get('question_id')}")


def load_train_utility_rows() -> list[dict]:
    rows: list[dict] = []
    for path in TRAIN_UTILITY_PATHS:
        rows.extend(read_jsonl(path))
    if len(rows) != 2245:
        raise RuntimeError(f"Expected 2,245 train utility rows, found {len(rows)}")
    ids = [str(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate train utility question IDs")
    for row in rows:
        validate_utility_row(row, "train")
    return rows


def load_validation_utility_rows() -> list[dict]:
    rows = read_jsonl(VALIDATION_UTILITY_PATH)
    if len(rows) != 924:
        raise RuntimeError(f"Expected 924 validation utility rows, found {len(rows)}")
    ids = [str(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate validation utility question IDs")
    for row in rows:
        validate_utility_row(row, "validation")
    return rows


def utility_matrix(
    feature_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    lookup = {str(row["question_id"]): row for row in utility_rows}
    if len(lookup) != len(utility_rows):
        raise RuntimeError("Utility rows contain duplicate IDs")
    matrix = np.zeros((len(feature_rows), len(CLASS_TOKENS)), dtype=np.float64)
    for index, feature in enumerate(feature_rows):
        question_id = str(feature["question_id"])
        if question_id not in lookup:
            raise RuntimeError(f"Missing utility row for {question_id}")
        utility = lookup[question_id]
        if str(feature["document_id"]) != str(utility["document_id"]):
            raise RuntimeError(f"Document mismatch for {question_id}")
        matrix[index] = [
            float(item["f1_joined_topk"])
            for item in utility["per_granularity_metrics"]
        ]
    extra = set(lookup) - {str(row["question_id"]) for row in feature_rows}
    if extra:
        raise RuntimeError(f"Unmatched utility rows: {sorted(extra)[:3]}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Utility matrix contains non-finite values")
    return matrix


def regret_matrix(utilities: np.ndarray) -> np.ndarray:
    utilities = np.asarray(utilities, dtype=np.float64)
    if utilities.ndim != 2 or utilities.shape[1] != len(CLASS_TOKENS):
        raise ValueError(f"Invalid utility matrix shape: {utilities.shape}")
    regrets = utilities.max(axis=1, keepdims=True) - utilities
    regrets[np.abs(regrets) < 1e-15] = 0.0
    if np.any(regrets < 0):
        raise RuntimeError("Regret cannot be negative")
    return regrets


def build_fusion_matrix(
    rows: Sequence[dict], logits_path: Path
) -> tuple[np.ndarray, list[str]]:
    arrays = phase3c.load_npz(logits_path)
    matrix, names = phase3c.fusion_matrix(rows, arrays)
    if matrix.shape != (len(rows), FEATURE_COUNT):
        raise RuntimeError(f"Invalid Phase 4 feature shape: {matrix.shape}")
    return matrix, names


def regression_parameters(seed: int) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "max_depth": int(FIXED_CANDIDATE["max_depth"]),
        "eta": float(FIXED_CANDIDATE["learning_rate"]),
        "min_child_weight": float(FIXED_CANDIDATE["min_child_weight"]),
        "subsample": float(FIXED_CANDIDATE["subsample"]),
        "colsample_bytree": float(FIXED_CANDIDATE["colsample_bytree"]),
        "lambda": float(FIXED_CANDIDATE["reg_lambda"]),
        "alpha": float(FIXED_CANDIDATE["reg_alpha"]),
        "eval_metric": "rmse",
        "seed": int(seed),
        "nthread": max(1, min(8, os.cpu_count() or 1)),
        "verbosity": 0,
    }


def train_regret_models(
    features: np.ndarray,
    regrets: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
):
    import xgboost as xgb

    models = []
    for action in range(len(CLASS_TOKENS)):
        matrix = xgb.DMatrix(
            features,
            label=np.asarray(regrets[:, action], dtype=np.float32),
            feature_names=list(feature_names),
        )
        models.append(
            xgb.train(
                regression_parameters(seed + action),
                matrix,
                num_boost_round=int(FIXED_CANDIDATE["n_estimators"]),
            )
        )
    return models


def predict_regrets(
    models: Sequence[Any], features: np.ndarray, feature_names: Sequence[str]
) -> np.ndarray:
    import xgboost as xgb

    if len(models) != len(CLASS_TOKENS):
        raise RuntimeError("Expected one regret model per action")
    matrix = xgb.DMatrix(features, feature_names=list(feature_names))
    predictions = np.column_stack(
        [np.asarray(model.predict(matrix), dtype=np.float64) for model in models]
    )
    if predictions.shape != (len(features), len(CLASS_TOKENS)):
        raise RuntimeError(f"Invalid predicted regret shape: {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise RuntimeError("Predicted regrets contain non-finite values")
    return predictions


def choose_actions(predicted_regrets: np.ndarray) -> np.ndarray:
    """Choose minimum predicted regret; stable argmin resolves ties to smaller chunks."""

    values = np.asarray(predicted_regrets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_TOKENS):
        raise ValueError(f"Invalid predicted regret shape: {values.shape}")
    return np.argmin(values, axis=1).astype(np.int64)


def action_summary(utilities: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    utilities = np.asarray(utilities, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    selected = utilities[np.arange(len(actions)), actions]
    best = utilities.max(axis=1)
    regret = best - selected
    optimal_mask = np.isclose(utilities, best[:, None], rtol=0.0, atol=1e-12)
    any_optimal = optimal_mask[np.arange(len(actions)), actions]
    single_optimal = utilities.argmax(axis=1)
    return {
        "examples": int(len(actions)),
        "mean_joined_retrieval_f1": float(selected.mean()),
        "median_joined_retrieval_f1": float(np.median(selected)),
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "zero_regret_rate": float(np.mean(any_optimal)),
        "retrieval_optimal_any_tie_accuracy": float(np.mean(any_optimal)),
        "retrieval_optimal_smaller_tie_accuracy": float(
            np.mean(actions == single_optimal)
        ),
        "oracle_upper_bound_mean_joined_retrieval_f1": float(best.mean()),
        "selected_distribution": {
            str(tokens): int(np.sum(actions == index))
            for index, tokens in enumerate(CLASS_TOKENS)
        },
        "retrieval_optimal_smaller_tie_distribution": {
            str(tokens): int(np.sum(single_optimal == index))
            for index, tokens in enumerate(CLASS_TOKENS)
        },
    }


def diagnostic_cross_validation(
    features: np.ndarray,
    regrets: np.ndarray,
    utilities: np.ndarray,
    feature_names: Sequence[str],
    folds: np.ndarray,
) -> dict[str, Any]:
    predicted_regrets = np.zeros_like(regrets, dtype=np.float64)
    fold_results = []
    started = time.perf_counter()
    for fold in range(FOLDS):
        train_mask = folds != fold
        heldout_mask = folds == fold
        models = train_regret_models(
            features[train_mask], regrets[train_mask], feature_names, SEED + 10 * fold
        )
        fold_predictions = predict_regrets(
            models, features[heldout_mask], feature_names
        )
        predicted_regrets[heldout_mask] = fold_predictions
        fold_actions = choose_actions(fold_predictions)
        fold_results.append(
            {
                "fold": fold,
                "train_examples": int(train_mask.sum()),
                "heldout_examples": int(heldout_mask.sum()),
                "metrics": action_summary(utilities[heldout_mask], fold_actions),
                "regret_rmse": float(
                    np.sqrt(np.mean((fold_predictions - regrets[heldout_mask]) ** 2))
                ),
            }
        )
    actions = choose_actions(predicted_regrets)
    return {
        "status": "descriptive_only_not_used_for_model_selection",
        "metrics": action_summary(utilities, actions),
        "regret_rmse": float(np.sqrt(np.mean((predicted_regrets - regrets) ** 2))),
        "folds": fold_results,
        "wall_seconds": time.perf_counter() - started,
        "dependency_caveat": (
            "The meta-level grouped diagnostic uses globally cross-fitted Qwen "
            "features. Some Qwen fits producing meta-training features may have "
            "seen papers in the current meta-held-out fold. Therefore this result "
            "is descriptive only and selected no Phase 4 setting."
        ),
    }


def prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    predicted_regrets: np.ndarray,
    actions: np.ndarray,
) -> list[dict]:
    output = []
    for index, row in enumerate(rows):
        action = int(actions[index])
        ranking = np.argsort(predicted_regrets[index], kind="stable")
        output.append(
            {
                "phase": PHASE,
                "formulation_version": FORMULATION_VERSION,
                "question_id": str(row["question_id"]),
                "document_id": str(row["document_id"]),
                "question_text": str(row["question_text"]),
                "predicted_label": int(CLASS_TOKENS[action]),
                "parsed_prediction": int(CLASS_TOKENS[action]),
                "predicted_class_index": action,
                "predicted_regret_by_label": {
                    str(tokens): float(predicted_regrets[index, class_index])
                    for class_index, tokens in enumerate(CLASS_TOKENS)
                },
                "ranked_actions_lowest_predicted_regret_first": [
                    int(CLASS_TOKENS[int(value)]) for value in ranking
                ],
                "prediction_status": "valid_frozen_before_gold_utility_join",
            }
        )
    return output


def selected_evaluation_rows(
    prediction_rows_: Sequence[Mapping[str, Any]], utilities: np.ndarray
) -> list[dict]:
    output = []
    for index, prediction in enumerate(prediction_rows_):
        action = int(prediction["predicted_class_index"])
        values = utilities[index]
        best = float(values.max())
        selected = float(values[action])
        optimal = [
            int(CLASS_TOKENS[i])
            for i, value in enumerate(values)
            if math.isclose(float(value), best, rel_tol=0.0, abs_tol=1e-12)
        ]
        output.append(
            {
                "question_id": str(prediction["question_id"]),
                "document_id": str(prediction["document_id"]),
                "predicted_granularity": int(CLASS_TOKENS[action]),
                "utility_by_granularity": {
                    str(tokens): float(values[i])
                    for i, tokens in enumerate(CLASS_TOKENS)
                },
                "selected_joined_retrieval_f1": selected,
                "best_achievable_joined_retrieval_f1": best,
                "retrieval_regret": best - selected,
                "retrieval_optimal_granularities": optimal,
                "selected_is_retrieval_optimal": int(CLASS_TOKENS[action]) in optimal,
                "source": str(VALIDATION_UTILITY_PATH),
                "top_k": TOP_K,
                "paper_restricted": True,
            }
        )
    return output


def label_from_prediction(row: Mapping[str, Any]) -> int:
    for key in ("predicted_label", "parsed_prediction", "predicted_granularity_tokens"):
        value = row.get(key)
        if value is not None:
            value = int(value)
            if value in CLASS_TOKENS:
                return value
    raise RuntimeError(f"No valid prediction in row {row.get('question_id')}")


def baseline_actions(path: Path, validation_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows = read_jsonl(path)
    lookup = {str(row["question_id"]): row for row in rows}
    if len(rows) != 924 or len(lookup) != 924:
        raise RuntimeError(f"Incomplete baseline predictions: {path}")
    actions = []
    token_to_index = {tokens: index for index, tokens in enumerate(CLASS_TOKENS)}
    for row in validation_rows:
        question_id = str(row["question_id"])
        if question_id not in lookup:
            raise RuntimeError(f"Missing baseline prediction for {question_id}: {path}")
        actions.append(token_to_index[label_from_prediction(lookup[question_id])])
    return np.asarray(actions, dtype=np.int64)


def paper_cluster_bootstrap(
    values: np.ndarray,
    document_ids: Sequence[str],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    documents = np.asarray([str(value) for value in document_ids])
    if values.ndim != 1 or len(values) != len(documents):
        raise ValueError("Bootstrap values and document IDs must be aligned vectors")
    unique = np.asarray(sorted(set(documents.tolist())))
    sums = np.asarray([values[documents == doc].sum() for doc in unique])
    counts = np.asarray([np.sum(documents == doc) for doc in unique], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(unique), size=len(unique))
        estimates[iteration] = sums[sampled].sum() / counts[sampled].sum()
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "point_estimate": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "iterations": int(iterations),
        "seed": int(seed),
        "resampling_unit": "paper",
        "paper_count": int(len(unique)),
        "question_count": int(len(values)),
    }


def selected_utilities(utilities: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return utilities[np.arange(len(actions)), actions]


def comparison_summary(
    validation_rows: Sequence[Mapping[str, Any]],
    utilities: np.ndarray,
    phase4_actions: np.ndarray,
) -> dict[str, Any]:
    document_ids = [str(row["document_id"]) for row in validation_rows]
    best = utilities.max(axis=1)
    actions_by_strategy: dict[str, np.ndarray] = {"phase4": phase4_actions}
    for name, path in BASELINE_PREDICTIONS.items():
        actions_by_strategy[name] = baseline_actions(path, validation_rows)
    for index, tokens in enumerate(CLASS_TOKENS):
        actions_by_strategy[f"fixed_{tokens}"] = np.full(
            len(validation_rows), index, dtype=np.int64
        )

    strategy_metrics = {}
    selected_by_strategy = {}
    for name, actions in actions_by_strategy.items():
        selected = selected_utilities(utilities, actions)
        selected_by_strategy[name] = selected
        strategy_metrics[name] = {
            **action_summary(utilities, actions),
            "mean_retrieval_f1_ci95": paper_cluster_bootstrap(
                selected, document_ids
            ),
        }
    strategy_metrics["retrieval_oracle_upper_bound"] = {
        "examples": int(len(best)),
        "mean_joined_retrieval_f1": float(best.mean()),
        "median_joined_retrieval_f1": float(np.median(best)),
        "mean_regret": 0.0,
        "median_regret": 0.0,
        "deployable": False,
        "mean_retrieval_f1_ci95": paper_cluster_bootstrap(best, document_ids),
    }

    paired = {}
    phase4_values = selected_by_strategy["phase4"]
    for name, values in selected_by_strategy.items():
        if name == "phase4":
            continue
        differences = phase4_values - values
        paired[f"phase4_minus_{name}"] = paper_cluster_bootstrap(
            differences, document_ids
        )
    return {
        "primary_metric": "mean_joined_retrieval_f1",
        "primary_operational_baseline": "fixed_40",
        "uncertainty_method": "paired paper-cluster bootstrap",
        "strategy_metrics": strategy_metrics,
        "paired_differences": paired,
    }


def write_distribution_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    selected = metrics["selected_distribution"]
    optimal = metrics["retrieval_optimal_smaller_tie_distribution"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["granularity", "phase4_selected", "retrieval_optimal_smaller_tie"])
        for tokens in CLASS_TOKENS:
            writer.writerow([tokens, selected[str(tokens)], optimal[str(tokens)]])
    temporary.replace(path)


def preflight(output_root: Path) -> dict[str, Any]:
    hashes = source_hashes()
    train_rows = phase3c.source_rows("train")
    train_utilities = load_train_utility_rows()
    utility = utility_matrix(train_rows, train_utilities)
    train_arrays = phase3c.load_npz(
        phase3c.DEFAULT_OUTPUT_ROOT / "qwen_features" / "train_oof_logits.npz"
    )
    features, names = phase3c.fusion_matrix(train_rows, train_arrays)
    folds = phase3c.training_folds(train_rows)
    audit = {
        "phase": PHASE,
        "status": "passed",
        "completed_at": utc_now(),
        "gpu_required": False,
        "qdrant_required": False,
        "retrieval_rerun_required": False,
        "train_examples": len(train_rows),
        "train_documents": len({str(row["document_id"]) for row in train_rows}),
        "feature_shape": list(features.shape),
        "feature_count": len(names),
        "utility_shape": list(utility.shape),
        "utility_source_rows": {
            "main": 2223,
            "recovery": 22,
            "combined": 2245,
        },
        "fold_count": FOLDS,
        "paper_grouping_verified": all(
            len(
                {
                    int(folds[index])
                    for index, row in enumerate(train_rows)
                    if str(row["document_id"]) == document
                }
            )
            == 1
            for document in {str(row["document_id"]) for row in train_rows}
        ),
        "qwen_oof_exactly_once": len(set(train_arrays["question_ids"].tolist()))
        == len(train_rows),
        "source_hashes": hashes,
        "mean_train_utility_by_granularity": {
            str(tokens): float(utility[:, index].mean())
            for index, tokens in enumerate(CLASS_TOKENS)
        },
        "train_retrieval_optimal_distribution_smaller_tie": {
            str(tokens): int(np.sum(utility.argmax(axis=1) == index))
            for index, tokens in enumerate(CLASS_TOKENS)
        },
    }
    atomic_json(output_root / "integrity" / "preflight_audit.json", audit)
    return audit


def lock_procedure(output_root: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    lock = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "status": "frozen_before_validation_utility_access",
        "locked_at": utc_now(),
        "objective": (
            "Estimate conditional retrieval regret for each of five actions and "
            "select the action with minimum predicted regret."
        ),
        "cost_definition": "C(q,g)=max_h U(q,h)-U(q,g)",
        "utility_definition": (
            "joined token-level F1 of unchanged paper-restricted top-5 retrieval"
        ),
        "decision_rule": "argmin_g predicted_E[C(q,g)|x]",
        "tie_rule": "smaller granularity via stable action order",
        "features": {
            "qwen_logits": 5,
            "similarity_tree": 173,
            "total": FEATURE_COUNT,
            "training_qwen_provenance": "paper-grouped OOF Phase 3C-OOF",
            "validation_qwen_provenance": "frozen all-train full refit",
        },
        "model": {
            "family": "five independent XGBoost regret regressors",
            "objective": "reg:squarederror",
            "fixed_candidate_inherited_from_phase3c_oof": FIXED_CANDIDATE,
            "regressor_count": 5,
            "seed": SEED,
            "hyperparameter_search": "none",
        },
        "training": {
            "examples": 2245,
            "paper_grouped_diagnostic_folds": FOLDS,
            "diagnostic_used_for_selection": False,
            "gold_evidence_role": "training retrieval-utility targets only",
        },
        "validation": {
            "examples": 924,
            "used_for_training_or_selection": False,
            "gold_utility_join": "only after prediction artifact is frozen and hashed",
        },
        "evaluation": {
            "primary": "mean joined retrieval F1",
            "secondary": [
                "mean and median retrieval regret",
                "retrieval-optimal agreement",
                "evidence-length Oracle classification diagnostics",
            ],
            "primary_operational_baseline": "fixed granularity 40",
            "uncertainty": "paired paper-cluster bootstrap",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "source_hashes": audit["source_hashes"],
    }
    atomic_json(output_root / "selection" / "selection_lock.json", lock)
    return lock


def environment_summary() -> dict[str, Any]:
    import xgboost

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "xgboost": xgboost.__version__,
        "device": "CPU",
    }


def run(output_root: Path) -> dict[str, Any]:
    total_started = time.perf_counter()
    audit = preflight(output_root)
    lock = lock_procedure(output_root, audit)
    lock_path = output_root / "selection" / "selection_lock.json"
    lock_hash = phase3b.sha256_file(lock_path)

    train_rows = phase3c.source_rows("train")
    train_utility_rows = load_train_utility_rows()
    train_utilities = utility_matrix(train_rows, train_utility_rows)
    train_regrets = regret_matrix(train_utilities)
    train_features, feature_names = build_fusion_matrix(
        train_rows,
        phase3c.DEFAULT_OUTPUT_ROOT / "qwen_features" / "train_oof_logits.npz",
    )
    folds = phase3c.training_folds(train_rows)

    diagnostic = diagnostic_cross_validation(
        train_features,
        train_regrets,
        train_utilities,
        feature_names,
        folds,
    )
    atomic_json(
        output_root / "cross_validation" / "fixed_procedure_diagnostic.json",
        diagnostic,
    )

    fit_started = time.perf_counter()
    models = train_regret_models(
        train_features, train_regrets, feature_names, SEED
    )
    fit_seconds = time.perf_counter() - fit_started
    model_dir = output_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_hashes = {}
    for action, model in enumerate(models):
        path = model_dir / f"regret_{CLASS_TOKENS[action]}.json"
        model.save_model(path)
        model_hashes[str(CLASS_TOKENS[action])] = phase3b.sha256_file(path)

    # Prediction uses only inference-safe validation features and frozen Qwen logits.
    # Validation utility/gold records are intentionally not loaded until after this
    # artifact is durably saved and hashed.
    validation_rows = phase3c.source_rows("validation")
    validation_features, validation_names = build_fusion_matrix(
        validation_rows,
        phase3c.DEFAULT_OUTPUT_ROOT / "qwen_features" / "validation_logits.npz",
    )
    if validation_names != feature_names:
        raise RuntimeError("Train/validation Phase 4 feature schemas differ")
    prediction_started = time.perf_counter()
    predicted_regrets = predict_regrets(models, validation_features, feature_names)
    prediction_seconds = time.perf_counter() - prediction_started
    actions = choose_actions(predicted_regrets)
    predictions = prediction_rows(validation_rows, predicted_regrets, actions)
    pre_evaluation_path = output_root / "validation" / "predictions_pre_evaluation.jsonl"
    atomic_jsonl(pre_evaluation_path, predictions)
    prediction_hash = phase3b.sha256_file(pre_evaluation_path)
    prediction_lock = {
        "status": "frozen_before_validation_gold_utility_join",
        "frozen_at": utc_now(),
        "prediction_path": str(pre_evaluation_path),
        "prediction_sha256": prediction_hash,
        "selection_lock_sha256": lock_hash,
        "examples": len(predictions),
        "gold_utility_fields_in_prediction_artifact": False,
    }
    atomic_json(output_root / "validation" / "prediction_lock.json", prediction_lock)

    # Evaluation-only gold utility access begins here.
    validation_utility_rows = load_validation_utility_rows()
    validation_utilities = utility_matrix(validation_rows, validation_utility_rows)
    evaluation = action_summary(validation_utilities, actions)
    evaluation_rows = selected_evaluation_rows(predictions, validation_utilities)
    atomic_jsonl(output_root / "retrieval" / "results.jsonl", evaluation_rows)
    atomic_json(output_root / "retrieval" / "summary.json", evaluation)
    write_distribution_csv(output_root / "retrieval" / "distribution.csv", evaluation)

    evidence_length_targets = phase3b.target_array(validation_rows)
    pseudo_probabilities = np.exp(
        -(predicted_regrets - predicted_regrets.min(axis=1, keepdims=True))
    )
    pseudo_probabilities /= pseudo_probabilities.sum(axis=1, keepdims=True)
    classification = phase3b.classification_metrics(
        evidence_length_targets, actions, pseudo_probabilities
    )
    classification["role"] = "secondary evidence-length-Oracle diagnostic"
    classification["probability_note"] = (
        "Softmax of negative predicted regrets is used only for ranking/top-2 "
        "diagnostics; it is not a calibrated class probability."
    )
    atomic_json(output_root / "classification" / "metrics.json", classification)
    phase3b.write_confusion_csv(
        output_root / "classification" / "confusion_matrix.csv", classification
    )
    phase3b.write_histogram_svg(
        output_root / "classification" / "predicted_vs_evidence_length_oracle.svg",
        classification["class_distribution"],
        classification["predicted_distribution"],
    )

    comparisons = comparison_summary(validation_rows, validation_utilities, actions)
    atomic_json(output_root / "comparison" / "baselines.json", comparisons)

    train_target_rows = []
    for index, row in enumerate(train_rows):
        train_target_rows.append(
            {
                "question_id": str(row["question_id"]),
                "document_id": str(row["document_id"]),
                "utility_by_granularity": {
                    str(tokens): float(train_utilities[index, action])
                    for action, tokens in enumerate(CLASS_TOKENS)
                },
                "regret_by_granularity": {
                    str(tokens): float(train_regrets[index, action])
                    for action, tokens in enumerate(CLASS_TOKENS)
                },
            }
        )
    atomic_jsonl(output_root / "targets" / "train_utility_targets.jsonl", train_target_rows)

    runtime = {
        "fit_seconds": fit_seconds,
        "validation_prediction_seconds": prediction_seconds,
        "cross_validation_diagnostic_seconds": diagnostic["wall_seconds"],
        "total_wall_seconds": time.perf_counter() - total_started,
        "device": "CPU",
        "gpu_used": False,
        "qdrant_used": False,
        "retrieval_rerun": False,
    }
    atomic_json(output_root / "runtime" / "summary.json", runtime)
    atomic_json(output_root / "environment" / "python_environment.json", environment_summary())

    final = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "objective": lock["objective"],
        "configuration": lock,
        "integrity": {
            "selection_lock_sha256": lock_hash,
            "prediction_sha256_before_gold_utility_join": prediction_hash,
            "validation_used_for_training_or_selection": False,
            "validation_gold_utility_accessed_only_after_predictions_frozen": True,
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "validation_documents": len(
                {str(row["document_id"]) for row in validation_rows}
            ),
            "top_k": TOP_K,
            "paper_restricted": True,
            "evaluation_config_hash": EVALUATION_CONFIG_HASH,
        },
        "train": {
            "mean_utility_by_granularity": audit[
                "mean_train_utility_by_granularity"
            ],
            "retrieval_optimal_distribution": audit[
                "train_retrieval_optimal_distribution_smaller_tie"
            ],
            "fixed_procedure_cross_validation_diagnostic": diagnostic,
        },
        "validation": {
            "retrieval_utility_metrics": evaluation,
            "evidence_length_oracle_classification_diagnostic": classification,
        },
        "comparisons": comparisons,
        "runtime": runtime,
        "model_hashes": model_hashes,
        "artifacts": {
            "preflight": str(output_root / "integrity" / "preflight_audit.json"),
            "selection_lock": str(lock_path),
            "prediction_lock": str(
                output_root / "validation" / "prediction_lock.json"
            ),
            "predictions": str(pre_evaluation_path),
            "retrieval_results": str(output_root / "retrieval" / "results.jsonl"),
            "comparison": str(output_root / "comparison" / "baselines.json"),
            "train_targets": str(
                output_root / "targets" / "train_utility_targets.jsonl"
            ),
        },
    }
    atomic_json(output_root / "final_summary.json", final)
    return final


def audit_command(args: argparse.Namespace) -> int:
    print(json.dumps(preflight(args.output_root), indent=2))
    return 0


def run_command(args: argparse.Namespace) -> int:
    final = run(args.output_root)
    print(json.dumps(final["validation"]["retrieval_utility_metrics"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.set_defaults(function=audit_command)
    run_parser = subparsers.add_parser("run")
    run_parser.set_defaults(function=run_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
