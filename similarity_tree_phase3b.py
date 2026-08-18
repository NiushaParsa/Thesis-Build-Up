#!/usr/bin/env python
"""Phase 3B nonlinear router over frozen Phase 3A similarity-tree features.

The training path has no Qdrant dependency and never recomputes Phase 3A
features. Hyperparameters and the primary feature variant are selected only
from paper-grouped out-of-fold predictions on the preserved training split.
The preserved validation split is evaluated after a durable selection lock.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PHASE = "Phase 3B"
FORMULATION_VERSION = "similarity-tree-xgboost-evidence-length-oracle-v1"
CLASS_TOKENS = (10, 20, 40, 80, 160)
SEED = 42
FOLDS = 5
TOP_K = 5
MODEL_SELECTION_METRIC = "macro_f1"
SOURCE_ORACLE_CONFIG_HASH = (
    "9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8"
)
SOURCE_FEATURE_ROOT = Path("outputs/similarity_tree_phase3a_evidence_length_oracle")
DEFAULT_OUTPUT_ROOT = Path("outputs/similarity_tree_phase3b_xgboost_evidence_length_oracle")
EXPECTED_SOURCE_HASHES = {
    "train": "6d55e1d10872c8db24cf9af9becfb8e2e6570e13a7697151febc7f44ecebdd9c",
    "validation": "548e3cccab3b19dee644eb9858081ff380b6375765433f1d2369c6d7d2ecb893",
}
FEATURE_VARIANTS = {
    "level_xgboost": "level_features",
    "tree_xgboost": "tree_features",
}
FORBIDDEN_FEATURE_TERMS = (
    "evidence",
    "oracle",
    "answer",
    "retrieval_f1",
    "joined_f1",
    "chunk_f1",
)
GRID = tuple(
    {
        "max_depth": depth,
        "learning_rate": learning_rate,
        "n_estimators": estimators,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    }
    for depth in (2, 3, 4)
    for learning_rate in (0.03, 0.05)
    for estimators in (200, 400)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
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
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed JSONL at {path}:{line_number}") from error
    return rows


def feature_path(source_root: Path, split: str) -> Path:
    return source_root / "features" / f"{split}_features.jsonl"


def assert_no_leakage_feature_names(names: Sequence[str]) -> None:
    prohibited = [
        name for name in names if any(term in name.lower() for term in FORBIDDEN_FEATURE_TERMS)
    ]
    if prohibited:
        raise RuntimeError(f"Leakage-prone feature names are forbidden: {prohibited}")


def validate_rows(rows: Sequence[dict], split: str, expected_count: int) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} {split} rows, found {len(rows)}")
    ids = [str(row.get("question_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"{split} question IDs are missing or duplicated")
    required = {
        "question_id",
        "document_id",
        "question_text",
        "oracle_label",
        "level_features",
        "tree_features",
    }
    for row in rows:
        if not required.issubset(row):
            raise RuntimeError(f"Incomplete {split} feature row: {row.get('question_id')}")
        if int(row["oracle_label"]) not in CLASS_TOKENS:
            raise RuntimeError(f"Invalid Oracle label in {split}: {row['oracle_label']}")
    for key, expected_features in (("level_features", 85), ("tree_features", 173)):
        names = sorted(rows[0][key])
        if len(names) != expected_features:
            raise RuntimeError(f"Unexpected {key} count: {len(names)}")
        assert_no_leakage_feature_names(names)
        for row in rows:
            if sorted(row[key]) != names:
                raise RuntimeError(f"Inconsistent {key} schema")
            values = np.asarray([row[key][name] for name in names], dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite {key} values")


def target_array(rows: Sequence[dict]) -> np.ndarray:
    mapping = {tokens: index for index, tokens in enumerate(CLASS_TOKENS)}
    return np.asarray([mapping[int(row["oracle_label"])] for row in rows], dtype=np.int64)


def feature_matrix(rows: Sequence[dict], feature_key: str) -> tuple[np.ndarray, list[str]]:
    names = sorted(rows[0][feature_key])
    assert_no_leakage_feature_names(names)
    matrix = np.asarray(
        [[float(row[feature_key][name]) for name in names] for row in rows],
        dtype=np.float32,
    )
    if not np.isfinite(matrix).all():
        raise RuntimeError("Feature matrix contains non-finite values")
    return matrix, names


def grouped_stratified_folds(rows: Sequence[dict], fold_count: int, seed: int) -> np.ndarray:
    label_to_index = {tokens: index for index, tokens in enumerate(CLASS_TOKENS)}
    groups: dict[str, np.ndarray] = {}
    for row in rows:
        document = str(row["document_id"])
        groups.setdefault(document, np.zeros(len(CLASS_TOKENS), dtype=np.int64))
        groups[document][label_to_index[int(row["oracle_label"])]] += 1
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
        observed = {
            int(folds[index])
            for index, row in enumerate(rows)
            if str(row["document_id"]) == document
        }
        if len(observed) != 1:
            raise RuntimeError(f"Paper {document} crosses grouped folds")
    return folds


def fold_manifest(rows: Sequence[dict], targets: np.ndarray, folds: np.ndarray) -> list[dict]:
    return [
        {
            "fold": fold,
            "examples": int(np.sum(folds == fold)),
            "documents": len(
                {
                    str(row["document_id"])
                    for index, row in enumerate(rows)
                    if folds[index] == fold
                }
            ),
            "class_distribution": {
                str(tokens): int(np.sum(targets[folds == fold] == index))
                for index, tokens in enumerate(CLASS_TOKENS)
            },
        }
        for fold in range(FOLDS)
    ]


def class_balance_weights(targets: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    counts = np.bincount(targets, minlength=len(CLASS_TOKENS)).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("Every class must occur in a training fold")
    maximum = float(counts.max())
    by_index = np.sqrt(maximum / counts)
    weights = by_index[targets].astype(np.float32)
    return weights, {
        str(tokens): float(by_index[index]) for index, tokens in enumerate(CLASS_TOKENS)
    }


def quadratic_weighted_kappa(targets: np.ndarray, predictions: np.ndarray) -> float:
    count = len(CLASS_TOKENS)
    observed = np.zeros((count, count), dtype=np.float64)
    for target, prediction in zip(targets, predictions):
        observed[int(target), int(prediction)] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(1, observed.sum())
    weights = np.fromfunction(lambda i, j: ((i - j) ** 2) / ((count - 1) ** 2), (count, count))
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else 0.0


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict:
    class_count = len(CLASS_TOKENS)
    if probabilities.shape != (len(targets), class_count):
        raise ValueError(f"Invalid probability shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values")
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(class_count, dtype=float),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(class_count, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=float),
        where=(precision + recall) != 0,
    )
    top2 = np.argsort(-probabilities, axis=1, kind="stable")[:, :2]
    distances = np.abs(targets - predictions)
    token_values = np.asarray(CLASS_TOKENS)
    total = int(support.sum())
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "top_2_accuracy": float(
            np.mean([target in candidates for target, candidates in zip(targets, top2)])
        ),
        "per_class": {
            str(tokens): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, tokens in enumerate(CLASS_TOKENS)
        },
        "confusion_matrix": confusion.tolist(),
        "class_distribution": {
            str(tokens): int(support[index]) for index, tokens in enumerate(CLASS_TOKENS)
        },
        "predicted_distribution": {
            str(tokens): int(np.sum(predictions == index))
            for index, tokens in enumerate(CLASS_TOKENS)
        },
        "top_2_accuracy_status": "available",
        "mean_absolute_class_distance": float(distances.mean()),
        "within_one_level_accuracy": float(np.mean(distances <= 1)),
        "mean_absolute_token_distance": float(
            np.abs(token_values[targets] - token_values[predictions]).mean()
        ),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(targets, predictions),
        "correct_count": int(np.sum(targets == predictions)),
        "example_count": int(len(targets)),
    }


def majority_reference(targets: np.ndarray, class_index: int) -> dict:
    predictions = np.full(len(targets), class_index, dtype=np.int64)
    probabilities = np.zeros((len(targets), len(CLASS_TOKENS)), dtype=np.float32)
    probabilities[:, class_index] = 1.0
    return classification_metrics(targets, predictions, probabilities)


def candidate_key(candidate: dict) -> tuple:
    metrics = candidate["oof_metrics"]
    parameters = candidate["parameters"]
    return (
        float(metrics[MODEL_SELECTION_METRIC]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        -int(parameters["max_depth"]),
        -int(parameters["n_estimators"]),
        -float(parameters["learning_rate"]),
    )


def variant_key(name: str, result: dict) -> tuple:
    metrics = result["selected_candidate"]["oof_metrics"]
    return (
        float(metrics[MODEL_SELECTION_METRIC]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        int(name == "tree_xgboost"),
    )


def xgboost_parameters(candidate: dict, seed: int) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": len(CLASS_TOKENS),
        "tree_method": "hist",
        "device": "cpu",
        "max_depth": int(candidate["max_depth"]),
        "eta": float(candidate["learning_rate"]),
        "min_child_weight": float(candidate["min_child_weight"]),
        "subsample": float(candidate["subsample"]),
        "colsample_bytree": float(candidate["colsample_bytree"]),
        "lambda": float(candidate["reg_lambda"]),
        "alpha": float(candidate["reg_alpha"]),
        "eval_metric": "mlogloss",
        "seed": int(seed),
        "nthread": max(1, min(8, os.cpu_count() or 1)),
        "verbosity": 0,
    }


def train_booster(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    feature_names: Sequence[str],
    candidate: dict,
    seed: int,
):
    import xgboost as xgb

    matrix = xgb.DMatrix(
        features,
        label=targets,
        weight=weights,
        feature_names=list(feature_names),
    )
    return xgb.train(
        xgboost_parameters(candidate, seed),
        matrix,
        num_boost_round=int(candidate["n_estimators"]),
    )


def predict_booster(booster, features: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    import xgboost as xgb

    probabilities = np.asarray(
        booster.predict(xgb.DMatrix(features, feature_names=list(feature_names))),
        dtype=np.float64,
    )
    if probabilities.shape != (len(features), len(CLASS_TOKENS)):
        raise RuntimeError(f"Unexpected XGBoost prediction shape: {probabilities.shape}")
    return probabilities


def candidate_identity(candidate: dict) -> str:
    return (
        f"depth{candidate['max_depth']}-lr{candidate['learning_rate']}-"
        f"trees{candidate['n_estimators']}"
    )


def cross_validate_variant(
    *,
    variant: str,
    features: np.ndarray,
    feature_names: Sequence[str],
    targets: np.ndarray,
    folds: np.ndarray,
    output_root: Path,
) -> dict:
    progress_path = output_root / "cross_validation" / f"{variant}.json"
    candidates = []
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("grid_fingerprint") != stable_hash(GRID):
            raise RuntimeError(f"Saved {variant} grid fingerprint changed")
        if saved.get("status") == "complete":
            return saved
        candidates = list(saved.get("completed_candidates", []))
    completed_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    started = time.perf_counter()
    for candidate_number, parameters in enumerate(GRID, start=1):
        identity = candidate_identity(parameters)
        if identity in completed_ids:
            continue
        oof_probabilities = np.zeros((len(targets), len(CLASS_TOKENS)), dtype=np.float64)
        fold_results = []
        candidate_started = time.perf_counter()
        for fold in range(FOLDS):
            train_mask = folds != fold
            held_mask = folds == fold
            weights, class_weights = class_balance_weights(targets[train_mask])
            booster = train_booster(
                features[train_mask],
                targets[train_mask],
                weights,
                feature_names,
                parameters,
                SEED + fold,
            )
            probabilities = predict_booster(booster, features[held_mask], feature_names)
            oof_probabilities[held_mask] = probabilities
            predictions = probabilities.argmax(axis=1)
            fold_results.append(
                {
                    "fold": fold,
                    "train_examples": int(train_mask.sum()),
                    "held_out_examples": int(held_mask.sum()),
                    "class_weights": class_weights,
                    "metrics": classification_metrics(
                        targets[held_mask], predictions, probabilities
                    ),
                }
            )
        oof_predictions = oof_probabilities.argmax(axis=1)
        result = {
            "candidate_id": identity,
            "parameters": parameters,
            "oof_metrics": classification_metrics(
                targets, oof_predictions, oof_probabilities
            ),
            "fold_metrics": fold_results,
            "wall_seconds": time.perf_counter() - candidate_started,
        }
        candidates.append(result)
        atomic_json(
            progress_path,
            {
                "phase": PHASE,
                "variant": variant,
                "grid_fingerprint": stable_hash(GRID),
                "completed_candidates": candidates,
                "status": "running",
            },
        )
        print(
            json.dumps(
                {
                    "event": "phase3b_cv_progress",
                    "variant": variant,
                    "candidate": candidate_number,
                    "candidate_count": len(GRID),
                    "candidate_id": result["candidate_id"],
                    "oof_macro_f1": result["oof_metrics"]["macro_f1"],
                    "oof_accuracy": result["oof_metrics"]["accuracy"],
                }
            ),
            flush=True,
        )
    selected = max(candidates, key=candidate_key)
    summary = {
        "phase": PHASE,
        "variant": variant,
        "grid_fingerprint": stable_hash(GRID),
        "grid_size": len(GRID),
        "selection_metric": MODEL_SELECTION_METRIC,
        "selected_candidate": selected,
        "candidates": candidates,
        "wall_seconds": sum(float(candidate["wall_seconds"]) for candidate in candidates),
        "current_invocation_wall_seconds": time.perf_counter() - started,
        "status": "complete",
    }
    atomic_json(progress_path, summary)
    return summary


def feature_importance_rows(booster, feature_names: Sequence[str]) -> list[dict]:
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    rows = [
        {
            "feature": name,
            "gain": float(gain.get(name, 0.0)),
            "split_count": float(weight.get(name, 0.0)),
        }
        for name in feature_names
    ]
    return sorted(rows, key=lambda row: (-row["gain"], -row["split_count"], row["feature"]))


def write_importance_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "gain", "split_count"])
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["oracle\\predicted", *CLASS_TOKENS])
        for tokens, values in zip(CLASS_TOKENS, metrics["confusion_matrix"]):
            writer.writerow([tokens, *values])


def write_histogram_svg(path: Path, oracle: dict, predicted: dict) -> None:
    width, height = 800, 480
    left, top, plot_width, plot_height = 70, 45, 680, 340
    maximum = max(max(oracle.values()), max(predicted.values()), 1)
    bars = []
    labels = []
    for index, tokens in enumerate(CLASS_TOKENS):
        center = left + (index + 0.5) * plot_width / len(CLASS_TOKENS)
        for offset, values, color in ((-18, oracle, "#4C78A8"), (18, predicted, "#54A24B")):
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
            f'<text x="{center:.1f}" y="{top + plot_height + 25}" '
            f'text-anchor="middle" font-size="13">{tokens}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="400" y="24" text-anchor="middle" font-size="18">Phase 3B Oracle vs predicted distribution</text>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>
{''.join(bars)}{''.join(labels)}
<rect x="260" y="440" width="14" height="14" fill="#4C78A8"/><text x="280" y="452" font-size="13">Oracle</text>
<rect x="390" y="440" width="14" height="14" fill="#54A24B"/><text x="410" y="452" font-size="13">Predicted</text>
</svg>'''
    atomic_text(path, svg)


def audit_command(args: argparse.Namespace) -> int:
    import xgboost as xgb

    train_path = feature_path(args.source_root, "train")
    validation_path = feature_path(args.source_root, "validation")
    observed_hashes = {
        "train": sha256_file(train_path),
        "validation": sha256_file(validation_path),
    }
    if observed_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"Frozen Phase 3A feature hashes changed: {observed_hashes}")
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    validate_rows(train, "train", 2245)
    validate_rows(validation, "validation", 924)
    train_documents = {str(row["document_id"]) for row in train}
    validation_documents = {str(row["document_id"]) for row in validation}
    if train_documents & validation_documents:
        raise RuntimeError("Train and validation papers overlap")
    train_targets = target_array(train)
    folds = grouped_stratified_folds(train, FOLDS, SEED)
    manifest = fold_manifest(train, train_targets, folds)
    phase3a_manifest = json.loads(
        (args.source_root / "cross_validation" / "paper_grouped_folds.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest != phase3a_manifest:
        raise RuntimeError("Phase 3B grouped folds do not reproduce Phase 3A")
    package_lock = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    atomic_text(args.output_root / "environment" / "package_lock.txt", package_lock)
    environment = {
        "environment_name": ".venv-phase3b",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "xgboost": xgb.__version__,
        "device": "CPU",
        "tree_method": "hist",
        "package_lock_path": str(args.output_root / "environment" / "package_lock.txt"),
        "legacy_venv_modified": False,
        "qwen_venv_modified": False,
    }
    atomic_json(args.output_root / "environment" / "python_environment.json", environment)
    configuration = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "created_at": utc_now(),
        "source_feature_root": str(args.source_root),
        "source_feature_hashes": observed_hashes,
        "source_oracle_config_hash": SOURCE_ORACLE_CONFIG_HASH,
        "classes": list(CLASS_TOKENS),
        "feature_variants": FEATURE_VARIANTS,
        "feature_counts": {"level_xgboost": 85, "tree_xgboost": 173},
        "classifier": "XGBoost gradient-boosted decision trees",
        "objective": "multi:softprob",
        "class_weighting": "sqrt(maximum_fold_class_count / fold_class_count)",
        "selection": "five-fold paper-grouped train-only out-of-fold macro-F1",
        "variant_selection": "train-only OOF macro-F1; tree preferred only on an exact tie",
        "grid": list(GRID),
        "grid_fingerprint": stable_hash(GRID),
        "random_seed": SEED,
        "validation_policy": "evaluate only after durable train-only selection lock",
        "qdrant_used_for_training": False,
        "retrieval": {"top_k": TOP_K, "paper_restricted": True, "read_only": True},
        "forbidden_features": list(FORBIDDEN_FEATURE_TERMS),
    }
    atomic_json(args.output_root / "configuration" / "experiment.json", configuration)
    preflight = {
        "phase": PHASE,
        "audited_at": utc_now(),
        "status": "passed",
        "source_phase3a_final_summary": str(args.source_root / "final_summary.json"),
        "source_phase3a_final_summary_sha256": sha256_file(args.source_root / "final_summary.json"),
        "source_feature_hashes": observed_hashes,
        "train_examples": len(train),
        "train_documents": len(train_documents),
        "validation_examples": len(validation),
        "validation_documents": len(validation_documents),
        "paper_overlap": 0,
        "train_distribution": {
            str(tokens): int(np.sum(train_targets == index))
            for index, tokens in enumerate(CLASS_TOKENS)
        },
        "fold_manifest_matches_phase3a": True,
        "feature_counts": {"level": 85, "tree": 173},
        "environment": environment,
        "qdrant_contacted": False,
        "phase3a_artifacts_modified": False,
    }
    atomic_json(args.output_root / "integrity" / "preflight_audit.json", preflight)
    atomic_json(args.output_root / "cross_validation" / "paper_grouped_folds.json", manifest)
    print(json.dumps(preflight, indent=2))
    return 0


def train_evaluate_command(args: argparse.Namespace) -> int:
    import xgboost as xgb

    started = time.perf_counter()
    preflight_path = args.output_root / "integrity" / "preflight_audit.json"
    if not preflight_path.exists():
        raise RuntimeError("Run Phase 3B audit before training")
    train_path = feature_path(args.source_root, "train")
    validation_path = feature_path(args.source_root, "validation")
    if {
        "train": sha256_file(train_path),
        "validation": sha256_file(validation_path),
    } != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("Frozen Phase 3A features changed after preflight")
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    validate_rows(train, "train", 2245)
    validate_rows(validation, "validation", 924)
    train_targets = target_array(train)
    folds = grouped_stratified_folds(train, FOLDS, SEED)

    variant_results = {}
    matrices = {}
    names_by_variant = {}
    for variant, feature_key in FEATURE_VARIANTS.items():
        matrix, names = feature_matrix(train, feature_key)
        matrices[variant] = matrix
        names_by_variant[variant] = names
        variant_results[variant] = cross_validate_variant(
            variant=variant,
            features=matrix,
            feature_names=names,
            targets=train_targets,
            folds=folds,
            output_root=args.output_root,
        )

    primary_variant = max(
        variant_results,
        key=lambda variant: variant_key(variant, variant_results[variant]),
    )
    selection_lock = {
        "phase": PHASE,
        "locked_at": utc_now(),
        "selection_data": "preserved train split only",
        "validation_metrics_observed_at_lock": False,
        "selection_metric": MODEL_SELECTION_METRIC,
        "primary_variant": primary_variant,
        "variant_oof_results": {
            variant: {
                "selected_candidate_id": result["selected_candidate"]["candidate_id"],
                "selected_parameters": result["selected_candidate"]["parameters"],
                "oof_metrics": result["selected_candidate"]["oof_metrics"],
            }
            for variant, result in variant_results.items()
        },
        "source_feature_hashes": EXPECTED_SOURCE_HASHES,
        "grid_fingerprint": stable_hash(GRID),
    }
    atomic_json(args.output_root / "selection" / "selection_lock.json", selection_lock)
    print(
        json.dumps(
            {
                "event": "phase3b_primary_locked",
                "primary_variant": primary_variant,
                "validation_metrics_observed_at_lock": False,
            }
        ),
        flush=True,
    )

    validation_targets = target_array(validation)
    final_results = {}
    prediction_rows_by_variant = {}
    for variant, feature_key in FEATURE_VARIANTS.items():
        selected = variant_results[variant]["selected_candidate"]
        train_weights, final_class_weights = class_balance_weights(train_targets)
        fit_started = time.perf_counter()
        booster = train_booster(
            matrices[variant],
            train_targets,
            train_weights,
            names_by_variant[variant],
            selected["parameters"],
            SEED,
        )
        final_fit_seconds = time.perf_counter() - fit_started
        model_path = args.output_root / "models" / f"{variant}.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(model_path)
        validation_matrix, validation_names = feature_matrix(validation, feature_key)
        if validation_names != names_by_variant[variant]:
            raise RuntimeError("Train/validation feature-name mismatch")
        prediction_started = time.perf_counter()
        probabilities = predict_booster(booster, validation_matrix, validation_names)
        prediction_seconds = time.perf_counter() - prediction_started
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(validation_targets, predictions, probabilities)
        importance = feature_importance_rows(booster, names_by_variant[variant])
        write_importance_csv(
            args.output_root / "feature_importance" / f"{variant}.csv", importance
        )
        model_metadata = {
            "phase": PHASE,
            "variant": variant,
            "feature_key": feature_key,
            "feature_count": len(names_by_variant[variant]),
            "feature_names": names_by_variant[variant],
            "selected_candidate": selected,
            "final_class_weights": final_class_weights,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "xgboost_version": xgb.__version__,
            "final_fit_seconds": final_fit_seconds,
            "validation_prediction_seconds": prediction_seconds,
            "top_feature_importance_by_gain": importance[:25],
        }
        atomic_json(args.output_root / "models" / f"{variant}_metadata.json", model_metadata)
        rows = []
        for index, row in enumerate(validation):
            predicted_index = int(predictions[index])
            rows.append(
                {
                    "phase": PHASE,
                    "formulation_version": FORMULATION_VERSION,
                    "variant": variant,
                    "question_id": str(row["question_id"]),
                    "document_id": str(row["document_id"]),
                    "question_text": str(row["question_text"]),
                    "oracle_label": int(row["oracle_label"]),
                    "predicted_label": CLASS_TOKENS[predicted_index],
                    "predicted_class_index": predicted_index,
                    "probabilities": {
                        str(tokens): float(probabilities[index, class_index])
                        for class_index, tokens in enumerate(CLASS_TOKENS)
                    },
                    "prediction_status": "valid_five_class_softprob",
                }
            )
        variant_prediction_path = args.output_root / "validation" / f"{variant}_predictions.jsonl"
        atomic_jsonl(variant_prediction_path, rows)
        prediction_rows_by_variant[variant] = rows
        final_results[variant] = {
            "feature_key": feature_key,
            "feature_count": len(names_by_variant[variant]),
            "selected_candidate": selected,
            "validation_metrics": metrics,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "prediction_path": str(variant_prediction_path),
            "prediction_sha256": sha256_file(variant_prediction_path),
            "final_fit_seconds": final_fit_seconds,
            "validation_prediction_seconds": prediction_seconds,
            "final_class_weights": final_class_weights,
        }

    primary_rows = prediction_rows_by_variant[primary_variant]
    primary_path = args.output_root / "validation" / "predictions.jsonl"
    atomic_jsonl(primary_path, primary_rows)
    primary_metrics = final_results[primary_variant]["validation_metrics"]
    atomic_json(args.output_root / "classification" / "metrics.json", primary_metrics)
    write_confusion_csv(args.output_root / "classification" / "confusion_matrix.csv", primary_metrics)
    write_histogram_svg(
        args.output_root / "classification" / "predicted_vs_oracle.svg",
        primary_metrics["class_distribution"],
        primary_metrics["predicted_distribution"],
    )
    train_majority_index = int(np.bincount(train_targets).argmax())
    validation_majority_index = int(np.bincount(validation_targets).argmax())
    source_phase3a = json.loads(
        (args.source_root / "final_summary.json").read_text(encoding="utf-8")
    )
    summary = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "completed_at": utc_now(),
        "status": "classification_complete",
        "primary_variant": primary_variant,
        "primary_variant_selection": "train-only paper-grouped OOF macro-F1",
        "selection_lock_path": str(args.output_root / "selection" / "selection_lock.json"),
        "dataset": {
            "train_examples": len(train),
            "train_documents": len({str(row["document_id"]) for row in train}),
            "validation_examples": len(validation),
            "validation_documents": len({str(row["document_id"]) for row in validation}),
            "train_distribution": {
                str(tokens): int(np.sum(train_targets == index))
                for index, tokens in enumerate(CLASS_TOKENS)
            },
            "validation_distribution": {
                str(tokens): int(np.sum(validation_targets == index))
                for index, tokens in enumerate(CLASS_TOKENS)
            },
        },
        "features": {
            "source": "frozen Phase 3A same-paper similarity-distribution features",
            "source_hashes": EXPECTED_SOURCE_HASHES,
            "evidence_used": False,
            "answers_used": False,
            "retrieval_f1_used": False,
            "oracle_used_as_feature": False,
        },
        "class_weighting": "square-root inverse frequency computed within each training fold",
        "models": final_results,
        "primary_validation_metrics": primary_metrics,
        "references": {
            "train_prior_majority": {
                "class": CLASS_TOKENS[train_majority_index],
                "selection_status": "deployable_train_only_reference",
                "validation_metrics": majority_reference(
                    validation_targets, train_majority_index
                ),
            },
            "validation_oracle_majority": {
                "class": CLASS_TOKENS[validation_majority_index],
                "selection_status": "descriptive_non_deployable_validation_label_reference",
                "validation_metrics": majority_reference(
                    validation_targets, validation_majority_index
                ),
            },
            "phase3a_primary": source_phase3a["classification"],
        },
        "primary_prediction_path": str(primary_path),
        "training_and_validation_wall_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_root / "classification_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def retrieve_command(args: argparse.Namespace) -> int:
    import qwen_phase1 as phase1
    import similarity_tree_phase3a as phase3a

    predictions = read_jsonl(args.output_root / "validation" / "predictions.jsonl")
    if len(predictions) != 924 or len({str(row["question_id"]) for row in predictions}) != 924:
        raise RuntimeError("Expected 924 unique Phase 3B validation predictions")
    result_path = args.output_root / "retrieval" / "results.jsonl"
    existing_rows = read_jsonl(result_path) if result_path.exists() else []
    existing = {str(row["question_id"]): row for row in existing_rows}
    expected_ids = {str(row["question_id"]) for row in predictions}
    if len(existing) != len(existing_rows) or not set(existing).issubset(expected_ids):
        raise RuntimeError("Invalid Phase 3B retrieval recovery records")
    before_client = phase3a.qdrant_client(prefer_grpc=False)
    try:
        before = phase3a.collection_snapshot(before_client)
    finally:
        before_client.close()
    client = phase1.qdrant_client()
    started = time.perf_counter()
    primary_variant = str(predictions[0]["variant"])
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
                    evaluation_run_id=f"phase3b-xgboost-{primary_variant}",
                    evaluation_config_hash=SOURCE_ORACLE_CONFIG_HASH,
                )
            )
            if len(records) != 1:
                raise RuntimeError(f"Expected one retrieval record for {question_id}")
            record = dict(records[0])
            record.update(
                {
                    "method_name": "phase3b-xgboost-similarity-tree-router",
                    "phase": PHASE,
                    "formulation_version": FORMULATION_VERSION,
                    "router_model": primary_variant,
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
                            "event": "phase3b_retrieval_progress",
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
    after_client = phase3a.qdrant_client(prefer_grpc=False)
    try:
        after = phase3a.collection_snapshot(after_client)
    finally:
        after_client.close()
    if before != after:
        raise RuntimeError("Qdrant collections changed during Phase 3B retrieval")
    values = [float(row["f1_joined_topk"]) for row in ordered]
    summary = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "router_model": primary_variant,
        "evaluated_examples": len(ordered),
        "valid_prediction_retrievals": len(ordered),
        "retrieval_coverage": len(ordered) / len(predictions),
        "mean_joined_retrieval_f1": statistics.fmean(values),
        "median_joined_retrieval_f1": statistics.median(values),
        "coverage_adjusted_full_set_mean_joined_retrieval_f1": sum(values) / len(predictions),
        "top_k": TOP_K,
        "paper_restricted": True,
        "embedding_model": phase1.OPENAI_EMBEDDING_MODEL,
        "embedding_dimension": phase1.EMBEDDING_DIM,
        "similarity": "cosine",
        "retrieval_wall_seconds_this_invocation": time.perf_counter() - started,
        "qdrant_collections_unchanged": True,
        "qdrant_snapshot_before": before,
        "qdrant_snapshot_after": after,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
    }
    atomic_json(args.output_root / "retrieval" / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def verify_primary_model(
    output_root: Path, source_root: Path, classification: dict
) -> dict:
    import xgboost as xgb

    variant = str(classification["primary_variant"])
    feature_key = str(classification["models"][variant]["feature_key"])
    rows = read_jsonl(feature_path(source_root, "validation"))
    features, names = feature_matrix(rows, feature_key)
    booster = xgb.Booster()
    model_path = Path(classification["models"][variant]["model_path"])
    booster.load_model(model_path)
    probabilities = predict_booster(booster, features, names)
    predictions = probabilities.argmax(axis=1)
    saved = read_jsonl(output_root / "validation" / "predictions.jsonl")
    if len(saved) != len(rows):
        raise RuntimeError("Primary prediction count changed")
    saved_probabilities = np.asarray(
        [
            [float(row["probabilities"][str(tokens)]) for tokens in CLASS_TOKENS]
            for row in saved
        ],
        dtype=np.float64,
    )
    maximum_difference = float(np.max(np.abs(probabilities - saved_probabilities)))
    identities_match = all(
        str(source["question_id"]) == str(prediction["question_id"])
        for source, prediction in zip(rows, saved)
    )
    labels_match = all(
        CLASS_TOKENS[int(predictions[index])] == int(saved[index]["predicted_label"])
        for index in range(len(saved))
    )
    targets = target_array(rows)
    recomputed_metrics = classification_metrics(targets, predictions, probabilities)
    metrics_match = stable_hash(recomputed_metrics) == stable_hash(
        classification["primary_validation_metrics"]
    )
    if not identities_match or not labels_match or maximum_difference > 1e-12 or not metrics_match:
        raise RuntimeError("Reloaded Phase 3B model does not reproduce saved validation results")
    canonical = output_root / "validation" / "predictions.jsonl"
    variant_path = output_root / "validation" / f"{variant}_predictions.jsonl"
    if sha256_file(canonical) != sha256_file(variant_path):
        raise RuntimeError("Canonical and selected-variant predictions differ")
    return {
        "model_reload_succeeded": True,
        "prediction_identities_match": identities_match,
        "predicted_labels_match": labels_match,
        "maximum_absolute_probability_difference": maximum_difference,
        "metrics_match": metrics_match,
        "canonical_predictions_match_selected_variant": True,
        "model_sha256": sha256_file(model_path),
        "prediction_sha256": sha256_file(canonical),
    }


def finalize_command(args: argparse.Namespace) -> int:
    if {
        "train": sha256_file(feature_path(args.source_root, "train")),
        "validation": sha256_file(feature_path(args.source_root, "validation")),
    } != EXPECTED_SOURCE_HASHES:
        raise RuntimeError("Frozen Phase 3A features changed before finalization")
    classification = json.loads(
        (args.output_root / "classification_summary.json").read_text(encoding="utf-8")
    )
    retrieval = json.loads(
        (args.output_root / "retrieval" / "summary.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (args.output_root / "selection" / "selection_lock.json").read_text(encoding="utf-8")
    )
    if not retrieval["qdrant_collections_unchanged"]:
        raise RuntimeError("Phase 3B retrieval integrity did not pass")
    model_verification = verify_primary_model(
        args.output_root, args.source_root, classification
    )
    cross_validation_times = {}
    for variant in FEATURE_VARIANTS:
        result = json.loads(
            (args.output_root / "cross_validation" / f"{variant}.json").read_text(
                encoding="utf-8"
            )
        )
        cross_validation_times[variant] = float(result["wall_seconds"])
    runtime = {
        "cross_validation_seconds_by_variant": cross_validation_times,
        "cross_validation_seconds_sum": sum(cross_validation_times.values()),
        "recovery_final_fit_and_validation_invocation_seconds": float(
            classification["training_and_validation_wall_seconds"]
        ),
        "retrieval_seconds": float(retrieval["retrieval_wall_seconds_this_invocation"]),
    }
    runtime["known_recorded_computational_stage_seconds"] = (
        runtime["cross_validation_seconds_sum"]
        + runtime["recovery_final_fit_and_validation_invocation_seconds"]
        + runtime["retrieval_seconds"]
    )
    runtime["note"] = (
        "The sum uses durable per-candidate CV times, the recovery invocation that wrote "
        "the selection lock and final validation, and retrieval. Interactive audit and "
        "orchestration overhead are not included."
    )
    atomic_json(args.output_root / "runtime" / "summary.json", runtime)
    artifacts = []
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path.name not in {"final_summary.json", "final_audit.json"}:
            artifacts.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    final = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "completed_at": utc_now(),
        "status": "complete",
        "primary_variant": classification["primary_variant"],
        "selection": selection,
        "classification": classification["primary_validation_metrics"],
        "models": classification["models"],
        "references": classification["references"],
        "retrieval": retrieval,
        "dataset": classification["dataset"],
        "features": classification["features"],
        "integrity": {
            "source_phase3a_features_unchanged": True,
            "qdrant_read_only": True,
            "qdrant_snapshot_unchanged": True,
            "legacy_venv_modified": False,
            "qwen_venv_modified": False,
            "model_reload_verification": model_verification,
        },
        "runtime": runtime,
        "artifacts": artifacts,
    }
    atomic_json(args.output_root / "final_summary.json", final)
    audit = {
        "verified_at": utc_now(),
        "status": "passed",
        "source_feature_hashes": EXPECTED_SOURCE_HASHES,
        "primary_prediction_rows": len(
            read_jsonl(args.output_root / "validation" / "predictions.jsonl")
        ),
        "retrieval_rows": len(read_jsonl(args.output_root / "retrieval" / "results.jsonl")),
        "qdrant_snapshot_matches": True,
        "model_reload_verification": model_verification,
        "artifact_count_before_final_files": len(artifacts),
    }
    atomic_json(args.output_root / "integrity" / "final_audit.json", audit)
    print(json.dumps(final, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit").set_defaults(func=audit_command)
    subparsers.add_parser("train-evaluate").set_defaults(func=train_evaluate_command)
    subparsers.add_parser("retrieve").set_defaults(func=retrieve_command)
    subparsers.add_parser("finalize").set_defaults(func=finalize_command)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
