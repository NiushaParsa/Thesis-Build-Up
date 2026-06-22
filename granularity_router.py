#!/usr/bin/env python
"""Train and run a question-embedding-only QASPER granularity router."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from qdrant_client.models import FieldCondition, Filter, MatchValue
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    EMBEDDING_DIM,
    PAPER_QUESTION_COLLECTION,
    ROUTER_DATASET_COLLECTION,
    ROUTER_MODEL_DIR,
    ROUTER_RANDOM_SEED,
)
from qdrant_schema import get_qdrant_client


logger = logging.getLogger(__name__)
ROUTER_ARTIFACT_VERSION = 1
CLASS_TOKENS = (10, 20, 40, 80, 160)
ALLOWED_SPLITS = {"train", "validation", "test"}
MODEL_SELECTION_METRIC = "macro_f1"


def _dense_vector(vector: Any, point_id: str) -> List[float]:
    if isinstance(vector, list):
        return [float(value) for value in vector]
    if isinstance(vector, dict) and len(vector) == 1:
        value = next(iter(vector.values()))
        if isinstance(value, list):
            return [float(item) for item in value]
    raise ValueError(f"Point {point_id} has no usable dense question vector")


def target_to_tokens(target: Any, class_tokens: Sequence[int] = CLASS_TOKENS) -> int:
    """Convert stored granularity levels (1..N) or token labels to token classes."""
    try:
        value = int(target)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid router target: {target!r}") from exc
    if value in class_tokens:
        return value
    if 1 <= value <= len(class_tokens):
        return int(class_tokens[value - 1])
    raise ValueError(f"Router target {value} is not a configured granularity")


def _scroll_split(client, collection: str, split: str, config_hash: Optional[str]) -> list:
    must = [FieldCondition(key="split", match=MatchValue(value=split))]
    if config_hash:
        must.append(
            FieldCondition(
                key="evaluation_config_hash", match=MatchValue(value=config_hash)
            )
        )
    points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=must),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    return points


def load_router_examples(
    client,
    *,
    collection: str,
    splits: Sequence[str],
    config_hash: Optional[str] = None,
    expected_dimension: Optional[int] = None,
) -> Tuple[List[dict], str]:
    """Load only requested splits and reject ambiguous oracle configurations."""
    invalid_splits = set(splits) - ALLOWED_SPLITS
    if invalid_splits:
        raise ValueError(f"Unknown QASPER splits: {sorted(invalid_splits)}")
    points = []
    for split in splits:
        points.extend(_scroll_split(client, collection, split, config_hash))
    hashes = {
        (point.payload or {}).get("evaluation_config_hash") for point in points
    } - {None, ""}
    if config_hash:
        selected_hash = config_hash
        if hashes and hashes != {config_hash}:
            raise ValueError("Qdrant returned records outside the requested config hash")
    elif len(hashes) == 1:
        selected_hash = next(iter(hashes))
    elif not hashes:
        raise ValueError(f"No router examples found for splits {list(splits)}")
    else:
        counts = Counter(
            (point.payload or {}).get("evaluation_config_hash") for point in points
        )
        raise ValueError(
            "Multiple oracle configurations are present; pass --evaluation-config-hash: "
            f"{dict(counts)}"
        )

    examples = []
    seen_questions = set()
    embedding_models = set()
    label_versions = set()
    dimensions = set()
    for point in points:
        payload = point.payload or {}
        if payload.get("evaluation_config_hash") != selected_hash:
            continue
        point_id = str(point.id)
        question_id = str(payload.get("question_id") or point_id)
        if question_id in seen_questions:
            raise ValueError(
                f"Duplicate question {question_id} for configuration {selected_hash}"
            )
        seen_questions.add(question_id)
        split = payload.get("split")
        if split not in splits:
            raise ValueError(f"Question {question_id} has unexpected split {split!r}")
        vector = _dense_vector(point.vector, point_id)
        if any(not math.isfinite(value) for value in vector):
            raise ValueError(f"Question {question_id} has non-finite vector values")
        dimensions.add(len(vector))
        embedding_models.add(payload.get("embedding_model"))
        label_versions.add(payload.get("label_version"))
        examples.append(
            {
                "question_id": question_id,
                "document_id": str(payload.get("document_id", "")),
                "split": split,
                "question_text": payload.get("question_text", ""),
                "vector": vector,
                "target_tokens": target_to_tokens(payload.get("router_target_granularity")),
                "embedding_model": payload.get("embedding_model"),
                "label_version": payload.get("label_version"),
            }
        )
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent question-vector dimensions: {sorted(dimensions)}")
    dimension = next(iter(dimensions))
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"Question-vector dimension {dimension} does not match {expected_dimension}"
        )
    if len(embedding_models) != 1 or None in embedding_models:
        raise ValueError(f"Inconsistent embedding model identities: {embedding_models}")
    if len(label_versions) != 1 or None in label_versions:
        raise ValueError(f"Inconsistent label versions: {label_versions}")
    validate_split_isolation(examples)
    return examples, selected_hash


def validate_split_isolation(examples: Sequence[dict]) -> None:
    """Reject question or document identities assigned to multiple splits."""
    question_splits: Dict[str, set] = {}
    document_splits: Dict[str, set] = {}
    for example in examples:
        question_splits.setdefault(example["question_id"], set()).add(example["split"])
        document_splits.setdefault(example["document_id"], set()).add(example["split"])
    question_leaks = {
        key: sorted(value) for key, value in question_splits.items() if len(value) > 1
    }
    document_leaks = {
        key: sorted(value) for key, value in document_splits.items() if len(value) > 1
    }
    if question_leaks or document_leaks:
        raise ValueError(
            "Split leakage detected: "
            f"questions={question_leaks}, documents={document_leaks}"
        )


def examples_to_arrays(
    examples: Sequence[dict], class_tokens: Sequence[int] = CLASS_TOKENS
) -> Tuple[np.ndarray, np.ndarray]:
    """Return question vectors and class indices; no payload metric is a feature."""
    if not examples:
        raise ValueError("Cannot build arrays from an empty example list")
    class_to_index = {tokens: index for index, tokens in enumerate(class_tokens)}
    features = np.asarray([example["vector"] for example in examples], dtype=np.float32)
    targets = np.asarray(
        [class_to_index[example["target_tokens"]] for example in examples],
        dtype=np.int64,
    )
    return features, targets


def fit_preprocessor(features: np.ndarray, standardize: bool) -> dict:
    if not standardize:
        return {"standardize": False, "mean": None, "scale": None}
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0.0] = 1.0
    return {"standardize": True, "mean": mean, "scale": scale}


def transform_features(features: np.ndarray, preprocessing: dict) -> np.ndarray:
    if not preprocessing["standardize"]:
        return features.astype(np.float32, copy=False)
    return ((features - preprocessing["mean"]) / preprocessing["scale"]).astype(
        np.float32
    )


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_tokens: Sequence[int] = CLASS_TOKENS,
) -> dict:
    """Compute fixed-label multiclass metrics without external metric packages."""
    class_count = len(class_tokens)
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
    total = int(support.sum())
    top2 = np.argsort(-probabilities, axis=1, kind="stable")[:, :2]
    top2_accuracy = float(
        np.mean([target in candidates for target, candidates in zip(targets, top2)])
    )
    present = support > 0
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "balanced_accuracy": float(recall[present].mean()),
        "top_2_accuracy": top2_accuracy,
        "per_class": {
            str(tokens): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, tokens in enumerate(class_tokens)
        },
        "confusion_matrix": confusion.tolist(),
        "class_distribution": {
            str(tokens): int(support[index])
            for index, tokens in enumerate(class_tokens)
        },
    }


class LinearRouter(nn.Module):
    def __init__(self, input_dimension: int, class_count: int):
        super().__init__()
        self.classifier = nn.Linear(input_dimension, class_count)

    def forward(self, features):
        return self.classifier(features)


class MLPRouter(nn.Module):
    def __init__(
        self, input_dimension: int, class_count: int, hidden_size: int, dropout: float
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, class_count),
        )

    def forward(self, features):
        return self.network(features)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_network(
    *,
    model_type: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    input_dimension: int,
    class_count: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    hidden_size: Optional[int] = None,
    dropout: float = 0.0,
) -> Tuple[nn.Module, List[float]]:
    _set_seed(seed)
    if model_type == "logistic_regression":
        model = LinearRouter(input_dimension, class_count)
    elif model_type == "mlp":
        model = MLPRouter(input_dimension, class_count, int(hidden_size), dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    dataset = TensorDataset(
        torch.from_numpy(train_features), torch.from_numpy(train_targets)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    losses = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for features, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(features)
        losses.append(epoch_loss / len(dataset))
    model.eval()
    return model, losses


def predict_probabilities(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(features)), dim=1).cpu().numpy()


def majority_predictions(
    train_targets: np.ndarray, count: int, class_count: int
) -> Tuple[int, np.ndarray, np.ndarray]:
    counts = np.bincount(train_targets, minlength=class_count)
    majority = int(np.flatnonzero(counts == counts.max())[0])
    predictions = np.full(count, majority, dtype=np.int64)
    class_probabilities = counts.astype(np.float32) / counts.sum()
    probabilities = np.repeat(class_probabilities[None, :], count, axis=0)
    return majority, predictions, probabilities


def _candidate_key(candidate: dict) -> tuple:
    metrics = candidate["validation_metrics"]
    return (metrics[MODEL_SELECTION_METRIC], metrics["accuracy"])


def mlp_is_justified(
    logistic_metrics: dict, mlp_metrics: dict, minimum_improvement: float
) -> Tuple[bool, float]:
    improvement = (
        mlp_metrics[MODEL_SELECTION_METRIC]
        - logistic_metrics[MODEL_SELECTION_METRIC]
    )
    return improvement >= minimum_improvement, improvement


def tune_models(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    logistic_learning_rates: Sequence[float],
    weight_decays: Sequence[float],
    epochs: int,
    batch_size: int,
    seed: int,
    enable_mlp: bool,
    mlp_hidden_sizes: Sequence[int],
    mlp_dropouts: Sequence[float],
    mlp_learning_rates: Sequence[float],
    mlp_min_improvement: float,
) -> dict:
    class_count = len(CLASS_TOKENS)
    input_dimension = train_features.shape[1]
    majority_index, majority_pred, majority_prob = majority_predictions(
        train_targets, len(validation_targets), class_count
    )
    majority_metrics = classification_metrics(
        validation_targets, majority_pred, majority_prob
    )

    logistic_candidates = []
    best_logistic_model = None
    candidate_index = 0
    for learning_rate in logistic_learning_rates:
        for weight_decay in weight_decays:
            candidate_seed = seed + candidate_index
            model, losses = train_network(
                model_type="logistic_regression",
                train_features=train_features,
                train_targets=train_targets,
                input_dimension=input_dimension,
                class_count=class_count,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                seed=candidate_seed,
            )
            probabilities = predict_probabilities(model, validation_features)
            predictions = probabilities.argmax(axis=1)
            candidate = {
                "model_type": "logistic_regression",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": candidate_seed,
                "final_train_loss": losses[-1],
                "validation_metrics": classification_metrics(
                    validation_targets, predictions, probabilities
                ),
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            }
            logistic_candidates.append(candidate)
            if best_logistic_model is None or _candidate_key(candidate) > _candidate_key(
                best_logistic_model
            ):
                best_logistic_model = candidate
            candidate_index += 1

    best_mlp_model = None
    mlp_candidates = []
    if enable_mlp:
        candidate_index = 0
        for hidden_size in mlp_hidden_sizes:
            for dropout in mlp_dropouts:
                for learning_rate in mlp_learning_rates:
                    candidate_seed = seed + 10_000 + candidate_index
                    model, losses = train_network(
                        model_type="mlp",
                        train_features=train_features,
                        train_targets=train_targets,
                        input_dimension=input_dimension,
                        class_count=class_count,
                        learning_rate=learning_rate,
                        weight_decay=weight_decays[0],
                        epochs=epochs,
                        batch_size=batch_size,
                        seed=candidate_seed,
                        hidden_size=hidden_size,
                        dropout=dropout,
                    )
                    probabilities = predict_probabilities(model, validation_features)
                    predictions = probabilities.argmax(axis=1)
                    candidate = {
                        "model_type": "mlp",
                        "hidden_size": hidden_size,
                        "dropout": dropout,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decays[0],
                        "seed": candidate_seed,
                        "final_train_loss": losses[-1],
                        "validation_metrics": classification_metrics(
                            validation_targets, predictions, probabilities
                        ),
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                    }
                    mlp_candidates.append(candidate)
                    if best_mlp_model is None or _candidate_key(candidate) > _candidate_key(
                        best_mlp_model
                    ):
                        best_mlp_model = candidate
                    candidate_index += 1

    selected_type = "logistic_regression"
    mlp_justified = False
    if best_mlp_model is not None:
        mlp_justified, improvement = mlp_is_justified(
            best_logistic_model["validation_metrics"],
            best_mlp_model["validation_metrics"],
            mlp_min_improvement,
        )
        if mlp_justified:
            selected_type = "mlp"
    else:
        improvement = None

    return {
        "majority_class_index": majority_index,
        "majority_class_probabilities": majority_prob[0].tolist(),
        "majority_validation_metrics": majority_metrics,
        "best_logistic": best_logistic_model,
        "logistic_candidates": logistic_candidates,
        "best_mlp": best_mlp_model,
        "mlp_candidates": mlp_candidates,
        "mlp_validation_improvement": improvement,
        "mlp_justified": mlp_justified,
        "selected_model_type": selected_type,
    }


def _json_safe_candidate(candidate: Optional[dict]) -> Optional[dict]:
    if candidate is None:
        return None
    return {key: value for key, value in candidate.items() if key != "state_dict"}


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_numbers(value: str, converter) -> list:
    parsed = [converter(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("At least one comma-separated value is required")
    return parsed


def train_command(args) -> int:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.mlp_min_improvement < 0:
        raise ValueError("mlp-min-improvement cannot be negative")
    if any(value <= 0 for value in args.logistic_learning_rates):
        raise ValueError("logistic learning rates must be positive")
    if any(value < 0 for value in args.weight_decays):
        raise ValueError("weight decays cannot be negative")
    if any(value <= 0 for value in args.mlp_hidden_sizes):
        raise ValueError("MLP hidden sizes must be positive")
    if any(value <= 0 for value in args.mlp_learning_rates):
        raise ValueError("MLP learning rates must be positive")
    if any(not 0.0 <= dropout < 1.0 for dropout in args.mlp_dropouts):
        raise ValueError("MLP dropout values must be in [0, 1)")
    client = get_qdrant_client()
    try:
        examples, config_hash = load_router_examples(
            client,
            collection=args.collection,
            splits=["train", "validation"],
            config_hash=args.evaluation_config_hash,
            expected_dimension=args.embedding_dimension,
        )
        test_examples = []
        if args.evaluate_test:
            test_examples, _ = load_router_examples(
                client,
                collection=args.collection,
                splits=["test"],
                config_hash=config_hash,
                expected_dimension=args.embedding_dimension,
            )
            validate_split_isolation([*examples, *test_examples])
    finally:
        client.close()

    train_examples = [item for item in examples if item["split"] == "train"]
    validation_examples = [item for item in examples if item["split"] == "validation"]
    if not train_examples:
        raise ValueError("No QASPER train router examples exist for the selected configuration")
    if not validation_examples:
        raise ValueError(
            "No QASPER validation router examples exist; validation is required for tuning"
        )

    train_x, train_y = examples_to_arrays(train_examples)
    validation_x, validation_y = examples_to_arrays(validation_examples)
    preprocessing = fit_preprocessor(train_x, args.standardize)
    train_x = transform_features(train_x, preprocessing)
    validation_x = transform_features(validation_x, preprocessing)

    tuned = tune_models(
        train_features=train_x,
        train_targets=train_y,
        validation_features=validation_x,
        validation_targets=validation_y,
        logistic_learning_rates=args.logistic_learning_rates,
        weight_decays=args.weight_decays,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        enable_mlp=args.enable_mlp,
        mlp_hidden_sizes=args.mlp_hidden_sizes,
        mlp_dropouts=args.mlp_dropouts,
        mlp_learning_rates=args.mlp_learning_rates,
        mlp_min_improvement=args.mlp_min_improvement,
    )

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "standardize": args.standardize,
        "logistic_learning_rates": args.logistic_learning_rates,
        "weight_decays": args.weight_decays,
        "enable_mlp": args.enable_mlp,
        "mlp_hidden_sizes": args.mlp_hidden_sizes,
        "mlp_dropouts": args.mlp_dropouts,
        "mlp_learning_rates": args.mlp_learning_rates,
        "mlp_min_improvement": args.mlp_min_improvement,
        "model_selection_metric": MODEL_SELECTION_METRIC,
        "test_evaluated": args.evaluate_test,
    }
    embedding_model = train_examples[0]["embedding_model"]
    label_version = train_examples[0]["label_version"]
    artifact = {
        "artifact_version": ROUTER_ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_model_type": tuned["selected_model_type"],
        "class_tokens": list(CLASS_TOKENS),
        "class_to_index": {str(tokens): index for index, tokens in enumerate(CLASS_TOKENS)},
        "index_to_class": {str(index): tokens for index, tokens in enumerate(CLASS_TOKENS)},
        "embedding_model": embedding_model,
        "embedding_dimension": train_x.shape[1],
        "oracle_evaluation_config_hash": config_hash,
        "oracle_label_version": label_version,
        "random_seed": args.seed,
        "training_config": training_config,
        "preprocessing": {
            "standardize": preprocessing["standardize"],
            "mean": preprocessing["mean"],
            "scale": preprocessing["scale"],
        },
        "majority_class_index": tuned["majority_class_index"],
        "majority_class_probabilities": tuned["majority_class_probabilities"],
        "logistic_config": {
            key: value
            for key, value in tuned["best_logistic"].items()
            if key not in {"state_dict", "validation_metrics"}
        },
        "logistic_state_dict": tuned["best_logistic"]["state_dict"],
        "mlp_config": (
            {
                key: value
                for key, value in tuned["best_mlp"].items()
                if key not in {"state_dict", "validation_metrics"}
            }
            if tuned["best_mlp"]
            else None
        ),
        "mlp_state_dict": (
            tuned["best_mlp"]["state_dict"] if tuned["best_mlp"] else None
        ),
        "git_revision": _git_revision(),
    }

    selected_validation_metrics = (
        tuned["best_mlp"]["validation_metrics"]
        if tuned["selected_model_type"] == "mlp"
        else tuned["best_logistic"]["validation_metrics"]
    )
    comparison_metrics = (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "top_2_accuracy",
    )
    report = {
        "artifact_version": ROUTER_ARTIFACT_VERSION,
        "selected_model_type": tuned["selected_model_type"],
        "mlp_justified": tuned["mlp_justified"],
        "mlp_validation_improvement": tuned["mlp_validation_improvement"],
        "oracle_evaluation_config_hash": config_hash,
        "embedding_model": embedding_model,
        "embedding_dimension": train_x.shape[1],
        "label_version": label_version,
        "random_seed": args.seed,
        "class_tokens": list(CLASS_TOKENS),
        "dataset": {
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "test_examples": len(test_examples) if args.evaluate_test else "untouched",
            "train_documents": len({item["document_id"] for item in train_examples}),
            "validation_documents": len(
                {item["document_id"] for item in validation_examples}
            ),
            "train_class_distribution": dict(
                Counter(str(item["target_tokens"]) for item in train_examples)
            ),
            "validation_class_distribution": dict(
                Counter(str(item["target_tokens"]) for item in validation_examples)
            ),
        },
        "majority_validation_metrics": tuned["majority_validation_metrics"],
        "logistic_validation_metrics": tuned["best_logistic"]["validation_metrics"],
        "mlp_validation_metrics": (
            tuned["best_mlp"]["validation_metrics"] if tuned["best_mlp"] else None
        ),
        "selected_validation_metrics": selected_validation_metrics,
        "selected_comparison_to_majority": {
            metric: selected_validation_metrics[metric]
            - tuned["majority_validation_metrics"][metric]
            for metric in comparison_metrics
        },
        "logistic_comparison_to_majority": {
            metric: tuned["best_logistic"]["validation_metrics"][metric]
            - tuned["majority_validation_metrics"][metric]
            for metric in comparison_metrics
        },
        "mlp_comparison_to_majority": (
            {
                metric: tuned["best_mlp"]["validation_metrics"][metric]
                - tuned["majority_validation_metrics"][metric]
                for metric in comparison_metrics
            }
            if tuned["best_mlp"]
            else None
        ),
        "logistic_candidates": [
            _json_safe_candidate(candidate) for candidate in tuned["logistic_candidates"]
        ],
        "mlp_candidates": [
            _json_safe_candidate(candidate) for candidate in tuned["mlp_candidates"]
        ],
        "training_config": training_config,
    }

    if args.evaluate_test:
        test_x, test_y = examples_to_arrays(test_examples)
        report["test_metrics"] = evaluate_artifact_models(artifact, test_x, test_y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "router_model.pt"
    report_path = args.output_dir / "training_report.json"
    metadata_path = args.output_dir / "metadata.json"
    torch.save(artifact, artifact_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metadata = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "logistic_state_dict",
            "mlp_state_dict",
            "preprocessing",
        }
    }
    metadata["preprocessing"] = {
        "standardize": preprocessing["standardize"],
        "feature_count": train_x.shape[1],
        "fitted_on": "QASPER train only",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(artifact_path), "report": report}, indent=2))
    return 0


def _build_model_from_artifact(artifact: dict, choice: str) -> Optional[nn.Module]:
    if choice == "majority":
        return None
    dimension = artifact["embedding_dimension"]
    class_count = len(artifact["class_tokens"])
    if choice == "logistic_regression":
        model = LinearRouter(dimension, class_count)
        model.load_state_dict(artifact["logistic_state_dict"])
    elif choice == "mlp":
        if artifact.get("mlp_state_dict") is None:
            raise ValueError("The artifact does not contain a trained MLP")
        config = artifact["mlp_config"]
        model = MLPRouter(
            dimension, class_count, config["hidden_size"], config["dropout"]
        )
        model.load_state_dict(artifact["mlp_state_dict"])
    else:
        raise ValueError(f"Unknown model choice: {choice}")
    model.eval()
    return model


def predict_with_artifact(
    artifact: dict, features: np.ndarray, model_choice: str = "primary"
) -> Tuple[np.ndarray, np.ndarray]:
    choice = (
        artifact["selected_model_type"] if model_choice == "primary" else model_choice
    )
    preprocessing = artifact["preprocessing"]
    transformed = transform_features(features, preprocessing)
    if choice == "majority":
        predictions = np.full(
            len(features), artifact["majority_class_index"], dtype=np.int64
        )
        probabilities = np.repeat(
            np.asarray(artifact["majority_class_probabilities"], dtype=np.float32)[
                None, :
            ],
            len(features),
            axis=0,
        )
        return predictions, probabilities
    model = _build_model_from_artifact(artifact, choice)
    probabilities = predict_probabilities(model, transformed)
    return probabilities.argmax(axis=1), probabilities


def evaluate_artifact_models(
    artifact: dict, features: np.ndarray, targets: np.ndarray
) -> dict:
    choices = ["majority", "logistic_regression"]
    if artifact.get("mlp_state_dict") is not None:
        choices.append("mlp")
    result = {}
    for choice in choices:
        predictions, probabilities = predict_with_artifact(artifact, features, choice)
        result[choice] = classification_metrics(targets, predictions, probabilities)
    return result


def _load_prediction_questions(client, args) -> list:
    if args.question_id:
        points = client.retrieve(
            collection_name=args.question_collection,
            ids=args.question_id,
            with_payload=True,
            with_vectors=True,
        )
    else:
        if not args.split:
            raise ValueError("Prediction requires --question-id or --split")
        points = []
        offset = None
        while len(points) < args.limit:
            batch, next_offset = client.scroll(
                collection_name=args.question_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="split", match=MatchValue(value=args.split))]
                ),
                limit=min(256, args.limit - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset
    return points


def predict_command(args) -> int:
    artifact = torch.load(args.model, map_location="cpu", weights_only=False)
    client = get_qdrant_client()
    try:
        points = _load_prediction_questions(client, args)
    finally:
        client.close()
    vectors = np.asarray(
        [_dense_vector(point.vector, str(point.id)) for point in points], dtype=np.float32
    )
    if vectors.shape[1] != artifact["embedding_dimension"]:
        raise ValueError(
            f"Prediction dimension {vectors.shape[1]} does not match model "
            f"dimension {artifact['embedding_dimension']}"
        )
    predictions, probabilities = predict_with_artifact(
        artifact, vectors, args.model_choice
    )
    class_tokens = artifact["class_tokens"]
    rows = []
    for point, prediction, probability in zip(points, predictions, probabilities):
        top2 = np.argsort(-probability, kind="stable")[:2]
        payload = point.payload or {}
        rows.append(
            {
                "question_id": str(point.id),
                "document_id": payload.get("document_id"),
                "split": payload.get("split"),
                "question_text": payload.get("question_text"),
                "predicted_granularity_tokens": class_tokens[int(prediction)],
                "top_2_granularities": [class_tokens[int(index)] for index in top2],
                "class_probabilities": {
                    str(tokens): float(probability[index])
                    for index, tokens in enumerate(class_tokens)
                },
                "model_choice": (
                    artifact["selected_model_type"]
                    if args.model_choice == "primary"
                    else args.model_choice
                ),
            }
        )
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    print(json.dumps(rows, indent=2))
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train from persisted oracle records")
    train.add_argument("--collection", default=ROUTER_DATASET_COLLECTION)
    train.add_argument("--evaluation-config-hash", default=None)
    train.add_argument("--output-dir", type=Path, default=Path(ROUTER_MODEL_DIR))
    train.add_argument("--embedding-dimension", type=int, default=EMBEDDING_DIM)
    train.add_argument("--seed", type=int, default=ROUTER_RANDOM_SEED)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument(
        "--logistic-learning-rates",
        type=lambda value: _parse_numbers(value, float),
        default=[0.01, 0.001],
    )
    train.add_argument(
        "--weight-decays",
        type=lambda value: _parse_numbers(value, float),
        default=[0.0, 0.0001],
    )
    standardization = train.add_mutually_exclusive_group()
    standardization.add_argument("--standardize", dest="standardize", action="store_true")
    standardization.add_argument(
        "--no-standardize", dest="standardize", action="store_false"
    )
    train.set_defaults(standardize=True)
    train.add_argument("--enable-mlp", action="store_true")
    train.add_argument(
        "--mlp-hidden-sizes",
        type=lambda value: _parse_numbers(value, int),
        default=[64],
    )
    train.add_argument(
        "--mlp-dropouts",
        type=lambda value: _parse_numbers(value, float),
        default=[0.1],
    )
    train.add_argument(
        "--mlp-learning-rates",
        type=lambda value: _parse_numbers(value, float),
        default=[0.001],
    )
    train.add_argument("--mlp-min-improvement", type=float, default=0.01)
    train.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicit final-only test evaluation; omitted by default",
    )
    train.set_defaults(func=train_command)

    predict = subparsers.add_parser("predict", help="Predict from PaperQuestion vectors")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--question-collection", default=PAPER_QUESTION_COLLECTION)
    predict.add_argument("--question-id", action="append")
    predict.add_argument("--split", choices=sorted(ALLOWED_SPLITS))
    predict.add_argument("--limit", type=int, default=100)
    predict.add_argument(
        "--model-choice",
        choices=["primary", "majority", "logistic_regression", "mlp"],
        default="primary",
    )
    predict.add_argument("--output-jsonl", type=Path, default=None)
    predict.set_defaults(func=predict_command)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    )
    args = parse_args()
    try:
        return args.func(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
