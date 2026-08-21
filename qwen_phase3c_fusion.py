#!/usr/bin/env python
"""Phase 3C fusion of frozen Phase 2D Qwen and Phase 3B tree features.

The Qwen checkpoint is never updated.  Its final non-padding hidden state and
five classifier logits are extracted for the preserved Phase 3A question rows.
Two XGBoost fusion variants are selected using the existing paper-grouped
training folds, before the canonical 924-example validation predictions are
written.  Retrieval reuses the unchanged Phase 3B read-only evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import similarity_tree_phase3b as phase3b


PHASE = "Phase 3C"
FORMULATION_VERSION = "qwen-phase2d-hidden-state-plus-phase3b-tree-xgboost-v1"
CLASS_TOKENS = phase3b.CLASS_TOKENS
SEED = phase3b.SEED
FOLDS = phase3b.FOLDS
TOP_K = phase3b.TOP_K
MAX_SEQUENCE_LENGTH = 128
SOURCE_FEATURE_ROOT = phase3b.SOURCE_FEATURE_ROOT
EXPECTED_SOURCE_HASHES = phase3b.EXPECTED_SOURCE_HASHES
DEFAULT_OUTPUT_ROOT = Path("outputs/qwen_phase3c_fusion_evidence_length_oracle")
PHASE2D_ROOT = Path(
    "outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle"
)
PHASE2D_CHECKPOINT = PHASE2D_ROOT / (
    "runs/qwen-phase2d-base-sequence-classifier-token-count-prompt-full-parameter-"
    "20260808-seed42-v1/checkpoints/step-000213/model"
)
PHASE3B_ROOT = Path("outputs/similarity_tree_phase3b_xgboost_evidence_length_oracle")
EXPECTED_CHECKPOINT_SHA256 = (
    "020af0a83af773239e7e60e9983afad29cae3f31493c7073e9162e040b732814"
)
SUPERVISOR_INSTRUCTION = (
    "You are a router for a retrieval-augmented generation system. Based only "
    "on the question, select the option representing the context size most "
    "suitable for retrieving the evidence required to answer it. Choose exactly "
    "one value from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, "
    "4 = 80 tokens, 5 = 160 tokens. Return only the number"
)
FUSION_VARIANTS = {
    "qwen_logits_tree": ("qwen_logits", "tree_features"),
    "qwen_hidden_tree": ("qwen_hidden", "tree_features"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classifier_text(question_text: str) -> str:
    return f"{SUPERVISOR_INSTRUCTION}\n\nQuestion: {str(question_text)}"


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def npz_path(output_root: Path, split: str) -> Path:
    return output_root / "qwen_features" / f"{split}_qwen_features.npz"


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_qwen_arrays(rows: Sequence[dict], arrays: Mapping[str, np.ndarray]) -> None:
    required = {"question_ids", "oracle_labels", "logits", "hidden", "token_counts"}
    if set(arrays) != required:
        raise RuntimeError(f"Unexpected Qwen feature arrays: {sorted(arrays)}")
    expected_ids = np.asarray([str(row["question_id"]) for row in rows])
    expected_labels = np.asarray([int(row["oracle_label"]) for row in rows], dtype=np.int64)
    if not np.array_equal(arrays["question_ids"], expected_ids):
        raise RuntimeError("Qwen feature question IDs do not align with Phase 3A rows")
    if not np.array_equal(arrays["oracle_labels"], expected_labels):
        raise RuntimeError("Qwen feature Oracle labels do not align with Phase 3A rows")
    if arrays["logits"].shape != (len(rows), len(CLASS_TOKENS)):
        raise RuntimeError(f"Unexpected Qwen logits shape: {arrays['logits'].shape}")
    if arrays["hidden"].shape[0] != len(rows) or arrays["hidden"].ndim != 2:
        raise RuntimeError(f"Unexpected Qwen hidden shape: {arrays['hidden'].shape}")
    if arrays["token_counts"].shape != (len(rows),):
        raise RuntimeError("Unexpected token-count shape")
    if int(arrays["token_counts"].max()) > MAX_SEQUENCE_LENGTH:
        raise RuntimeError("Saved Qwen input exceeded the Phase 2D sequence limit")
    if not np.isfinite(arrays["logits"]).all() or not np.isfinite(arrays["hidden"]).all():
        raise RuntimeError("Qwen features contain non-finite values")


def fusion_matrix(
    rows: Sequence[dict], arrays: Mapping[str, np.ndarray], variant: str
) -> tuple[np.ndarray, list[str]]:
    """Build one explicitly named Qwen/tree feature matrix."""

    validate_qwen_arrays(rows, arrays)
    if variant not in FUSION_VARIANTS:
        raise ValueError(f"Unknown fusion variant: {variant}")
    tree, tree_names = phase3b.feature_matrix(rows, "tree_features")
    sources = FUSION_VARIANTS[variant]
    parts: list[np.ndarray] = []
    names: list[str] = []
    for source in sources:
        if source == "qwen_logits":
            parts.append(np.asarray(arrays["logits"], dtype=np.float32))
            names.extend(f"qwen_logit_{tokens}" for tokens in CLASS_TOKENS)
        elif source == "qwen_hidden":
            hidden = np.asarray(arrays["hidden"], dtype=np.float32)
            parts.append(hidden)
            names.extend(f"qwen_hidden_{index:04d}" for index in range(hidden.shape[1]))
        elif source == "tree_features":
            parts.append(tree)
            names.extend(f"tree__{name}" for name in tree_names)
        else:
            raise RuntimeError(f"Unhandled feature source: {source}")
    matrix = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if matrix.shape != (len(rows), len(names)) or not np.isfinite(matrix).all():
        raise RuntimeError("Invalid fusion matrix")
    return matrix, names


def variant_key(name: str, result: dict) -> tuple[float, float, float, int]:
    metrics = result["selected_candidate"]["oof_metrics"]
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["accuracy"]),
        int(name == "qwen_hidden_tree"),
    )


def audit_command(args: argparse.Namespace) -> int:
    import xgboost as xgb

    train_path = phase3b.feature_path(args.source_root, "train")
    validation_path = phase3b.feature_path(args.source_root, "validation")
    hashes = {
        "train": phase3b.sha256_file(train_path),
        "validation": phase3b.sha256_file(validation_path),
    }
    if hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"Frozen Phase 3A feature hashes changed: {hashes}")
    checkpoint_file = args.checkpoint / "model.safetensors"
    if not checkpoint_file.exists():
        raise RuntimeError(f"Missing Phase 2D checkpoint: {checkpoint_file}")
    checkpoint_hash = phase3b.sha256_file(checkpoint_file)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Phase 2D checkpoint hash changed: {checkpoint_hash}")
    for path in (
        args.checkpoint / "config.json",
        args.checkpoint / "tokenizer.json",
        PHASE2D_ROOT / "final_summary.json",
        PHASE2D_ROOT / "validation" / "predictions.jsonl",
        PHASE3B_ROOT / "final_summary.json",
    ):
        if not path.exists():
            raise RuntimeError(f"Missing required preserved artifact: {path}")
    train = phase3b.read_jsonl(train_path)
    validation = phase3b.read_jsonl(validation_path)
    phase3b.validate_rows(train, "train", 2245)
    phase3b.validate_rows(validation, "validation", 924)
    train_documents = {str(row["document_id"]) for row in train}
    validation_documents = {str(row["document_id"]) for row in validation}
    if train_documents & validation_documents:
        raise RuntimeError("Train and validation papers overlap")
    targets = phase3b.target_array(train)
    folds = phase3b.grouped_stratified_folds(train, FOLDS, SEED)
    manifest = phase3b.fold_manifest(train, targets, folds)
    preserved_manifest = json.loads(
        (args.source_root / "cross_validation" / "paper_grouped_folds.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest != preserved_manifest:
        raise RuntimeError("Phase 3C grouped folds do not reproduce Phase 3A/3B")
    package_lock = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    phase3b.atomic_text(args.output_root / "environment" / "package_lock.txt", package_lock)
    environment = {
        "environment_name": ".venv-fusion",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "xgboost": xgb.__version__,
        "qwen_extraction_device": "CUDA",
        "fusion_classifier_device": "CPU",
        "legacy_venv_modified": False,
        "phase2d_artifacts_modified": False,
        "phase3a_artifacts_modified": False,
        "phase3b_artifacts_modified": False,
    }
    phase3b.atomic_json(args.output_root / "environment" / "python_environment.json", environment)
    configuration = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "created_at": utc_now(),
        "objective": "combine Phase 2D question representation with Phase 3B paper-specific similarity tree",
        "classes": list(CLASS_TOKENS),
        "source_phase2d_checkpoint": str(args.checkpoint),
        "source_phase2d_checkpoint_sha256": checkpoint_hash,
        "source_phase3a_feature_hashes": hashes,
        "source_phase3b_final_summary": str(PHASE3B_ROOT / "final_summary.json"),
        "qwen_frozen": True,
        "gradient_computation": False,
        "parameter_updates_to_qwen": 0,
        "prompt": SUPERVISOR_INSTRUCTION,
        "prompt_sha256": text_sha256(SUPERVISOR_INSTRUCTION),
        "input_template": "{instruction}\\n\\nQuestion: {original_question_text}",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "fusion_variants": FUSION_VARIANTS,
        "fusion_classifier": "XGBoost multi:softprob",
        "xgboost_grid": list(phase3b.GRID),
        "xgboost_grid_fingerprint": phase3b.stable_hash(phase3b.GRID),
        "class_weighting": "sqrt(maximum_fold_class_count / fold_class_count)",
        "selection": "five-fold paper-grouped train-only OOF macro-F1",
        "validation_policy": "fusion model selection locked before Phase 3C validation inference",
        "development_set_caveat": (
            "The frozen Phase 2D checkpoint was previously selected on this same validation split; "
            "Phase 3C is therefore a development-set result, not an unbiased final test estimate."
        ),
        "retrieval": {"top_k": TOP_K, "paper_restricted": True, "read_only": True},
    }
    phase3b.atomic_json(args.output_root / "configuration" / "experiment.json", configuration)
    preflight = {
        "phase": PHASE,
        "status": "passed",
        "audited_at": utc_now(),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "paper_overlap": 0,
        "source_feature_hashes": hashes,
        "checkpoint_sha256": checkpoint_hash,
        "fold_manifest_matches_phase3a_phase3b": True,
        "qdrant_contacted": False,
        "source_artifacts_modified": False,
        "environment": environment,
    }
    phase3b.atomic_json(args.output_root / "integrity" / "preflight_audit.json", preflight)
    phase3b.atomic_json(args.output_root / "cross_validation" / "paper_grouped_folds.json", manifest)
    print(json.dumps(preflight, indent=2))
    return 0


def format_input_ids(tokenizer: Any, row: Mapping[str, Any]) -> list[int]:
    encoded = tokenizer(
        classifier_text(str(row["question_text"])),
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    values = [int(value) for value in input_ids]
    if not values or len(values) > MAX_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Invalid Phase 2D input length {len(values)} for {row['question_id']}"
        )
    if tokenizer.pad_token_id is None or int(tokenizer.pad_token_id) in values:
        raise RuntimeError("Phase 2D padding contract is not reproducible")
    return values


def extract_split(
    *, model: Any, tokenizer: Any, torch: Any, rows: Sequence[dict], batch_size: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    all_logits: list[np.ndarray] = []
    all_hidden: list[np.ndarray] = []
    token_counts: list[int] = []
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        ids = [format_input_ids(tokenizer, row) for row in batch_rows]
        maximum = max(len(values) for values in ids)
        input_ids = torch.full(
            (len(ids), maximum), int(tokenizer.pad_token_id), dtype=torch.long
        )
        attention_mask = torch.zeros((len(ids), maximum), dtype=torch.long)
        last_positions = []
        for index, values in enumerate(ids):
            length = len(values)
            input_ids[index, :length] = torch.tensor(values, dtype=torch.long)
            attention_mask[index, :length] = 1
            last_positions.append(length - 1)
            token_counts.append(length)
        input_ids = input_ids.to("cuda", non_blocking=True)
        attention_mask = attention_mask.to("cuda", non_blocking=True)
        last_tensor = torch.tensor(last_positions, device="cuda", dtype=torch.long)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        if not outputs.hidden_states:
            raise RuntimeError("Qwen did not return hidden states")
        final_hidden = outputs.hidden_states[-1]
        pooled = final_hidden[
            torch.arange(len(batch_rows), device=final_hidden.device), last_tensor
        ]
        all_logits.append(outputs.logits.detach().float().cpu().numpy())
        all_hidden.append(pooled.detach().float().cpu().numpy())
        if (start + len(batch_rows)) % 256 == 0 or start + len(batch_rows) == len(rows):
            print(
                json.dumps(
                    {
                        "event": "phase3c_qwen_extraction_progress",
                        "complete": start + len(batch_rows),
                        "expected": len(rows),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    arrays = {
        "question_ids": np.asarray([str(row["question_id"]) for row in rows]),
        "oracle_labels": np.asarray([int(row["oracle_label"]) for row in rows], dtype=np.int64),
        "logits": np.concatenate(all_logits).astype(np.float32),
        "hidden": np.concatenate(all_hidden).astype(np.float32),
        "token_counts": np.asarray(token_counts, dtype=np.int64),
    }
    validate_qwen_arrays(rows, arrays)
    return arrays, {
        "examples": len(rows),
        "wall_seconds": time.perf_counter() - started,
        "mean_seconds_per_example": (time.perf_counter() - started) / len(rows),
        "hidden_size": int(arrays["hidden"].shape[1]),
        "minimum_token_count": int(arrays["token_counts"].min()),
        "maximum_token_count": int(arrays["token_counts"].max()),
    }


def extract_qwen_command(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (args.output_root / "integrity" / "preflight_audit.json").exists():
        raise RuntimeError("Run Phase 3C audit before Qwen extraction")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3C Qwen extraction requires the configured CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        dtype=torch.bfloat16,
    )
    model.requires_grad_(False)
    model.eval().to("cuda")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen must remain frozen during Phase 3C")
    if int(model.config.num_labels) != len(CLASS_TOKENS):
        raise RuntimeError("The transferred Phase 2D checkpoint is not the five-class model")
    post_model_load_started = time.perf_counter()
    metadata: dict[str, Any] = {
        "phase": PHASE,
        "extracted_at": utc_now(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": str(next(model.parameters()).dtype),
        "model_training_mode": model.training,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "gradient_computation": False,
        "parameter_updates": 0,
        "prompt_sha256": text_sha256(SUPERVISOR_INSTRUCTION),
        "splits": {},
    }
    for split, expected in (("train", 2245), ("validation", 924)):
        rows = phase3b.read_jsonl(phase3b.feature_path(args.source_root, split))
        phase3b.validate_rows(rows, split, expected)
        destination = npz_path(args.output_root, split)
        if destination.exists() and not args.force:
            arrays = load_npz(destination)
            validate_qwen_arrays(rows, arrays)
            split_metadata = {
                "examples": len(rows),
                "resumed_existing_features": True,
                "hidden_size": int(arrays["hidden"].shape[1]),
                "minimum_token_count": int(arrays["token_counts"].min()),
                "maximum_token_count": int(arrays["token_counts"].max()),
            }
        else:
            arrays, split_metadata = extract_split(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                rows=rows,
                batch_size=args.batch_size,
            )
            save_npz_atomic(destination, **arrays)
        split_metadata.update(
            {
                "path": str(destination),
                "sha256": phase3b.sha256_file(destination),
                "logits_shape": list(arrays["logits"].shape),
                "hidden_shape": list(arrays["hidden"].shape),
            }
        )
        metadata["splits"][split] = split_metadata
    source_predictions = phase3b.read_jsonl(PHASE2D_ROOT / "validation" / "predictions.jsonl")
    validation_arrays = load_npz(npz_path(args.output_root, "validation"))
    extracted_labels = np.asarray(validation_arrays["logits"]).argmax(axis=1)
    saved_labels = np.asarray(
        [int(row["predicted_class_id"]) for row in source_predictions], dtype=np.int64
    )
    if len(saved_labels) != 924 or not np.array_equal(extracted_labels, saved_labels):
        mismatch = int(np.sum(extracted_labels != saved_labels))
        raise RuntimeError(f"Reloaded Phase 2D logits disagree with {mismatch} saved predictions")
    metadata["phase2d_prediction_reproduction"] = {
        "status": "passed",
        "matching_predictions": 924,
        "mismatches": 0,
    }
    metadata["post_model_load_extraction_and_verification_seconds"] = (
        time.perf_counter() - post_model_load_started
    )
    phase3b.atomic_json(args.output_root / "qwen_features" / "manifest.json", metadata)
    print(json.dumps(metadata, indent=2))
    return 0


def train_evaluate_command(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    train = phase3b.read_jsonl(phase3b.feature_path(args.source_root, "train"))
    validation = phase3b.read_jsonl(phase3b.feature_path(args.source_root, "validation"))
    phase3b.validate_rows(train, "train", 2245)
    phase3b.validate_rows(validation, "validation", 924)
    train_arrays = load_npz(npz_path(args.output_root, "train"))
    validation_arrays = load_npz(npz_path(args.output_root, "validation"))
    validate_qwen_arrays(train, train_arrays)
    validate_qwen_arrays(validation, validation_arrays)
    train_targets = phase3b.target_array(train)
    validation_targets = phase3b.target_array(validation)
    folds = phase3b.grouped_stratified_folds(train, FOLDS, SEED)
    phase3b.PHASE = PHASE
    matrices: dict[str, np.ndarray] = {}
    names_by_variant: dict[str, list[str]] = {}
    cv_results: dict[str, dict] = {}
    for variant in FUSION_VARIANTS:
        matrix, names = fusion_matrix(train, train_arrays, variant)
        matrices[variant] = matrix
        names_by_variant[variant] = names
        cv_results[variant] = phase3b.cross_validate_variant(
            variant=variant,
            features=matrix,
            feature_names=names,
            targets=train_targets,
            folds=folds,
            output_root=args.output_root,
        )
    primary_variant = max(
        cv_results, key=lambda name: variant_key(name, cv_results[name])
    )
    selection_lock = {
        "phase": PHASE,
        "locked_at": utc_now(),
        "fusion_selection_data": "preserved train split only",
        "phase3c_validation_labels_observed_for_fusion_selection": False,
        "primary_variant": primary_variant,
        "selection_metric": "paper-grouped OOF macro-F1",
        "variant_oof_results": {
            name: {
                "selected_candidate_id": result["selected_candidate"]["candidate_id"],
                "selected_parameters": result["selected_candidate"]["parameters"],
                "oof_metrics": result["selected_candidate"]["oof_metrics"],
            }
            for name, result in cv_results.items()
        },
        "development_set_caveat": (
            "The frozen Phase 2D checkpoint had already been selected on the preserved "
            "validation split before Phase 3C."
        ),
    }
    phase3b.atomic_json(args.output_root / "selection" / "selection_lock.json", selection_lock)
    print(json.dumps({"event": "phase3c_primary_locked", "primary_variant": primary_variant}), flush=True)
    models: dict[str, Any] = {}
    predictions_by_variant: dict[str, list[dict]] = {}
    for variant in FUSION_VARIANTS:
        selected = cv_results[variant]["selected_candidate"]
        weights, class_weights = phase3b.class_balance_weights(train_targets)
        fit_started = time.perf_counter()
        booster = phase3b.train_booster(
            matrices[variant], train_targets, weights, names_by_variant[variant],
            selected["parameters"], SEED
        )
        fit_seconds = time.perf_counter() - fit_started
        model_path = args.output_root / "models" / f"{variant}.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(model_path)
        validation_matrix, validation_names = fusion_matrix(
            validation, validation_arrays, variant
        )
        if validation_names != names_by_variant[variant]:
            raise RuntimeError("Train/validation fusion schemas differ")
        prediction_started = time.perf_counter()
        probabilities = phase3b.predict_booster(booster, validation_matrix, validation_names)
        prediction_seconds = time.perf_counter() - prediction_started
        predicted = probabilities.argmax(axis=1)
        metrics = phase3b.classification_metrics(validation_targets, predicted, probabilities)
        importance = phase3b.feature_importance_rows(booster, names_by_variant[variant])
        phase3b.write_importance_csv(
            args.output_root / "feature_importance" / f"{variant}.csv", importance
        )
        rows: list[dict] = []
        for index, source in enumerate(validation):
            predicted_index = int(predicted[index])
            ranked = np.argsort(-probabilities[index], kind="stable")
            rows.append(
                {
                    "phase": PHASE,
                    "formulation_version": FORMULATION_VERSION,
                    "variant": variant,
                    "question_id": str(source["question_id"]),
                    "document_id": str(source["document_id"]),
                    "question_text": str(source["question_text"]),
                    "oracle_label": int(source["oracle_label"]),
                    "predicted_label": CLASS_TOKENS[predicted_index],
                    "parsed_prediction": CLASS_TOKENS[predicted_index],
                    "predicted_class_index": predicted_index,
                    "probabilities": {
                        str(tokens): float(probabilities[index, class_index])
                        for class_index, tokens in enumerate(CLASS_TOKENS)
                    },
                    "ranked_predictions": [CLASS_TOKENS[int(value)] for value in ranked],
                    "top_2_predictions": [CLASS_TOKENS[int(value)] for value in ranked[:2]],
                    "prediction_status": "valid_phase3c_fusion_softprob",
                }
            )
        variant_path = args.output_root / "validation" / f"{variant}_predictions.jsonl"
        phase3b.atomic_jsonl(variant_path, rows)
        predictions_by_variant[variant] = rows
        metadata = {
            "variant": variant,
            "feature_sources": list(FUSION_VARIANTS[variant]),
            "feature_count": len(validation_names),
            "selected_candidate": selected,
            "class_weights": class_weights,
            "validation_metrics": metrics,
            "model_path": str(model_path),
            "model_sha256": phase3b.sha256_file(model_path),
            "prediction_path": str(variant_path),
            "prediction_sha256": phase3b.sha256_file(variant_path),
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
            "top_feature_importance_by_gain": importance[:30],
        }
        phase3b.atomic_json(args.output_root / "models" / f"{variant}_metadata.json", metadata)
        models[variant] = metadata
    canonical_path = args.output_root / "validation" / "predictions.jsonl"
    phase3b.atomic_jsonl(canonical_path, predictions_by_variant[primary_variant])
    primary_metrics = models[primary_variant]["validation_metrics"]
    phase3b.atomic_json(args.output_root / "classification" / "metrics.json", primary_metrics)
    phase3b.write_confusion_csv(args.output_root / "classification" / "confusion_matrix.csv", primary_metrics)
    histogram = args.output_root / "classification" / "predicted_vs_oracle.svg"
    phase3b.write_histogram_svg(
        histogram,
        primary_metrics["class_distribution"],
        primary_metrics["predicted_distribution"],
    )
    phase3b.atomic_text(
        histogram,
        histogram.read_text(encoding="utf-8").replace("Phase 3B", "Phase 3C"),
    )
    train_majority = int(np.bincount(train_targets).argmax())
    source_phase2d = json.loads((PHASE2D_ROOT / "final_summary.json").read_text(encoding="utf-8"))
    source_phase3b = json.loads((PHASE3B_ROOT / "final_summary.json").read_text(encoding="utf-8"))
    summary = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "status": "classification_complete",
        "completed_at": utc_now(),
        "primary_variant": primary_variant,
        "primary_variant_selection": "train-only paper-grouped OOF macro-F1",
        "dataset": {
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_documents": len({str(row['document_id']) for row in train}),
            "validation_documents": len({str(row['document_id']) for row in validation}),
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
            "qwen_source": "frozen Phase 2D last non-padding hidden state or five logits",
            "tree_source": "frozen Phase 3A/3B same-paper similarity-tree features",
            "qwen_hidden_size": int(train_arrays["hidden"].shape[1]),
            "tree_feature_count": 173,
            "gold_evidence_used_as_feature": False,
            "answer_used_as_feature": False,
            "retrieval_f1_used_as_feature": False,
            "oracle_used_only_as_supervised_target": True,
        },
        "models": models,
        "primary_validation_metrics": primary_metrics,
        "references": {
            "phase2d": source_phase2d["classification"],
            "phase3b": source_phase3b["classification"],
            "train_prior_majority": {
                "class": CLASS_TOKENS[train_majority],
                "validation_metrics": phase3b.majority_reference(validation_targets, train_majority),
            },
        },
        "selection_lock_path": str(args.output_root / "selection" / "selection_lock.json"),
        "primary_prediction_path": str(canonical_path),
        "primary_prediction_sha256": phase3b.sha256_file(canonical_path),
        "development_set_caveat": selection_lock["development_set_caveat"],
        "training_and_validation_wall_seconds": time.perf_counter() - started,
    }
    phase3b.atomic_json(args.output_root / "classification_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def retrieve_command(args: argparse.Namespace) -> int:
    phase3b.PHASE = PHASE
    phase3b.FORMULATION_VERSION = FORMULATION_VERSION
    return int(phase3b.retrieve_command(args))


def verify_saved_models(args: argparse.Namespace, classification: dict) -> dict[str, Any]:
    """Reload both boosters and reproduce every saved validation probability."""

    import xgboost as xgb

    validation = phase3b.read_jsonl(phase3b.feature_path(args.source_root, "validation"))
    arrays = load_npz(npz_path(args.output_root, "validation"))
    targets = phase3b.target_array(validation)
    checks: dict[str, Any] = {}

    def values_close(left: Any, right: Any) -> bool:
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(
                values_close(left[key], right[key]) for key in left
            )
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(
                values_close(a, b) for a, b in zip(left, right)
            )
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))
        return left == right

    for variant in FUSION_VARIANTS:
        matrix, names = fusion_matrix(validation, arrays, variant)
        model_path = Path(classification["models"][variant]["model_path"])
        booster = xgb.Booster()
        booster.load_model(model_path)
        probabilities = phase3b.predict_booster(booster, matrix, names)
        saved = phase3b.read_jsonl(
            args.output_root / "validation" / f"{variant}_predictions.jsonl"
        )
        saved_probabilities = np.asarray(
            [
                [float(row["probabilities"][str(tokens)]) for tokens in CLASS_TOKENS]
                for row in saved
            ],
            dtype=np.float64,
        )
        maximum_difference = float(np.max(np.abs(probabilities - saved_probabilities)))
        predicted = probabilities.argmax(axis=1)
        labels_match = all(
            CLASS_TOKENS[int(predicted[index])] == int(saved[index]["predicted_label"])
            for index in range(len(saved))
        )
        metrics = phase3b.classification_metrics(targets, predicted, probabilities)
        metrics_match = values_close(
            metrics, classification["models"][variant]["validation_metrics"]
        )
        if len(saved) != 924 or maximum_difference > 1e-7 or not labels_match or not metrics_match:
            raise RuntimeError(f"Reloaded Phase 3C model failed reproduction: {variant}")
        checks[variant] = {
            "status": "passed",
            "prediction_rows": len(saved),
            "predicted_labels_match": labels_match,
            "metrics_match": metrics_match,
            "maximum_absolute_probability_difference": maximum_difference,
            "model_sha256": phase3b.sha256_file(model_path),
        }
    return checks


def finalize_command(args: argparse.Namespace) -> int:
    classification = json.loads(
        (args.output_root / "classification_summary.json").read_text(encoding="utf-8")
    )
    retrieval = json.loads(
        (args.output_root / "retrieval" / "summary.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (args.output_root / "selection" / "selection_lock.json").read_text(encoding="utf-8")
    )
    if not retrieval.get("qdrant_collections_unchanged"):
        raise RuntimeError("Phase 3C retrieval integrity did not pass")
    predictions = phase3b.read_jsonl(args.output_root / "validation" / "predictions.jsonl")
    records = phase3b.read_jsonl(args.output_root / "retrieval" / "results.jsonl")
    if len(predictions) != 924 or len(records) != 924:
        raise RuntimeError("Phase 3C final artifacts are incomplete")
    model_verification = verify_saved_models(args, classification)
    qwen_manifest = json.loads(
        (args.output_root / "qwen_features" / "manifest.json").read_text(encoding="utf-8")
    )
    extraction_seconds = float(
        qwen_manifest["post_model_load_extraction_and_verification_seconds"]
    )
    train_evaluate_seconds = float(classification["training_and_validation_wall_seconds"])
    retrieval_seconds = float(retrieval["retrieval_wall_seconds_this_invocation"])
    runtime = {
        "qwen_extraction_splits": qwen_manifest["splits"],
        "post_model_load_extraction_and_verification_seconds": extraction_seconds,
        "cross_validation_seconds_by_variant": {
            variant: float(
                json.loads(
                    (args.output_root / "cross_validation" / f"{variant}.json").read_text(
                        encoding="utf-8"
                    )
                )["wall_seconds"]
            )
            for variant in FUSION_VARIANTS
        },
        "train_evaluate_command_seconds_including_cross_validation": train_evaluate_seconds,
        "final_fit_and_validation_seconds_by_variant": {
            variant: float(classification["models"][variant]["fit_seconds"])
            + float(classification["models"][variant]["prediction_seconds"])
            for variant in FUSION_VARIANTS
        },
        "retrieval_seconds": retrieval_seconds,
        "known_recorded_sequential_stage_seconds": (
            extraction_seconds + train_evaluate_seconds + retrieval_seconds
        ),
        "runtime_accounting_note": (
            "The train-evaluate command time already includes both cross-validation searches "
            "and final fitting, so the per-variant CV times are descriptive and are not added "
            "again. Model-loading time before the extraction timer was not recorded and is not "
            "fabricated."
        ),
    }
    phase3b.atomic_json(args.output_root / "runtime" / "summary.json", runtime)
    final_audit = {
        "phase": PHASE,
        "verified_at": utc_now(),
        "status": "passed",
        "source_feature_hashes": EXPECTED_SOURCE_HASHES,
        "phase2d_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "prediction_rows": len(predictions),
        "retrieval_rows": len(records),
        "qdrant_snapshot_matches": True,
        "qwen_parameter_updates": 0,
        "model_reload_verification": model_verification,
    }
    phase3b.atomic_json(args.output_root / "integrity" / "final_audit.json", final_audit)
    artifacts = []
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path.name != "final_summary.json":
            artifacts.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": phase3b.sha256_file(path)}
            )
    final = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "primary_variant": classification["primary_variant"],
        "classification": classification["primary_validation_metrics"],
        "models": classification["models"],
        "references": classification["references"],
        "retrieval": retrieval,
        "dataset": classification["dataset"],
        "features": classification["features"],
        "selection": selection,
        "runtime": runtime,
        "development_set_caveat": classification["development_set_caveat"],
        "integrity": {
            "phase2d_checkpoint_unchanged": True,
            "phase3a_phase3b_features_unchanged": True,
            "qwen_parameter_updates": 0,
            "qdrant_read_only": True,
            "qdrant_snapshot_unchanged": True,
            "prediction_rows": len(predictions),
            "retrieval_rows": len(records),
            "model_reload_verification": model_verification,
        },
        "artifacts": artifacts,
    }
    phase3b.atomic_json(args.output_root / "final_summary.json", final)
    print(json.dumps(final, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=PHASE2D_CHECKPOINT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit").set_defaults(func=audit_command)
    extract = subparsers.add_parser("extract-qwen")
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--force", action="store_true")
    extract.set_defaults(func=extract_qwen_command)
    subparsers.add_parser("train-evaluate").set_defaults(func=train_evaluate_command)
    subparsers.add_parser("retrieve").set_defaults(func=retrieve_command)
    subparsers.add_parser("finalize").set_defaults(func=finalize_command)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
