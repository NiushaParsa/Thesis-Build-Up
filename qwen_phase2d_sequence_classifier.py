#!/usr/bin/env python
"""Phase 2D Qwen Base sequence-classification granularity router.

This experiment is intentionally isolated from the completed generative Phase 1,
Phase 2, and Phase 2B experiments.  It uses the same preserved evidence-length
Oracle records, but trains ``Qwen/Qwen3.5-0.8B-Base`` as a conventional
five-label sequence classifier.  Inputs contain only the supervisor-approved
instruction and the original question text.  No chat template, target-token
generation, output parser, retrieval result, or evidence-derived feature is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os

# Required before torch initializes for deterministic CUDA/cuBLAS kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

import qwen_phase2 as phase2
import qwen_phase2b as phase2b


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
TRANSFORMERS_COMMIT = phase2.TRANSFORMERS_COMMIT
ORACLE_VERSION = phase2.ORACLE_VERSION
CHUNK_SIZES = phase2.CLASS_TOKENS
LABEL_TO_ID = phase2.CLASS_TO_INDEX
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
NUM_LABELS = len(CHUNK_SIZES)

SUPERVISOR_INSTRUCTION = (
    "You are a router for a retrieval-augmented generation system. Based only "
    "on the question, select the option representing the context size most "
    "suitable for retrieving the evidence required to answer it. Choose exactly "
    "one value from: 1 = 10 tokens, 2 = 20 tokens, 3 = 40 tokens, "
    "4 = 80 tokens, 5 = 160 tokens. Return only the number"
)
FORMULATION_VERSION = "qwen-phase2d-base-sequence-classifier-token-count-prompt-v1"
DECISION_METHOD = "five_logit_sequence_classifier_argmax"
PREDICTION_STATUS = "valid_sequence_classifier_argmax"
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle"
)


@dataclass(frozen=True)
class TrainingConfig:
    """Predeclared Phase 2D optimization and model configuration."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    architecture: str = "AutoModelForSequenceClassification"
    training_method: str = "full_parameter_sequence_classification"
    formulation_version: str = FORMULATION_VERSION
    objective: str = "uniform_five_class_cross_entropy"
    problem_type: str = "single_label_classification"
    dtype: str = "torch.bfloat16"
    device: str = "cuda"
    max_sequence_length: int = 128
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    epochs: int = 3
    gradient_clipping: float = 1.0
    seed: int = 42
    logging_steps: int = 1
    evaluation_frequency: str = "end_of_epoch"
    checkpoint_frequency: str = "end_of_epoch"
    checkpoint_retention_policy: str = (
        "retain_current_and_best_during_training_selected_only_at_completion"
    )
    checkpoint_selection_metric: str = "validation_macro_f1"
    checkpoint_tie_break: str = (
        "accuracy, weighted_f1, balanced_accuracy, "
        "lower_validation_loss, earlier_step"
    )
    early_stopping: str = "none_fixed_three_epochs"
    quantization: None = None
    class_weights: str = "uniform"


class FormattedDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return phase2b.canonical_json_sha256(value)


def classifier_text(
    question_text: str,
    instruction: str = SUPERVISOR_INSTRUCTION,
) -> str:
    """Return the complete and only semantic model input."""

    return phase2.prompt_text(str(question_text), instruction)


def configure_classifier_config(config: Any, pad_token_id: int) -> Any:
    """Apply the exact five-label mapping and padding contract.

    Transformers' generic decoder sequence-classifier selects its pooled token
    from ``config.get_text_config().pad_token_id`` rather than from the attention
    mask.  The Qwen Base repository leaves that nested value unset, so both the
    nested and top-level values are set explicitly and later verified.
    """

    if pad_token_id is None:
        raise RuntimeError("Qwen Base tokenizer has no pad token ID")
    config.num_labels = NUM_LABELS
    config.problem_type = "single_label_classification"
    config.id2label = {index: str(label) for index, label in ID_TO_LABEL.items()}
    config.label2id = {str(label): index for label, index in LABEL_TO_ID.items()}
    config.pad_token_id = int(pad_token_id)
    config.use_cache = False
    text_config = config.get_text_config()
    text_config.pad_token_id = int(pad_token_id)
    text_config.use_cache = False
    return config


def verify_model_config(model_or_config: Any, pad_token_id: int) -> dict[str, Any]:
    """Fail closed if the runtime classifier configuration drifts."""

    config = getattr(model_or_config, "config", model_or_config)
    text_config = config.get_text_config()
    expected_id2label = {
        index: str(label) for index, label in ID_TO_LABEL.items()
    }
    expected_label2id = {
        str(label): index for label, index in LABEL_TO_ID.items()
    }
    actual_id2label = {
        int(index): str(label) for index, label in config.id2label.items()
    }
    actual_label2id = {
        str(label): int(index) for label, index in config.label2id.items()
    }
    checks = {
        "num_labels": int(config.num_labels) == NUM_LABELS,
        "problem_type": config.problem_type == "single_label_classification",
        "id2label": actual_id2label == expected_id2label,
        "label2id": actual_label2id == expected_label2id,
        "top_level_pad_token_id": int(config.pad_token_id) == int(pad_token_id),
        "text_config_pad_token_id": int(text_config.pad_token_id)
        == int(pad_token_id),
        "top_level_use_cache": config.use_cache is False,
        "text_config_use_cache": text_config.use_cache is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Classifier configuration audit failed: {failed}")
    return {
        "status": "passed",
        "checks": checks,
        "num_labels": NUM_LABELS,
        "id2label": {str(key): value for key, value in expected_id2label.items()},
        "label2id": expected_label2id,
        "pad_token_id": int(pad_token_id),
    }


def tokenizer_input_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise RuntimeError("Expected exactly one tokenized classifier sequence")
        input_ids = input_ids[0]
    return [int(value) for value in input_ids]


def format_classification_example(
    tokenizer: Any,
    record: Mapping[str, Any],
    instruction: str = SUPERVISOR_INSTRUCTION,
    max_sequence_length: int = 128,
) -> dict[str, Any]:
    """Format plain prompt-plus-question input without silent truncation."""

    oracle_label = int(record["oracle_label"])
    if oracle_label not in LABEL_TO_ID:
        raise ValueError(f"Unexpected Oracle class: {oracle_label}")
    text = classifier_text(str(record["question_text"]), instruction)
    input_ids = tokenizer_input_ids(tokenizer, text)
    if not input_ids:
        raise RuntimeError("Tokenizer produced an empty classifier sequence")
    if len(input_ids) > max_sequence_length:
        raise RuntimeError(
            f"Classifier input has {len(input_ids)} tokens; maximum is "
            f"{max_sequence_length}. Phase 2D never silently truncates input."
        )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("Qwen Base tokenizer has no pad token ID")
    if int(pad_token_id) in input_ids:
        raise RuntimeError(
            "An unpadded input contains the configured pad token ID; generic "
            "Qwen sequence-classifier pooling would become ambiguous"
        )
    return {
        "question_id": str(record["question_id"]),
        "document_id": str(record["document_id"]),
        "question_text": str(record["question_text"]),
        "oracle_label": oracle_label,
        "target_class_id": LABEL_TO_ID[oracle_label],
        "input_text": text,
        "input_text_sha256": text_sha256(text),
        "input_ids": input_ids,
        "sequence_token_count": len(input_ids),
    }


def format_records(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    instruction: str,
    maximum: int,
) -> list[dict[str, Any]]:
    return [
        format_classification_example(tokenizer, row, instruction, maximum)
        for row in records
    ]


def collate_classification_batch(
    rows: Sequence[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot collate an empty classifier batch")
    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    last_non_pad_positions: list[int] = []
    for row in rows:
        length = len(row["input_ids"])
        padding = maximum - length
        input_ids.append(row["input_ids"] + [int(pad_token_id)] * padding)
        attention_masks.append([1] * length + [0] * padding)
        last_non_pad_positions.append(length - 1)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(
            [row["target_class_id"] for row in rows], dtype=torch.long
        ),
        "last_non_pad_positions": torch.tensor(
            last_non_pad_positions, dtype=torch.long
        ),
        "question_ids": [row["question_id"] for row in rows],
        "rows": list(rows),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def audit_classifier_head(
    model: Any,
    loading_info: Mapping[str, Any],
    *,
    initial_base_load: bool,
    seed: int,
) -> dict[str, Any]:
    """Audit deterministic initialization and checkpoint loading of the head."""

    if not hasattr(model, "score") or not hasattr(model.score, "weight"):
        raise RuntimeError("Sequence-classification model has no score.weight head")
    weight = model.score.weight
    expected_shape = (NUM_LABELS, int(model.config.get_text_config().hidden_size))
    if tuple(weight.shape) != expected_shape:
        raise RuntimeError(
            f"Classifier head shape is {tuple(weight.shape)}; expected {expected_shape}"
        )
    if not torch.isfinite(weight.detach().float()).all():
        raise RuntimeError("Classifier head initialization contains non-finite values")

    missing = sorted(str(value) for value in loading_info.get("missing_keys", []))
    unexpected = sorted(
        str(value) for value in loading_info.get("unexpected_keys", [])
    )
    mismatched = list(loading_info.get("mismatched_keys", []))
    errors = list(loading_info.get("error_msgs", []))
    expected_missing = ["score.weight"] if initial_base_load else []
    if missing != expected_missing or unexpected or mismatched or errors:
        raise RuntimeError(
            "Classifier model-loading audit failed: "
            f"missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}, errors={errors}"
        )
    return {
        "status": "passed",
        "initial_base_load": initial_base_load,
        "seed_set_before_model_load": int(seed),
        "head_parameter_name": "score.weight",
        "head_shape": list(expected_shape),
        "head_bias": getattr(model.score, "bias", None) is not None,
        "head_weight_dtype": str(weight.dtype),
        "head_weight_sha256_float32": tensor_sha256(weight),
        "loading_info": {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "mismatched_keys": mismatched,
            "error_msgs": errors,
        },
    }


def load_tokenizer_config(
    source: str = MODEL_ID,
    revision: str | None = MODEL_REVISION,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Official Qwen Base tokenizer has no pad token")
    config = AutoConfig.from_pretrained(source, revision=revision)
    configure_classifier_config(config, int(tokenizer.pad_token_id))
    verify_model_config(config, int(tokenizer.pad_token_id))
    return tokenizer, config


def load_classifier_model(
    source: str = MODEL_ID,
    revision: str | None = MODEL_REVISION,
    *,
    initial_base_load: bool,
    seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the exact classifier and return its explicit initialization audit."""

    phase2.set_deterministic_seed(seed)
    tokenizer, config = load_tokenizer_config(source, revision)
    loaded = AutoModelForSequenceClassification.from_pretrained(
        source,
        revision=revision,
        config=config,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    model, loading_info = loaded
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    model.requires_grad_(True)
    config_audit = verify_model_config(model, int(tokenizer.pad_token_id))
    head_audit = audit_classifier_head(
        model,
        loading_info,
        initial_base_load=initial_base_load,
        seed=seed,
    )
    return tokenizer, model, {
        "configuration": config_audit,
        "classifier_head": head_audit,
    }


def uniform_ce_components(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[1] != NUM_LABELS:
        raise ValueError("Expected classifier logits shaped [batch, 5]")
    targets = targets.to(device=logits.device, dtype=torch.long)
    if targets.shape != (logits.shape[0],):
        raise ValueError("Classifier target count does not match batch size")
    per_example = F.cross_entropy(logits.float(), targets, reduction="none")
    return {
        "per_example_ce": per_example,
        "loss_sum": per_example.sum(),
        "example_count": torch.tensor(
            logits.shape[0], device=logits.device, dtype=torch.float32
        ),
        "mean": per_example.mean(),
    }


def deterministic_ranking(scores: Sequence[float]) -> list[int]:
    return phase2b.deterministic_ranking(scores)


def metrics_with_top2(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = phase2.fixed_classification_metrics(rows)
    if rows:
        top2_correct = sum(
            int(int(row["oracle_label"]) in [int(x) for x in row["top_2_predictions"]])
            for row in rows
        )
        metrics["top_2_accuracy"] = float(top2_correct / len(rows))
        metrics["top_2_accuracy_status"] = (
            "available_from_comparable_five_class_head_logits"
        )
    else:
        metrics["top_2_accuracy"] = None
        metrics["top_2_accuracy_status"] = "unavailable_no_examples"
    return metrics


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_classifier(
    model: Any,
    loader: DataLoader[Any],
    source_records: Mapping[str, Mapping[str, Any]],
    checkpoint_id: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate CE, five logits, deterministic top-1, and exact top-2."""

    model.eval()
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    predictions: list[dict[str, Any]] = []
    loss_sum = 0.0
    example_count = 0
    wall_started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            _synchronize(device)
            batch_started = time.perf_counter()
            outputs = model(
                input_ids=inputs,
                attention_mask=attention,
                labels=targets,
                use_cache=False,
            )
            logits = outputs.logits.float()
            _synchronize(device)
            batch_wall = time.perf_counter() - batch_started
            components = uniform_ce_components(logits, targets)
            loss_sum += float(components["loss_sum"].detach().cpu())
            example_count += int(logits.shape[0])
            probabilities = torch.softmax(logits, dim=-1).detach().cpu()
            cpu_logits = logits.detach().cpu()
            per_example_wall = batch_wall / max(1, int(logits.shape[0]))
            for row_index, question_id in enumerate(batch["question_ids"]):
                scores = [float(value) for value in cpu_logits[row_index].tolist()]
                probs = [float(value) for value in probabilities[row_index].tolist()]
                ranked_ids = deterministic_ranking(scores)
                predicted_id = ranked_ids[0]
                predicted_label = ID_TO_LABEL[predicted_id]
                record = dict(source_records[str(question_id)])
                predictions.append(
                    {
                        **record,
                        "decision_method": DECISION_METHOD,
                        "predicted_class_id": predicted_id,
                        "predicted_label": predicted_label,
                        "parsed_prediction": predicted_label,
                        "prediction_status": PREDICTION_STATUS,
                        "class_logits_by_label": {
                            str(ID_TO_LABEL[index]): scores[index]
                            for index in range(NUM_LABELS)
                        },
                        "class_probabilities_by_label": {
                            str(ID_TO_LABEL[index]): probs[index]
                            for index in range(NUM_LABELS)
                        },
                        "ranked_class_ids": ranked_ids,
                        "ranked_predictions": [
                            ID_TO_LABEL[index] for index in ranked_ids
                        ],
                        "top_2_class_ids": ranked_ids[:2],
                        "top_2_predictions": [
                            ID_TO_LABEL[index] for index in ranked_ids[:2]
                        ],
                        "inference_seconds": per_example_wall,
                        "inference_timing_basis": (
                            "synchronized_batch_forward_wall_divided_by_batch_size"
                        ),
                        "selected_checkpoint": checkpoint_id,
                    }
                )
    model.train()
    if example_count != len(predictions):
        raise RuntimeError("Classifier evaluation prediction and loss counts disagree")
    return predictions, {
        "uniform_cross_entropy": loss_sum / example_count,
        "loss_sum": loss_sum,
        "evaluated_examples": example_count,
        "wall_seconds": time.perf_counter() - wall_started,
    }


def ensure_output_root(output_root: Path) -> dict[str, Any]:
    forbidden = {
        phase2.PHASE1_ROOT.resolve(),
        phase2.DEFAULT_OUTPUT_ROOT.resolve(),
        *(path.resolve() for path in phase2b.DEFAULT_OUTPUT_ROOTS.values()),
        Path(
            "outputs/qwen_phase2c_sequence_classifier_evidence_length_oracle"
        ).resolve(),
    }
    resolved_output = output_root.resolve()
    if any(
        resolved_output == protected or protected in resolved_output.parents
        for protected in forbidden
    ):
        raise RuntimeError(
            "Phase 2D output root must not reuse a completed experiment root"
        )
    marker_path = output_root / "configuration" / "experiment.json"
    expected = {
        "phase": "Phase 2D",
        "formulation_version": FORMULATION_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "architecture": "AutoModelForSequenceClassification",
        "instruction": SUPERVISOR_INSTRUCTION,
        "instruction_sha256": text_sha256(SUPERVISOR_INSTRUCTION),
        "input_template": "{instruction}\\n\\nQuestion: {original_question_text}",
        "id2label": {str(index): str(label) for index, label in ID_TO_LABEL.items()},
        "label2id": {
            str(label): index for label, index in LABEL_TO_ID.items()
        },
        "objective": "uniform_five_class_cross_entropy",
    }
    if marker_path.exists():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"Phase 2D output-root configuration mismatch at {key}"
                )
        return existing
    value = {**expected, "created_at": utc_now()}
    phase2.atomic_json(marker_path, value)
    return value


def experiment_fingerprint(data_manifest: Mapping[str, Any], pad_id: int) -> str:
    return canonical_json_sha256(
        {
            "formulation_version": FORMULATION_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_commit": TRANSFORMERS_COMMIT,
            "instruction_sha256": text_sha256(SUPERVISOR_INSTRUCTION),
            "input_template": "plain_instruction_blankline_question_prefix",
            "max_sequence_length": TrainingConfig().max_sequence_length,
            "pad_token_id": int(pad_id),
            "id2label": ID_TO_LABEL,
            "objective": "uniform_five_class_cross_entropy",
            "train_oracle_sha256": data_manifest["train_oracle_sha256"],
            "validation_oracle_sha256": data_manifest["validation_oracle_sha256"],
        }
    )


def select_subset(
    records: Sequence[dict[str, Any]],
    mode: str,
    per_class: int,
    seed: int,
    *,
    validation: bool = False,
) -> list[dict[str, Any]]:
    if mode == "full":
        return list(records)
    return phase2.select_balanced_subset(
        records, 1 if validation else per_class, seed
    )


def inspect_phase2d(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    ensure_output_root(output_root)
    data_manifest = phase2.validate_frozen_data(args.phase1_root)
    train_records = phase2.load_oracle_records("train", args.phase1_root)
    validation_records = phase2.load_oracle_records("validation", args.phase1_root)
    tokenizer, config = load_tokenizer_config()
    formatted_train = format_records(
        tokenizer,
        train_records,
        SUPERVISOR_INSTRUCTION,
        TrainingConfig().max_sequence_length,
    )
    formatted_validation = format_records(
        tokenizer,
        validation_records,
        SUPERVISOR_INSTRUCTION,
        TrainingConfig().max_sequence_length,
    )

    def lengths(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        values = [int(row["sequence_token_count"]) for row in rows]
        return {
            "minimum": min(values),
            "maximum": max(values),
            "mean": float(np.mean(values)),
            "over_maximum": sum(
                value > TrainingConfig().max_sequence_length for value in values
            ),
        }

    result = {
        "status": "passed",
        "phase": "Phase 2D",
        "formulation_version": FORMULATION_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "data": data_manifest,
        "instruction": SUPERVISOR_INSTRUCTION,
        "instruction_sha256": text_sha256(SUPERVISOR_INSTRUCTION),
        "model_inputs": ["fixed_supervisor_instruction", "original_question_text"],
        "excluded_inputs": [
            "evidence",
            "evidence_length",
            "answer",
            "paper_text",
            "retrieved_chunks",
            "retrieval_scores",
            "metadata",
            "handcrafted_features",
            "chat_template",
            "assistant_target_tokens",
        ],
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "padding_side": tokenizer.padding_side,
            "add_special_tokens": True,
            "truncation": False,
        },
        "configuration_audit": verify_model_config(
            config, int(tokenizer.pad_token_id)
        ),
        "sequence_lengths": {
            "train": lengths(formatted_train),
            "validation": lengths(formatted_validation),
        },
        "pooling_safety": {
            "unpadded_inputs_containing_pad_token_id": 0,
            "right_padding": True,
            "nested_text_config_pad_token_id_explicit": True,
        },
        "experiment_fingerprint": experiment_fingerprint(
            data_manifest, int(tokenizer.pad_token_id)
        ),
        "verified_at": utc_now(),
    }
    phase2.atomic_json(
        output_root / "configuration" / "preflight_manifest.json", result
    )
    print(json.dumps(result, indent=2))
    return result


def _run_id(mode: str) -> str:
    return (
        f"qwen-phase2d-sequence-classifier-{mode}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-seed42"
    )


def _optimizer(model: Any, config: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def load_classifier_checkpoint(
    checkpoint_dir: Path,
    config: TrainingConfig,
    total_steps: int,
    warmup_steps: int,
    data_generator: torch.Generator,
) -> tuple[Any, Any, torch.optim.Optimizer, Any, dict[str, Any], dict[str, Any]]:
    """Classifier-specific resumable checkpoint loader."""

    tokenizer, model, load_audit = load_classifier_model(
        str(checkpoint_dir / "model"),
        None,
        initial_base_load=False,
        seed=config.seed,
    )
    optimizer = _optimizer(model, config)
    optimizer.load_state_dict(
        torch.load(
            checkpoint_dir / "optimizer.pt",
            map_location="cpu",
            weights_only=False,
        )
    )
    scheduler = phase2.cosine_scheduler(optimizer, total_steps, warmup_steps)
    scheduler.load_state_dict(
        torch.load(
            checkpoint_dir / "scheduler.pt",
            map_location="cpu",
            weights_only=False,
        )
    )
    state = json.loads(
        (checkpoint_dir / "training_state.json").read_text(encoding="utf-8")
    )
    random_states = torch.load(
        checkpoint_dir / "random_states.pt",
        map_location="cpu",
        weights_only=False,
    )
    torch.set_rng_state(random_states["torch_rng_state"])
    torch.cuda.set_rng_state_all(random_states["cuda_rng_state_all"])
    import random

    random.setstate(random_states["python_random_state"])
    np.random.set_state(random_states["numpy_random_state"])
    if "data_loader_generator_state" in random_states:
        data_generator.set_state(random_states["data_loader_generator_state"])
    return tokenizer, model, optimizer, scheduler, state, load_audit


def _gradient_audit(model: Any) -> dict[str, Any]:
    names_with_grad: list[str] = []
    names_without_grad: list[str] = []
    parameters_with_grad = 0
    parameters_without_grad = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            names_without_grad.append(name)
            parameters_without_grad += parameter.numel()
        else:
            names_with_grad.append(name)
            parameters_with_grad += parameter.numel()
    if "score.weight" not in names_with_grad:
        raise RuntimeError("Classifier score.weight received no gradient")
    if not any(name.startswith("model.language_model") for name in names_with_grad):
        raise RuntimeError("Qwen language backbone received no gradient")
    return {
        "status": "passed",
        "classifier_head_received_gradient": True,
        "language_backbone_received_gradient": True,
        "parameters_with_gradient": parameters_with_grad,
        "parameters_without_gradient": parameters_without_grad,
        "tensors_with_gradient": len(names_with_grad),
        "tensors_without_gradient": len(names_without_grad),
        "without_gradient_note": (
            "The composite qwen3_5 model contains a vision tower; text-only "
            "inputs may intentionally leave vision parameters without gradients."
        ),
        "sample_names_with_gradient": names_with_grad[:10],
        "sample_names_without_gradient": names_without_grad[:10],
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    from torch.utils.tensorboard import SummaryWriter

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 2D training requires a BF16 CUDA GPU")
    if args.mode == "full" and args.max_steps is not None:
        raise ValueError("A full Phase 2D run cannot use --max-steps")
    config = TrainingConfig()
    phase2.set_deterministic_seed(config.seed)
    output_root = Path(args.output_root)
    ensure_output_root(output_root)
    data_manifest = phase2.validate_frozen_data(args.phase1_root)
    train_records = select_subset(
        phase2.load_oracle_records("train", args.phase1_root),
        args.mode,
        args.per_class,
        config.seed,
    )
    validation_records = select_subset(
        phase2.load_oracle_records("validation", args.phase1_root),
        args.mode,
        args.per_class,
        config.seed,
        validation=True,
    )
    active_batch_size = (
        1
        if args.mode == "tiny-overfit"
        else 2
        if args.mode == "smoke"
        else config.per_device_batch_size
    )
    active_accumulation = (
        1
        if args.mode == "tiny-overfit"
        else 2
        if args.mode == "smoke"
        else config.gradient_accumulation_steps
    )
    run_id = args.run_id or _run_id(args.mode)
    run_dir = output_root / "runs" / run_id
    if run_dir.exists() and args.resume is None:
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_root / "tensorboard" / run_id

    tokenizer, model, initial_load_audit = load_classifier_model(
        initial_base_load=True,
        seed=config.seed,
    )
    pad_id = int(tokenizer.pad_token_id)
    fingerprint = experiment_fingerprint(data_manifest, pad_id)
    formatted_train = format_records(
        tokenizer,
        train_records,
        SUPERVISOR_INSTRUCTION,
        config.max_sequence_length,
    )
    formatted_validation = format_records(
        tokenizer,
        validation_records,
        SUPERVISOR_INSTRUCTION,
        config.max_sequence_length,
    )
    manifest = {
        **data_manifest,
        "active_train_examples": len(formatted_train),
        "active_validation_examples": len(formatted_validation),
        "active_train_distribution": dict(
            Counter(row["oracle_label"] for row in formatted_train)
        ),
        "active_validation_distribution": dict(
            Counter(row["oracle_label"] for row in formatted_validation)
        ),
        "sequence_length": {
            "train_minimum": min(row["sequence_token_count"] for row in formatted_train),
            "train_maximum": max(row["sequence_token_count"] for row in formatted_train),
            "validation_minimum": min(
                row["sequence_token_count"] for row in formatted_validation
            ),
            "validation_maximum": max(
                row["sequence_token_count"] for row in formatted_validation
            ),
        },
        "model_inputs": ["fixed_supervisor_instruction", "original_question_text"],
        "experiment_fingerprint": fingerprint,
        "created_at": utc_now(),
    }
    phase2.atomic_json(run_dir / "dataset_manifest.json", manifest)
    phase2.atomic_json(
        run_dir / "formatted_example_inspection.json",
        {
            "train_first": {
                key: value
                for key, value in formatted_train[0].items()
                if key != "input_ids"
            },
            "validation_first": {
                key: value
                for key, value in formatted_validation[0].items()
                if key != "input_ids"
            },
            "input_ids_are_excluded_from_inspection_only": True,
        },
    )

    data_generator = torch.Generator()
    data_generator.manual_seed(config.seed)
    train_loader = DataLoader(
        FormattedDataset(formatted_train),
        batch_size=active_batch_size,
        shuffle=True,
        generator=data_generator,
        collate_fn=lambda rows: collate_classification_batch(rows, pad_id),
    )
    validation_loader = DataLoader(
        FormattedDataset(formatted_validation),
        batch_size=active_batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_classification_batch(rows, pad_id),
    )
    validation_by_id = {
        str(row["question_id"]): row for row in validation_records
    }
    steps_per_epoch = phase2.optimizer_steps_for_batches(
        len(train_loader), active_accumulation
    )
    planned_steps = steps_per_epoch * config.epochs
    total_steps = (
        min(planned_steps, int(args.max_steps))
        if args.max_steps is not None
        else planned_steps
    )
    if total_steps < 1:
        raise ValueError("Phase 2D requires at least one optimizer step")
    warmup_steps = math.ceil(total_steps * config.warmup_ratio)
    optimizer = _optimizer(model, config)
    scheduler = phase2.cosine_scheduler(optimizer, total_steps, warmup_steps)
    state: dict[str, Any] = {
        "epoch": 0,
        "global_step": 0,
        "micro_step": 0,
        "initial_loss": None,
        "last_loss": None,
        "cumulative_elapsed_seconds": 0.0,
        "gradient_audit": None,
    }
    resume_contract = {
        "training_config": asdict(config),
        "run_mode": args.mode,
        "active_per_device_batch_size": active_batch_size,
        "active_gradient_accumulation_steps": active_accumulation,
        "maximum_optimizer_steps": args.max_steps,
        "total_optimizer_steps": total_steps,
        "experiment_fingerprint": fingerprint,
    }
    resume_contract_sha256 = canonical_json_sha256(resume_contract)
    state["resume_contract_sha256"] = resume_contract_sha256

    script_path = Path(__file__)
    run_config = {
        **asdict(config),
        "run_mode": args.mode,
        "active_per_device_batch_size": active_batch_size,
        "active_gradient_accumulation_steps": active_accumulation,
        "active_effective_batch_size": active_batch_size * active_accumulation,
        "maximum_optimizer_steps": args.max_steps,
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "run_id": run_id,
        "output_root": str(output_root),
        "repository_commit": os.getenv("PHASE2D_REPOSITORY_COMMIT", "unavailable"),
        "training_script_sha256": phase2.sha256_file(script_path),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "transformers_commit": TRANSFORMERS_COMMIT,
        "tensorboard_version": importlib.metadata.version("tensorboard"),
        "gpu": torch.cuda.get_device_name(0),
        "instruction": SUPERVISOR_INSTRUCTION,
        "instruction_sha256": text_sha256(SUPERVISOR_INSTRUCTION),
        "pad_token_id": pad_id,
        "id2label": {str(index): str(label) for index, label in ID_TO_LABEL.items()},
        "label2id": {
            str(label): index for label, index in LABEL_TO_ID.items()
        },
        "initial_model_loading_audit": initial_load_audit,
        "experiment_fingerprint": fingerprint,
        "resume_contract_sha256": resume_contract_sha256,
        "created_at": utc_now(),
    }
    config_path = run_dir / "training_config.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config.get("resume_contract_sha256") != resume_contract_sha256:
            raise RuntimeError("Existing Phase 2D run has a different resume contract")
    else:
        phase2.atomic_json(config_path, run_config)

    if args.resume is not None:
        del model
        del optimizer
        tokenizer, model, optimizer, scheduler, state, checkpoint_load_audit = (
            load_classifier_checkpoint(
                args.resume,
                config,
                total_steps,
                warmup_steps,
                data_generator,
            )
        )
        if state.get("resume_contract_sha256") != resume_contract_sha256:
            raise RuntimeError("Checkpoint Phase 2D resume contract mismatch")
        phase2.truncate_jsonl_after_step(
            run_dir / "training_history.jsonl", int(state["global_step"])
        )
        phase2.truncate_jsonl_after_step(
            run_dir / "validation_history.jsonl", int(state["global_step"])
        )
        phase2.append_jsonl(
            run_dir / "resume_history.jsonl",
            {
                "checkpoint": str(args.resume),
                "global_step": int(state["global_step"]),
                "checkpoint_model_loading_audit": checkpoint_load_audit,
                "experiment_fingerprint": fingerprint,
                "timestamp": utc_now(),
            },
        )

    writer = SummaryWriter(
        log_dir=str(tensorboard_dir),
        purge_step=int(state["global_step"]) + 1 if args.resume else None,
    )
    device = torch.device("cuda")
    model.to(device)
    phase2.move_optimizer_state(optimizer, device)
    model.train()
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError("Phase 2D unexpectedly contains frozen parameters")
    writer.add_text(
        "configuration/run",
        "```json\n"
        + json.dumps(
            {
                **asdict(config),
                "run_id": run_id,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                **manifest,
            },
            indent=2,
        )
        + "\n```",
        int(state["global_step"]),
    )

    optimizer.zero_grad(set_to_none=True)
    process = psutil.Process()
    started = time.perf_counter()
    elapsed_before_resume = float(state.get("cumulative_elapsed_seconds", 0.0))
    initial_loss: float | None = state.get("initial_loss")
    final_loss: float | None = state.get("last_loss")
    evaluation_records: list[dict[str, Any]] = []
    checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
    if args.resume is not None and checkpoint_manifest_path.exists():
        evaluation_records = [
            row
            for row in json.loads(
                checkpoint_manifest_path.read_text(encoding="utf-8")
            )
            if int(row["global_step"]) <= int(state["global_step"])
        ]

    def evaluate_and_checkpoint(completed_epoch: int) -> dict[str, Any]:
        checkpoint_id = f"step-{state['global_step']:06d}"
        predictions, loss_summary = evaluate_classifier(
            model,
            validation_loader,
            validation_by_id,
            checkpoint_id,
            device,
        )
        metrics = metrics_with_top2(predictions)
        distribution = Counter(row["predicted_label"] for row in predictions)
        event = {
            "event": "validation",
            "global_step": state["global_step"],
            "epoch": completed_epoch,
            "loss": loss_summary["uniform_cross_entropy"],
            "uniform_cross_entropy": loss_summary["uniform_cross_entropy"],
            **metrics,
            "predicted_distribution": {
                str(chunk): distribution[chunk] for chunk in CHUNK_SIZES
            },
            "wall_seconds": loss_summary["wall_seconds"],
            "timestamp": utc_now(),
        }
        phase2.append_jsonl(run_dir / "validation_history.jsonl", event)
        for tag, value in (
            ("validation/loss", loss_summary["uniform_cross_entropy"]),
            ("validation/accuracy", metrics["accuracy"]),
            ("validation/macro_f1", metrics["macro_f1"]),
            ("validation/weighted_f1", metrics["weighted_f1"]),
            ("validation/balanced_accuracy", metrics["balanced_accuracy"]),
            ("validation/top_2_accuracy", metrics["top_2_accuracy"]),
        ):
            writer.add_scalar(tag, value, state["global_step"])
        for label, values in metrics["per_class"].items():
            for metric_name in ("precision", "recall", "f1"):
                writer.add_scalar(
                    f"validation/class_{label}_{metric_name}",
                    values[metric_name],
                    state["global_step"],
                )
        for chunk in CHUNK_SIZES:
            writer.add_scalar(
                f"validation/predicted_class_{chunk}_count",
                distribution[chunk],
                state["global_step"],
            )
        writer.flush()

        checkpoint = run_dir / "checkpoints" / checkpoint_id
        prior_best_id = (
            phase2.select_best_evaluation(evaluation_records)["checkpoint_id"]
            if evaluation_records
            else None
        )
        pruned_before = phase2b.prune_checkpoints(
            run_dir / "checkpoints",
            {prior_best_id} if prior_best_id is not None else set(),
        )
        state.update(
            {
                "epoch": completed_epoch,
                "validation_metrics": metrics,
                "validation_loss": loss_summary["uniform_cross_entropy"],
                "initial_loss": initial_loss,
                "last_loss": final_loss,
                "cumulative_elapsed_seconds": (
                    elapsed_before_resume + time.perf_counter() - started
                ),
            }
        )
        phase2.save_checkpoint(
            checkpoint,
            model,
            tokenizer,
            optimizer,
            scheduler,
            state,
            data_generator,
        )
        prediction_path = (
            run_dir / "validation" / f"predictions_{checkpoint_id}.jsonl"
        )
        phase2.atomic_jsonl(prediction_path, predictions)
        record = {
            "checkpoint": str(checkpoint),
            "checkpoint_id": checkpoint_id,
            "global_step": state["global_step"],
            "epoch": completed_epoch,
            "validation_loss": loss_summary["uniform_cross_entropy"],
            "classification_metrics": metrics,
            "predicted_distribution": event["predicted_distribution"],
            "predictions": str(prediction_path),
            "validation_wall_seconds": loss_summary["wall_seconds"],
            "experiment_fingerprint": fingerprint,
            "checkpoints_pruned_before_save": pruned_before,
        }
        evaluation_records.append(record)
        current_best = phase2.select_best_evaluation(evaluation_records)
        retained_ids = {checkpoint_id, current_best["checkpoint_id"]}
        removed_ids = phase2b.prune_checkpoints(
            run_dir / "checkpoints", retained_ids
        )
        for evaluation in evaluation_records:
            evaluation["checkpoint_retained"] = (
                evaluation["checkpoint_id"] in retained_ids
            )
        record["checkpoints_pruned_after_evaluation"] = removed_ids
        phase2.atomic_json(checkpoint_manifest_path, evaluation_records)
        return record

    stop = False
    active_epochs = max(
        config.epochs, math.ceil(total_steps / max(steps_per_epoch, 1))
    )
    for epoch in range(int(state["epoch"]), active_epochs):
        accumulated_batches = 0
        accumulated_examples = 0
        accumulated_tokens = 0
        accumulated_loss_sum = 0.0
        accumulation_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            batch_examples = int(batch["input_ids"].shape[0])
            batch_tokens = int(batch["attention_mask"].sum().item())
            inputs = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            outputs = model(
                input_ids=inputs,
                attention_mask=attention,
                labels=targets,
                use_cache=False,
            )
            components = uniform_ce_components(outputs.logits, targets)
            loss_sum = components["loss_sum"]
            if not torch.isfinite(loss_sum):
                raise RuntimeError(
                    f"Non-finite classifier loss at step {state['global_step']}"
                )
            loss_sum.backward()
            state["micro_step"] += 1
            accumulated_batches += 1
            accumulated_examples += batch_examples
            accumulated_tokens += batch_tokens
            accumulated_loss_sum += float(loss_sum.detach().cpu())
            end_of_epoch = batch_index + 1 == len(train_loader)
            if accumulated_batches < active_accumulation and not end_of_epoch:
                continue

            if state.get("gradient_audit") is None:
                state["gradient_audit"] = _gradient_audit(model)
                phase2.atomic_json(
                    run_dir / "gradient_coverage_audit.json",
                    state["gradient_audit"],
                )
            phase2b.normalize_accumulated_gradients(
                model.parameters(), float(accumulated_examples)
            )
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clipping
                )
                .detach()
                .cpu()
            )
            used_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            state["epoch"] = epoch
            duration = time.perf_counter() - accumulation_started
            objective_loss = accumulated_loss_sum / accumulated_examples
            initial_loss = objective_loss if initial_loss is None else initial_loss
            final_loss = objective_loss
            log = {
                "event": "train_step",
                "global_step": state["global_step"],
                "epoch": epoch + (batch_index + 1) / max(1, len(train_loader)),
                "loss": objective_loss,
                "uniform_cross_entropy": objective_loss,
                "learning_rate": used_lr,
                "gradient_norm": gradient_norm,
                "step_duration_seconds": duration,
                "examples_per_second": accumulated_examples / max(duration, 1e-9),
                "tokens_per_second": accumulated_tokens / max(duration, 1e-9),
                "examples_in_step": accumulated_examples,
                "microbatches_in_step": accumulated_batches,
                "cpu_ram_gib": process.memory_info().rss / 2**30,
                "gpu_memory_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "gpu_memory_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "timestamp": utc_now(),
            }
            phase2.append_jsonl(run_dir / "training_history.jsonl", log)
            for tag, key in (
                ("train/loss", "loss"),
                ("train/uniform_cross_entropy", "uniform_cross_entropy"),
                ("train/learning_rate", "learning_rate"),
                ("train/epoch", "epoch"),
                ("train/gradient_norm", "gradient_norm"),
                ("train/step_duration_seconds", "step_duration_seconds"),
                ("train/examples_per_second", "examples_per_second"),
                ("train/tokens_per_second", "tokens_per_second"),
                ("system/cpu_ram_gib", "cpu_ram_gib"),
                ("system/gpu_memory_allocated_gib", "gpu_memory_allocated_gib"),
                ("system/gpu_memory_reserved_gib", "gpu_memory_reserved_gib"),
            ):
                writer.add_scalar(tag, log[key], state["global_step"])
            writer.add_scalar(
                "train/global_step", state["global_step"], state["global_step"]
            )
            accumulated_batches = 0
            accumulated_examples = 0
            accumulated_tokens = 0
            accumulated_loss_sum = 0.0
            accumulation_started = time.perf_counter()
            if state["global_step"] >= total_steps:
                stop = True
                break
        state["epoch"] = epoch + 1
        if args.mode == "full" or stop:
            evaluate_and_checkpoint(epoch + 1)
        if stop:
            break

    if not evaluation_records:
        evaluate_and_checkpoint(int(state["epoch"]))
    if int(state["global_step"]) != total_steps:
        raise RuntimeError(
            f"Phase 2D ended at step {state['global_step']}; expected {total_steps}"
        )
    best = phase2.select_best_evaluation(evaluation_records)
    pruned_completion = phase2b.prune_checkpoints(
        run_dir / "checkpoints", {best["checkpoint_id"]}
    )
    for evaluation in evaluation_records:
        evaluation["checkpoint_retained"] = (
            evaluation["checkpoint_id"] == best["checkpoint_id"]
        )
    phase2.atomic_json(checkpoint_manifest_path, evaluation_records)
    phase2.atomic_json(
        run_dir / "best_checkpoint.json",
        {
            **best,
            "selection_metric": config.checkpoint_selection_metric,
            "tie_break": config.checkpoint_tie_break,
            "experiment_fingerprint": fingerprint,
            "selected_at": utc_now(),
        },
    )
    summary = {
        "status": "complete",
        "phase": "Phase 2D",
        "mode": args.mode,
        "run_id": run_id,
        "global_step": state["global_step"],
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "validation_loss": best["validation_loss"],
        "validation_metrics": best["classification_metrics"],
        "validation_events": len(evaluation_records),
        "selected_checkpoint": best["checkpoint"],
        "selected_checkpoint_id": best["checkpoint_id"],
        "selection_reason": config.checkpoint_selection_metric,
        "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rss_gib": process.memory_info().rss / 2**30,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "gradient_audit": state["gradient_audit"],
        "experiment_fingerprint": fingerprint,
        "checkpoints_pruned_at_completion": pruned_completion,
        "tensorboard_directory": str(tensorboard_dir),
        "created_at": utc_now(),
    }
    phase2.atomic_json(run_dir / "summary.json", summary)
    writer.flush()
    writer.close()
    print(json.dumps(summary, indent=2))
    return summary


def _prediction_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "question_id",
        "predicted_class_id",
        "predicted_label",
        "parsed_prediction",
        "prediction_status",
        "class_logits_by_label",
        "class_probabilities_by_label",
        "ranked_class_ids",
        "ranked_predictions",
        "top_2_class_ids",
        "top_2_predictions",
    )
    return {key: row[key] for key in keys}


def validate_prediction_identity(
    predictions: Sequence[dict[str, Any]],
    frozen: Sequence[dict[str, Any]],
    checkpoint_id: str,
) -> None:
    if len(predictions) != len(frozen):
        raise RuntimeError("Phase 2D predictions and frozen validation lengths differ")
    ids = [row["question_id"] for row in predictions]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Phase 2D validation contains duplicate prediction IDs")
    if ids != [row["question_id"] for row in frozen]:
        raise RuntimeError("Phase 2D predictions do not preserve validation order")
    for prediction, oracle in zip(predictions, frozen):
        for key in ("question_id", "document_id", "question_text", "oracle_label"):
            if prediction[key] != oracle[key]:
                raise RuntimeError(
                    f"Phase 2D prediction differs from frozen data at {key}: "
                    f"{prediction['question_id']}"
                )
        if prediction["selected_checkpoint"] != checkpoint_id:
            raise RuntimeError("Phase 2D prediction checkpoint ID mismatch")
        class_id = int(prediction["predicted_class_id"])
        if ID_TO_LABEL.get(class_id) != int(prediction["predicted_label"]):
            raise RuntimeError("Classifier class-ID-to-label mapping mismatch")
        scores = [
            prediction["class_logits_by_label"][str(label)]
            for label in CHUNK_SIZES
        ]
        ranking = deterministic_ranking(scores)
        if prediction["ranked_class_ids"] != ranking:
            raise RuntimeError("Saved classifier ranking is not reproducible")


def materialize_final_classification(
    output_root: Path,
    phase1_root: Path,
    run_id: str,
    predictions: Sequence[dict[str, Any]],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if len(predictions) != phase2.EXPECTED_COUNTS["validation"]:
        raise RuntimeError(f"Expected 924 predictions, got {len(predictions)}")
    run_dir = output_root / "runs" / run_id
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    training_config_path = run_dir / "training_config.json"
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    metrics = metrics_with_top2(predictions)
    if metrics != best["classification_metrics"]:
        raise RuntimeError("Final Phase 2D metrics do not reproduce selected checkpoint")
    if metrics["invalid_predictions"]:
        raise RuntimeError("Five-logit classifier unexpectedly produced invalid output")

    oracle_counter = Counter(int(row["oracle_label"]) for row in predictions)
    predicted_counter = Counter(int(row["predicted_label"]) for row in predictions)
    oracle_distribution = {
        str(chunk): oracle_counter[chunk] for chunk in CHUNK_SIZES
    }
    predicted_distribution = {
        str(chunk): predicted_counter[chunk] for chunk in CHUNK_SIZES
    }
    majority_class = min(
        CHUNK_SIZES, key=lambda chunk: (-oracle_counter[chunk], chunk)
    )
    majority_metrics = phase2.fixed_classification_metrics(
        [{**row, "parsed_prediction": majority_class} for row in predictions]
    )
    checkpoint_path = run_dir / "checkpoints" / best["checkpoint_id"]
    canonical: list[dict[str, Any]] = []
    for prediction in predictions:
        row = dict(prediction)
        row.pop("selected_checkpoint", None)
        row["selected_checkpoint_id"] = best["checkpoint_id"]
        row["selected_checkpoint_path"] = str(checkpoint_path)
        row["formulation_version"] = FORMULATION_VERSION
        row["experiment_fingerprint"] = best["experiment_fingerprint"]
        canonical.append(row)

    validation_dir = output_root / "validation"
    classification_dir = output_root / "classification"
    phase2.atomic_jsonl(validation_dir / "predictions.jsonl", canonical)
    phase2.atomic_jsonl(
        validation_dir / "raw_outputs.jsonl",
        (
            {
                "question_id": row["question_id"],
                "document_id": row["document_id"],
                "class_logits_by_label": row["class_logits_by_label"],
                "class_probabilities_by_label": row[
                    "class_probabilities_by_label"
                ],
                "raw_output_semantics": "five_sequence_classifier_head_logits",
            }
            for row in canonical
        ),
    )
    phase2.atomic_jsonl(
        validation_dir / "parsed_predictions.jsonl",
        (
            {
                "question_id": row["question_id"],
                "document_id": row["document_id"],
                "predicted_class_id": row["predicted_class_id"],
                "predicted_label": row["predicted_label"],
                "parsed_prediction": row["parsed_prediction"],
                "prediction_status": row["prediction_status"],
                "top_2_class_ids": row["top_2_class_ids"],
                "top_2_predictions": row["top_2_predictions"],
            }
            for row in canonical
        ),
    )
    phase2.atomic_jsonl(validation_dir / "invalid_outputs.jsonl", [])
    phase2.atomic_json(validation_dir / "runtime_summary.json", dict(runtime))
    classification_payload = {
        "classification_metrics": metrics,
        "oracle_distribution": oracle_distribution,
        "predicted_distribution": predicted_distribution,
        "valid_output_count": len(canonical),
        "valid_output_rate": 1.0,
        "invalid_output_count": 0,
        "invalid_output_percentage": 0.0,
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_metrics["accuracy"],
        "majority_baseline_macro_f1": majority_metrics["macro_f1"],
        "selected_checkpoint_id": best["checkpoint_id"],
        "checkpoint_selection_metric": best["selection_metric"],
        "confusion_matrix_note": (
            "Rows are Oracle labels and columns are classifier predictions, "
            "ordered 10, 20, 40, 80, 160."
        ),
    }
    phase2.atomic_json(classification_dir / "metrics.json", classification_payload)
    confusion_path = classification_dir / "confusion_matrix.csv"
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["oracle\\predicted", *CHUNK_SIZES])
        for label, values in zip(CHUNK_SIZES, metrics["confusion_matrix"]):
            writer.writerow([label, *values])
    histogram_path = phase2b.write_classification_histogram(
        output_root, oracle_distribution, predicted_distribution
    )

    final = {
        "status": "classification_complete_retrieval_pending",
        "experiment_status": "classification_complete_retrieval_pending",
        "phase": "Phase 2D Base sequence-classification fine-tuning",
        "formulation_version": FORMULATION_VERSION,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "evaluated_examples": len(canonical),
        "selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_id": best["checkpoint_id"],
        "experiment_fingerprint": best["experiment_fingerprint"],
        "classification": metrics,
        "classification_metrics": metrics,
        "oracle_distribution": oracle_distribution,
        "predicted_distribution": predicted_distribution,
        "valid_outputs": len(canonical),
        "valid_output_rate": 1.0,
        "invalid_outputs": 0,
        "invalid_output_percentage": 0.0,
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_metrics["accuracy"],
        "majority_baseline_macro_f1": majority_metrics["macro_f1"],
        "id2label": {str(index): label for index, label in ID_TO_LABEL.items()},
        "training": {
            "global_steps": run_summary["global_step"],
            "parameter_updates": run_summary["global_step"],
            "total_parameters": run_summary["total_parameters"],
            "trainable_parameters": run_summary["trainable_parameters"],
            "initial_loss": run_summary["initial_loss"],
            "final_loss": run_summary["final_loss"],
            "validation_loss": run_summary["validation_loss"],
            "elapsed_seconds": run_summary["elapsed_seconds"],
            "gradient_audit": run_summary["gradient_audit"],
            "experiment_fingerprint": run_summary["experiment_fingerprint"],
        },
        "environment": {
            "python_version": training_config["python_version"],
            "python_executable": training_config["python_executable"],
            "torch_version": training_config["torch_version"],
            "torch_cuda_version": training_config["torch_cuda_version"],
            "transformers_version": training_config["transformers_version"],
            "transformers_commit": training_config["transformers_commit"],
            "gpu": training_config["gpu"],
            "dtype": training_config["dtype"],
            "quantization": training_config["quantization"],
        },
        "runtime": dict(runtime),
        "retrieval": None,
        "artifacts": {
            "training_config": str(training_config_path),
            "dataset_manifest": str(run_dir / "dataset_manifest.json"),
            "best_checkpoint": str(run_dir / "best_checkpoint.json"),
            "validation_predictions": str(validation_dir / "predictions.jsonl"),
            "canonical_predictions": str(validation_dir / "predictions.jsonl"),
            "raw_outputs": str(validation_dir / "raw_outputs.jsonl"),
            "parsed_predictions": str(
                validation_dir / "parsed_predictions.jsonl"
            ),
            "invalid_outputs": str(validation_dir / "invalid_outputs.jsonl"),
            "classification_metrics": str(classification_dir / "metrics.json"),
            "confusion_matrix": str(confusion_path),
            "predicted_vs_oracle_histogram": str(histogram_path),
            "validation_runtime": str(validation_dir / "runtime_summary.json"),
            "gradient_coverage_audit": str(
                run_dir / "gradient_coverage_audit.json"
            ),
            "phase1_train_oracle": str(
                phase1_root / "oracle" / "train_oracle.jsonl"
            ),
            "phase1_validation_oracle": str(
                phase1_root / "oracle" / "validation_oracle.jsonl"
            ),
        },
        "created_at": utc_now(),
    }
    phase2.atomic_json(output_root / "final_summary.json", final)
    return final


def final_validation(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 2D final validation requires a BF16 CUDA GPU")
    config = TrainingConfig()
    phase2.set_deterministic_seed(config.seed)
    output_root = Path(args.output_root)
    ensure_output_root(output_root)
    run_dir = output_root / "runs" / args.run_id
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    run_config = json.loads(
        (run_dir / "training_config.json").read_text(encoding="utf-8")
    )
    checkpoint = run_dir / "checkpoints" / best["checkpoint_id"]
    frozen = phase2.load_oracle_records("validation", args.phase1_root)
    load_started = time.perf_counter()
    tokenizer, model, load_audit = load_classifier_model(
        str(checkpoint / "model"),
        None,
        initial_base_load=False,
        seed=config.seed,
    )
    data_manifest = phase2.validate_frozen_data(args.phase1_root)
    fingerprint = experiment_fingerprint(data_manifest, int(tokenizer.pad_token_id))
    if (
        fingerprint != best.get("experiment_fingerprint")
        or fingerprint != run_config.get("experiment_fingerprint")
    ):
        raise RuntimeError("Selected Phase 2D checkpoint fingerprint mismatch")
    device = torch.device("cuda")
    model.to(device)
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    formatted = format_records(
        tokenizer,
        frozen,
        SUPERVISOR_INSTRUCTION,
        config.max_sequence_length,
    )
    loader = DataLoader(
        FormattedDataset(formatted),
        batch_size=config.per_device_batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_classification_batch(
            rows, int(tokenizer.pad_token_id)
        ),
    )
    by_id = {str(row["question_id"]): row for row in frozen}
    inference_started = time.perf_counter()
    predictions, loss_summary = evaluate_classifier(
        model, loader, by_id, best["checkpoint_id"], device
    )
    inference_wall = time.perf_counter() - inference_started
    validate_prediction_identity(predictions, frozen, best["checkpoint_id"])
    selected_predictions = phase2b.read_jsonl(
        run_dir
        / "validation"
        / f"predictions_{best['checkpoint_id']}.jsonl"
    )
    if [_prediction_signature(row) for row in selected_predictions] != [
        _prediction_signature(row) for row in predictions
    ]:
        raise RuntimeError(
            "Reloaded classifier checkpoint does not exactly reproduce "
            "selected-epoch scores"
        )
    timings = [float(row["inference_seconds"]) for row in predictions]
    runtime = {
        "source": "post_training_selected_checkpoint_reload",
        "new_inference_performed": True,
        "inference_method": DECISION_METHOD,
        "model_load_seconds": model_load_seconds,
        "isolated_inference_wall_seconds": inference_wall,
        "selected_epoch_validation_wall_seconds": best[
            "validation_wall_seconds"
        ],
        "selected_epoch_exact_score_match": True,
        "selected_epoch_outputs_compared": len(predictions),
        "mean_inference_seconds": float(np.mean(timings)),
        "median_inference_seconds": float(np.median(timings)),
        "total_allocated_batch_forward_seconds": float(sum(timings)),
        "inference_timing_basis": (
            "synchronized_batch_forward_wall_divided_evenly_per_batch"
        ),
        "uniform_validation_cross_entropy": loss_summary[
            "uniform_cross_entropy"
        ],
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rss_gib": psutil.Process().memory_info().rss / 2**30,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_id": best["checkpoint_id"],
        "checkpoint_model_loading_audit": load_audit,
        "evaluated_examples": len(predictions),
        "completed_at": utc_now(),
    }
    result = materialize_final_classification(
        output_root,
        args.phase1_root,
        args.run_id,
        predictions,
        runtime,
    )
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=phase2.PHASE1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect")
    train = subparsers.add_parser("train")
    train.add_argument(
        "--mode", choices=("full", "smoke", "tiny-overfit"), default="full"
    )
    train.add_argument("--run-id")
    train.add_argument("--resume", type=Path)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--per-class", type=int, default=2)
    final = subparsers.add_parser("final-validation")
    final.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        inspect_phase2d(args)
    elif args.command == "train":
        if args.mode == "tiny-overfit" and args.max_steps is None:
            args.max_steps = 20
        if args.mode == "smoke" and args.max_steps is None:
            args.max_steps = 4
        run_training(args)
    elif args.command == "final-validation":
        final_validation(args)
    else:  # pragma: no cover - argparse enforces supported commands
        raise RuntimeError(f"Unsupported Phase 2D command: {args.command}")


if __name__ == "__main__":
    main()
