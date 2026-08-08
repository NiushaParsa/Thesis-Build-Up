#!/usr/bin/env python
"""Phase 2B restricted-alias Qwen granularity-router training.

This module is deliberately separate from :mod:`qwen_phase2`.  It consumes the
same frozen evidence-length Oracle records, but reformulates routing as a flat
five-way decision over verified single-token aliases.  No Phase 1 or Phase 2
artifact is modified.

Two predeclared variants are supported:

``alias-unweighted``
    Uniform five-class cross-entropy.

``alias-classbalanced``
    Effective-number class weighting with beta=0.999, calculated from the
    preserved training split only.

Both variants use the language-model vocabulary logits at the first assistant
answer position.  Inference is a deterministic argmax restricted to the five
alias tokens; unrestricted text generation and the legacy text parser are not
part of this formulation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os

# Required by CUDA/cuBLAS before torch initializes for deterministic kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import re
import shutil
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
from transformers import AutoModelForMultimodalLM, AutoProcessor

import qwen_phase2 as phase2


MODEL_ID = phase2.MODEL_ID
MODEL_REVISION = phase2.MODEL_REVISION
TRANSFORMERS_COMMIT = phase2.TRANSFORMERS_COMMIT
CHUNK_SIZES = phase2.CLASS_TOKENS
CHUNK_TO_INDEX = phase2.CLASS_TO_INDEX
INDEX_TO_CHUNK = {index: chunk for chunk, index in CHUNK_TO_INDEX.items()}

ALIASES = (1, 2, 3, 4, 5)
ALIAS_TO_CHUNK = dict(zip(ALIASES, CHUNK_SIZES))
CHUNK_TO_ALIAS = {chunk: alias for alias, chunk in ALIAS_TO_CHUNK.items()}
EXPECTED_ALIAS_TOKEN_IDS = {1: 16, 2: 17, 3: 18, 4: 19, 5: 20}

VARIANT_UNWEIGHTED = "alias-unweighted"
VARIANT_CLASSBALANCED = "alias-classbalanced"
VARIANTS = (VARIANT_UNWEIGHTED, VARIANT_CLASSBALANCED)
EFFECTIVE_NUMBER_BETA = 0.999

DEFAULT_OUTPUT_ROOTS = {
    VARIANT_UNWEIGHTED: Path(
        "outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle"
    ),
    VARIANT_CLASSBALANCED: Path(
        "outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle"
    ),
}

PHASE2B_INSTRUCTION = (
    "You are a router for a retrieval-augmented generation system. Based only "
    "on the question, select the chunk size most suitable for retrieving the "
    "evidence required to answer it. Return only its alias: 1=10, 2=20, "
    "3=40, 4=80, 5=160."
)

FORMULATION_VERSION = "qwen-phase2b-restricted-five-alias-next-token-v1"
PREDICTION_STATUS = "valid_restricted_alias_argmax"
DECISION_METHOD = "restricted_five_alias_next_token_argmax"


@dataclass(frozen=True)
class TrainingConfig:
    """The fixed Phase 2B optimization configuration."""

    variant: str
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    training_method: str = "full_parameter_restricted_alias_classification"
    formulation_version: str = FORMULATION_VERSION
    label_encoding: str = "single_token_digit_aliases_1_to_5"
    objective: str = "restricted_five_logit_cross_entropy"
    gradient_window_normalization: str = (
        "weighted_numerator_divided_by_accumulated_target_weight_sum"
    )
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
        "lower_unweighted_validation_ce, earlier_step"
    )
    early_stopping: str = "none_fixed_three_epochs"
    quantization: None = None
    class_weight_beta: float | None = None
    class_weight_source: str = "preserved_training_split_only"

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unsupported Phase 2B variant: {self.variant}")
        expected_beta = (
            EFFECTIVE_NUMBER_BETA
            if self.variant == VARIANT_CLASSBALANCED
            else None
        )
        if self.class_weight_beta != expected_beta:
            object.__setattr__(self, "class_weight_beta", expected_beta)


class FormattedDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a non-empty JSONL artifact without relying on posttraining helpers."""

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"JSONL artifact is empty: {path}")
    return rows


def prune_checkpoints(
    checkpoint_root: Path,
    keep_checkpoint_ids: Iterable[str],
) -> list[str]:
    """Delete only validated stale ``step-NNNNNN`` directories below one run.

    The caller controls the small keep-set.  Path validation deliberately
    happens before any recursive deletion so this helper cannot target an
    output root, workspace, or unrelated directory.
    """

    keep = {str(value) for value in keep_checkpoint_ids}
    if any(not re.fullmatch(r"step-\d{6}", value) for value in keep):
        raise ValueError("Checkpoint keep-set contains an invalid checkpoint ID")
    root = checkpoint_root.resolve()
    if checkpoint_root.name != "checkpoints":
        raise RuntimeError("Checkpoint pruning requires a directory named checkpoints")
    removed: list[str] = []
    if not checkpoint_root.exists():
        return removed
    candidates = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and re.fullmatch(r"step-\d{6}", path.name)
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.parent != root:
            raise RuntimeError(f"Checkpoint path escaped its run root: {path}")
        if path.name in keep:
            continue
        shutil.rmtree(resolved)
        removed.append(path.name)
    return sorted(removed)


def write_classification_histogram(
    output_root: Path,
    oracle_distribution: Mapping[str, int],
    predicted_distribution: Mapping[str, int],
) -> Path:
    """Write the Phase 2B distribution chart without Phase 2 postprocessing."""

    os.environ.setdefault("MPLCONFIGDIR", str((Path("tmp") / "matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(CHUNK_SIZES))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        x - width / 2,
        [oracle_distribution[str(chunk)] for chunk in CHUNK_SIZES],
        width,
        label="Evidence-length Oracle",
    )
    axis.bar(
        x + width / 2,
        [predicted_distribution[str(chunk)] for chunk in CHUNK_SIZES],
        width,
        label="Phase 2B Qwen",
    )
    axis.set_xticks(x, [str(chunk) for chunk in CHUNK_SIZES])
    axis.set_xlabel("Chunk size (tokens)")
    axis.set_ylabel("Validation examples")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = output_root / "classification" / "predicted_vs_oracle.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path


def phase2b_user_messages(
    question_text: str,
    instruction: str = PHASE2B_INSTRUCTION,
) -> list[dict[str, Any]]:
    return phase2.user_messages(question_text, instruction)


def _token_ids(template_output: Any) -> list[int]:
    return phase2._token_ids(template_output)


def effective_number_class_weights(
    counts: Mapping[int, int] | None = None,
    beta: float = EFFECTIVE_NUMBER_BETA,
) -> tuple[float, ...]:
    """Return effective-number weights in canonical chunk-size order.

    The values are normalized to arithmetic class mean one for transparent
    reporting.  Training uses a weighted-mean objective, so a common rescaling
    of all weights has no effect on the gradients.
    """

    counts = counts or phase2.EXPECTED_DISTRIBUTIONS["train"]
    if not 0.0 <= beta < 1.0:
        raise ValueError("Effective-number beta must satisfy 0 <= beta < 1")
    observed = {int(chunk): int(counts[chunk]) for chunk in CHUNK_SIZES}
    if any(value <= 0 for value in observed.values()):
        raise ValueError("Every class count must be positive")
    raw = [
        (1.0 - beta) / (1.0 - beta ** observed[chunk])
        if beta
        else 1.0
        for chunk in CHUNK_SIZES
    ]
    mean = float(sum(raw) / len(raw))
    return tuple(float(value / mean) for value in raw)


def class_weights_for_variant(variant: str) -> tuple[float, ...]:
    if variant == VARIANT_UNWEIGHTED:
        return tuple(1.0 for _ in CHUNK_SIZES)
    if variant == VARIANT_CLASSBALANCED:
        return effective_number_class_weights()
    raise ValueError(f"Unsupported Phase 2B variant: {variant}")


def class_weight_manifest(variant: str) -> dict[str, Any]:
    weights = class_weights_for_variant(variant)
    return {
        "variant": variant,
        "scheme": (
            "uniform"
            if variant == VARIANT_UNWEIGHTED
            else "effective_number"
        ),
        "beta": (
            None
            if variant == VARIANT_UNWEIGHTED
            else EFFECTIVE_NUMBER_BETA
        ),
        "source": "preserved_training_split_only",
        "training_counts": {
            str(chunk): phase2.EXPECTED_DISTRIBUTIONS["train"][chunk]
            for chunk in CHUNK_SIZES
        },
        "weights_normalized_to_arithmetic_class_mean_one": {
            str(chunk): weights[index]
            for index, chunk in enumerate(CHUNK_SIZES)
        },
        "gradient_window_reduction": (
            "sum(weight[target] * per_example_ce) / sum(weight[target])"
        ),
    }


def verify_alias_tokenization(
    processor: Any,
    question_texts: Sequence[str] | None = None,
    instruction: str = PHASE2B_INSTRUCTION,
) -> dict[str, Any]:
    """Prove exact standalone and chat-template alias tokenization."""

    tokenizer = processor.tokenizer
    standalone: dict[int, list[int]] = {}
    for alias in ALIASES:
        ids = [
            int(value)
            for value in tokenizer.encode(str(alias), add_special_tokens=False)
        ]
        standalone[alias] = ids
        if ids != [EXPECTED_ALIAS_TOKEN_IDS[alias]]:
            raise RuntimeError(
                f"Alias {alias} tokenized as {ids}; expected exactly "
                f"[{EXPECTED_ALIAS_TOKEN_IDS[alias]}]"
            )

    questions = list(question_texts or ["What was measured?"])
    if not questions:
        raise ValueError("At least one tokenization-check question is required")
    checks: list[dict[str, Any]] = []
    for question in questions:
        messages = phase2b_user_messages(str(question), instruction)
        prompt_ids = _token_ids(
            processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
        )
        suffixes: dict[int, list[int]] = {}
        for alias in ALIASES:
            conversation = messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(alias)}],
                }
            ]
            full_ids = _token_ids(
                processor.apply_chat_template(
                    conversation, add_generation_prompt=False, tokenize=True
                )
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(
                    "Assistant conversation does not preserve the prompt prefix"
                )
            extension = full_ids[len(prompt_ids) :]
            if not extension or extension[0] != EXPECTED_ALIAS_TOKEN_IDS[alias]:
                raise RuntimeError(
                    f"Alias {alias} is not the first assistant-content token: "
                    f"{extension}"
                )
            suffixes[alias] = extension[1:]
        first_suffix = suffixes[ALIASES[0]]
        if any(suffixes[alias] != first_suffix for alias in ALIASES[1:]):
            raise RuntimeError("Alias targets do not share one common template suffix")
        checks.append(
            {
                "question_sha256": text_sha256(str(question)),
                "prompt_token_count": len(prompt_ids),
                "assistant_template_suffix_token_ids": first_suffix,
                "extensions": {
                    str(alias): [EXPECTED_ALIAS_TOKEN_IDS[alias], *first_suffix]
                    for alias in ALIASES
                },
            }
        )
    return {
        "status": "passed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "alias_to_chunk_size": {
            str(alias): ALIAS_TO_CHUNK[alias] for alias in ALIASES
        },
        "standalone_alias_token_ids": {
            str(alias): standalone[alias] for alias in ALIASES
        },
        "expected_alias_token_ids": {
            str(alias): EXPECTED_ALIAS_TOKEN_IDS[alias] for alias in ALIASES
        },
        "chat_template_checks": checks,
        "verified_at": utc_now(),
    }


def format_classification_example(
    processor: Any,
    record: Mapping[str, Any],
    instruction: str = PHASE2B_INSTRUCTION,
    max_sequence_length: int = 128,
) -> dict[str, Any]:
    """Format prompt-only input and a five-class target index."""

    chunk = int(record["oracle_label"])
    if chunk not in CHUNK_TO_ALIAS:
        raise ValueError(f"Unexpected Oracle class: {chunk}")
    alias = CHUNK_TO_ALIAS[chunk]
    messages = phase2b_user_messages(str(record["question_text"]), instruction)
    input_ids = _token_ids(
        processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    )
    if not input_ids:
        raise RuntimeError("Chat template produced an empty prompt")
    if len(input_ids) > max_sequence_length:
        raise RuntimeError(
            f"Formatted prompt has {len(input_ids)} tokens; maximum is "
            f"{max_sequence_length}"
        )
    return {
        "question_id": str(record["question_id"]),
        "document_id": str(record["document_id"]),
        "question_text": str(record["question_text"]),
        "oracle_label": chunk,
        "target_alias": alias,
        "target_class_index": CHUNK_TO_INDEX[chunk],
        "target_alias_token_id": EXPECTED_ALIAS_TOKEN_IDS[alias],
        "input_ids": input_ids,
        "prompt_token_count": len(input_ids),
        "sequence_token_count": len(input_ids),
    }


def format_records(
    processor: Any,
    records: Sequence[Mapping[str, Any]],
    instruction: str,
    maximum: int,
) -> list[dict[str, Any]]:
    return [
        format_classification_example(processor, row, instruction, maximum)
        for row in records
    ]


def collate_classification_batch(
    rows: Sequence[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Any]:
    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    positions: list[int] = []
    for row in rows:
        length = len(row["input_ids"])
        padding = maximum - length
        input_ids.append(row["input_ids"] + [pad_token_id] * padding)
        attention.append([1] * length + [0] * padding)
        positions.append(length - 1)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "last_prompt_positions": torch.tensor(positions, dtype=torch.long),
        "class_indices": torch.tensor(
            [row["target_class_index"] for row in rows], dtype=torch.long
        ),
        "question_ids": [row["question_id"] for row in rows],
        "rows": list(rows),
    }


def restricted_alias_logits(
    vocabulary_logits: torch.Tensor,
    last_prompt_positions: torch.Tensor,
    alias_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Gather next-token logits and restrict them to the five aliases."""

    if vocabulary_logits.ndim != 3:
        raise ValueError("Expected vocabulary logits shaped [batch, sequence, vocab]")
    if last_prompt_positions.ndim != 1:
        raise ValueError("Expected one last-prompt position per row")
    if vocabulary_logits.shape[0] != last_prompt_positions.shape[0]:
        raise ValueError("Batch size and prompt-position count disagree")
    device = vocabulary_logits.device
    positions = last_prompt_positions.to(device=device, dtype=torch.long)
    if torch.any(positions < 0) or torch.any(positions >= vocabulary_logits.shape[1]):
        raise ValueError("A last-prompt position is outside the sequence")
    rows = torch.arange(vocabulary_logits.shape[0], device=device)
    next_token = vocabulary_logits[rows, positions]
    ids = list(alias_token_ids or [EXPECTED_ALIAS_TOKEN_IDS[a] for a in ALIASES])
    if len(ids) != len(ALIASES) or len(set(ids)) != len(ALIASES):
        raise ValueError("Exactly five distinct alias token IDs are required")
    alias_ids = torch.tensor(ids, device=device, dtype=torch.long)
    return next_token.index_select(-1, alias_ids).float()


def weighted_loss_components(
    class_logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return differentiable numerator/denominator loss components."""

    targets = targets.to(device=class_logits.device, dtype=torch.long)
    weights = class_weights.to(device=class_logits.device, dtype=torch.float32)
    if class_logits.ndim != 2 or class_logits.shape[1] != len(CHUNK_SIZES):
        raise ValueError("Expected five class logits per example")
    if targets.shape != (class_logits.shape[0],):
        raise ValueError("Target count does not match batch size")
    if weights.shape != (len(CHUNK_SIZES),):
        raise ValueError("Expected exactly five class weights")
    per_example = F.cross_entropy(
        class_logits.float(), targets, reduction="none"
    )
    target_weights = weights.index_select(0, targets)
    numerator = torch.sum(per_example * target_weights)
    denominator = torch.sum(target_weights)
    if not torch.isfinite(denominator) or float(denominator.detach().cpu()) <= 0.0:
        raise RuntimeError("Class-weight denominator must be finite and positive")
    return {
        "per_example_unweighted_ce": per_example,
        "target_weights": target_weights,
        "weighted_numerator": numerator,
        "weight_denominator": denominator,
        "weighted_mean": numerator / denominator,
        "unweighted_mean": per_example.mean(),
    }


def normalize_accumulated_gradients(
    parameters: Iterable[torch.nn.Parameter],
    accumulated_weight_sum: float,
) -> None:
    if not math.isfinite(accumulated_weight_sum) or accumulated_weight_sum <= 0.0:
        raise ValueError("Accumulated class-weight sum must be finite and positive")
    scale = 1.0 / accumulated_weight_sum
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def deterministic_ranking(scores: Sequence[float]) -> list[int]:
    """Rank class indices by score, resolving exact ties by smaller alias."""

    if len(scores) != len(ALIASES):
        raise ValueError("Exactly five scores are required")
    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("All restricted class scores must be finite")
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


def metrics_with_top2(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = phase2.fixed_classification_metrics(rows)
    top2_correct = sum(
        int(int(row["oracle_label"]) in [int(x) for x in row["top_2_predictions"]])
        for row in rows
    )
    metrics["top_2_accuracy"] = float(top2_correct / len(rows)) if rows else None
    metrics["top_2_accuracy_status"] = (
        "available_from_comparable_restricted_five_class_logits"
        if rows
        else "unavailable_no_examples"
    )
    return metrics


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_classifier(
    model: Any,
    loader: DataLoader[Any],
    source_records: Mapping[str, Mapping[str, Any]],
    class_weights: torch.Tensor,
    checkpoint_id: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate loss and deterministic five-class scores in one pass."""

    model.eval()
    model.config.use_cache = False
    predictions: list[dict[str, Any]] = []
    weighted_numerator = 0.0
    weight_denominator = 0.0
    unweighted_sum = 0.0
    example_count = 0
    wall_started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            targets = batch["class_indices"].to(device)
            _synchronize(device)
            batch_started = time.perf_counter()
            outputs = model(input_ids=inputs, attention_mask=attention)
            logits = restricted_alias_logits(
                outputs.logits,
                batch["last_prompt_positions"],
            )
            _synchronize(device)
            batch_wall = time.perf_counter() - batch_started
            components = weighted_loss_components(logits, targets, class_weights)
            weighted_numerator += float(
                components["weighted_numerator"].detach().cpu()
            )
            weight_denominator += float(
                components["weight_denominator"].detach().cpu()
            )
            unweighted_sum += float(
                components["per_example_unweighted_ce"].sum().detach().cpu()
            )
            example_count += int(logits.shape[0])
            probabilities = torch.softmax(logits.float(), dim=-1).detach().cpu()
            cpu_logits = logits.detach().cpu()
            per_example_wall = batch_wall / max(1, int(logits.shape[0]))
            for row_index, question_id in enumerate(batch["question_ids"]):
                scores = [float(value) for value in cpu_logits[row_index].tolist()]
                probs = [float(value) for value in probabilities[row_index].tolist()]
                ranked_indices = deterministic_ranking(scores)
                predicted_index = ranked_indices[0]
                predicted_alias = ALIASES[predicted_index]
                predicted_chunk = INDEX_TO_CHUNK[predicted_index]
                record = dict(source_records[str(question_id)])
                predictions.append(
                    {
                        **record,
                        "decision_method": DECISION_METHOD,
                        "raw_qwen_output": str(predicted_alias),
                        "raw_output_semantics": (
                            "restricted_argmax_alias_not_unrestricted_generation"
                        ),
                        "predicted_alias": predicted_alias,
                        "parsed_prediction": predicted_chunk,
                        "prediction_status": PREDICTION_STATUS,
                        "class_logits_by_alias": {
                            str(alias): scores[index]
                            for index, alias in enumerate(ALIASES)
                        },
                        "restricted_probabilities_by_alias": {
                            str(alias): probs[index]
                            for index, alias in enumerate(ALIASES)
                        },
                        "ranked_aliases": [
                            ALIASES[index] for index in ranked_indices
                        ],
                        "ranked_predictions": [
                            INDEX_TO_CHUNK[index] for index in ranked_indices
                        ],
                        "top_2_aliases": [
                            ALIASES[index] for index in ranked_indices[:2]
                        ],
                        "top_2_predictions": [
                            INDEX_TO_CHUNK[index] for index in ranked_indices[:2]
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
        raise RuntimeError("Evaluation prediction and loss counts disagree")
    return predictions, {
        "objective_weighted_ce": weighted_numerator / weight_denominator,
        "unweighted_ce": unweighted_sum / example_count,
        "weighted_numerator": weighted_numerator,
        "weight_denominator": weight_denominator,
        "evaluated_examples": example_count,
        "wall_seconds": time.perf_counter() - wall_started,
    }


def load_model_processor(
    source: str = MODEL_ID,
    revision: str | None = MODEL_REVISION,
) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(source, revision=revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        source, revision=revision, dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.requires_grad_(True)
    return processor, model


def resolve_output_root(args: argparse.Namespace) -> Path:
    root = args.output_root or DEFAULT_OUTPUT_ROOTS[args.variant]
    root = Path(root)
    forbidden = {
        phase2.PHASE1_ROOT.resolve(),
        phase2.DEFAULT_OUTPUT_ROOT.resolve(),
    }
    if root.resolve() in forbidden:
        raise RuntimeError(
            "Phase 2B output root must not be a Phase 1 or Phase 2 artifact root"
        )
    return root


def ensure_experiment_root(output_root: Path, variant: str) -> dict[str, Any]:
    marker_path = output_root / "configuration" / "experiment.json"
    expected = {
        "phase": "Phase 2B",
        "variant": variant,
        "formulation_version": FORMULATION_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "instruction": PHASE2B_INSTRUCTION,
        "instruction_sha256": text_sha256(PHASE2B_INSTRUCTION),
        "alias_to_chunk_size": {
            str(alias): ALIAS_TO_CHUNK[alias] for alias in ALIASES
        },
        "expected_alias_token_ids": {
            str(alias): EXPECTED_ALIAS_TOKEN_IDS[alias] for alias in ALIASES
        },
        "class_weights": class_weight_manifest(variant),
    }
    if marker_path.exists():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"Phase 2B output-root configuration mismatch at {key}"
                )
        return existing
    value = {**expected, "created_at": utc_now()}
    phase2.atomic_json(marker_path, value)
    return value


def experiment_fingerprint(
    variant: str,
    data_manifest: Mapping[str, Any],
    alias_tokenization: Mapping[str, Any],
) -> str:
    payload = {
        "variant": variant,
        "formulation_version": FORMULATION_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "instruction_sha256": text_sha256(PHASE2B_INSTRUCTION),
        "alias_to_chunk_size": ALIAS_TO_CHUNK,
        "alias_token_ids": alias_tokenization["expected_alias_token_ids"],
        "class_weights": class_weight_manifest(variant),
        "train_oracle_sha256": data_manifest["train_oracle_sha256"],
        "validation_oracle_sha256": data_manifest[
            "validation_oracle_sha256"
        ],
    }
    return canonical_json_sha256(payload)


def select_subset(
    records: Sequence[dict[str, Any]],
    mode: str,
    per_class: int,
    seed: int,
    validation: bool = False,
) -> list[dict[str, Any]]:
    if mode == "full":
        return list(records)
    return phase2.select_balanced_subset(
        records, 1 if validation else per_class, seed
    )


def inspect_phase2b(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve_output_root(args)
    ensure_experiment_root(output_root, args.variant)
    data_manifest = phase2.validate_frozen_data(args.phase1_root)
    records = phase2.load_oracle_records("train", args.phase1_root)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenization = verify_alias_tokenization(
        processor,
        [records[0]["question_text"], records[-1]["question_text"]],
    )
    formatted = format_records(
        processor, records, PHASE2B_INSTRUCTION, TrainingConfig(args.variant).max_sequence_length
    )
    result = {
        "status": "passed",
        "variant": args.variant,
        "formulation_version": FORMULATION_VERSION,
        "data": data_manifest,
        "instruction": PHASE2B_INSTRUCTION,
        "instruction_sha256": text_sha256(PHASE2B_INSTRUCTION),
        "alias_tokenization": tokenization,
        "class_weights": class_weight_manifest(args.variant),
        "prompt_lengths": {
            "minimum": min(row["prompt_token_count"] for row in formatted),
            "maximum": max(row["prompt_token_count"] for row in formatted),
            "mean": float(
                np.mean([row["prompt_token_count"] for row in formatted])
            ),
        },
        "model_inputs": ["fixed_phase2b_instruction", "original_question_text"],
        "excluded_inputs": [
            "evidence",
            "evidence_length",
            "answer",
            "paper_text",
            "retrieved_chunks",
            "retrieval_scores",
            "metadata",
            "handcrafted_features",
        ],
        "verified_at": utc_now(),
    }
    phase2.atomic_json(
        output_root / "configuration" / "preflight_manifest.json", result
    )
    print(json.dumps(result, indent=2))
    return result


def _run_id(variant: str, mode: str) -> str:
    return (
        f"qwen-phase2b-{variant}-{mode}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-seed42"
    )


def _training_metadata(
    config: TrainingConfig,
    manifest: Mapping[str, Any],
    run_id: str,
    total_parameters: int,
    trainable_parameters: int,
) -> str:
    value = {
        **asdict(config),
        "run_id": run_id,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": (
            100.0 * trainable_parameters / total_parameters
        ),
        **manifest,
    }
    return "```json\n" + json.dumps(value, indent=2) + "\n```"


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    from torch.utils.tensorboard import SummaryWriter

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 2B full-parameter training requires a BF16 CUDA GPU")
    config = TrainingConfig(args.variant)
    phase2.set_deterministic_seed(config.seed)
    output_root = resolve_output_root(args)
    ensure_experiment_root(output_root, args.variant)
    data_manifest = phase2.validate_frozen_data(args.phase1_root)
    train_records = phase2.load_oracle_records("train", args.phase1_root)
    validation_records = phase2.load_oracle_records(
        "validation", args.phase1_root
    )
    train_records = select_subset(
        train_records, args.mode, args.per_class, config.seed
    )
    validation_records = select_subset(
        validation_records,
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
    run_id = args.run_id or _run_id(args.variant, args.mode)
    run_dir = output_root / "runs" / run_id
    if run_dir.exists() and args.resume is None:
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_root / "tensorboard" / run_id

    processor, model = load_model_processor()
    alias_tokenization = verify_alias_tokenization(
        processor,
        [train_records[0]["question_text"], validation_records[0]["question_text"]],
    )
    fingerprint = experiment_fingerprint(
        args.variant, data_manifest, alias_tokenization
    )
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("Official tokenizer has no pad token")
    formatted_train = format_records(
        processor,
        train_records,
        PHASE2B_INSTRUCTION,
        config.max_sequence_length,
    )
    formatted_validation = format_records(
        processor,
        validation_records,
        PHASE2B_INSTRUCTION,
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
        "prompt_length": {
            "minimum": min(row["prompt_token_count"] for row in formatted_train),
            "maximum": max(row["prompt_token_count"] for row in formatted_train),
            "mean": float(
                np.mean([row["prompt_token_count"] for row in formatted_train])
            ),
        },
        "alias_tokenization": alias_tokenization,
        "class_weights": class_weight_manifest(args.variant),
        "experiment_fingerprint": fingerprint,
        "created_at": utc_now(),
    }
    phase2.atomic_json(run_dir / "dataset_manifest.json", manifest)

    script_path = Path(__file__)
    resume_contract = {
        "training_config": asdict(config),
        "run_mode": args.mode,
        "active_per_device_batch_size": active_batch_size,
        "active_gradient_accumulation_steps": active_accumulation,
        "active_effective_batch_size": active_batch_size * active_accumulation,
        "maximum_optimizer_steps": args.max_steps,
        "experiment_fingerprint": fingerprint,
    }
    resume_contract_sha256 = canonical_json_sha256(resume_contract)
    run_config = {
        **asdict(config),
        "run_mode": args.mode,
        "active_per_device_batch_size": active_batch_size,
        "active_gradient_accumulation_steps": active_accumulation,
        "active_effective_batch_size": active_batch_size * active_accumulation,
        "maximum_optimizer_steps": args.max_steps,
        "run_id": run_id,
        "output_root": str(output_root),
        "repository_commit": os.getenv("PHASE2B_REPOSITORY_COMMIT", "unavailable"),
        "training_script_sha256": phase2.sha256_file(script_path),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "transformers_commit": TRANSFORMERS_COMMIT,
        "tensorboard_version": importlib.metadata.version("tensorboard"),
        "gpu": torch.cuda.get_device_name(0),
        "tensorboard_directory": str(tensorboard_dir),
        "instruction": PHASE2B_INSTRUCTION,
        "instruction_sha256": text_sha256(PHASE2B_INSTRUCTION),
        "alias_to_chunk_size": {
            str(alias): ALIAS_TO_CHUNK[alias] for alias in ALIASES
        },
        "alias_token_ids": {
            str(alias): EXPECTED_ALIAS_TOKEN_IDS[alias] for alias in ALIASES
        },
        "class_weights": class_weight_manifest(args.variant),
        "experiment_fingerprint": fingerprint,
        "resume_contract": resume_contract,
        "resume_contract_sha256": resume_contract_sha256,
    }
    config_path = run_dir / "training_config.json"
    if args.resume is not None and config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError("Resume training configuration fingerprint mismatch")
        if existing_config.get("variant") != args.variant:
            raise RuntimeError("Resume variant does not match the existing run")
        if existing_config.get("resume_contract_sha256") != resume_contract_sha256:
            raise RuntimeError("Resume optimizer/schedule contract mismatch")
    else:
        phase2.atomic_json(config_path, run_config)
        phase2.atomic_json(
            run_dir / "formatted_example_inspection.json", formatted_train[:5]
        )

    generator = torch.Generator().manual_seed(config.seed)
    collator = lambda rows: collate_classification_batch(rows, pad_id)
    train_loader = DataLoader(
        FormattedDataset(formatted_train),
        batch_size=active_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        FormattedDataset(formatted_validation),
        batch_size=active_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    validation_by_id = {
        str(row["question_id"]): row for row in validation_records
    }
    steps_per_epoch = phase2.optimizer_steps_for_batches(
        len(train_loader), active_accumulation
    )
    configured_total = steps_per_epoch * config.epochs
    total_steps = args.max_steps or configured_total
    warmup_steps = round(total_steps * config.warmup_ratio)

    def optimizer_factory(current_model: Any) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            current_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def scheduler_factory(current_optimizer: torch.optim.Optimizer) -> Any:
        return phase2.cosine_scheduler(
            current_optimizer, total_steps, warmup_steps
        )

    optimizer = optimizer_factory(model)
    scheduler = scheduler_factory(optimizer)
    state: dict[str, Any] = {
        "global_step": 0,
        "epoch": 0,
        "micro_step": 0,
        "run_id": run_id,
        "variant": args.variant,
        "experiment_fingerprint": fingerprint,
        "resume_contract_sha256": resume_contract_sha256,
    }
    if args.resume is not None:
        processor, model, optimizer, scheduler, state = phase2.load_checkpoint(
            args.resume, optimizer_factory, scheduler_factory, generator
        )
        if state.get("run_id") != run_id:
            raise RuntimeError("Resume run ID does not match the requested run ID")
        if state.get("variant") != args.variant:
            raise RuntimeError("Checkpoint belongs to another Phase 2B variant")
        if state.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError("Checkpoint experiment fingerprint mismatch")
        if state.get("resume_contract_sha256") != resume_contract_sha256:
            raise RuntimeError("Checkpoint optimizer/schedule contract mismatch")
        verify_alias_tokenization(processor)
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
                "variant": args.variant,
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
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError("Phase 2B unexpectedly contains frozen parameters")
    if args.resume is None:
        writer.add_text(
            "configuration/run",
            _training_metadata(
                config,
                manifest,
                run_id,
                total_parameters,
                trainable_parameters,
            ),
            0,
        )

    class_weights = torch.tensor(
        class_weights_for_variant(args.variant), dtype=torch.float32, device=device
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
            class_weights,
            checkpoint_id,
            device,
        )
        metrics = metrics_with_top2(predictions)
        distribution = Counter(row["parsed_prediction"] for row in predictions)
        validation_event = {
            "event": "validation",
            "global_step": state["global_step"],
            "epoch": completed_epoch,
            "loss": loss_summary["unweighted_ce"],
            "unweighted_ce": loss_summary["unweighted_ce"],
            "objective_weighted_ce": loss_summary["objective_weighted_ce"],
            **metrics,
            "predicted_distribution": {
                str(chunk): distribution[chunk] for chunk in CHUNK_SIZES
            },
            "wall_seconds": loss_summary["wall_seconds"],
            "timestamp": utc_now(),
        }
        phase2.append_jsonl(run_dir / "validation_history.jsonl", validation_event)
        for tag, value in (
            ("validation/loss", loss_summary["unweighted_ce"]),
            ("validation/unweighted_ce", loss_summary["unweighted_ce"]),
            (
                "validation/objective_weighted_ce",
                loss_summary["objective_weighted_ce"],
            ),
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
        # Before writing the next full checkpoint, retain only the best prior
        # checkpoint.  The model currently being trained is already resident in
        # memory; if writing fails, that prior best remains a recoverable state.
        prior_best_id = (
            phase2.select_best_evaluation(evaluation_records)["checkpoint_id"]
            if evaluation_records
            else None
        )
        pruned_before_save = prune_checkpoints(
            run_dir / "checkpoints",
            {prior_best_id} if prior_best_id is not None else set(),
        )
        state.update(
            {
                "epoch": completed_epoch,
                "validation_metrics": metrics,
                "validation_unweighted_ce": loss_summary["unweighted_ce"],
                "validation_objective_weighted_ce": loss_summary[
                    "objective_weighted_ce"
                ],
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
            processor,
            optimizer,
            scheduler,
            state,
            generator,
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
            "validation_loss": loss_summary["unweighted_ce"],
            "validation_unweighted_ce": loss_summary["unweighted_ce"],
            "validation_objective_weighted_ce": loss_summary[
                "objective_weighted_ce"
            ],
            "classification_metrics": metrics,
            "predicted_distribution": validation_event[
                "predicted_distribution"
            ],
            "predictions": str(prediction_path),
            "validation_wall_seconds": loss_summary["wall_seconds"],
            "variant": args.variant,
            "experiment_fingerprint": fingerprint,
            "checkpoints_pruned_before_save": pruned_before_save,
        }
        evaluation_records.append(record)
        current_best = phase2.select_best_evaluation(evaluation_records)
        retained_ids = {checkpoint_id, current_best["checkpoint_id"]}
        removed_ids = prune_checkpoints(
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
        accumulated_weighted_numerator = 0.0
        accumulated_weight_sum = 0.0
        accumulated_unweighted_sum = 0.0
        accumulation_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            batch_examples = int(batch["input_ids"].shape[0])
            batch_tokens = int(batch["attention_mask"].sum().item())
            inputs = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            targets = batch["class_indices"].to(device)
            outputs = model(input_ids=inputs, attention_mask=attention)
            logits = restricted_alias_logits(
                outputs.logits, batch["last_prompt_positions"]
            )
            components = weighted_loss_components(logits, targets, class_weights)
            numerator = components["weighted_numerator"]
            if not torch.isfinite(numerator):
                raise RuntimeError(
                    f"Non-finite loss numerator at step {state['global_step']}"
                )
            numerator.backward()
            state["micro_step"] += 1
            accumulated_batches += 1
            accumulated_examples += batch_examples
            accumulated_tokens += batch_tokens
            accumulated_weighted_numerator += float(numerator.detach().cpu())
            accumulated_weight_sum += float(
                components["weight_denominator"].detach().cpu()
            )
            accumulated_unweighted_sum += float(
                components["per_example_unweighted_ce"].sum().detach().cpu()
            )
            end_of_epoch = batch_index + 1 == len(train_loader)
            if accumulated_batches < active_accumulation and not end_of_epoch:
                continue

            normalize_accumulated_gradients(
                model.parameters(), accumulated_weight_sum
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
            objective_loss = (
                accumulated_weighted_numerator / accumulated_weight_sum
            )
            unweighted_ce = accumulated_unweighted_sum / accumulated_examples
            initial_loss = objective_loss if initial_loss is None else initial_loss
            final_loss = objective_loss
            log = {
                "event": "train_step",
                "global_step": state["global_step"],
                "epoch": epoch + (batch_index + 1) / max(1, len(train_loader)),
                "loss": objective_loss,
                "objective_weighted_ce": objective_loss,
                "unweighted_ce": unweighted_ce,
                "accumulated_weight_sum": accumulated_weight_sum,
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
                ("train/objective_weighted_ce", "objective_weighted_ce"),
                ("train/unweighted_ce", "unweighted_ce"),
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
            accumulated_weighted_numerator = 0.0
            accumulated_weight_sum = 0.0
            accumulated_unweighted_sum = 0.0
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
    if args.mode == "full" and int(state["global_step"]) != total_steps:
        raise RuntimeError(
            f"Full Phase 2B run ended at {state['global_step']}; expected "
            f"{total_steps}"
        )
    best = phase2.select_best_evaluation(evaluation_records)
    checkpoints_pruned_at_completion = prune_checkpoints(
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
            "variant": args.variant,
            "experiment_fingerprint": fingerprint,
            "selected_at": utc_now(),
        },
    )
    summary = {
        "status": "complete",
        "phase": "Phase 2B",
        "variant": args.variant,
        "mode": args.mode,
        "run_id": run_id,
        "global_step": state["global_step"],
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "validation_unweighted_ce": best["validation_unweighted_ce"],
        "validation_objective_weighted_ce": best[
            "validation_objective_weighted_ce"
        ],
        "validation_metrics": best["classification_metrics"],
        "validation_events": len(evaluation_records),
        "selected_checkpoint": best["checkpoint"],
        "selected_checkpoint_id": best["checkpoint_id"],
        "selection_reason": config.checkpoint_selection_metric,
        "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rss_gib": process.memory_info().rss / 2**30,
        "retained_checkpoint": best["checkpoint"],
        "checkpoint_retention": (
            "at_most_current_and_best_during_training_selected_only_at_completion"
        ),
        "checkpoints_pruned_at_completion": checkpoints_pruned_at_completion,
        "tensorboard_directory": str(tensorboard_dir),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "experiment_fingerprint": fingerprint,
        "created_at": utc_now(),
    }
    phase2.atomic_json(run_dir / "summary.json", summary)
    writer.close()
    print(json.dumps(summary, indent=2))
    return summary


def _validate_prediction_identity(
    predictions: Sequence[dict[str, Any]],
    frozen: Sequence[dict[str, Any]],
    checkpoint_id: str,
) -> None:
    if len(predictions) != len(frozen):
        raise RuntimeError("Final predictions and frozen validation lengths differ")
    ids = [row["question_id"] for row in predictions]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Final validation predictions contain duplicate IDs")
    if ids != [row["question_id"] for row in frozen]:
        raise RuntimeError("Final predictions do not preserve validation order")
    for prediction, oracle in zip(predictions, frozen):
        for key in ("question_id", "document_id", "question_text", "oracle_label"):
            if prediction[key] != oracle[key]:
                raise RuntimeError(
                    f"Final prediction differs from frozen data at {key}: "
                    f"{prediction['question_id']}"
                )
        if prediction["selected_checkpoint"] != checkpoint_id:
            raise RuntimeError("Final prediction checkpoint ID mismatch")
        alias = int(prediction["predicted_alias"])
        if prediction["parsed_prediction"] != ALIAS_TO_CHUNK[alias]:
            raise RuntimeError("Alias-to-chunk mapping mismatch")
        ranking = deterministic_ranking(
            [
                prediction["class_logits_by_alias"][str(alias_value)]
                for alias_value in ALIASES
            ]
        )
        if prediction["ranked_aliases"] != [ALIASES[index] for index in ranking]:
            raise RuntimeError("Saved class ranking is not reproducible")


def _prediction_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "question_id",
        "raw_qwen_output",
        "predicted_alias",
        "parsed_prediction",
        "prediction_status",
        "class_logits_by_alias",
        "restricted_probabilities_by_alias",
        "ranked_aliases",
        "ranked_predictions",
        "top_2_aliases",
        "top_2_predictions",
    )
    return {key: row[key] for key in keys}


def materialize_final_classification(
    output_root: Path,
    phase1_root: Path,
    variant: str,
    run_id: str,
    predictions: Sequence[dict[str, Any]],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if len(predictions) != phase2.EXPECTED_COUNTS["validation"]:
        raise RuntimeError(f"Expected 924 final predictions, got {len(predictions)}")
    final_path = output_root / "final_summary.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing.get("run_id") != run_id or existing.get("variant") != variant:
            raise RuntimeError("Refusing to overwrite another Phase 2B final result")
    run_dir = output_root / "runs" / run_id
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    training_config_path = run_dir / "training_config.json"
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    if best["variant"] != variant or training_config["variant"] != variant:
        raise RuntimeError("Final materialization variant provenance mismatch")
    metrics = metrics_with_top2(predictions)
    if metrics != best["classification_metrics"]:
        raise RuntimeError("Final metrics do not reproduce the selected checkpoint")
    if metrics["invalid_predictions"]:
        raise RuntimeError("Restricted alias argmax unexpectedly produced invalid output")

    oracle_counter = Counter(int(row["oracle_label"]) for row in predictions)
    predicted_counter = Counter(int(row["parsed_prediction"]) for row in predictions)
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

    canonical: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "checkpoints" / best["checkpoint_id"]
    for prediction in predictions:
        row = dict(prediction)
        row.pop("selected_checkpoint", None)
        row["selected_checkpoint_id"] = best["checkpoint_id"]
        row["selected_checkpoint_path"] = str(checkpoint_path)
        row["phase2b_variant"] = variant
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
                "raw_qwen_output": row["raw_qwen_output"],
                "raw_output_semantics": row["raw_output_semantics"],
                "class_logits_by_alias": row["class_logits_by_alias"],
                "restricted_probabilities_by_alias": row[
                    "restricted_probabilities_by_alias"
                ],
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
                "predicted_alias": row["predicted_alias"],
                "parsed_prediction": row["parsed_prediction"],
                "prediction_status": row["prediction_status"],
                "top_2_aliases": row["top_2_aliases"],
                "top_2_predictions": row["top_2_predictions"],
            }
            for row in canonical
        ),
    )
    phase2.atomic_jsonl(validation_dir / "invalid_outputs.jsonl", [])
    phase2.atomic_json(validation_dir / "runtime_summary.json", dict(runtime))
    classification = {
        "classification_metrics": metrics,
        "oracle_distribution": oracle_distribution,
        "predicted_distribution": predicted_distribution,
        "valid_output_count": len(canonical),
        "valid_output_rate": 1.0,
        "invalid_output_count": 0,
        "invalid_output_percentage": 0.0,
        "top_2_accuracy_note": (
            "Available from five comparable restricted next-token class logits."
        ),
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_metrics["accuracy"],
        "majority_baseline_macro_f1": majority_metrics["macro_f1"],
        "selected_checkpoint_id": best["checkpoint_id"],
        "checkpoint_selection_metric": best["selection_metric"],
        "confusion_matrix_note": (
            "Rows are Oracle labels and columns are restricted-alias predictions, "
            "ordered 10, 20, 40, 80, 160."
        ),
    }
    phase2.atomic_json(classification_dir / "metrics.json", classification)
    confusion_path = classification_dir / "confusion_matrix.csv"
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["oracle\\predicted", *CHUNK_SIZES])
        for label, values in zip(CHUNK_SIZES, metrics["confusion_matrix"]):
            writer.writerow([label, *values])
    write_classification_histogram(
        output_root, oracle_distribution, predicted_distribution
    )

    final = {
        "status": "classification_complete_retrieval_pending",
        "phase": "Phase 2B restricted-alias full-parameter fine-tuning",
        "variant": variant,
        "formulation_version": FORMULATION_VERSION,
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "evaluated_examples": len(canonical),
        "selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_id": best["checkpoint_id"],
        "classification": metrics,
        "oracle_distribution": oracle_distribution,
        "predicted_distribution": predicted_distribution,
        "valid_outputs": len(canonical),
        "valid_output_rate": 1.0,
        "invalid_outputs": 0,
        "invalid_output_percentage": 0.0,
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_metrics["accuracy"],
        "majority_baseline_macro_f1": majority_metrics["macro_f1"],
        "alias_to_chunk_size": {
            str(alias): ALIAS_TO_CHUNK[alias] for alias in ALIASES
        },
        "alias_token_ids": {
            str(alias): EXPECTED_ALIAS_TOKEN_IDS[alias] for alias in ALIASES
        },
        "class_weights": class_weight_manifest(variant),
        "training": {
            "global_steps": run_summary["global_step"],
            "parameter_updates": run_summary["global_step"],
            "total_parameters": run_summary["total_parameters"],
            "trainable_parameters": run_summary["trainable_parameters"],
            "initial_loss": run_summary["initial_loss"],
            "final_loss": run_summary["final_loss"],
            "validation_unweighted_ce": run_summary[
                "validation_unweighted_ce"
            ],
            "validation_objective_weighted_ce": run_summary[
                "validation_objective_weighted_ce"
            ],
            "elapsed_seconds": run_summary["elapsed_seconds"],
            "experiment_fingerprint": run_summary[
                "experiment_fingerprint"
            ],
        },
        "runtime": dict(runtime),
        "retrieval": None,
        "artifacts": {
            "training_config": str(training_config_path),
            "dataset_manifest": str(run_dir / "dataset_manifest.json"),
            "best_checkpoint": str(run_dir / "best_checkpoint.json"),
            "canonical_predictions": str(validation_dir / "predictions.jsonl"),
            "raw_outputs": str(validation_dir / "raw_outputs.jsonl"),
            "parsed_predictions": str(
                validation_dir / "parsed_predictions.jsonl"
            ),
            "invalid_outputs": str(validation_dir / "invalid_outputs.jsonl"),
            "classification_metrics": str(classification_dir / "metrics.json"),
            "confusion_matrix": str(classification_dir / "confusion_matrix.csv"),
            "predicted_vs_oracle_histogram": str(
                classification_dir / "predicted_vs_oracle.svg"
            ),
            "validation_runtime": str(validation_dir / "runtime_summary.json"),
            "phase1_train_oracle": str(
                phase1_root / "oracle" / "train_oracle.jsonl"
            ),
            "phase1_validation_oracle": str(
                phase1_root / "oracle" / "validation_oracle.jsonl"
            ),
        },
        "created_at": utc_now(),
    }
    phase2.atomic_json(final_path, final)
    return final


def final_validation(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 2B final validation requires a BF16 CUDA GPU")
    config = TrainingConfig(args.variant)
    phase2.set_deterministic_seed(config.seed)
    output_root = resolve_output_root(args)
    ensure_experiment_root(output_root, args.variant)
    run_dir = output_root / "runs" / args.run_id
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    run_config = json.loads((run_dir / "training_config.json").read_text(encoding="utf-8"))
    if best.get("variant") != args.variant or run_config.get("variant") != args.variant:
        raise RuntimeError("Requested variant does not match the selected checkpoint")
    checkpoint = run_dir / "checkpoints" / best["checkpoint_id"]
    frozen = phase2.load_oracle_records("validation", args.phase1_root)
    load_started = time.perf_counter()
    processor, model = load_model_processor(str(checkpoint / "model"), None)
    tokenization = verify_alias_tokenization(
        processor, [frozen[0]["question_text"], frozen[-1]["question_text"]]
    )
    data_manifest = phase2.validate_frozen_data(args.phase1_root)
    fingerprint = experiment_fingerprint(
        args.variant, data_manifest, tokenization
    )
    if fingerprint != best.get("experiment_fingerprint"):
        raise RuntimeError("Selected checkpoint experiment fingerprint mismatch")
    device = torch.device("cuda")
    model.to(device)
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    formatted = format_records(
        processor, frozen, PHASE2B_INSTRUCTION, config.max_sequence_length
    )
    pad_id = processor.tokenizer.pad_token_id
    loader = DataLoader(
        FormattedDataset(formatted),
        batch_size=config.per_device_batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_classification_batch(rows, pad_id),
    )
    by_id = {str(row["question_id"]): row for row in frozen}
    inference_started = time.perf_counter()
    predictions, loss_summary = evaluate_classifier(
        model,
        loader,
        by_id,
        torch.tensor(
            class_weights_for_variant(args.variant), dtype=torch.float32
        ),
        best["checkpoint_id"],
        device,
    )
    inference_wall = time.perf_counter() - inference_started
    _validate_prediction_identity(predictions, frozen, best["checkpoint_id"])
    selected_predictions = read_jsonl(
        run_dir
        / "validation"
        / f"predictions_{best['checkpoint_id']}.jsonl"
    )
    if [_prediction_signature(row) for row in selected_predictions] != [
        _prediction_signature(row) for row in predictions
    ]:
        raise RuntimeError(
            "Reloaded checkpoint does not exactly reproduce selected-epoch scores"
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
        "unweighted_validation_ce": loss_summary["unweighted_ce"],
        "objective_weighted_validation_ce": loss_summary[
            "objective_weighted_ce"
        ],
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rss_gib": psutil.Process().memory_info().rss / 2**30,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_id": best["checkpoint_id"],
        "evaluated_examples": len(predictions),
        "completed_at": utc_now(),
    }
    result = materialize_final_classification(
        output_root,
        args.phase1_root,
        args.variant,
        args.run_id,
        predictions,
        runtime,
    )
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=phase2.PHASE1_ROOT)
    parser.add_argument("--output-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--variant", choices=VARIANTS, required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--variant", choices=VARIANTS, required=True)
    train_parser.add_argument("--run-id")
    train_parser.add_argument(
        "--mode", choices=("full", "smoke", "tiny-overfit"), default="full"
    )
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--per-class", type=int, default=2)

    final_parser = subparsers.add_parser("final-validation")
    final_parser.add_argument("--variant", choices=VARIANTS, required=True)
    final_parser.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        inspect_phase2b(args)
    elif args.command == "train":
        if args.mode == "tiny-overfit" and args.max_steps is None:
            args.max_steps = 20
        if args.mode == "smoke" and args.max_steps is None:
            args.max_steps = 4
        run_training(args)
    elif args.command == "final-validation":
        final_validation(args)
    else:  # pragma: no cover - argparse enforces the choices
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
