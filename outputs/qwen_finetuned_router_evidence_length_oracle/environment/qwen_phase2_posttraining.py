#!/usr/bin/env python
"""Full-parameter supervised Qwen Phase 2 granularity-router training.

The pipeline consumes only the frozen Phase 1 evidence-length Oracle records.
It never regenerates labels or writes into the Phase 1 experiment directory.
Training loss is restricted to the assistant class response and its official
chat-template ending tokens.
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
from typing import Any, Iterable, Sequence

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMultimodalLM, AutoProcessor


MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TRANSFORMERS_COMMIT = "2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7"
CLASS_TOKENS = (10, 20, 40, 80, 160)
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_TOKENS)}
EXPECTED_COUNTS = {"train": 2245, "validation": 924}
EXPECTED_DISTRIBUTIONS = {
    "train": {10: 55, 20: 267, 40: 586, 80: 687, 160: 650},
    "validation": {10: 13, 20: 81, 40: 178, 80: 232, 160: 420},
}
ORACLE_VERSION = "oracle-evidence-length-gpt2-smaller-midpoint-v1"
RETRIEVAL_CONFIG_HASH = "9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8"
RETRIEVAL_SCHEMA_VERSION = 2
RETRIEVAL_METRIC_VERSION = "qasper-token-prf-v2"
RETRIEVAL_NORMALIZATION_VERSION = "lowercase-remove-punctuation-collapse-whitespace-v1"
VALID_CLASS_PATTERN = re.compile(r"(?<!\d)(10|20|40|80|160)(?!\d)")
PHASE1_ROOT = Path("outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle")
DEFAULT_OUTPUT_ROOT = Path("outputs/qwen_finetuned_router_evidence_length_oracle")


@dataclass(frozen=True)
class TrainingConfig:
    """One predeclared full-parameter training configuration."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    training_method: str = "full_parameter_sft"
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
    checkpoint_retention_policy: str = "retain_all_three_epoch_checkpoints"
    checkpoint_selection_metric: str = "validation_macro_f1"
    checkpoint_tie_break: str = (
        "accuracy, weighted_f1, balanced_accuracy, lower_validation_loss, earlier_step"
    )
    early_stopping: str = "none_fixed_three_epochs"
    quantization: None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact without silently accepting blank-only files."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"JSONL artifact is empty: {path}")
    return rows


def truncate_jsonl_after_step(path: Path, maximum_step: int) -> int:
    """Atomically discard post-checkpoint events before a resumed run."""
    if not path.exists():
        return 0
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retained = [row for row in rows if int(row.get("global_step", 0)) <= maximum_step]
    removed = len(rows) - len(retained)
    if removed:
        atomic_jsonl(path, retained)
    return removed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_qwen_class(raw_output: str) -> tuple[int | None, str]:
    """Apply the frozen Phase 1 parser."""
    found = {int(match) for match in VALID_CLASS_PATTERN.findall(raw_output)}
    if len(found) == 1:
        return next(iter(found)), "valid"
    if not found:
        return None, "invalid_no_valid_class"
    return None, "invalid_multiple_classes"


def load_fixed_instruction(phase1_root: Path = PHASE1_ROOT) -> str:
    path = phase1_root / "configuration" / "fixed_prompt.json"
    value = json.loads(path.read_text(encoding="utf-8"))["instruction"]
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Invalid frozen instruction: {path}")
    return value


def prompt_text(question_text: str, instruction: str) -> str:
    return f"{instruction}\n\nQuestion: {question_text}"


def user_messages(question_text: str, instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text(question_text, instruction)}],
        }
    ]


def _token_ids(template_output: Any) -> list[int]:
    if isinstance(template_output, dict):
        template_output = template_output["input_ids"]
    if isinstance(template_output, torch.Tensor):
        template_output = template_output.detach().cpu().tolist()
    if template_output and isinstance(template_output[0], list):
        if len(template_output) != 1:
            raise RuntimeError("Expected one chat-template sequence")
        template_output = template_output[0]
    return [int(value) for value in template_output]


def format_training_example(
    processor: Any,
    record: dict[str, Any],
    instruction: str,
    max_sequence_length: int = 128,
) -> dict[str, Any]:
    """Format one example and mask every token except the assistant target."""
    target = int(record["oracle_label"])
    if target not in CLASS_TO_INDEX:
        raise ValueError(f"Unexpected Oracle class: {target}")
    messages = user_messages(str(record["question_text"]), instruction)
    prompt_ids = _token_ids(
        processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    )
    conversation = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": str(target)}]}
    ]
    full_ids = _token_ids(
        processor.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=True
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Assistant conversation does not preserve prompt token prefix")
    if len(full_ids) > max_sequence_length:
        raise RuntimeError(
            f"Formatted sequence has {len(full_ids)} tokens; maximum is {max_sequence_length}"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not labels or all(value == -100 for value in labels):
        raise RuntimeError("No assistant target tokens contribute to loss")
    target_ids = [value for value in labels if value != -100]
    return {
        "question_id": str(record["question_id"]),
        "document_id": str(record["document_id"]),
        "oracle_label": target,
        "input_ids": full_ids,
        "labels": labels,
        "prompt_token_count": len(prompt_ids),
        "target_token_count": len(target_ids),
        "sequence_token_count": len(full_ids),
        "target_token_ids": target_ids,
    }


def load_oracle_records(split: str, phase1_root: Path = PHASE1_ROOT) -> list[dict[str, Any]]:
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"Unsupported split: {split}")
    path = phase1_root / "oracle" / f"{split}_oracle.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"Expected {EXPECTED_COUNTS[split]} {split} rows, got {len(records)}")
    ids = [str(row["question_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate {split} question IDs")
    distribution = Counter(int(row["oracle_label"]) for row in records)
    if dict(distribution) != EXPECTED_DISTRIBUTIONS[split]:
        raise RuntimeError(f"Unexpected {split} distribution: {dict(distribution)}")
    for row in records:
        if row["split"] != split or row["label_version"] != ORACLE_VERSION:
            raise RuntimeError(f"Frozen Oracle metadata mismatch: {row['question_id']}")
    return records


def validate_frozen_data(phase1_root: Path = PHASE1_ROOT) -> dict[str, Any]:
    train = load_oracle_records("train", phase1_root)
    validation = load_oracle_records("validation", phase1_root)
    train_questions = {row["question_id"] for row in train}
    validation_questions = {row["question_id"] for row in validation}
    train_documents = {row["document_id"] for row in train}
    validation_documents = {row["document_id"] for row in validation}
    if train_questions & validation_questions:
        raise RuntimeError("Train-validation question overlap")
    if train_documents & validation_documents:
        raise RuntimeError("Train-validation document overlap")
    return {
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "train_distribution": EXPECTED_DISTRIBUTIONS["train"],
        "validation_distribution": EXPECTED_DISTRIBUTIONS["validation"],
        "train_oracle_sha256": sha256_file(phase1_root / "oracle" / "train_oracle.jsonl"),
        "validation_oracle_sha256": sha256_file(
            phase1_root / "oracle" / "validation_oracle.jsonl"
        ),
        "verified_at": utc_now(),
    }


class FormattedDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_training_batch(rows: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention: list[list[int]] = []
    for row in rows:
        padding = maximum - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * padding)
        labels.append(row["labels"] + [-100] * padding)
        attention.append([1] * len(row["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "question_ids": [row["question_id"] for row in rows],
    }


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def fixed_classification_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Match the frozen Phase 1 complete-set metric semantics."""
    labels = list(CLASS_TOKENS)
    index = CLASS_TO_INDEX
    confusion = np.zeros((5, 5), dtype=np.int64)
    support = np.zeros(5, dtype=np.int64)
    correct = 0
    valid = 0
    for row in rows:
        gold = int(row["oracle_label"])
        prediction = row.get("parsed_prediction")
        support[index[gold]] += 1
        if prediction is not None:
            prediction = int(prediction)
            confusion[index[gold], index[prediction]] += 1
            valid += 1
            correct += int(gold == prediction)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(float)
    precision = np.divide(true_positive, predicted, out=np.zeros(5), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) != 0)
    total = len(rows)
    present = support != 0
    return {
        "accuracy": float(correct / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "balanced_accuracy": float(recall[present].mean()),
        "top_2_accuracy": None,
        "top_2_accuracy_status": "unavailable_no_comparable_class_scores",
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion.tolist(),
        "evaluated_examples": total,
        "valid_predictions": valid,
        "invalid_predictions": total - valid,
    }


def cosine_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> Any:
    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def optimizer_steps_for_batches(batch_count: int, accumulation_steps: int) -> int:
    if batch_count < 0 or accumulation_steps < 1:
        raise ValueError("Invalid batch or accumulation count")
    return math.ceil(batch_count / accumulation_steps)


def partial_window_gradient_scale(
    accumulation_steps: int,
    nominal_batch_size: int,
    observed_examples: int,
) -> float:
    """Normalize accumulated batch-mean gradients by the examples observed."""
    if accumulation_steps < 1 or nominal_batch_size < 1 or observed_examples < 1:
        raise ValueError("Invalid accumulation-window size")
    return accumulation_steps * nominal_batch_size / observed_examples


def select_best_evaluation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("At least one checkpoint evaluation is required")
    return max(
        records,
        key=lambda record: (
            record["classification_metrics"]["macro_f1"],
            record["classification_metrics"]["accuracy"],
            record["classification_metrics"]["weighted_f1"],
            record["classification_metrics"]["balanced_accuracy"],
            -record["validation_loss"],
            -record["global_step"],
        ),
    )


def select_balanced_subset(records: Sequence[dict[str, Any]], per_class: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in CLASS_TOKENS:
        candidates = [row for row in records if int(row["oracle_label"]) == label]
        selected.extend(rng.sample(candidates, min(per_class, len(candidates))))
    rng.shuffle(selected)
    return selected


def _load_model_processor(source: str = MODEL_ID, revision: str | None = MODEL_REVISION) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(source, revision=revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        source, revision=revision, dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.requires_grad_(True)
    return processor, model


def _format_records(processor: Any, records: Sequence[dict[str, Any]], instruction: str, maximum: int) -> list[dict[str, Any]]:
    return [format_training_example(processor, row, instruction, maximum) for row in records]


def _metadata_text(config: TrainingConfig, data_manifest: dict[str, Any], run_id: str, total: int, trainable: int) -> str:
    value = {
        **asdict(config),
        "run_id": run_id,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total,
        "oracle_version": ORACLE_VERSION,
        **data_manifest,
    }
    return "```json\n" + json.dumps(value, indent=2) + "\n```"


def save_checkpoint(
    checkpoint_dir: Path,
    model: Any,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: dict[str, Any],
    data_generator: torch.Generator | None = None,
) -> None:
    if checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_dir}")
    temporary = checkpoint_dir.with_name(checkpoint_dir.name + ".writing")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "model", safe_serialization=True)
    processor.save_pretrained(temporary / "model")
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    random_states = {
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
    }
    if data_generator is not None:
        random_states["data_loader_generator_state"] = data_generator.get_state()
    torch.save(random_states, temporary / "random_states.pt")
    atomic_json(temporary / "training_state.json", state)
    os.replace(temporary, checkpoint_dir)


def load_checkpoint(
    checkpoint_dir: Path,
    optimizer_factory: Any,
    scheduler_factory: Any,
    data_generator: torch.Generator | None = None,
) -> tuple[Any, Any, torch.optim.Optimizer, Any, dict[str, Any]]:
    processor, model = _load_model_processor(str(checkpoint_dir / "model"), None)
    optimizer = optimizer_factory(model)
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler = scheduler_factory(optimizer)
    scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu", weights_only=False))
    state = json.loads((checkpoint_dir / "training_state.json").read_text(encoding="utf-8"))
    random_states = torch.load(checkpoint_dir / "random_states.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(random_states["torch_rng_state"])
    torch.cuda.set_rng_state_all(random_states["cuda_rng_state_all"])
    random.setstate(random_states["python_random_state"])
    np.random.set_state(random_states["numpy_random_state"])
    if data_generator is not None and "data_loader_generator_state" in random_states:
        data_generator.set_state(random_states["data_loader_generator_state"])
    return processor, model, optimizer, scheduler, state


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move every tensor in a restored optimizer state to the model device."""
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[key] = value.to(device)


def generate_predictions(
    processor: Any,
    model: Any,
    records: Sequence[dict[str, Any]],
    instruction: str,
    checkpoint_id: str,
) -> list[dict[str, Any]]:
    model.eval()
    model.config.use_cache = True
    rows: list[dict[str, Any]] = []
    for record in records:
        messages = user_messages(str(record["question_text"]), instruction)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        elapsed = time.perf_counter() - started
        raw = processor.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        prediction, status = parse_qwen_class(raw)
        rows.append(
            {
                **record,
                "raw_qwen_output": raw,
                "parsed_prediction": prediction,
                "prediction_status": status,
                "inference_seconds": elapsed,
                "selected_checkpoint": checkpoint_id,
            }
        )
    model.config.use_cache = False
    model.train()
    return rows


def evaluate_loss(model: Any, loader: DataLoader[Any]) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                labels=batch["labels"].to("cuda"),
            )
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return float(np.mean(losses))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    from torch.utils.tensorboard import SummaryWriter

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 2 full-parameter training requires a BF16 CUDA GPU")
    config = TrainingConfig()
    set_deterministic_seed(config.seed)
    phase1_root = args.phase1_root
    output_root = args.output_root
    data_manifest = validate_frozen_data(phase1_root)
    instruction = load_fixed_instruction(phase1_root)
    train_records = load_oracle_records("train", phase1_root)
    validation_records = load_oracle_records("validation", phase1_root)
    if args.mode in {"tiny-overfit", "smoke"}:
        train_records = select_balanced_subset(train_records, args.per_class, config.seed)
        validation_records = select_balanced_subset(validation_records, 1, config.seed)
    active_batch_size = (
        1 if args.mode == "tiny-overfit" else 2 if args.mode == "smoke" else config.per_device_batch_size
    )
    active_gradient_accumulation = (
        1 if args.mode == "tiny-overfit" else 2 if args.mode == "smoke" else config.gradient_accumulation_steps
    )
    run_id = args.run_id or (
        f"qwen-phase2-{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-seed{config.seed}"
    )
    run_dir = output_root / "runs" / run_id
    if run_dir.exists() and args.resume is None:
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_root / "tensorboard" / run_id

    processor, model = _load_model_processor()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("Official tokenizer has no pad token")
    formatted_train = _format_records(processor, train_records, instruction, config.max_sequence_length)
    formatted_validation = _format_records(processor, validation_records, instruction, config.max_sequence_length)
    manifest = {
        **data_manifest,
        "active_train_examples": len(formatted_train),
        "active_validation_examples": len(formatted_validation),
        "active_train_distribution": dict(Counter(row["oracle_label"] for row in formatted_train)),
        "sequence_length": {
            "minimum": min(row["sequence_token_count"] for row in formatted_train),
            "maximum": max(row["sequence_token_count"] for row in formatted_train),
            "mean": float(np.mean([row["sequence_token_count"] for row in formatted_train])),
        },
        "created_at": utc_now(),
    }
    atomic_json(run_dir / "dataset_manifest.json", manifest)
    run_config = {
        **asdict(config),
        "run_mode": args.mode,
        "active_per_device_batch_size": active_batch_size,
        "active_gradient_accumulation_steps": active_gradient_accumulation,
        "active_effective_batch_size": active_batch_size * active_gradient_accumulation,
        "maximum_optimizer_steps": args.max_steps,
        "run_id": run_id,
        "repository_commit": os.getenv("PHASE2_REPOSITORY_COMMIT", "unavailable"),
        "training_script_sha256": sha256_file(Path(__file__)),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "transformers_commit": TRANSFORMERS_COMMIT,
        "tensorboard_version": importlib.metadata.version("tensorboard"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tensorboard_directory": str(tensorboard_dir),
    }
    atomic_json(run_dir / "training_config.json", run_config)
    atomic_json(run_dir / "formatted_example_inspection.json", formatted_train[:5])

    generator = torch.Generator().manual_seed(config.seed)
    collator = lambda rows: collate_training_batch(rows, pad_id)
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
    steps_per_epoch = optimizer_steps_for_batches(
        len(train_loader), active_gradient_accumulation
    )
    configured_total = steps_per_epoch * config.epochs
    total_steps = args.max_steps or configured_total
    warmup_steps = round(total_steps * config.warmup_ratio)

    def optimizer_factory(current_model: Any) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            current_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def scheduler_factory(current_optimizer: torch.optim.Optimizer) -> Any:
        return cosine_scheduler(current_optimizer, total_steps, warmup_steps)

    optimizer = optimizer_factory(model)
    scheduler = scheduler_factory(optimizer)
    state = {"global_step": 0, "epoch": 0, "micro_step": 0, "run_id": run_id}
    if args.resume is not None:
        processor, model, optimizer, scheduler, state = load_checkpoint(
            args.resume, optimizer_factory, scheduler_factory, generator
        )
        if state["run_id"] != run_id:
            raise RuntimeError("Resume run ID does not match requested run ID")
        removed_train = truncate_jsonl_after_step(
            run_dir / "training_history.jsonl", int(state["global_step"])
        )
        removed_validation = truncate_jsonl_after_step(
            run_dir / "validation_history.jsonl", int(state["global_step"])
        )
        append_jsonl(
            run_dir / "resume_history.jsonl",
            {
                "checkpoint": str(args.resume),
                "resumed_global_step": int(state["global_step"]),
                "removed_post_checkpoint_train_events": removed_train,
                "removed_post_checkpoint_validation_events": removed_validation,
                "timestamp": utc_now(),
            },
        )
    writer = SummaryWriter(
        log_dir=str(tensorboard_dir),
        purge_step=int(state["global_step"]) + 1 if args.resume is not None else None,
    )
    model.to("cuda")
    move_optimizer_state(optimizer, torch.device("cuda"))
    model.train()
    model.config.use_cache = False
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_parameters != total_parameters:
        raise RuntimeError("Full-parameter run unexpectedly contains frozen parameters")
    if args.resume is None:
        writer.add_text("configuration/run", _metadata_text(config, manifest, run_id, total_parameters, trainable_parameters), 0)
    optimizer.zero_grad(set_to_none=True)
    process = psutil.Process()
    started = time.perf_counter()
    initial_loss: float | None = state.get("initial_loss")
    final_loss: float | None = state.get("last_loss")
    elapsed_before_resume = float(state.get("cumulative_elapsed_seconds", 0.0))
    stop = False
    checkpoint_manifest_path = run_dir / "checkpoint_manifest.json"
    evaluation_records: list[dict[str, Any]] = []
    if args.resume is not None and checkpoint_manifest_path.exists():
        evaluation_records = [
            record
            for record in json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
            if int(record["global_step"]) <= int(state["global_step"])
        ]

    def evaluate_and_checkpoint(completed_epoch: int) -> dict[str, Any]:
        validation_started = time.perf_counter()
        validation_loss = evaluate_loss(model, validation_loader)
        checkpoint_id = f"step-{state['global_step']:06d}"
        predictions = generate_predictions(
            processor, model, validation_records, instruction, checkpoint_id
        )
        metrics = fixed_classification_metrics(predictions)
        prediction_distribution = Counter(
            row["parsed_prediction"]
            for row in predictions
            if row["parsed_prediction"] is not None
        )
        validation_event = {
            "event": "validation",
            "global_step": state["global_step"],
            "epoch": completed_epoch,
            "loss": validation_loss,
            **metrics,
            "predicted_distribution": {
                str(label): prediction_distribution[label] for label in CLASS_TOKENS
            },
            "wall_seconds": time.perf_counter() - validation_started,
            "timestamp": utc_now(),
        }
        append_jsonl(run_dir / "validation_history.jsonl", validation_event)
        writer.add_scalar("validation/loss", validation_loss, state["global_step"])
        for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
            writer.add_scalar(f"validation/{key}", metrics[key], state["global_step"])
        writer.add_scalar(
            "validation/invalid_output_count",
            metrics["invalid_predictions"],
            state["global_step"],
        )
        writer.add_scalar(
            "validation/invalid_output_percentage",
            100.0 * metrics["invalid_predictions"] / len(predictions),
            state["global_step"],
        )
        for label, values in metrics["per_class"].items():
            for metric_name in ("precision", "recall", "f1"):
                writer.add_scalar(
                    f"validation/class_{label}_{metric_name}",
                    values[metric_name],
                    state["global_step"],
                )
        for label in CLASS_TOKENS:
            writer.add_scalar(
                f"validation/predicted_class_{label}_count",
                prediction_distribution[label],
                state["global_step"],
            )
        writer.flush()

        checkpoint = run_dir / "checkpoints" / checkpoint_id
        state.update(
            {
                "epoch": completed_epoch,
                "validation_metrics": metrics,
                "validation_loss": validation_loss,
                "initial_loss": initial_loss,
                "last_loss": final_loss,
                "cumulative_elapsed_seconds": (
                    elapsed_before_resume + time.perf_counter() - started
                ),
            }
        )
        save_checkpoint(
            checkpoint, model, processor, optimizer, scheduler, state, generator
        )
        prediction_path = run_dir / "validation" / f"predictions_{checkpoint_id}.jsonl"
        atomic_jsonl(prediction_path, predictions)
        record = {
            "checkpoint": str(checkpoint),
            "checkpoint_id": checkpoint_id,
            "global_step": state["global_step"],
            "epoch": completed_epoch,
            "validation_loss": validation_loss,
            "classification_metrics": metrics,
            "predicted_distribution": validation_event["predicted_distribution"],
            "predictions": str(prediction_path),
            "validation_wall_seconds": validation_event["wall_seconds"],
        }
        evaluation_records.append(record)
        atomic_json(checkpoint_manifest_path, evaluation_records)
        return record

    active_epochs = max(config.epochs, math.ceil(total_steps / max(steps_per_epoch, 1)))
    for epoch in range(int(state["epoch"]), active_epochs):
        accumulated_batches = 0
        accumulated_examples = 0
        accumulated_tokens = 0
        accumulated_weighted_loss = 0.0
        accumulation_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            batch_examples = int(batch["input_ids"].shape[0])
            batch_tokens = int(batch["attention_mask"].sum().item())
            outputs = model(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                labels=batch["labels"].to("cuda"),
            )
            loss = (
                outputs.loss
                * batch_examples
                / (active_gradient_accumulation * active_batch_size)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at global step {state['global_step']}")
            loss.backward()
            observed_loss = float(outputs.loss.detach().cpu())
            state["micro_step"] += 1
            accumulated_batches += 1
            accumulated_examples += batch_examples
            accumulated_tokens += batch_tokens
            accumulated_weighted_loss += observed_loss * batch_examples
            end_of_epoch = batch_index + 1 == len(train_loader)
            if accumulated_batches < active_gradient_accumulation and not end_of_epoch:
                continue
            correction = partial_window_gradient_scale(
                active_gradient_accumulation,
                active_batch_size,
                accumulated_examples,
            )
            if not math.isclose(correction, 1.0):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping).detach().cpu())
            used_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            state["epoch"] = epoch
            duration = time.perf_counter() - accumulation_started
            mean_loss = accumulated_weighted_loss / accumulated_examples
            initial_loss = mean_loss if initial_loss is None else initial_loss
            final_loss = mean_loss
            log = {
                "event": "train_step",
                "global_step": state["global_step"],
                "epoch": epoch + (batch_index + 1) / max(1, len(train_loader)),
                "loss": mean_loss,
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
            append_jsonl(run_dir / "training_history.jsonl", log)
            for tag, key in (
                ("train/loss", "loss"), ("train/learning_rate", "learning_rate"),
                ("train/epoch", "epoch"), ("train/gradient_norm", "gradient_norm"),
                ("train/step_duration_seconds", "step_duration_seconds"),
                ("train/examples_per_second", "examples_per_second"),
                ("train/tokens_per_second", "tokens_per_second"),
                ("system/cpu_ram_gib", "cpu_ram_gib"),
                ("system/gpu_memory_allocated_gib", "gpu_memory_allocated_gib"),
                ("system/gpu_memory_reserved_gib", "gpu_memory_reserved_gib"),
            ):
                writer.add_scalar(tag, log[key], state["global_step"])
            writer.add_scalar("train/global_step", state["global_step"], state["global_step"])
            accumulated_batches = 0
            accumulated_examples = 0
            accumulated_tokens = 0
            accumulated_weighted_loss = 0.0
            accumulation_started = time.perf_counter()
            if state["global_step"] >= total_steps:
                stop = True
                break
        state["epoch"] = epoch + 1
        if args.mode == "train" or stop:
            evaluate_and_checkpoint(epoch + 1)
        if stop:
            break

    if not evaluation_records:
        evaluate_and_checkpoint(int(state["epoch"]))
    if args.mode == "train" and int(state["global_step"]) != total_steps:
        raise RuntimeError(
            f"Full run ended at step {state['global_step']}; expected {total_steps}"
        )
    best = select_best_evaluation(evaluation_records)
    atomic_json(
        run_dir / "best_checkpoint.json",
        {
            **best,
            "selection_metric": config.checkpoint_selection_metric,
            "tie_break": config.checkpoint_tie_break,
            "selected_at": utc_now(),
        },
    )
    summary = {
        "status": "complete",
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
        "latest_checkpoint": evaluation_records[-1]["checkpoint"],
        "tensorboard_directory": str(tensorboard_dir),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "created_at": utc_now(),
    }
    atomic_json(run_dir / "summary.json", summary)
    writer.close()
    print(json.dumps(summary, indent=2))
    return summary


def inspect_data(args: argparse.Namespace) -> None:
    manifest = validate_frozen_data(args.phase1_root)
    instruction = load_fixed_instruction(args.phase1_root)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    train = load_oracle_records("train", args.phase1_root)
    formatted = _format_records(processor, train, instruction, 128)
    manifest["formatted_lengths"] = {
        "minimum": min(row["sequence_token_count"] for row in formatted),
        "maximum": max(row["sequence_token_count"] for row in formatted),
        "mean": float(np.mean([row["sequence_token_count"] for row in formatted])),
        "target_token_counts": sorted(set(row["target_token_count"] for row in formatted)),
    }
    manifest["label_target_token_ids"] = {
        str(label): next(row["target_token_ids"] for row in formatted if row["oracle_label"] == label)
        for label in CLASS_TOKENS
    }
    atomic_json(args.output_root / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def audit_tensorboard(args: argparse.Namespace) -> None:
    """Verify all required TensorBoard scalars against authoritative JSONL logs."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_dir = args.output_root / "runs" / args.run_id
    event_dir = args.output_root / "tensorboard" / args.run_id
    accumulator = EventAccumulator(str(event_dir))
    accumulator.Reload()
    tags = accumulator.Tags()
    scalar_tags = sorted(tags.get("scalars", []))
    train_rows = [
        json.loads(line)
        for line in (run_dir / "training_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation_path = run_dir / "validation_history.jsonl"
    validation_rows = read_jsonl(validation_path) if validation_path.exists() else []
    tensorboard_values = {
        tag: {int(event.step): float(event.value) for event in accumulator.Scalars(tag)}
        for tag in scalar_tags
    }
    expected: list[tuple[str, int, float]] = []
    train_mapping = {
        "train/loss": "loss",
        "train/learning_rate": "learning_rate",
        "train/epoch": "epoch",
        "train/gradient_norm": "gradient_norm",
        "train/step_duration_seconds": "step_duration_seconds",
        "train/examples_per_second": "examples_per_second",
        "train/tokens_per_second": "tokens_per_second",
        "system/cpu_ram_gib": "cpu_ram_gib",
        "system/gpu_memory_allocated_gib": "gpu_memory_allocated_gib",
        "system/gpu_memory_reserved_gib": "gpu_memory_reserved_gib",
    }
    for row in train_rows:
        step = int(row["global_step"])
        for tag, key in train_mapping.items():
            expected.append((tag, step, float(row[key])))
        expected.append(("train/global_step", step, float(step)))
    validation_mapping = {
        "validation/loss": "loss",
        "validation/accuracy": "accuracy",
        "validation/macro_f1": "macro_f1",
        "validation/weighted_f1": "weighted_f1",
        "validation/balanced_accuracy": "balanced_accuracy",
        "validation/invalid_output_count": "invalid_predictions",
    }
    for row in validation_rows:
        step = int(row["global_step"])
        for tag, key in validation_mapping.items():
            expected.append((tag, step, float(row[key])))
        expected.append(
            (
                "validation/invalid_output_percentage",
                step,
                100.0 * float(row["invalid_predictions"]) / EXPECTED_COUNTS["validation"],
            )
        )
        for label in CLASS_TOKENS:
            values = row["per_class"][str(label)]
            for metric_name in ("precision", "recall", "f1"):
                expected.append(
                    (
                        f"validation/class_{label}_{metric_name}",
                        step,
                        float(values[metric_name]),
                    )
                )
            expected.append(
                (
                    f"validation/predicted_class_{label}_count",
                    step,
                    float(row["predicted_distribution"][str(label)]),
                )
            )
    mismatches = []
    expected_counts: Counter[str] = Counter()
    for tag, step, structured in expected:
        expected_counts[tag] += 1
        observed = tensorboard_values.get(tag, {}).get(step)
        if observed is None or not math.isclose(
            observed, structured, rel_tol=1e-5, abs_tol=1e-9
        ):
            mismatches.append(
                {
                    "tag": tag,
                    "step": step,
                    "structured": structured,
                    "tensorboard": observed,
                }
            )
    count_mismatches = {
        tag: {
            "expected": count,
            "tensorboard": len(accumulator.Scalars(tag)) if tag in scalar_tags else 0,
        }
        for tag, count in expected_counts.items()
        if (len(accumulator.Scalars(tag)) if tag in scalar_tags else 0) != count
    }
    best_checkpoint = json.loads(
        (run_dir / "best_checkpoint.json").read_text(encoding="utf-8")
    )
    tensorboard_evaluations = (
        [
            {
                "global_step": int(row["global_step"]),
                "validation_loss": tensorboard_values["validation/loss"][
                    int(row["global_step"])
                ],
                "classification_metrics": {
                    key: tensorboard_values[f"validation/{key}"][
                        int(row["global_step"])
                    ]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "balanced_accuracy",
                    )
                },
            }
            for row in validation_rows
        ]
        if not mismatches and not count_mismatches
        else []
    )
    tensorboard_selected_step = (
        int(select_best_evaluation(tensorboard_evaluations)["global_step"])
        if tensorboard_evaluations
        else None
    )
    selected_checkpoint_agrees = (
        tensorboard_selected_step is not None
        and tensorboard_selected_step == int(best_checkpoint["global_step"])
    )
    result = {
        "run_id": args.run_id,
        "event_directory": str(event_dir),
        "scalar_tags": scalar_tags,
        "scalar_event_counts": {
            tag: len(accumulator.Scalars(tag)) for tag in scalar_tags
        },
        "structured_train_steps": len(train_rows),
        "structured_validation_events": len(validation_rows),
        "tensorboard_train_loss_steps": len(
            tensorboard_values.get("train/loss", {})
        ),
        "loss_mismatch_count": sum(
            row["tag"] == "train/loss" for row in mismatches
        ),
        "loss_mismatches": [
            row for row in mismatches if row["tag"] == "train/loss"
        ],
        "required_scalar_value_count": len(expected),
        "required_scalar_value_mismatch_count": len(mismatches),
        "required_scalar_value_mismatches": mismatches,
        "required_scalar_count_mismatch_count": len(count_mismatches),
        "required_scalar_count_mismatches": count_mismatches,
        "structured_selected_checkpoint_id": best_checkpoint["checkpoint_id"],
        "structured_selected_checkpoint_step": int(best_checkpoint["global_step"]),
        "tensorboard_derived_selected_checkpoint_step": tensorboard_selected_step,
        "selected_checkpoint_agrees": selected_checkpoint_agrees,
        "verified_at": utc_now(),
    }
    if mismatches or count_mismatches or not selected_checkpoint_agrees:
        raise RuntimeError(
            "TensorBoard and structured logs disagree: "
            f"values={mismatches[:3]}, counts={count_mismatches}"
        )
    atomic_json(run_dir / "tensorboard_scalar_inventory.json", result)
    print(json.dumps(result, indent=2))


def verify_checkpoint(args: argparse.Namespace) -> None:
    """Reload a saved model and prove deterministic parser-compatible generation."""
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint verification requires CUDA")
    set_deterministic_seed(42)
    instruction = load_fixed_instruction(args.phase1_root)
    records = select_balanced_subset(
        load_oracle_records("validation", args.phase1_root), 1, 42
    )
    processor, model = _load_model_processor(str(args.checkpoint / "model"), None)
    model.to("cuda")
    checkpoint_id = args.checkpoint.name
    first = generate_predictions(processor, model, records, instruction, checkpoint_id)
    set_deterministic_seed(42)
    second = generate_predictions(processor, model, records, instruction, checkpoint_id)
    comparisons = []
    for left, right in zip(first, second):
        same = (
            left["raw_qwen_output"] == right["raw_qwen_output"]
            and left["parsed_prediction"] == right["parsed_prediction"]
            and left["prediction_status"] == right["prediction_status"]
        )
        comparisons.append(
            {
                "question_id": left["question_id"],
                "raw_output": left["raw_qwen_output"],
                "parsed_prediction": left["parsed_prediction"],
                "prediction_status": left["prediction_status"],
                "deterministic_repeat_match": same,
            }
        )
    if not all(row["deterministic_repeat_match"] for row in comparisons):
        raise RuntimeError("Reloaded checkpoint generation is not deterministic")
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": {
            path.name: sha256_file(path)
            for path in sorted((args.checkpoint / "model").glob("*.safetensors"))
        },
        "optimizer_state_present": (args.checkpoint / "optimizer.pt").is_file(),
        "scheduler_state_present": (args.checkpoint / "scheduler.pt").is_file(),
        "random_state_present": (args.checkpoint / "random_states.pt").is_file(),
        "all_generation_repeats_match": True,
        "examples": comparisons,
        "verified_at": utc_now(),
    }
    run_dir = args.output_root / "runs" / args.run_id
    atomic_json(run_dir / "checkpoint_verification.json", result)
    print(json.dumps(result, indent=2))


def _write_classification_histogram(
    output_root: Path,
    oracle_distribution: dict[str, int],
    predicted_distribution: dict[str, int],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path("tmp") / "matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(CLASS_TOKENS))
    width = 0.38
    oracle = [oracle_distribution[str(label)] for label in CLASS_TOKENS]
    predicted = [predicted_distribution[str(label)] for label in CLASS_TOKENS]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(x - width / 2, oracle, width, label="Evidence-length Oracle")
    axis.bar(x + width / 2, predicted, width, label="Fine-tuned Qwen")
    axis.set_xticks(x, [str(label) for label in CLASS_TOKENS])
    axis.set_xlabel("Chunk size (tokens)")
    axis.set_ylabel("Validation examples")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = output_root / "classification" / "predicted_vs_oracle.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def materialize_final_classification(
    output_root: Path,
    phase1_root: Path,
    run_id: str,
    predictions: Sequence[dict[str, Any]],
    validation_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Create the complete selected-checkpoint Phase 2 artifact set."""
    if len(predictions) != EXPECTED_COUNTS["validation"]:
        raise RuntimeError(f"Expected 924 final predictions, got {len(predictions)}")
    final_summary_path = output_root / "final_summary.json"
    if final_summary_path.exists():
        existing_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
        if existing_summary.get("run_id") != run_id:
            raise RuntimeError(
                "Refusing to overwrite canonical Phase 2 artifacts from another run: "
                f"{existing_summary.get('run_id')}"
            )
    run_dir = output_root / "runs" / run_id
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    training_config_path = run_dir / "training_config.json"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    current_data = validate_frozen_data(phase1_root)
    for key in (
        "train_examples",
        "validation_examples",
        "train_documents",
        "validation_documents",
        "train_oracle_sha256",
        "validation_oracle_sha256",
    ):
        if dataset_manifest[key] != current_data[key]:
            raise RuntimeError(f"Frozen data changed since training: {key}")
    metrics = fixed_classification_metrics(predictions)
    if metrics != best["classification_metrics"]:
        raise RuntimeError("Selected predictions do not reproduce best-checkpoint metrics")
    if metrics != run_summary["validation_metrics"]:
        raise RuntimeError("Selected predictions do not reproduce run-summary metrics")
    oracle_counter = Counter(int(row["oracle_label"]) for row in predictions)
    predicted_counter = Counter(
        int(row["parsed_prediction"])
        for row in predictions
        if row["parsed_prediction"] is not None
    )
    oracle_distribution = {
        str(label): oracle_counter[label] for label in CLASS_TOKENS
    }
    predicted_distribution = {
        str(label): predicted_counter[label] for label in CLASS_TOKENS
    }
    majority_class = min(
        CLASS_TOKENS, key=lambda label: (-oracle_counter[label], label)
    )
    majority_rows = [
        {**row, "majority_prediction": majority_class} for row in predictions
    ]
    majority_metrics = fixed_classification_metrics(
        [
            {**row, "parsed_prediction": row["majority_prediction"]}
            for row in majority_rows
        ]
    )
    classification = {
        "classification_metrics": metrics,
        "oracle_distribution": oracle_distribution,
        "predicted_distribution": predicted_distribution,
        "invalid_output_count": metrics["invalid_predictions"],
        "invalid_output_percentage": (
            100.0 * metrics["invalid_predictions"] / len(predictions)
        ),
        "valid_output_count": metrics["valid_predictions"],
        "valid_output_rate": metrics["valid_predictions"] / len(predictions),
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_metrics["accuracy"],
        "majority_baseline_macro_f1": majority_metrics["macro_f1"],
        "majority_baseline_metrics": majority_metrics,
        "selected_checkpoint": best["checkpoint"],
        "selected_checkpoint_id": best["checkpoint_id"],
        "checkpoint_selection_metric": best["selection_metric"],
        "confusion_matrix_note": (
            "Rows are evidence-length Oracle labels and columns are parsed Qwen "
            "predictions, both ordered as 10, 20, 40, 80, 160. Invalid outputs "
            "are excluded from matrix cells but remain incorrect in complete-set metrics."
        ),
    }
    validation_dir = output_root / "validation"
    atomic_jsonl(validation_dir / "predictions.jsonl", predictions)
    atomic_jsonl(
        validation_dir / "raw_outputs.jsonl",
        (
            {
                "question_id": row["question_id"],
                "document_id": row["document_id"],
                "raw_qwen_output": row["raw_qwen_output"],
            }
            for row in predictions
        ),
    )
    atomic_jsonl(
        validation_dir / "parsed_predictions.jsonl",
        (
            {
                "question_id": row["question_id"],
                "document_id": row["document_id"],
                "parsed_prediction": row["parsed_prediction"],
                "prediction_status": row["prediction_status"],
            }
            for row in predictions
        ),
    )
    atomic_jsonl(
        validation_dir / "invalid_outputs.jsonl",
        (row for row in predictions if row["parsed_prediction"] is None),
    )
    atomic_json(validation_dir / "runtime_summary.json", validation_runtime)
    classification_dir = output_root / "classification"
    atomic_json(classification_dir / "metrics.json", classification)
    classification_dir.mkdir(parents=True, exist_ok=True)
    with (classification_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["oracle\\predicted", *CLASS_TOKENS])
        for label, values in zip(CLASS_TOKENS, metrics["confusion_matrix"]):
            writer.writerow([label, *values])
    _write_classification_histogram(
        output_root, oracle_distribution, predicted_distribution
    )
    phase1_summary_path = phase1_root / "final_summary.json"
    phase1_summary = json.loads(phase1_summary_path.read_text(encoding="utf-8"))
    final_summary = {
        "status": "classification_complete_retrieval_pending",
        "phase": "Phase 2 full-parameter supervised fine-tuning",
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "training_method": "full_parameter_sft",
        "evaluated_examples": len(predictions),
        "selected_checkpoint": str(run_dir / "checkpoints" / best["checkpoint_id"]),
        "selected_checkpoint_id": best["checkpoint_id"],
        "checkpoint_selection": {
            "metric": best["selection_metric"],
            "tie_break": best["tie_break"],
            "validation_loss": best["validation_loss"],
        },
        "classification": metrics,
        "oracle_distribution": oracle_distribution,
        "predicted_distribution": predicted_distribution,
        "valid_outputs": metrics["valid_predictions"],
        "valid_output_rate": metrics["valid_predictions"] / len(predictions),
        "invalid_outputs": metrics["invalid_predictions"],
        "invalid_output_percentage": classification["invalid_output_percentage"],
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_metrics["accuracy"],
        "majority_baseline_macro_f1": majority_metrics["macro_f1"],
        "phase1_comparison": {
            "description": (
                "Same preserved split and same evidence-length Oracle; pretrained "
                "zero-shot Phase 1 baseline, not an old retrieval-F1-Oracle result."
            ),
            "source": str(phase1_summary_path),
            "accuracy": phase1_summary["classification"]["accuracy"],
            "macro_f1": phase1_summary["classification"]["macro_f1"],
            "weighted_f1": phase1_summary["classification"]["weighted_f1"],
            "balanced_accuracy": phase1_summary["classification"]["balanced_accuracy"],
            "mean_joined_retrieval_f1": phase1_summary["retrieval"]["valid_only_mean_joined_f1"],
        },
        "data": {
            "train_examples": dataset_manifest["train_examples"],
            "validation_examples": dataset_manifest["validation_examples"],
            "train_distribution": dataset_manifest["train_distribution"],
            "validation_distribution": dataset_manifest["validation_distribution"],
            "train_oracle_sha256": dataset_manifest["train_oracle_sha256"],
            "validation_oracle_sha256": dataset_manifest["validation_oracle_sha256"],
        },
        "environment": {
            "environment_name": ".venv-qwen",
            "python_version": training_config["python_version"],
            "python_executable": training_config["python_executable"],
            "torch_version": training_config["torch_version"],
            "torch_cuda_version": training_config["torch_cuda_version"],
            "transformers_version": training_config["transformers_version"],
            "transformers_commit": training_config["transformers_commit"],
            "tensorboard_version": training_config["tensorboard_version"],
            "gpu": training_config["gpu"],
            "device": training_config["device"],
            "dtype": training_config["dtype"],
            "quantization": training_config["quantization"],
        },
        "training": {
            "global_steps": run_summary["global_step"],
            "parameter_updates": run_summary["global_step"],
            "total_parameters": run_summary["total_parameters"],
            "trainable_parameters": run_summary["trainable_parameters"],
            "initial_loss": run_summary["initial_loss"],
            "final_loss": run_summary["final_loss"],
            "validation_events": run_summary["validation_events"],
            "peak_gpu_allocated_gib": run_summary["peak_gpu_allocated_gib"],
            "peak_gpu_reserved_gib": run_summary["peak_gpu_reserved_gib"],
            "final_rss_gib": run_summary["rss_gib"],
            "configuration_path": str(training_config_path),
            "configuration_sha256": sha256_file(training_config_path),
            "dataset_manifest_path": str(dataset_manifest_path),
            "training_script_sha256": training_config["training_script_sha256"],
            "repository_commit": training_config["repository_commit"],
            "trainable_percentage": (
                100.0
                * run_summary["trainable_parameters"]
                / run_summary["total_parameters"]
            ),
        },
        "runtime": {
            "training_wall_seconds": run_summary["elapsed_seconds"],
            "training_wall_includes_epoch_validation_and_checkpointing": True,
            "selected_validation": validation_runtime,
            "final_validation_load_plus_inference_wall_seconds": (
                float(validation_runtime.get("model_load_seconds") or 0.0)
                + float(
                    validation_runtime.get("isolated_generation_wall_seconds")
                    or 0.0
                )
            ),
            "known_training_plus_final_validation_wall_seconds": (
                float(run_summary["elapsed_seconds"])
                + float(validation_runtime.get("model_load_seconds") or 0.0)
                + float(
                    validation_runtime.get("isolated_generation_wall_seconds")
                    or 0.0
                )
            ),
            "retrieval_wall_seconds": None,
            "known_training_final_validation_and_retrieval_wall_seconds": None,
        },
        "retrieval": None,
        "artifacts": {
            "training_config": str(training_config_path),
            "dataset_manifest": str(dataset_manifest_path),
            "best_checkpoint": str(run_dir / "best_checkpoint.json"),
            "selected_epoch_predictions": best["predictions"],
            "canonical_predictions": str(validation_dir / "predictions.jsonl"),
            "raw_outputs": str(validation_dir / "raw_outputs.jsonl"),
            "parsed_predictions": str(
                validation_dir / "parsed_predictions.jsonl"
            ),
            "invalid_outputs": str(validation_dir / "invalid_outputs.jsonl"),
            "validation_runtime": str(validation_dir / "runtime_summary.json"),
            "classification_metrics": str(classification_dir / "metrics.json"),
            "confusion_matrix": str(classification_dir / "confusion_matrix.csv"),
            "predicted_vs_oracle_histogram": str(
                classification_dir / "predicted_vs_oracle.svg"
            ),
            "fixed_prompt": str(phase1_root / "configuration" / "fixed_prompt.json"),
            "fixed_prompt_sha256": sha256_file(
                phase1_root / "configuration" / "fixed_prompt.json"
            ),
        },
        "created_at": utc_now(),
    }
    atomic_json(final_summary_path, final_summary)
    return final_summary


def validate_and_canonicalize_final_predictions(
    predictions: Sequence[dict[str, Any]],
    frozen: Sequence[dict[str, Any]],
    best: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Verify final inference identity, parser output, and checkpoint provenance."""
    if len(predictions) != EXPECTED_COUNTS["validation"]:
        raise RuntimeError(f"Expected 924 final predictions, got {len(predictions)}")
    ids = [row["question_id"] for row in predictions]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Final validation predictions contain duplicate IDs")
    if ids != [row["question_id"] for row in frozen]:
        raise RuntimeError("Final predictions do not match frozen validation order")
    canonical_predictions: list[dict[str, Any]] = []
    for prediction, oracle in zip(predictions, frozen):
        for key in ("question_id", "document_id", "question_text", "oracle_label"):
            if prediction[key] != oracle[key]:
                raise RuntimeError(
                    f"Selected prediction differs from frozen Oracle at {key}: "
                    f"{prediction['question_id']}"
                )
        if prediction.get("selected_checkpoint") != best["checkpoint_id"]:
            raise RuntimeError("Selected prediction checkpoint ID mismatch")
        reparsed = parse_qwen_class(prediction["raw_qwen_output"])
        if reparsed != (
            prediction["parsed_prediction"], prediction["prediction_status"]
        ):
            raise RuntimeError(
                f"Saved parser result is not reproducible: {prediction['question_id']}"
            )
        canonical = dict(prediction)
        canonical.pop("selected_checkpoint", None)
        canonical["selected_checkpoint_id"] = best["checkpoint_id"]
        canonical["selected_checkpoint_path"] = str(
            run_dir / "checkpoints" / best["checkpoint_id"]
        )
        canonical_predictions.append(canonical)
    return canonical_predictions


def final_validation(args: argparse.Namespace) -> None:
    """Reload the selected checkpoint and run required final deterministic inference."""
    if not torch.cuda.is_available():
        raise RuntimeError("Final validation requires CUDA")
    set_deterministic_seed(42)
    run_dir = args.output_root / "runs" / args.run_id
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "checkpoints" / best["checkpoint_id"]
    frozen = load_oracle_records("validation", args.phase1_root)
    load_started = time.perf_counter()
    processor, model = _load_model_processor(str(checkpoint / "model"), None)
    model.to("cuda")
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    predictions = generate_predictions(
        processor, model, frozen, load_fixed_instruction(args.phase1_root),
        best["checkpoint_id"],
    )
    inference_wall = time.perf_counter() - inference_started
    canonical_predictions = validate_and_canonicalize_final_predictions(
        predictions, frozen, best, run_dir
    )
    selected_epoch_predictions = read_jsonl(
        run_dir / "validation" / f"predictions_{best['checkpoint_id']}.jsonl"
    )
    if len(selected_epoch_predictions) != len(predictions):
        raise RuntimeError("Selected epoch and final inference lengths disagree")
    for saved, final in zip(selected_epoch_predictions, predictions):
        comparable = ("question_id", "raw_qwen_output", "parsed_prediction", "prediction_status")
        if any(saved[key] != final[key] for key in comparable):
            raise RuntimeError(
                "Reloaded selected checkpoint does not reproduce its epoch output: "
                f"{final['question_id']}"
            )
    timings = [float(row["inference_seconds"]) for row in predictions]
    runtime = {
        "source": "post_training_selected_checkpoint_reload",
        "new_inference_performed": True,
        "model_load_seconds": model_load_seconds,
        "isolated_generation_wall_seconds": inference_wall,
        "selected_epoch_validation_wall_seconds": best["validation_wall_seconds"],
        "selected_epoch_validation_wall_includes": (
            "validation loss pass, deterministic generation, parsing, and metrics"
        ),
        "selected_epoch_exact_output_match": True,
        "selected_epoch_outputs_compared": len(predictions),
        "mean_inference_seconds": float(np.mean(timings)),
        "median_inference_seconds": float(np.median(timings)),
        "total_recorded_generate_call_seconds": float(sum(timings)),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rss_gib": psutil.Process().memory_info().rss / 2**30,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_id": best["checkpoint_id"],
        "evaluated_examples": len(predictions),
        "completed_at": utc_now(),
    }
    summary = materialize_final_classification(
        args.output_root,
        args.phase1_root,
        args.run_id,
        canonical_predictions,
        runtime,
    )
    print(json.dumps(summary, indent=2))


def validate_phase2_retrieval_record(
    record: dict[str, Any],
    prediction: dict[str, Any],
    run_id: str,
    checkpoint_id: str,
    embedding_model: str,
    embedding_dimension: int,
    tokenizer_name: str,
) -> None:
    """Reject stale incremental retrieval rows before they can be reused."""
    predicted_tokens = int(prediction["parsed_prediction"])
    predicted_level = CLASS_TOKENS.index(predicted_tokens) + 1
    expected = {
        "method_name": "qwen-full-parameter-finetuned-router",
        "phase2_run_id": run_id,
        "evaluation_run_id": f"{run_id}-retrieval-top5-paper",
        "question_id": prediction["question_id"],
        "document_id": prediction["document_id"],
        "split": "validation",
        "granularity_tokens": predicted_tokens,
        "granularity_level": predicted_level,
        "predicted_granularity_tokens": predicted_tokens,
        "predicted_granularity_level": predicted_level,
        "evidence_length_oracle": int(prediction["oracle_label"]),
        "oracle_label_version": ORACLE_VERSION,
        "selected_checkpoint_id": checkpoint_id,
        "selected_checkpoint_path": prediction["selected_checkpoint_path"],
        "k_requested": 5,
        "top_k": 5,
        "paper_restricted": True,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "tokenizer_identity": tokenizer_name,
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
            f"Stale or incompatible retrieval record for {prediction['question_id']}: "
            f"{mismatches}"
        )
def build_phase2_retrieval_summary(
    total_predictions: int,
    records: Sequence[dict[str, Any]],
    current_segment_wall_seconds: float,
    cumulative_recorded_wall_seconds: float,
    embedding_model: str,
    tokenizer_name: str,
) -> dict[str, Any]:
    """Build JSON-safe valid-only and coverage-adjusted retrieval summaries."""
    f1_values = [float(row["f1_joined_topk"]) for row in records]
    valid_count = len(f1_values)
    coverage = valid_count / total_predictions if total_predictions else 0.0
    return {
        "evaluated_examples": total_predictions,
        "valid_prediction_retrievals": valid_count,
        "invalid_predictions_without_retrieval": total_predictions - valid_count,
        "retrieval_coverage": coverage,
        "valid_only_mean_joined_retrieval_f1": (
            float(np.mean(f1_values)) if f1_values else None
        ),
        "valid_only_median_joined_retrieval_f1": (
            float(np.median(f1_values)) if f1_values else None
        ),
        "coverage_adjusted_full_set_mean_joined_retrieval_f1": (
            float(sum(f1_values) / total_predictions) if total_predictions else None
        ),
        "full_set_note": (
            "Invalid Qwen outputs receive no retrieval record and no default "
            "granularity. Valid-only F1 summarizes retrieved valid predictions; "
            "the coverage-adjusted full-set mean assigns zero contribution to "
            "invalid predictions solely for a transparent complete-set summary."
        ),
        "top_k": 5,
        "paper_restricted": True,
        "embedding_model": embedding_model,
        "tokenizer": tokenizer_name,
        "metric": "f1_joined_topk",
        "current_segment_wall_seconds": current_segment_wall_seconds,
        "cumulative_durable_question_processing_seconds": (
            cumulative_recorded_wall_seconds
        ),
    }


def evaluate_phase2_retrieval(args: argparse.Namespace) -> None:
    """Reuse the unchanged Phase 1 same-paper retrieval implementation locally."""
    import qwen_phase1 as phase1

    prediction_path = args.output_root / "validation" / "predictions.jsonl"
    predictions = read_jsonl(prediction_path)
    if len(predictions) != EXPECTED_COUNTS["validation"]:
        raise RuntimeError(f"Expected 924 predictions, got {len(predictions)}")
    prediction_ids = [row["question_id"] for row in predictions]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise RuntimeError("Canonical predictions contain duplicate question IDs")
    final_summary_path = args.output_root / "final_summary.json"
    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    if final_summary.get("run_id") != args.run_id:
        raise RuntimeError("Requested retrieval run ID differs from final summary")
    run_dir = args.output_root / "runs" / args.run_id
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    checkpoint_id = best["checkpoint_id"]
    if final_summary.get("selected_checkpoint_id") != checkpoint_id:
        raise RuntimeError("Final summary and best checkpoint disagree")
    for row in predictions:
        if row.get("selected_checkpoint_id") != checkpoint_id:
            raise RuntimeError("Canonical prediction checkpoint ID mismatch")
    valid = [row for row in predictions if row["parsed_prediction"] is not None]
    result_path = args.output_root / "retrieval" / "results.jsonl"
    existing = (
        read_jsonl(result_path)
        if result_path.exists() and result_path.stat().st_size
        else []
    )
    existing_ids = [row["question_id"] for row in existing]
    if len(set(existing_ids)) != len(existing_ids):
        raise RuntimeError("Retrieval artifact contains duplicate question IDs")
    by_id = {row["question_id"]: row for row in existing}
    valid_ids = {row["question_id"] for row in valid}
    if not set(by_id).issubset(valid_ids):
        raise RuntimeError("Retrieval file contains unknown or invalid predictions")
    prediction_by_id = {row["question_id"]: row for row in valid}
    for question_id, record in by_id.items():
        validate_phase2_retrieval_record(
            record,
            prediction_by_id[question_id],
            args.run_id,
            checkpoint_id,
            phase1.OPENAI_EMBEDDING_MODEL,
            phase1.EMBEDDING_DIM,
            phase1.TOKENIZER_NAME,
        )
        if "phase2_retrieval_wall_seconds" not in record:
            raise RuntimeError(
                "Retrieval record has no durable per-question runtime: "
                f"{question_id}"
            )
    previous_summary_path = args.output_root / "retrieval" / "summary.json"
    if len(by_id) == len(valid) and previous_summary_path.exists():
        previous_summary = json.loads(
            previous_summary_path.read_text(encoding="utf-8")
        )
        if previous_summary.get("run_id") != args.run_id:
            raise RuntimeError("Completed retrieval summary belongs to another run")
        print(json.dumps(previous_summary, indent=2))
        return
    started = time.perf_counter()
    retrieval_run_id = f"{args.run_id}-retrieval-top5-paper"
    client = phase1.qdrant_client() if len(by_id) < len(valid) else None
    completed_this_segment = 0
    for prediction in valid:
        question_id = prediction["question_id"]
        if question_id in by_id:
            continue
        question_started = time.perf_counter()
        if client is None:
            raise RuntimeError("Qdrant client was not initialized")
        points = client.retrieve(
            collection_name=phase1.PAPER_QUESTION_COLLECTION,
            ids=[question_id],
            with_payload=True,
            with_vectors=True,
        )
        if len(points) != 1:
            raise RuntimeError(f"Question point lookup failed: {question_id}")
        point = points[0]
        predicted_tokens = int(prediction["parsed_prediction"])
        level = CLASS_TOKENS.index(predicted_tokens) + 1
        records = list(
            phase1.evaluate_question(
                client=client,
                question_point_id=question_id,
                question_vector=point.vector,
                document_id=prediction["document_id"],
                question_text=prediction["question_text"],
                split="validation",
                top_k=5,
                granularity_levels=[level],
                store_retrieved_text=False,
                chunk_sizes=list(CLASS_TOKENS),
                embedding_model=phase1.OPENAI_EMBEDDING_MODEL,
                embedding_dimension=phase1.EMBEDDING_DIM,
                tokenizer_name=phase1.TOKENIZER_NAME,
                evaluation_run_id=retrieval_run_id,
            )
        )
        if len(records) != 1:
            raise RuntimeError(f"Expected one retrieval result: {question_id}")
        record = records[0]
        record.update(
            {
                "method_name": "qwen-full-parameter-finetuned-router",
                "phase2_run_id": args.run_id,
                "predicted_granularity_tokens": predicted_tokens,
                "predicted_granularity_level": level,
                "qwen_raw_output": prediction["raw_qwen_output"],
                "qwen_prediction_status": prediction["prediction_status"],
                "evidence_length_oracle": prediction["oracle_label"],
                "oracle_label_version": ORACLE_VERSION,
                "selected_checkpoint_id": checkpoint_id,
                "selected_checkpoint_path": prediction["selected_checkpoint_path"],
                "top_k": 5,
                "paper_restricted": True,
                "phase2_retrieval_wall_seconds": (
                    time.perf_counter() - question_started
                ),
            }
        )
        validate_phase2_retrieval_record(
            record,
            prediction,
            args.run_id,
            checkpoint_id,
            phase1.OPENAI_EMBEDDING_MODEL,
            phase1.EMBEDDING_DIM,
            phase1.TOKENIZER_NAME,
        )
        append_jsonl(result_path, record)
        by_id[question_id] = record
        completed_this_segment += 1
    ordered = [by_id[row["question_id"]] for row in valid]
    atomic_jsonl(result_path, ordered)
    segment_wall = time.perf_counter() - started
    segment_path = args.output_root / "retrieval" / "runtime_segments.jsonl"
    append_jsonl(
        segment_path,
        {
            "run_id": args.run_id,
            "evaluation_run_id": retrieval_run_id,
            "new_records": completed_this_segment,
            "records_after_segment": len(ordered),
            "wall_seconds": segment_wall,
            "completed_at": utc_now(),
        },
    )
    segments = read_jsonl(segment_path)
    if any(row.get("run_id") != args.run_id for row in segments):
        raise RuntimeError("Retrieval runtime history contains another run ID")
    cumulative_question_wall = float(
        sum(float(row["phase2_retrieval_wall_seconds"]) for row in ordered)
    )
    summary = build_phase2_retrieval_summary(
        len(predictions),
        ordered,
        segment_wall,
        cumulative_question_wall,
        phase1.OPENAI_EMBEDDING_MODEL,
        phase1.TOKENIZER_NAME,
    )
    summary["run_id"] = args.run_id
    summary["evaluation_run_id"] = retrieval_run_id
    summary["runtime_segments"] = len(segments)
    summary["completed_invocation_wall_seconds"] = float(
        sum(float(row["wall_seconds"]) for row in segments)
    )
    summary["complete_uninterrupted_run_wall_seconds"] = (
        segment_wall if len(existing) == 0 else None
    )
    summary["reported_retrieval_wall_seconds"] = (
        summary["complete_uninterrupted_run_wall_seconds"]
        if summary["complete_uninterrupted_run_wall_seconds"] is not None
        else cumulative_question_wall
    )
    summary["reported_retrieval_wall_basis"] = (
        "complete_uninterrupted_invocation"
        if summary["complete_uninterrupted_run_wall_seconds"] is not None
        else "durable_sum_of_per_question_processing_times_after_resume"
    )
    summary["runtime_note"] = (
        "Each completed question stores its own fsynced processing duration. "
        "Completed invocation walls are also retained; after an interrupted "
        "resume, the durable per-question sum is the non-fabricated runtime basis."
    )
    atomic_json(args.output_root / "retrieval" / "summary.json", summary)
    final_summary["status"] = "complete"
    final_summary["retrieval"] = summary
    final_summary["runtime"]["retrieval_wall_seconds"] = summary[
        "reported_retrieval_wall_seconds"
    ]
    final_summary["runtime"][
        "known_training_final_validation_and_retrieval_wall_seconds"
    ] = (
        float(
            final_summary["runtime"][
                "known_training_plus_final_validation_wall_seconds"
            ]
        )
        + float(summary["reported_retrieval_wall_seconds"])
    )
    final_summary["artifacts"]["retrieval_results"] = str(result_path)
    final_summary["artifacts"]["retrieval_summary"] = str(
        args.output_root / "retrieval" / "summary.json"
    )
    final_summary["completed_at"] = utc_now()
    atomic_json(final_summary_path, final_summary)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect-data")
    tensorboard = subparsers.add_parser("audit-tensorboard")
    tensorboard.add_argument("--run-id", required=True)
    verification = subparsers.add_parser("verify-checkpoint")
    verification.add_argument("--run-id", required=True)
    verification.add_argument("--checkpoint", type=Path, required=True)
    final = subparsers.add_parser("final-validation")
    final.add_argument("--run-id", required=True)
    retrieval = subparsers.add_parser("evaluate-retrieval")
    retrieval.add_argument("--run-id", required=True)
    for command in ("tiny-overfit", "smoke", "train"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-id")
        child.add_argument("--resume", type=Path)
        child.add_argument("--max-steps", type=int)
        child.add_argument("--per-class", type=int, default=1 if command == "tiny-overfit" else 2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect-data":
        inspect_data(args)
    elif args.command == "audit-tensorboard":
        audit_tensorboard(args)
    elif args.command == "verify-checkpoint":
        verify_checkpoint(args)
    elif args.command == "final-validation":
        final_validation(args)
    elif args.command == "evaluate-retrieval":
        evaluate_phase2_retrieval(args)
    else:
        args.mode = args.command
        if args.mode == "tiny-overfit" and args.max_steps is None:
            args.max_steps = 20
        if args.mode == "smoke" and args.max_steps is None:
            args.max_steps = 4
        run_training(args)


if __name__ == "__main__":
    main()
