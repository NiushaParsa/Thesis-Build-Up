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
import json
import math
import os
# Required by CUDA/cuBLAS before torch initializes for deterministic kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import re
import shutil
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
    logging_steps: int = 10
    evaluation_frequency: str = "end_of_epoch"
    checkpoint_frequency: str = "end_of_epoch"
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
    torch.save(
        {
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
        },
        temporary / "random_states.pt",
    )
    atomic_json(temporary / "training_state.json", state)
    os.replace(temporary, checkpoint_dir)


def load_checkpoint(
    checkpoint_dir: Path,
    optimizer_factory: Any,
    scheduler_factory: Any,
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
    writer = SummaryWriter(log_dir=str(tensorboard_dir), purge_step=None)

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
    steps_per_epoch = math.ceil(len(train_loader) / active_gradient_accumulation)
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
            args.resume, optimizer_factory, scheduler_factory
        )
        if state["run_id"] != run_id:
            raise RuntimeError("Resume run ID does not match requested run ID")
    model.to("cuda")
    move_optimizer_state(optimizer, torch.device("cuda"))
    model.train()
    model.config.use_cache = False
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_parameters != total_parameters:
        raise RuntimeError("Full-parameter run unexpectedly contains frozen parameters")
    writer.add_text("configuration/run", _metadata_text(config, manifest, run_id, total_parameters, trainable_parameters), 0)
    optimizer.zero_grad(set_to_none=True)
    process = psutil.Process()
    started = time.perf_counter()
    initial_loss: float | None = None
    final_loss: float | None = None
    stop = False
    active_epochs = max(config.epochs, math.ceil(total_steps / max(steps_per_epoch, 1)))
    for epoch in range(int(state["epoch"]), active_epochs):
        for batch in train_loader:
            micro_started = time.perf_counter()
            outputs = model(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                labels=batch["labels"].to("cuda"),
            )
            loss = outputs.loss / active_gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at global step {state['global_step']}")
            loss.backward()
            observed_loss = float(outputs.loss.detach().cpu())
            initial_loss = observed_loss if initial_loss is None else initial_loss
            final_loss = observed_loss
            state["micro_step"] += 1
            if state["micro_step"] % active_gradient_accumulation != 0:
                continue
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping).detach().cpu())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_step"] += 1
            state["epoch"] = epoch
            duration = time.perf_counter() - micro_started
            current_lr = float(optimizer.param_groups[0]["lr"])
            log = {
                "event": "train_step",
                "global_step": state["global_step"],
                "epoch": epoch + state["micro_step"] / max(1, len(train_loader)),
                "loss": observed_loss,
                "learning_rate": current_lr,
                "gradient_norm": gradient_norm,
                "step_duration_seconds": duration,
                "examples_per_second": active_batch_size * active_gradient_accumulation / max(duration, 1e-9),
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
                ("system/cpu_ram_gib", "cpu_ram_gib"),
                ("system/gpu_memory_allocated_gib", "gpu_memory_allocated_gib"),
                ("system/gpu_memory_reserved_gib", "gpu_memory_reserved_gib"),
            ):
                writer.add_scalar(tag, log[key], state["global_step"])
            writer.add_scalar("train/global_step", state["global_step"], state["global_step"])
            if state["global_step"] >= total_steps:
                stop = True
                break
        state["epoch"] = epoch + 1
        if stop:
            break

    validation_loss = evaluate_loss(model, validation_loader)
    predictions = generate_predictions(processor, model, validation_records, instruction, f"step-{state['global_step']:06d}")
    metrics = fixed_classification_metrics(predictions)
    validation_event = {
        "event": "validation",
        "global_step": state["global_step"],
        "epoch": state["epoch"],
        "loss": validation_loss,
        **metrics,
        "timestamp": utc_now(),
    }
    append_jsonl(run_dir / "validation_history.jsonl", validation_event)
    writer.add_scalar("validation/loss", validation_loss, state["global_step"])
    for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
        writer.add_scalar(f"validation/{key}", metrics[key], state["global_step"])
    writer.add_scalar("validation/invalid_output_count", metrics["invalid_predictions"], state["global_step"])
    writer.add_scalar("validation/invalid_output_percentage", 100.0 * metrics["invalid_predictions"] / len(predictions), state["global_step"])
    for label, values in metrics["per_class"].items():
        for metric_name in ("precision", "recall", "f1"):
            writer.add_scalar(f"validation/class_{label}_{metric_name}", values[metric_name], state["global_step"])
    prediction_distribution = Counter(row["parsed_prediction"] for row in predictions if row["parsed_prediction"] is not None)
    for label in CLASS_TOKENS:
        writer.add_scalar(f"validation/predicted_class_{label}_count", prediction_distribution[label], state["global_step"])
    writer.flush()

    checkpoint = run_dir / "checkpoints" / f"step-{state['global_step']:06d}"
    state.update({"validation_metrics": metrics, "validation_loss": validation_loss})
    save_checkpoint(checkpoint, model, processor, optimizer, scheduler, state)
    prediction_path = run_dir / "validation_predictions.jsonl"
    for row in predictions:
        append_jsonl(prediction_path, row)
    summary = {
        "status": "complete",
        "mode": args.mode,
        "run_id": run_id,
        "global_step": state["global_step"],
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "validation_loss": validation_loss,
        "validation_metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rss_gib": process.memory_info().rss / 2**30,
        "checkpoint": str(checkpoint),
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
    """Verify TensorBoard train scalars against the authoritative JSONL log."""
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
    tensorboard_loss = {
        int(event.step): float(event.value)
        for event in accumulator.Scalars("train/loss")
    }
    mismatches = []
    for row in train_rows:
        step = int(row["global_step"])
        observed = tensorboard_loss.get(step)
        if observed is None or not math.isclose(observed, float(row["loss"]), rel_tol=1e-6, abs_tol=1e-7):
            mismatches.append({"step": step, "structured": row["loss"], "tensorboard": observed})
    result = {
        "run_id": args.run_id,
        "event_directory": str(event_dir),
        "scalar_tags": scalar_tags,
        "scalar_event_counts": {
            tag: len(accumulator.Scalars(tag)) for tag in scalar_tags
        },
        "structured_train_steps": len(train_rows),
        "tensorboard_train_loss_steps": len(tensorboard_loss),
        "loss_mismatch_count": len(mismatches),
        "loss_mismatches": mismatches,
        "verified_at": utc_now(),
    }
    if mismatches:
        raise RuntimeError(f"TensorBoard and structured loss disagree: {mismatches[:3]}")
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
    else:
        args.mode = args.command
        if args.mode == "tiny-overfit" and args.max_steps is None:
            args.max_steps = 20
        if args.mode == "smoke" and args.max_steps is None:
            args.max_steps = 4
        run_training(args)


if __name__ == "__main__":
    main()
