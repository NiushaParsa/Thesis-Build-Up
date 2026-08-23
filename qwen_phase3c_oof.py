#!/usr/bin/env python
"""Leakage-safe Phase 3C stacking with paper-grouped OOF Qwen logits.

The five outer Qwen fits in this experiment are trained only on papers outside
their held-out fold.  Their held-out logits are assembled into one 2,245-row
OOF matrix.  A separate Qwen model is then refit on all preserved training
questions, without reading validation data, and is frozen before the canonical
924 validation rows are opened.  The Phase 3C XGBoost fusion model uses the
same five logits, 173 inference-safe tree features, fixed candidate, class
weights, grouped folds, preprocessing, and seed as the original primary
Phase 3C variant.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import similarity_tree_phase3b as phase3b


PHASE = "Phase 3C-OOF"
FORMULATION_VERSION = "qwen-phase2d-paper-oof-logits-plus-phase3b-tree-xgboost-v1"
CLASS_TOKENS = phase3b.CLASS_TOKENS
SEED = phase3b.SEED
FOLDS = phase3b.FOLDS
TOP_K = phase3b.TOP_K
SOURCE_ROOT = phase3b.SOURCE_FEATURE_ROOT
EXPECTED_SOURCE_HASHES = phase3b.EXPECTED_SOURCE_HASHES
DEFAULT_OUTPUT_ROOT = Path("outputs/qwen_phase3c_oof_fusion_evidence_length_oracle")
ORIGINAL_PHASE3C_ROOT = Path("outputs/qwen_phase3c_fusion_evidence_length_oracle")
PHASE2D_ROOT = Path(
    "outputs/qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle"
)
PHASE3B_ROOT = Path("outputs/similarity_tree_phase3b_xgboost_evidence_length_oracle")
FIXED_BASELINES_PATH = Path("reports/final_validation_comparison/strategy_metrics.json")
VARIANT = "qwen_oof_logits_tree"
TREE_FEATURE_COUNT = 173
QWEN_LOGIT_COUNT = 5
FUSION_FEATURE_COUNT = TREE_FEATURE_COUNT + QWEN_LOGIT_COUNT
FIXED_CANDIDATE = {
    "max_depth": 2,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def source_rows(split: str, *, verify_hash: bool = True) -> list[dict]:
    path = phase3b.feature_path(SOURCE_ROOT, split)
    if not path.exists():
        raise FileNotFoundError(path)
    if verify_hash:
        observed = phase3b.sha256_file(path)
        if observed != EXPECTED_SOURCE_HASHES[split]:
            raise RuntimeError(
                f"Frozen {split} feature hash changed: {observed} != "
                f"{EXPECTED_SOURCE_HASHES[split]}"
            )
    rows = phase3b.read_jsonl(path)
    phase3b.validate_rows(rows, split, 2245 if split == "train" else 924)
    return rows


def training_folds(rows: Sequence[dict]) -> np.ndarray:
    folds = phase3b.grouped_stratified_folds(rows, FOLDS, SEED)
    if folds.shape != (len(rows),) or set(folds.tolist()) != set(range(FOLDS)):
        raise RuntimeError("Invalid paper-grouped fold assignment")
    by_document: dict[str, set[int]] = {}
    for row, fold in zip(rows, folds):
        by_document.setdefault(str(row["document_id"]), set()).add(int(fold))
    crossing = [document for document, values in by_document.items() if len(values) != 1]
    if crossing:
        raise RuntimeError(f"Papers cross OOF folds: {crossing[:3]}")
    return folds


def write_lines(path: Path, values: Iterable[str]) -> None:
    phase3b.atomic_text(path, "".join(f"{value}\n" for value in values))


def fold_dir(output_root: Path, fold: int) -> Path:
    return output_root / "qwen_oof" / f"fold-{fold}"


def fold_logits_path(output_root: Path, fold: int) -> Path:
    return fold_dir(output_root, fold) / "heldout_logits.npz"


def audit_command(args: argparse.Namespace) -> int:
    train = source_rows("train")
    # Availability and integrity are checked here; validation values are not used
    # in any training, model-selection, checkpoint-selection, or threshold decision.
    validation = source_rows("validation")
    folds = training_folds(train)
    required = [
        Path("qwen_phase2d_sequence_classifier.py"),
        Path("qwen_phase3c_fusion.py"),
        Path("similarity_tree_phase3b.py"),
        PHASE2D_ROOT / "final_summary.json",
        PHASE3B_ROOT / "final_summary.json",
        ORIGINAL_PHASE3C_ROOT / "final_summary.json",
        FIXED_BASELINES_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required preserved artifacts: {missing}")
    original = json.loads(
        (ORIGINAL_PHASE3C_ROOT / "final_summary.json").read_text(encoding="utf-8")
    )
    original_parameters = original["models"]["qwen_logits_tree"][
        "selected_candidate"
    ]["parameters"]
    if original_parameters != FIXED_CANDIDATE:
        raise RuntimeError(
            f"Original Phase 3C primary parameters changed: {original_parameters}"
        )
    targets = phase3b.target_array(train)
    manifest = {
        "phase": PHASE,
        "status": "passed",
        "audited_at": utc_now(),
        "repository_commit": _git_commit(),
        "source_feature_hashes": EXPECTED_SOURCE_HASHES,
        "train_examples": len(train),
        "train_documents": len({str(row['document_id']) for row in train}),
        "validation_examples": len(validation),
        "validation_documents": len({str(row['document_id']) for row in validation}),
        "folds": phase3b.fold_manifest(train, targets, folds),
        "fold_assignment_sha256": stable_hash(folds.tolist()),
        "fixed_primary_variant": "five_Qwen_logits_plus_173_tree_features",
        "fixed_candidate": FIXED_CANDIDATE,
        "required_artifacts": {
            str(path): phase3b.sha256_file(path) for path in required
        },
        "validation_preflight_scope": (
            "existence, row count, schema, and frozen-file hash only; no values were "
            "used for training or any decision"
        ),
        "unavoidable_deviations": [
            (
                "Each outer Qwen fit sees four fifths of the training papers, so its "
                "fixed three epochs contain fewer optimizer steps than the original "
                "all-training-data Phase 2D fit."
            ),
            (
                "The original Phase 2D checkpoint was selected by validation macro-F1. "
                "The clean protocol instead freezes the third/final epoch endpoint using "
                "training data only. The original selected checkpoint was also epoch 3."
            ),
            (
                "A sixth Qwen fit on all 2,245 training questions is required to create "
                "validation-time base logits after the stacking procedure is frozen."
            ),
            (
                "Only the original primary Phase 3C five-logit fusion is rerun; the "
                "non-primary 1,024-hidden-state exploratory variant is outside scope."
            ),
        ],
    }
    phase3b.atomic_json(args.output_root / "integrity" / "preflight_audit.json", manifest)
    phase3b.atomic_json(
        args.output_root / "cross_validation" / "paper_grouped_folds.json",
        {
            "seed": SEED,
            "fold_count": FOLDS,
            "assignment_sha256": stable_hash(folds.tolist()),
            "folds": phase3b.fold_manifest(train, targets, folds),
        },
    )
    print(json.dumps(manifest, indent=2))
    return 0


def _gpu_modules():
    # Kept lazy so audit/assembly/meta-evaluation can run without a CUDA stack.
    import psutil
    import torch
    from torch.utils.data import DataLoader

    import qwen_phase2 as phase2
    import qwen_phase2b as phase2b
    import qwen_phase2d_sequence_classifier as phase2d

    return torch, DataLoader, psutil, phase2, phase2b, phase2d


def _qwen_environment(torch: Any, phase2d: Any) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "transformers_commit": phase2d.TRANSFORMERS_COMMIT,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "torch.bfloat16",
    }


def train_qwen(
    rows: Sequence[dict],
    run_dir: Path,
    *,
    role: str,
) -> tuple[Any, Any, list[dict], dict[str, Any]]:
    torch, DataLoader, psutil, phase2, phase2b, phase2d = _gpu_modules()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Clean Phase 3C Qwen training requires a BF16 CUDA GPU")
    config = phase2d.TrainingConfig()
    phase2.set_deterministic_seed(config.seed)
    tokenizer, model, loading_audit = phase2d.load_classifier_model(
        initial_base_load=True,
        seed=config.seed,
    )
    formatted = phase2d.format_records(
        tokenizer,
        rows,
        phase2d.SUPERVISOR_INSTRUCTION,
        config.max_sequence_length,
    )
    pad_id = int(tokenizer.pad_token_id)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        phase2d.FormattedDataset(formatted),
        batch_size=config.per_device_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda batch: phase2d.collate_classification_batch(batch, pad_id),
    )
    steps_per_epoch = phase2.optimizer_steps_for_batches(
        len(loader), config.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = math.ceil(total_steps * config.warmup_ratio)
    optimizer = phase2d._optimizer(model, config)
    scheduler = phase2.cosine_scheduler(optimizer, total_steps, warmup_steps)
    device = torch.device("cuda")
    model.to(device)
    model.train()
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    optimizer.zero_grad(set_to_none=True)
    process = psutil.Process()
    history_path = run_dir / "training_history.jsonl"
    started = time.perf_counter()
    global_step = 0
    micro_step = 0
    first_loss: float | None = None
    last_loss: float | None = None
    peak_rss = process.memory_info().rss
    peak_allocated = 0
    peak_reserved = 0
    for epoch in range(config.epochs):
        accumulated_batches = 0
        accumulated_examples = 0
        accumulated_tokens = 0
        accumulated_loss_sum = 0.0
        window_started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            inputs = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            outputs = model(
                input_ids=inputs,
                attention_mask=attention,
                labels=targets,
                use_cache=False,
            )
            components = phase2d.uniform_ce_components(outputs.logits, targets)
            loss_sum = components["loss_sum"]
            if not torch.isfinite(loss_sum):
                raise RuntimeError(f"Non-finite loss at step {global_step}")
            loss_sum.backward()
            micro_step += 1
            accumulated_batches += 1
            accumulated_examples += int(inputs.shape[0])
            accumulated_tokens += int(attention.sum().item())
            accumulated_loss_sum += float(loss_sum.detach().cpu())
            end_of_epoch = batch_index + 1 == len(loader)
            if (
                accumulated_batches < config.gradient_accumulation_steps
                and not end_of_epoch
            ):
                continue
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
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            duration = time.perf_counter() - window_started
            loss = accumulated_loss_sum / accumulated_examples
            first_loss = loss if first_loss is None else first_loss
            last_loss = loss
            peak_rss = max(peak_rss, process.memory_info().rss)
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
            phase3b.append_jsonl(
                history_path,
                {
                    "event": "train_step",
                    "role": role,
                    "epoch": epoch + (batch_index + 1) / len(loader),
                    "global_step": global_step,
                    "micro_step": micro_step,
                    "loss": loss,
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm,
                    "examples_in_step": accumulated_examples,
                    "tokens_in_step": accumulated_tokens,
                    "step_duration_seconds": duration,
                    "timestamp": utc_now(),
                },
            )
            accumulated_batches = 0
            accumulated_examples = 0
            accumulated_tokens = 0
            accumulated_loss_sum = 0.0
            window_started = time.perf_counter()
    if global_step != total_steps:
        raise RuntimeError(f"Ended at step {global_step}; expected {total_steps}")
    summary = {
        "role": role,
        "status": "trained_fixed_final_epoch_endpoint",
        "training_examples": len(rows),
        "training_documents": len({str(row['document_id']) for row in rows}),
        "training_distribution": dict(
            sorted(Counter(int(row["oracle_label"]) for row in rows).items())
        ),
        "configuration": asdict(config),
        "active_evaluation_frequency": "none",
        "active_checkpoint_selection": "none_fixed_epoch_3_endpoint",
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "initial_step_loss": first_loss,
        "final_step_loss": last_loss,
        "training_wall_seconds": time.perf_counter() - started,
        "approximate_peak_process_rss_gib": peak_rss / 2**30,
        "peak_cuda_allocated_gib": peak_allocated / 2**30,
        "peak_cuda_reserved_gib": peak_reserved / 2**30,
        "model_loading_audit": loading_audit,
        "environment": _qwen_environment(torch, phase2d),
        "completed_at": utc_now(),
    }
    return tokenizer, model, formatted, summary


def extract_logits(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict],
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    torch, DataLoader, _psutil, _phase2, _phase2b, phase2d = _gpu_modules()
    formatted = phase2d.format_records(
        tokenizer,
        rows,
        phase2d.SUPERVISOR_INSTRUCTION,
        phase2d.TrainingConfig().max_sequence_length,
    )
    loader = DataLoader(
        phase2d.FormattedDataset(formatted),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: phase2d.collate_classification_batch(
            batch, int(tokenizer.pad_token_id)
        ),
    )
    model.requires_grad_(False)
    model.eval()
    values: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                use_cache=False,
                return_dict=True,
            )
            values.append(outputs.logits.detach().float().cpu().numpy())
    logits = np.concatenate(values).astype(np.float32)
    if logits.shape != (len(rows), QWEN_LOGIT_COUNT) or not np.isfinite(logits).all():
        raise RuntimeError(f"Invalid extracted Qwen logits: {logits.shape}")
    arrays = {
        "question_ids": np.asarray([str(row["question_id"]) for row in rows]),
        "document_ids": np.asarray([str(row["document_id"]) for row in rows]),
        "oracle_labels": np.asarray(
            [int(row["oracle_label"]) for row in rows], dtype=np.int64
        ),
        "logits": logits,
        "token_counts": np.asarray(
            [int(row["sequence_token_count"]) for row in formatted], dtype=np.int64
        ),
    }
    return arrays, {
        "examples": len(rows),
        "wall_seconds": time.perf_counter() - started,
        "batch_size": batch_size,
        "minimum_token_count": int(arrays["token_counts"].min()),
        "maximum_token_count": int(arrays["token_counts"].max()),
    }


def train_fold_command(args: argparse.Namespace) -> int:
    if not (args.output_root / "integrity" / "preflight_audit.json").exists():
        raise RuntimeError("Run audit before training OOF Qwen folds")
    if args.fold not in range(FOLDS):
        raise ValueError(f"Fold must be 0..{FOLDS - 1}")
    output = fold_dir(args.output_root, args.fold)
    complete = output / "complete.json"
    if complete.exists():
        print(complete.read_text(encoding="utf-8"))
        return 0
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Incomplete fold directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = source_rows("train")
    folds = training_folds(rows)
    train = [row for row, fold in zip(rows, folds) if int(fold) != args.fold]
    heldout = [row for row, fold in zip(rows, folds) if int(fold) == args.fold]
    train_documents = {str(row["document_id"]) for row in train}
    heldout_documents = {str(row["document_id"]) for row in heldout}
    overlap = train_documents & heldout_documents
    if overlap:
        raise RuntimeError(f"Fold {args.fold} paper leakage: {sorted(overlap)[:3]}")
    write_lines(output / "train_document_ids.txt", sorted(train_documents))
    write_lines(output / "heldout_document_ids.txt", sorted(heldout_documents))
    write_lines(
        output / "heldout_question_ids.txt",
        [str(row["question_id"]) for row in heldout],
    )
    tokenizer, model, _formatted, training = train_qwen(
        train, output, role=f"outer_fold_{args.fold}"
    )
    arrays, extraction = extract_logits(
        model, tokenizer, heldout, batch_size=args.extraction_batch_size
    )
    arrays["outer_fold"] = np.full(len(heldout), args.fold, dtype=np.int64)
    atomic_npz(fold_logits_path(args.output_root, args.fold), **arrays)
    result = {
        "phase": PHASE,
        "fold": args.fold,
        "status": "complete",
        "training": training,
        "heldout_extraction": extraction,
        "training_question_ids_sha256": stable_hash(
            [str(row["question_id"]) for row in train]
        ),
        "heldout_question_ids_sha256": stable_hash(
            [str(row["question_id"]) for row in heldout]
        ),
        "training_document_ids_sha256": stable_hash(sorted(train_documents)),
        "heldout_document_ids_sha256": stable_hash(sorted(heldout_documents)),
        "paper_overlap_count": 0,
        "heldout_logits_path": str(fold_logits_path(args.output_root, args.fold)),
        "heldout_logits_sha256": phase3b.sha256_file(
            fold_logits_path(args.output_root, args.fold)
        ),
        "fold_model_retention": (
            "not retained after held-out logit extraction; deterministic training "
            "configuration and history are retained"
        ),
        "completed_at": utc_now(),
    }
    phase3b.atomic_json(output / "complete.json", result)
    print(json.dumps(result, indent=2))
    del model, tokenizer
    gc.collect()
    return 0


def validate_logit_arrays(rows: Sequence[dict], arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "question_ids",
        "document_ids",
        "oracle_labels",
        "logits",
        "token_counts",
    }
    if not required.issubset(arrays):
        raise RuntimeError(f"Missing logit arrays: {sorted(required - set(arrays))}")
    expected_ids = np.asarray([str(row["question_id"]) for row in rows])
    expected_documents = np.asarray([str(row["document_id"]) for row in rows])
    expected_labels = np.asarray([int(row["oracle_label"]) for row in rows])
    if not np.array_equal(arrays["question_ids"], expected_ids):
        raise RuntimeError("Qwen logits are not aligned by question ID")
    if not np.array_equal(arrays["document_ids"], expected_documents):
        raise RuntimeError("Qwen logits are not aligned by document ID")
    if not np.array_equal(arrays["oracle_labels"], expected_labels):
        raise RuntimeError("Qwen logits are not aligned by Oracle label")
    if arrays["logits"].shape != (len(rows), QWEN_LOGIT_COUNT):
        raise RuntimeError(f"Unexpected logit shape: {arrays['logits'].shape}")
    if not np.isfinite(arrays["logits"]).all():
        raise RuntimeError("Qwen logits contain non-finite values")


def assemble_oof_command(args: argparse.Namespace) -> int:
    rows = source_rows("train")
    folds = training_folds(rows)
    by_id: dict[str, dict[str, Any]] = {}
    fold_hashes: dict[str, str] = {}
    for fold in range(FOLDS):
        path = fold_logits_path(args.output_root, fold)
        complete = fold_dir(args.output_root, fold) / "complete.json"
        if not path.exists() or not complete.exists():
            raise RuntimeError(f"Fold {fold} is incomplete")
        arrays = load_npz(path)
        fold_hashes[str(fold)] = phase3b.sha256_file(path)
        for index, question_id in enumerate(arrays["question_ids"]):
            key = str(question_id)
            if key in by_id:
                raise RuntimeError(f"Duplicate OOF question: {key}")
            by_id[key] = {
                "document_id": str(arrays["document_ids"][index]),
                "oracle_label": int(arrays["oracle_labels"][index]),
                "logits": arrays["logits"][index],
                "token_count": int(arrays["token_counts"][index]),
                "outer_fold": int(arrays["outer_fold"][index]),
            }
    if len(by_id) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} OOF rows, found {len(by_id)}")
    ordered = []
    for index, row in enumerate(rows):
        question_id = str(row["question_id"])
        value = by_id.get(question_id)
        if value is None:
            raise RuntimeError(f"Missing OOF logits: {question_id}")
        if value["outer_fold"] != int(folds[index]):
            raise RuntimeError(f"Incorrect OOF provenance: {question_id}")
        if value["document_id"] != str(row["document_id"]):
            raise RuntimeError(f"Incorrect OOF document: {question_id}")
        ordered.append(value)
    arrays = {
        "question_ids": np.asarray([str(row["question_id"]) for row in rows]),
        "document_ids": np.asarray([str(row["document_id"]) for row in rows]),
        "oracle_labels": np.asarray(
            [int(row["oracle_label"]) for row in rows], dtype=np.int64
        ),
        "logits": np.stack([value["logits"] for value in ordered]).astype(np.float32),
        "token_counts": np.asarray(
            [value["token_count"] for value in ordered], dtype=np.int64
        ),
        "outer_folds": folds.astype(np.int64),
    }
    validate_logit_arrays(rows, arrays)
    path = args.output_root / "qwen_features" / "train_oof_logits.npz"
    atomic_npz(path, **arrays)
    manifest = {
        "status": "complete",
        "examples": len(rows),
        "documents": len({str(row['document_id']) for row in rows}),
        "shape": list(arrays["logits"].shape),
        "exactly_once_oof_coverage": True,
        "paper_grouped": True,
        "paper_overlap_within_each_fit": 0,
        "fold_assignment_sha256": stable_hash(folds.tolist()),
        "fold_file_hashes": fold_hashes,
        "assembled_path": str(path),
        "assembled_sha256": phase3b.sha256_file(path),
        "completed_at": utc_now(),
    }
    phase3b.atomic_json(args.output_root / "qwen_features" / "train_oof_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def train_full_command(args: argparse.Namespace) -> int:
    run_dir = args.output_root / "qwen_full_refit"
    complete = run_dir / "complete.json"
    if complete.exists():
        print(complete.read_text(encoding="utf-8"))
        return 0
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Incomplete full-refit directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Deliberately only open the preserved training rows in this command.
    rows = source_rows("train")
    tokenizer, model, _formatted, training = train_qwen(
        rows, run_dir, role="full_training_refit_for_frozen_validation_inference"
    )
    model_path = run_dir / "model"
    model.save_pretrained(model_path, safe_serialization=True)
    tokenizer.save_pretrained(model_path)
    result = {
        "phase": PHASE,
        "status": "complete",
        "training": training,
        "model_path": str(model_path),
        "model_sha256": phase3b.sha256_file(model_path / "model.safetensors"),
        "validation_rows_opened": False,
        "checkpoint_selection": "none_fixed_epoch_3_endpoint",
        "completed_at": utc_now(),
    }
    phase3b.atomic_json(complete, result)
    print(json.dumps(result, indent=2))
    return 0


def freeze_command(args: argparse.Namespace) -> int:
    train = source_rows("train")
    folds = training_folds(train)
    oof_path = args.output_root / "qwen_features" / "train_oof_logits.npz"
    full_complete = args.output_root / "qwen_full_refit" / "complete.json"
    full_model = args.output_root / "qwen_full_refit" / "model" / "model.safetensors"
    for path in (oof_path, full_complete, full_model):
        if not path.exists():
            raise RuntimeError(f"Cannot freeze; missing {path}")
    arrays = load_npz(oof_path)
    validate_logit_arrays(train, arrays)
    if not np.array_equal(arrays["outer_folds"], folds):
        raise RuntimeError("Assembled OOF fold provenance changed")
    lock = {
        "phase": PHASE,
        "status": "frozen_before_validation_evaluation",
        "locked_at": utc_now(),
        "selection_data": "2,245-question preserved training split only",
        "validation_used_for_any_decision": False,
        "base_model_protocol": {
            "outer_folds": FOLDS,
            "paper_grouped": True,
            "seed": SEED,
            "each_training_logit_from_model_excluding_question_and_source_paper": True,
            "validation_model": "fixed third-epoch full-training refit",
            "checkpoint_selection": "none",
        },
        "fusion_protocol": {
            "variant": VARIANT,
            "features": ["five_oof_Qwen_logits", "173_Phase3B_tree_features"],
            "feature_count": FUSION_FEATURE_COUNT,
            "preprocessing": "raw concatenation; no scaling or thresholding",
            "fixed_candidate_source": "original Phase 3C primary qwen_logits_tree",
            "fixed_candidate": FIXED_CANDIDATE,
            "class_weighting": "square-root inverse frequency",
            "seed": SEED,
        },
        "fold_assignment_sha256": stable_hash(folds.tolist()),
        "train_oof_logits_sha256": phase3b.sha256_file(oof_path),
        "full_refit_model_sha256": phase3b.sha256_file(full_model),
        "source_train_features_sha256": EXPECTED_SOURCE_HASHES["train"],
        "procedure_script_sha256": phase3b.sha256_file(Path(__file__)),
    }
    phase3b.atomic_json(args.output_root / "selection" / "selection_lock.json", lock)
    print(json.dumps(lock, indent=2))
    return 0


def extract_validation_command(args: argparse.Namespace) -> int:
    lock_path = args.output_root / "selection" / "selection_lock.json"
    if not lock_path.exists():
        raise RuntimeError("Freeze the complete training-only procedure first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    model_path = args.output_root / "qwen_full_refit" / "model"
    if phase3b.sha256_file(model_path / "model.safetensors") != lock[
        "full_refit_model_sha256"
    ]:
        raise RuntimeError("Frozen full-refit Qwen model changed")
    # This is the first command that opens validation rows for model evaluation.
    rows = source_rows("validation")
    torch, _DataLoader, _psutil, _phase2, _phase2b, phase2d = _gpu_modules()
    tokenizer, model, loading_audit = phase2d.load_classifier_model(
        str(model_path), None, initial_base_load=False, seed=SEED
    )
    model.to("cuda")
    arrays, extraction = extract_logits(
        model, tokenizer, rows, batch_size=args.extraction_batch_size
    )
    validate_logit_arrays(rows, arrays)
    path = args.output_root / "qwen_features" / "validation_logits.npz"
    atomic_npz(path, **arrays)
    marker = {
        "phase": PHASE,
        "status": "complete",
        "validation_examples": len(rows),
        "validation_logits_path": str(path),
        "validation_logits_sha256": phase3b.sha256_file(path),
        "frozen_selection_lock_sha256": phase3b.sha256_file(lock_path),
        "extraction": extraction,
        "model_loading_audit": loading_audit,
        "validation_used_for_training_or_selection": False,
        "completed_at": utc_now(),
    }
    phase3b.atomic_json(args.output_root / "qwen_features" / "validation_manifest.json", marker)
    print(json.dumps(marker, indent=2))
    return 0


def fusion_matrix(
    rows: Sequence[dict], arrays: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    validate_logit_arrays(rows, arrays)
    tree, tree_names = phase3b.feature_matrix(rows, "tree_features")
    matrix = np.concatenate(
        [np.asarray(arrays["logits"], dtype=np.float32), tree], axis=1
    ).astype(np.float32, copy=False)
    names = [f"qwen_logit_{tokens}" for tokens in CLASS_TOKENS] + [
        f"tree__{name}" for name in tree_names
    ]
    if matrix.shape != (len(rows), FUSION_FEATURE_COUNT) or len(names) != FUSION_FEATURE_COUNT:
        raise RuntimeError(f"Invalid fusion feature shape: {matrix.shape}")
    return matrix, names


def fixed_candidate_cv(
    features: np.ndarray,
    names: Sequence[str],
    targets: np.ndarray,
    folds: np.ndarray,
) -> dict:
    probabilities = np.zeros((len(targets), len(CLASS_TOKENS)), dtype=np.float64)
    fold_results = []
    started = time.perf_counter()
    for fold in range(FOLDS):
        train_mask = folds != fold
        heldout_mask = folds == fold
        weights, class_weights = phase3b.class_balance_weights(targets[train_mask])
        booster = phase3b.train_booster(
            features[train_mask],
            targets[train_mask],
            weights,
            names,
            FIXED_CANDIDATE,
            SEED + fold,
        )
        fold_probabilities = phase3b.predict_booster(
            booster, features[heldout_mask], names
        )
        probabilities[heldout_mask] = fold_probabilities
        fold_results.append(
            {
                "fold": fold,
                "train_examples": int(train_mask.sum()),
                "heldout_examples": int(heldout_mask.sum()),
                "class_weights": class_weights,
                "metrics": phase3b.classification_metrics(
                    targets[heldout_mask],
                    fold_probabilities.argmax(axis=1),
                    fold_probabilities,
                ),
            }
        )
    return {
        "status": "complete_fixed_candidate_diagnostic_not_model_selection",
        "candidate": FIXED_CANDIDATE,
        "metrics": phase3b.classification_metrics(
            targets, probabilities.argmax(axis=1), probabilities
        ),
        "folds": fold_results,
        "wall_seconds": time.perf_counter() - started,
        "dependency_caveat": (
            "This meta-level grouped-CV diagnostic uses cross-fitted base logits, but "
            "base fits used for other meta-training rows may have seen papers in the "
            "current meta held-out fold. It is descriptive only and did not select "
            "hyperparameters; the untouched validation evaluation is definitive."
        ),
    }


def train_evaluate_command(args: argparse.Namespace) -> int:
    lock_path = args.output_root / "selection" / "selection_lock.json"
    validation_manifest_path = (
        args.output_root / "qwen_features" / "validation_manifest.json"
    )
    if not lock_path.exists() or not validation_manifest_path.exists():
        raise RuntimeError("Missing frozen lock or validation logits")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["fusion_protocol"]["fixed_candidate"] != FIXED_CANDIDATE:
        raise RuntimeError("Frozen fusion candidate changed")
    started = time.perf_counter()
    train = source_rows("train")
    validation = source_rows("validation")
    train_arrays = load_npz(
        args.output_root / "qwen_features" / "train_oof_logits.npz"
    )
    validation_arrays = load_npz(
        args.output_root / "qwen_features" / "validation_logits.npz"
    )
    train_matrix, names = fusion_matrix(train, train_arrays)
    validation_matrix, validation_names = fusion_matrix(validation, validation_arrays)
    if names != validation_names:
        raise RuntimeError("Train/validation feature schemas differ")
    targets = phase3b.target_array(train)
    validation_targets = phase3b.target_array(validation)
    folds = training_folds(train)
    cv = fixed_candidate_cv(train_matrix, names, targets, folds)
    phase3b.atomic_json(
        args.output_root / "cross_validation" / "fixed_candidate_diagnostic.json", cv
    )
    weights, class_weights = phase3b.class_balance_weights(targets)
    fit_started = time.perf_counter()
    booster = phase3b.train_booster(
        train_matrix, targets, weights, names, FIXED_CANDIDATE, SEED
    )
    fit_seconds = time.perf_counter() - fit_started
    model_path = args.output_root / "models" / f"{VARIANT}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    prediction_started = time.perf_counter()
    probabilities = phase3b.predict_booster(booster, validation_matrix, names)
    prediction_seconds = time.perf_counter() - prediction_started
    predicted = probabilities.argmax(axis=1)
    metrics = phase3b.classification_metrics(
        validation_targets, predicted, probabilities
    )
    predictions = []
    for index, source in enumerate(validation):
        ranked = np.argsort(-probabilities[index], kind="stable")
        prediction_index = int(predicted[index])
        predictions.append(
            {
                "phase": PHASE,
                "formulation_version": FORMULATION_VERSION,
                "variant": VARIANT,
                "question_id": str(source["question_id"]),
                "document_id": str(source["document_id"]),
                "question_text": str(source["question_text"]),
                "oracle_label": int(source["oracle_label"]),
                "predicted_label": CLASS_TOKENS[prediction_index],
                "parsed_prediction": CLASS_TOKENS[prediction_index],
                "predicted_class_index": prediction_index,
                "probabilities": {
                    str(tokens): float(probabilities[index, class_index])
                    for class_index, tokens in enumerate(CLASS_TOKENS)
                },
                "ranked_predictions": [
                    CLASS_TOKENS[int(value)] for value in ranked
                ],
                "top_2_predictions": [
                    CLASS_TOKENS[int(value)] for value in ranked[:2]
                ],
                "prediction_status": "valid_clean_oof_phase3c_fusion_softprob",
            }
        )
    prediction_path = args.output_root / "validation" / "predictions.jsonl"
    phase3b.atomic_jsonl(prediction_path, predictions)
    phase3b.atomic_json(args.output_root / "classification" / "metrics.json", metrics)
    phase3b.write_confusion_csv(
        args.output_root / "classification" / "confusion_matrix.csv", metrics
    )
    histogram = args.output_root / "classification" / "predicted_vs_oracle.svg"
    phase3b.write_histogram_svg(
        histogram,
        metrics["class_distribution"],
        metrics["predicted_distribution"],
    )
    phase3b.atomic_text(
        histogram,
        histogram.read_text(encoding="utf-8").replace("Phase 3B", PHASE),
    )
    importance = phase3b.feature_importance_rows(booster, names)
    phase3b.write_importance_csv(
        args.output_root / "feature_importance" / f"{VARIANT}.csv", importance
    )
    metadata = {
        "variant": VARIANT,
        "architecture": "XGBoost multi:softprob",
        "feature_sources": ["five_paper_grouped_OOF_Qwen_logits", "tree_features"],
        "feature_count": len(names),
        "fixed_candidate": FIXED_CANDIDATE,
        "class_weights": class_weights,
        "meta_oof_diagnostic": cv,
        "validation_metrics": metrics,
        "model_path": str(model_path),
        "model_sha256": phase3b.sha256_file(model_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": phase3b.sha256_file(prediction_path),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "top_feature_importance_by_gain": importance[:30],
    }
    phase3b.atomic_json(args.output_root / "models" / f"{VARIANT}_metadata.json", metadata)
    summary = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "status": "classification_complete",
        "completed_at": utc_now(),
        "primary_variant": VARIANT,
        "selection": (
            "Original Phase 3C primary architecture and hyperparameters frozen before "
            "validation; no validation-based selection in this rerun"
        ),
        "dataset": {
            "train_examples": len(train),
            "train_documents": len({str(row['document_id']) for row in train}),
            "validation_examples": len(validation),
            "validation_documents": len({str(row['document_id']) for row in validation}),
        },
        "features": {
            "qwen_logit_count": QWEN_LOGIT_COUNT,
            "tree_feature_count": TREE_FEATURE_COUNT,
            "total_feature_count": FUSION_FEATURE_COUNT,
            "training_qwen_logits": "strict paper-grouped OOF",
            "validation_qwen_logits": "frozen all-training-data Qwen refit",
        },
        "classification": metrics,
        "model": metadata,
        "selection_lock_path": str(lock_path),
        "validation_prediction_path": str(prediction_path),
        "validation_prediction_sha256": phase3b.sha256_file(prediction_path),
        "training_and_validation_wall_seconds": time.perf_counter() - started,
    }
    phase3b.atomic_json(args.output_root / "classification_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def retrieve_command(args: argparse.Namespace) -> int:
    phase3b.PHASE = PHASE
    phase3b.FORMULATION_VERSION = FORMULATION_VERSION
    return int(phase3b.retrieve_command(args))


def reference_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    classification = value.get("classification") or value.get(
        "primary_validation_metrics"
    )
    retrieval = value.get("retrieval") or value.get("retrieval_summary")
    if path == PHASE2D_ROOT / "final_summary.json":
        retrieval = value.get("retrieval")
    mean_retrieval = None
    if retrieval:
        mean_retrieval = retrieval.get("mean_joined_retrieval_f1")
        if mean_retrieval is None:
            mean_retrieval = retrieval.get("valid_only_mean_joined_retrieval_f1")
    return {
        "classification_accuracy": classification["accuracy"],
        "classification_macro_f1": classification["macro_f1"],
        "classification_weighted_f1": classification["weighted_f1"],
        "classification_balanced_accuracy": classification["balanced_accuracy"],
        "mean_joined_retrieval_f1": mean_retrieval,
    }


def finalize_command(args: argparse.Namespace) -> int:
    classification = json.loads(
        (args.output_root / "classification_summary.json").read_text(encoding="utf-8")
    )
    retrieval = json.loads(
        (args.output_root / "retrieval" / "summary.json").read_text(encoding="utf-8")
    )
    predictions = phase3b.read_jsonl(
        args.output_root / "validation" / "predictions.jsonl"
    )
    records = phase3b.read_jsonl(args.output_root / "retrieval" / "results.jsonl")
    if len(predictions) != 924 or len(records) != 924:
        raise RuntimeError("Final Phase 3C-OOF artifacts are incomplete")
    if not retrieval.get("qdrant_collections_unchanged"):
        raise RuntimeError("Qdrant collection snapshot changed")
    metrics = classification["classification"]
    clean = {
        "classification_accuracy": metrics["accuracy"],
        "classification_macro_f1": metrics["macro_f1"],
        "classification_weighted_f1": metrics["weighted_f1"],
        "classification_balanced_accuracy": metrics["balanced_accuracy"],
        "mean_joined_retrieval_f1": retrieval["mean_joined_retrieval_f1"],
    }
    comparisons = {
        "clean_phase3c_oof": clean,
        "original_phase3c": reference_summary(
            ORIGINAL_PHASE3C_ROOT / "final_summary.json"
        ),
        "phase2d": reference_summary(PHASE2D_ROOT / "final_summary.json"),
        "phase3b": reference_summary(PHASE3B_ROOT / "final_summary.json"),
    }
    baselines = json.loads(FIXED_BASELINES_PATH.read_text(encoding="utf-8"))
    comparisons["fixed_granularity_baselines"] = {
        key: {
            "mean_joined_retrieval_f1": baselines[key]["mean_f1"],
            "median_joined_retrieval_f1": baselines[key]["median_f1"],
        }
        for key in ("fixed_10", "fixed_20", "fixed_40", "fixed_80", "fixed_160")
    }
    for key in ("original_phase3c", "phase2d", "phase3b"):
        comparisons[key]["delta_clean_minus_reference_accuracy"] = (
            clean["classification_accuracy"]
            - comparisons[key]["classification_accuracy"]
        )
        comparisons[key]["delta_clean_minus_reference_macro_f1"] = (
            clean["classification_macro_f1"]
            - comparisons[key]["classification_macro_f1"]
        )
        comparisons[key]["delta_clean_minus_reference_retrieval_f1"] = (
            clean["mean_joined_retrieval_f1"]
            - comparisons[key]["mean_joined_retrieval_f1"]
        )
    comparisons["fixed_granularity_deltas"] = {
        key: clean["mean_joined_retrieval_f1"] - value["mean_joined_retrieval_f1"]
        for key, value in comparisons["fixed_granularity_baselines"].items()
    }
    runtime = {
        "outer_qwen_folds": {
            str(fold): json.loads(
                (fold_dir(args.output_root, fold) / "complete.json").read_text(
                    encoding="utf-8"
                )
            )["training"]["training_wall_seconds"]
            for fold in range(FOLDS)
        },
        "full_qwen_refit_seconds": json.loads(
            (args.output_root / "qwen_full_refit" / "complete.json").read_text(
                encoding="utf-8"
            )
        )["training"]["training_wall_seconds"],
        "validation_qwen_extraction_seconds": json.loads(
            (
                args.output_root
                / "qwen_features"
                / "validation_manifest.json"
            ).read_text(encoding="utf-8")
        )["extraction"]["wall_seconds"],
        "meta_training_and_validation_seconds": classification[
            "training_and_validation_wall_seconds"
        ],
        "retrieval_seconds": retrieval["retrieval_wall_seconds_this_invocation"],
    }
    runtime["known_sequential_stage_seconds"] = (
        sum(runtime["outer_qwen_folds"].values())
        + runtime["full_qwen_refit_seconds"]
        + runtime["validation_qwen_extraction_seconds"]
        + runtime["meta_training_and_validation_seconds"]
        + runtime["retrieval_seconds"]
    )
    phase3b.atomic_json(args.output_root / "runtime" / "summary.json", runtime)
    phase3b.atomic_json(args.output_root / "comparison" / "baselines.json", comparisons)
    audit = {
        "status": "passed",
        "verified_at": utc_now(),
        "prediction_rows": len(predictions),
        "retrieval_rows": len(records),
        "qdrant_collections_unchanged": True,
        "training_oof_logit_rows": 2245,
        "paper_grouped_oof": True,
        "validation_used_for_training_or_selection": False,
        "fixed_candidate_matches_original_phase3c": True,
    }
    phase3b.atomic_json(args.output_root / "integrity" / "final_audit.json", audit)
    artifacts = []
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path.name != "final_summary.json":
            artifacts.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": phase3b.sha256_file(path),
                }
            )
    final = {
        "phase": PHASE,
        "formulation_version": FORMULATION_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "methodology": {
            "training_qwen_logits": "five-fold paper-grouped OOF",
            "validation_qwen_logits": "training-only full Qwen refit frozen before validation",
            "fusion": "same original Phase 3C primary five-logit-plus-173-tree-feature XGBoost",
            "fixed_candidate": FIXED_CANDIDATE,
            "validation_model_selection": False,
        },
        "classification": metrics,
        "retrieval": retrieval,
        "comparisons": comparisons,
        "runtime": runtime,
        "integrity": audit,
        "artifacts": artifacts,
    }
    phase3b.atomic_json(args.output_root / "final_summary.json", final)
    print(json.dumps(final, indent=2))
    return 0


def _git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit").set_defaults(function=audit_command)
    train_fold = subparsers.add_parser("train-fold")
    train_fold.add_argument("--fold", type=int, required=True)
    train_fold.add_argument("--extraction-batch-size", type=int, default=16)
    train_fold.set_defaults(function=train_fold_command)
    subparsers.add_parser("assemble-oof").set_defaults(function=assemble_oof_command)
    subparsers.add_parser("train-full").set_defaults(function=train_full_command)
    subparsers.add_parser("freeze").set_defaults(function=freeze_command)
    extract = subparsers.add_parser("extract-validation")
    extract.add_argument("--extraction-batch-size", type=int, default=16)
    extract.set_defaults(function=extract_validation_command)
    subparsers.add_parser("train-evaluate").set_defaults(
        function=train_evaluate_command
    )
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    retrieve.set_defaults(function=retrieve_command)
    subparsers.add_parser("finalize").set_defaults(function=finalize_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_root = args.output_root.resolve()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
