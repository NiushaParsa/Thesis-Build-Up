#!/usr/bin/env python
"""Phase 5B: tree-local routing with a frozen zero-shot Qwen context feature.

The experiment is deliberately separate from Phase 5A.  A frozen, task-
untrained Qwen/Qwen3.5-0.8B produces one of ``short``, ``medium``, or ``long``
once per question.  That result is one-hot encoded and appended to the 173
inference-safe local similarity-tree features for every tree belonging to the
question.  Phase 5A training examples, local labels, classifier settings,
paper-grouped folds, tree ranking, and retrieval evaluation remain fixed.

Run Qwen generation with the preserved local .venv-qwen or the isolated remote
.venv-phase5b-gpu environment, and tree training/evaluation with .venv-phase5a.
None of the preserved local environments is modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import similarity_tree_phase5a as phase5a


PHASE = "Phase 5B"
EXPERIMENT_NAME = "Tree-Local Similarity Router with Zero-Shot Qwen Context Feature"
FORMULATION_VERSION = "phase5b-tree-local-zero-shot-qwen-context-v1"
OUTPUT_ROOT = Path("outputs/similarity_tree_phase5b_zero_shot_qwen_context_feature")
REPORT_PATH = Path(
    "reports/similarity_tree_phase5b_zero_shot_qwen_context_feature/experiment_report.md"
)
DOC_PATH = Path("docs/SIMILARITY_TREE_PHASE5B_RESULTS.md")
ENTRYPOINT_SCRIPT = "similarity_tree_phase5b.py"
PHASE5A_ROOT = Path("outputs/similarity_tree_phase5a_local_gold_overlap_router")
PHASE3A_RAW_ROOT = Path("outputs/similarity_tree_phase3a_evidence_length_oracle/features")
LOCAL_QWEN_CACHE = Path("tmp/huggingface_qwen_cache/hub")
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TRANSFORMERS_COMMIT = "2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7"
MODEL_WEIGHT_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
LOCAL_QWEN_SNAPSHOT = (
    LOCAL_QWEN_CACHE / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / MODEL_REVISION
)
CONTEXT_LABELS = ("short", "medium", "long")
CONTEXT_PATTERN = re.compile(r"(?<![A-Za-z])(short|medium|long)(?![A-Za-z])", re.IGNORECASE)
FIXED_INSTRUCTION = (
    "You are providing an auxiliary context-length indication for a "
    "retrieval-augmented generation router. Based only on the question, "
    "decide whether the supporting context required to answer it is short, "
    "medium, or long. Choose exactly one label from: short, medium, long. "
    "Return only the label."
)
MAX_NEW_TOKENS = 8
DECISION_CONFIG = {
    "method": "deterministic free generation",
    "do_sample": False,
    "max_new_tokens": MAX_NEW_TOKENS,
}
PROCEDURE_DEVELOPMENT_NOTE = (
    "none; no prompt or hyperparameter search was performed for this procedure"
)
TEST_COMMAND = "tests/test_similarity_tree_phase5b.py"
SEED = 42
EXPECTED_TRAIN_QUESTIONS = 2101
EXPECTED_TRAIN_TREES = 4081
EXPECTED_VALIDATION_QUESTIONS = 924
EXPECTED_VALIDATION_TREES = EXPECTED_VALIDATION_QUESTIONS * phase5a.TOP_K_TREES
QWEN_EXECUTION_PROFILES = {
    False: {
        "environment_name": ".venv-qwen",
        "torch": "2.8.0+cpu",
        "device": "cpu",
    },
    True: {
        "environment_name": ".venv-phase5b-gpu",
        "torch": "2.8.0+cu128",
        "device": "cuda:0",
    },
}


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
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt(question_text: str) -> str:
    return f"{FIXED_INSTRUCTION}\n\nQuestion: {question_text}"


def parse_context_label(raw_output: str) -> tuple[str | None, str]:
    found = {match.casefold() for match in CONTEXT_PATTERN.findall(raw_output)}
    if len(found) == 1:
        return next(iter(found)), "valid"
    if not found:
        return None, "invalid_no_context_label"
    return None, "invalid_multiple_context_labels"


def qwen_execution_profile(cuda_available: bool) -> dict[str, str]:
    """Return the frozen package/device profile for the selected backend."""
    return dict(QWEN_EXECUTION_PROFILES[bool(cuda_available)])


def categorical_features(label: str) -> dict[str, float]:
    if label not in CONTEXT_LABELS:
        raise ValueError(f"Unknown Qwen context label: {label}")
    return {f"qwen_context_{candidate}": float(label == candidate) for candidate in CONTEXT_LABELS}


def combine_features(local_features: Mapping[str, Any], qwen_label: str) -> dict[str, float]:
    combined = {str(name): float(value) for name, value in local_features.items()}
    overlap = set(combined) & set(categorical_features(qwen_label))
    if overlap:
        raise RuntimeError(f"Qwen/local feature-name collision: {sorted(overlap)}")
    combined.update(categorical_features(qwen_label))
    phase5a.assert_inference_safe_feature_names(sorted(combined))
    if len(combined) != 176:
        raise RuntimeError(f"Expected 176 Phase 5B features, got {len(combined)}")
    return combined


def source_paths() -> dict[str, Path]:
    return {
        "phase5a_final_summary": PHASE5A_ROOT / "final_summary.json",
        "phase5a_train_tree_features": PHASE5A_ROOT / "features/train_gold_overlap_tree_features.jsonl.gz",
        "phase5a_validation_predictions": PHASE5A_ROOT / "validation/predictions_pre_evaluation.jsonl",
        "phase5a_retrieval_results": PHASE5A_ROOT / "retrieval/results.jsonl",
        "phase3a_validation_scores": PHASE3A_RAW_ROOT / "validation_similarity_trees.jsonl.gz",
    }


def load_train_tree_rows() -> list[dict[str, Any]]:
    rows = phase5a.read_jsonl(source_paths()["phase5a_train_tree_features"])
    if len(rows) != EXPECTED_TRAIN_TREES:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN_TREES} frozen Phase 5A train trees, got {len(rows)}")
    return rows


def question_manifest(split: str) -> list[dict[str, str]]:
    if split == "train":
        source = load_train_tree_rows()
    elif split == "validation":
        source = read_jsonl(source_paths()["phase5a_validation_predictions"])
    else:
        raise ValueError(f"Unknown split: {split}")
    questions: dict[str, dict[str, str]] = {}
    for row in source:
        question_id = str(row["question_id"])
        candidate = {
            "split": split,
            "question_id": question_id,
            "document_id": str(row["document_id"]),
            "question_text": str(row["question_text"]),
        }
        if question_id in questions and questions[question_id] != candidate:
            raise RuntimeError(f"Inconsistent question manifest row: {question_id}")
        questions[question_id] = candidate
    result = [questions[key] for key in sorted(questions)]
    expected = EXPECTED_TRAIN_QUESTIONS if split == "train" else EXPECTED_VALIDATION_QUESTIONS
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} {split} questions, got {len(result)}")
    return result


def qwen_output_path(output_root: Path, split: str) -> Path:
    return output_root / "qwen_features" / f"{split}_outputs.jsonl"


def validate_qwen_outputs(
    rows: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]], split: str
) -> dict[str, Any]:
    expected = {str(row["question_id"]): row for row in manifest}
    if len(rows) != len({str(row["question_id"]) for row in rows}):
        raise RuntimeError(f"Duplicate resumable Qwen outputs in {split}")
    observed = {str(row["question_id"]): row for row in rows}
    extra = set(observed) - set(expected)
    if extra:
        raise RuntimeError(f"Unexpected Qwen question IDs in {split}: {len(extra)}")
    for question_id, row in observed.items():
        if str(row["question_text"]) != str(expected[question_id]["question_text"]):
            raise RuntimeError(f"Question text changed for resumable output {question_id}")
        if str(row["model_id"]) != MODEL_ID or str(row["model_revision"]) != MODEL_REVISION:
            raise RuntimeError(f"Qwen identity changed for resumable output {question_id}")
        if str(row["prompt_sha256"]) != stable_hash(build_prompt(str(row["question_text"]))):
            raise RuntimeError(f"Prompt changed for resumable output {question_id}")
        parsed, status = parse_context_label(str(row["raw_qwen_output"]))
        if parsed != row.get("parsed_context_label") or status != row.get("prediction_status"):
            raise RuntimeError(f"Saved parser result is not reproducible: {question_id}")
    invalid = [row for row in rows if row.get("parsed_context_label") not in CONTEXT_LABELS]
    return {
        "split": split,
        "expected_questions": len(manifest),
        "completed_questions": len(rows),
        "remaining_questions": len(manifest) - len(rows),
        "valid_outputs": len(rows) - len(invalid),
        "invalid_outputs": len(invalid),
        "distribution": {
            label: sum(row.get("parsed_context_label") == label for row in rows)
            for label in CONTEXT_LABELS
        },
    }


def qwen_environment() -> dict[str, Any]:
    import psutil
    import torch
    import transformers

    distribution = importlib.metadata.distribution("transformers")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    installed_commit = str(direct_url.get("vcs_info", {}).get("commit_id", ""))
    cuda_available = torch.cuda.is_available()
    profile = qwen_execution_profile(cuda_available)
    environment_name = Path(sys.prefix).name
    environment = {
        "captured_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "environment_name": environment_name,
        "expected_environment": profile["environment_name"],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_source": direct_url.get("url"),
        "transformers_commit": installed_commit,
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_runtime": torch.version.cuda,
        "execution_device": profile["device"],
        "execution_provider": "Vast.ai" if os.environ.get("CONTAINER_ID") else "local",
        "vast_container_detected": bool(os.environ.get("CONTAINER_ID")),
    }
    if cuda_available:
        environment.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_compute_capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    expected = {
        "python": (3, 10, 7),
        "environment_name": profile["environment_name"],
        "torch": profile["torch"],
        "transformers": "5.15.0.dev0",
        "transformers_commit": TRANSFORMERS_COMMIT,
        "cuda_available": cuda_available,
    }
    observed = {
        "python": tuple(sys.version_info[:3]),
        "environment_name": environment_name,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_commit": installed_commit,
        "cuda_available": cuda_available,
    }
    if observed != expected:
        raise RuntimeError(f"Frozen Qwen environment mismatch: expected={expected}, observed={observed}")
    environment["exact_environment_check"] = "passed"
    return environment


def load_frozen_qwen() -> tuple[Any, Any, dict[str, Any]]:
    import psutil
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    process = psutil.Process()
    environment = qwen_environment()
    device = torch.device(environment["execution_device"])
    model_file = LOCAL_QWEN_SNAPSHOT / "model.safetensors-00001-of-00001.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"Frozen local Qwen weights are missing: {model_file}")
    observed_weight_hash = sha256_file(model_file)
    if observed_weight_hash != MODEL_WEIGHT_SHA256:
        raise RuntimeError(
            f"Frozen Qwen weight hash mismatch: expected={MODEL_WEIGHT_SHA256}, "
            f"observed={observed_weight_hash}"
        )
    processor_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(LOCAL_QWEN_CACHE),
        local_files_only=True,
    )
    processor_seconds = time.perf_counter() - processor_started
    model_started = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        cache_dir=str(LOCAL_QWEN_CACHE),
        local_files_only=True,
    )
    model.to(device)
    model_seconds = time.perf_counter() - model_started
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A frozen Qwen parameter unexpectedly requires gradients")
    first_parameter = next(model.parameters())
    model_info = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "processor_load_seconds": processor_seconds,
        "model_load_seconds": model_seconds,
        "dtype": str(first_parameter.dtype),
        "device": str(first_parameter.device),
        "quantization": None,
        "all_parameters_frozen": True,
        "parameter_updates": 0,
        "optimizer_created": False,
        "backward_passes": 0,
        "rss_gib_after_load": process.memory_info().rss / 2**30,
        "local_snapshot": str(LOCAL_QWEN_SNAPSHOT),
        "model_weight_sha256": observed_weight_hash,
        "environment": environment,
    }
    if device.type == "cuda":
        model_info.update(
            {
                "cuda_allocated_gib_after_load": torch.cuda.memory_allocated(0) / 2**30,
                "cuda_reserved_gib_after_load": torch.cuda.memory_reserved(0) / 2**30,
            }
        )
    return processor, model, model_info


def predict_context(processor: Any, model: Any, question_text: str) -> tuple[str, str | None, str, float]:
    import torch

    prompt = build_prompt(question_text)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    elapsed = time.perf_counter() - started
    generated_tokens = output[0][inputs["input_ids"].shape[-1] :].detach().cpu()
    raw = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    parsed, status = parse_context_label(raw)
    return raw, parsed, status, elapsed


def run_qwen_inference(output_root: Path, splits: Sequence[str]) -> dict[str, Any]:
    import psutil

    manifests = {split: question_manifest(split) for split in splits}
    for split, rows in manifests.items():
        atomic_jsonl(output_root / "manifests" / f"{split}_questions.jsonl", rows)
    existing = {
        split: read_jsonl(qwen_output_path(output_root, split)) for split in splits
    }
    for split in splits:
        validate_qwen_outputs(existing[split], manifests[split], split)
    pending = {
        split: [
            row
            for row in manifests[split]
            if str(row["question_id"])
            not in {str(item["question_id"]) for item in existing[split]}
        ]
        for split in splits
    }
    total_pending = sum(len(rows) for rows in pending.values())
    if total_pending:
        processor, model, model_info = load_frozen_qwen()
        atomic_json(output_root / "qwen_features" / "model_info.json", model_info)
    else:
        model_info = json.loads((output_root / "qwen_features" / "model_info.json").read_text(encoding="utf-8"))
        processor = model = None
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    invocation_started = time.perf_counter()
    completed_this_invocation = 0
    for split in splits:
        path = qwen_output_path(output_root, split)
        for manifest_row in pending[split]:
            raw, parsed, status, elapsed = predict_context(
                processor, model, str(manifest_row["question_text"])
            )
            result = {
                **manifest_row,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "transformers_commit": TRANSFORMERS_COMMIT,
                "prompt_sha256": stable_hash(build_prompt(str(manifest_row["question_text"]))),
                "raw_qwen_output": raw,
                "parsed_context_label": parsed,
                "prediction_status": status,
                "inference_seconds": elapsed,
                "completed_at": utc_now(),
            }
            append_jsonl(path, result)
            existing[split].append(result)
            completed_this_invocation += 1
            peak_rss = max(peak_rss, process.memory_info().rss)
            if completed_this_invocation % 10 == 0 or completed_this_invocation == total_pending:
                print(
                    json.dumps(
                        {
                            "event": "phase5b_qwen_progress",
                            "split": split,
                            "completed_this_invocation": completed_this_invocation,
                            "pending_this_invocation": total_pending,
                            "split_completed": len(existing[split]),
                            "split_expected": len(manifests[split]),
                            "last_seconds": elapsed,
                        }
                    ),
                    flush=True,
                )
            atomic_json(
                output_root / "qwen_features" / "checkpoint.json",
                {
                    "updated_at": utc_now(),
                    "completed_this_invocation": completed_this_invocation,
                    "pending_this_invocation": total_pending,
                    "split_counts": {name: len(rows) for name, rows in existing.items()},
                    "last_question_id": result["question_id"],
                },
            )
    summaries = {}
    all_timings = []
    invalid_records = []
    for split in splits:
        ordered_by_id = {str(row["question_id"]): row for row in existing[split]}
        ordered = [ordered_by_id[str(row["question_id"])] for row in manifests[split]]
        atomic_jsonl(qwen_output_path(output_root, split), ordered)
        invalid_records.extend(
            {**row, "split": split}
            for row in ordered
            if row.get("prediction_status") != "valid"
        )
        summaries[split] = validate_qwen_outputs(ordered, manifests[split], split)
        summaries[split]["output_path"] = str(qwen_output_path(output_root, split))
        summaries[split]["output_sha256"] = sha256_file(qwen_output_path(output_root, split))
        all_timings.extend(float(row["inference_seconds"]) for row in ordered)
    invalid_path = output_root / "qwen_features" / "invalid_outputs.jsonl"
    atomic_jsonl(invalid_path, invalid_records)
    invalid_total = sum(summary["invalid_outputs"] for summary in summaries.values())
    summary = {
        "status": "complete" if not invalid_total else "complete_with_invalid_outputs",
        "completed_at": utc_now(),
        "model": model_info,
        "prompt": {
            "instruction": FIXED_INSTRUCTION,
            "question_format": "{instruction}\\n\\nQuestion: {original_question_text}",
            "prompt_template_sha256": stable_hash(
                {"instruction": FIXED_INSTRUCTION, "question_format": "{instruction}\\n\\nQuestion: {original_question_text}"}
            ),
            "allowed_labels": list(CONTEXT_LABELS),
        },
        "decoding": dict(DECISION_CONFIG),
        "parser": "exactly one distinct standalone short/medium/long label, case-insensitive",
        "invalid_policy": "no default category; stop before classifier training if any output is invalid",
        "splits": summaries,
        "total_questions": sum(len(rows) for rows in manifests.values()),
        "valid_outputs": sum(summary["valid_outputs"] for summary in summaries.values()),
        "invalid_outputs": invalid_total,
        "invalid_output_records": {
            "path": str(invalid_path),
            "sha256": sha256_file(invalid_path),
        },
        "mean_inference_seconds": statistics.fmean(all_timings) if all_timings else 0.0,
        "median_inference_seconds": statistics.median(all_timings) if all_timings else 0.0,
        "total_generation_seconds": sum(all_timings),
        "peak_rss_gib_this_invocation": peak_rss / 2**30,
        "wall_seconds_this_invocation": time.perf_counter() - invocation_started,
    }
    atomic_json(output_root / "qwen_features" / "summary.json", summary)
    if invalid_total:
        raise RuntimeError(
            f"Phase 5B has {invalid_total} invalid Qwen outputs; no default category was assigned"
        )
    return summary


def run_qwen_smoke(output_root: Path, count: int) -> dict[str, Any]:
    import psutil

    if count < 1:
        raise ValueError("Smoke count must be positive")
    manifest = question_manifest("train")[:count]
    processor, model, model_info = load_frozen_qwen()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    rows = []
    for item in manifest:
        raw, parsed, status, elapsed = predict_context(
            processor, model, str(item["question_text"])
        )
        rows.append(
            {
                **item,
                "raw_qwen_output": raw,
                "parsed_context_label": parsed,
                "prediction_status": status,
                "inference_seconds": elapsed,
            }
        )
        peak_rss = max(peak_rss, process.memory_info().rss)
    summary = {
        "status": "passed" if all(row["prediction_status"] == "valid" for row in rows) else "failed",
        "examples": len(rows),
        "valid": sum(row["prediction_status"] == "valid" for row in rows),
        "distribution": dict(Counter(row["parsed_context_label"] for row in rows)),
        "mean_inference_seconds": statistics.fmean(row["inference_seconds"] for row in rows),
        "peak_rss_gib": peak_rss / 2**30,
        "model": model_info,
        "rows": rows,
    }
    atomic_json(output_root / "smoke" / "summary.json", summary)
    if summary["status"] != "passed":
        raise RuntimeError("Phase 5B Qwen smoke test produced an invalid output")
    return summary


def qwen_lookup(output_root: Path, split: str) -> dict[str, dict[str, Any]]:
    manifest = question_manifest(split)
    rows = read_jsonl(qwen_output_path(output_root, split))
    summary = validate_qwen_outputs(rows, manifest, split)
    if summary["completed_questions"] != summary["expected_questions"] or summary["invalid_outputs"]:
        raise RuntimeError(f"Complete valid {split} Qwen features are required")
    return {str(row["question_id"]): row for row in rows}


def combined_train_rows(output_root: Path) -> list[dict[str, Any]]:
    qwen = qwen_lookup(output_root, "train")
    rows = []
    for source in load_train_tree_rows():
        question_id = str(source["question_id"])
        label = str(qwen[question_id]["parsed_context_label"])
        row = dict(source)
        row["features"] = combine_features(source["features"], label)
        row["qwen_context_label"] = label
        rows.append(row)
    if len(rows) != EXPECTED_TRAIN_TREES:
        raise RuntimeError("Combined Phase 5B training-tree count changed")
    return rows


def build_validation_predictions(
    output_root: Path, booster: Any, feature_names: Sequence[str]
) -> list[dict[str, Any]]:
    qwen = qwen_lookup(output_root, "validation")
    raw_rows = phase5a.read_jsonl(source_paths()["phase3a_validation_scores"])
    predictions = []
    for raw in raw_rows:
        question_id = str(raw["question_id"])
        qwen_row = qwen[question_id]
        qwen_label = str(qwen_row["parsed_context_label"])
        scores = {int(tokens): values for tokens, values in raw["scores_by_tokens"].items()}
        ranked = phase5a.rank_roots(scores, phase5a.TOP_K_TREES)
        classifier_rows = [
            {
                "features": combine_features(
                    phase5a.extract_local_features(item["local_scores"]), qwen_label
                ),
                "question_id": question_id,
                "document_id": str(raw["document_id"]),
            }
            for item in ranked
        ]
        indices, probabilities = phase5a.predict_labels(
            booster, classifier_rows, feature_names
        )
        selected_trees = []
        for rank, (item, class_index, probability) in enumerate(
            zip(ranked, indices, probabilities), start=1
        ):
            root_index = int(item["root_index"])
            tokens = phase5a.CLASS_TOKENS[int(class_index)]
            selected_trees.append(
                {
                    "tree_rank": rank,
                    "root_index": root_index,
                    "tree_score": float(item["tree_score"]),
                    "predicted_granularity": tokens,
                    "class_probabilities": {
                        str(candidate): float(probability[index])
                        for index, candidate in enumerate(phase5a.CLASS_TOKENS)
                    },
                    "selected_chunk": phase5a.choose_single_chunk(
                        item["local_scores"], root_index, tokens
                    ),
                }
            )
        predictions.append(
            {
                "phase": PHASE,
                "formulation_version": FORMULATION_VERSION,
                "split": "validation",
                "question_id": question_id,
                "document_id": str(raw["document_id"]),
                "question_text": str(raw["question_text"]),
                "qwen_context_label": qwen_label,
                "raw_qwen_output": str(qwen_row["raw_qwen_output"]),
                "qwen_prediction_status": str(qwen_row["prediction_status"]),
                "tree_score_definition": "mean of level means at 10,20,40,80,160",
                "top_n_trees": phase5a.TOP_K_TREES,
                "selected_trees": selected_trees,
                "gold_fields_used": False,
            }
        )
    if len(predictions) != EXPECTED_VALIDATION_QUESTIONS:
        raise RuntimeError("Expected 924 Phase 5B validation prediction rows")
    return predictions


def fetch_selected_chunks(
    client: Any, predictions: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    identities = sorted(
        {
            phase5a.chunk_id(
                str(row["document_id"]),
                int(tree["selected_chunk"]["tokens"]),
                int(tree["selected_chunk"]["global_chunk_index"]),
            )
            for row in predictions
            for tree in row["selected_trees"]
        }
    )
    found: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(identities), 256):
        points = client.retrieve(
            collection_name=phase5a.PAPER_CHUNK_COLLECTION,
            ids=identities[offset : offset + 256],
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            found[str(point.id)] = dict(point.payload or {})
    if set(identities) != set(found):
        raise RuntimeError(f"Missing {len(set(identities) - set(found))} selected chunk payloads")
    return found


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    chunks: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from metrics import count_tokens, token_precision_recall_f1

    results = []
    precisions, recalls, f1_values, token_counts = [], [], [], []
    for prediction in predictions:
        question_id = str(prediction["question_id"])
        document_id = str(prediction["document_id"])
        selected = []
        for tree in prediction["selected_trees"]:
            choice = tree["selected_chunk"]
            identity = phase5a.chunk_id(
                document_id, int(choice["tokens"]), int(choice["global_chunk_index"])
            )
            payload = chunks[identity]
            selected.append(
                {
                    "chunk_id": identity,
                    "tree_rank": int(tree["tree_rank"]),
                    "root_index": int(tree["root_index"]),
                    "granularity_tokens": int(choice["tokens"]),
                    "chunk_idx": int(payload["chunk_idx"]),
                    "span_start": int(payload["span_start"]),
                    "span_end": int(payload["span_end"]),
                    "query_similarity": float(choice["similarity"]),
                    "content": str(payload["content"]),
                }
            )
        retrieved_text = "\n".join(item["content"] for item in selected)
        evidence_text = phase5a.deduplicated_evidence_text(evidence[question_id])
        precision, recall, f1 = token_precision_recall_f1(retrieved_text, evidence_text)
        retrieved_tokens = count_tokens(retrieved_text)
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        token_counts.append(retrieved_tokens)
        results.append(
            {
                "phase": PHASE,
                "question_id": question_id,
                "document_id": document_id,
                "qwen_context_label": prediction["qwen_context_label"],
                "precision_joined_top5_trees": precision,
                "recall_joined_top5_trees": recall,
                "f1_joined_top5_trees": f1,
                "retrieved_token_count": retrieved_tokens,
                "selected_chunks": selected,
                "paper_restricted": True,
                "top_n_trees": phase5a.TOP_K_TREES,
                "one_chunk_per_tree": True,
            }
        )
    return results, {
        "evaluated_questions": len(results),
        "retrieval_coverage": len(results) / EXPECTED_VALIDATION_QUESTIONS,
        "mean_joined_precision": statistics.fmean(precisions),
        "mean_joined_recall": statistics.fmean(recalls),
        "mean_joined_f1": statistics.fmean(f1_values),
        "median_joined_f1": statistics.median(f1_values),
        "mean_retrieved_token_count": statistics.fmean(token_counts),
        "top_n_trees": phase5a.TOP_K_TREES,
        "selection_per_tree": "single most similar chunk at predicted level",
        "paper_restricted": True,
    }


def paired_paper_cluster_bootstrap(
    phase5b_rows: Sequence[Mapping[str, Any]],
    baseline_by_question: Mapping[str, float],
    baseline_name: str,
    replicates: int = 10000,
) -> dict[str, Any]:
    by_paper: dict[str, list[float]] = defaultdict(list)
    for row in phase5b_rows:
        question_id = str(row["question_id"])
        by_paper[str(row["document_id"])].append(
            float(row["f1_joined_top5_trees"]) - float(baseline_by_question[question_id])
        )
    papers = sorted(by_paper)
    rng = np.random.default_rng(SEED)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(papers), size=len(papers))
        values = [value for index in sampled for value in by_paper[papers[int(index)]]]
        draws[replicate] = float(np.mean(values))
    observed = statistics.fmean(value for values in by_paper.values() for value in values)
    return {
        "comparison": f"phase5b_minus_{baseline_name}",
        "observed_mean_difference": observed,
        "confidence_interval_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "resampling_unit": "source paper",
        "papers": len(papers),
        "replicates": replicates,
        "seed": SEED,
    }


def phase5a_baselines() -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    rows = read_jsonl(source_paths()["phase5a_retrieval_results"])
    if len(rows) != EXPECTED_VALIDATION_QUESTIONS:
        raise RuntimeError("Phase 5A retrieval baseline is incomplete")
    phase5a_values = {
        str(row["question_id"]): float(row["methods"]["phase5a"]["f1_joined_top5_trees"])
        for row in rows
    }
    fixed40_values = {
        str(row["question_id"]): float(row["methods"]["same_tree_fixed_40"]["f1_joined_top5_trees"])
        for row in rows
    }
    return {"phase5a": phase5a_values, "same_tree_fixed_40": fixed40_values}, {
        "phase5a_mean_joined_f1": statistics.fmean(phase5a_values.values()),
        "same_tree_fixed_40_mean_joined_f1": statistics.fmean(fixed40_values.values()),
        "source_path": str(source_paths()["phase5a_retrieval_results"]),
        "source_sha256": sha256_file(source_paths()["phase5a_retrieval_results"]),
    }


def tree_environment() -> dict[str, Any]:
    import scipy
    import xgboost

    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "expected_environment": ".venv-phase5a",
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "xgboost": xgboost.__version__,
        "device": "CPU",
    }


def preflight(output_root: Path, *, require_qdrant: bool) -> dict[str, Any]:
    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Phase 5B inputs: {missing}")
    phase5a_summary = json.loads(paths["phase5a_final_summary"].read_text(encoding="utf-8"))
    if phase5a_summary.get("status") != "complete":
        raise RuntimeError("Phase 5A source experiment is not complete")
    model_file = LOCAL_QWEN_SNAPSHOT / "model.safetensors-00001-of-00001.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"Frozen local Qwen weights are missing: {model_file}")
    observed_model_weight_hash = sha256_file(model_file)
    if observed_model_weight_hash != MODEL_WEIGHT_SHA256:
        raise RuntimeError("Frozen local Qwen weights changed before Phase 5B")
    train_manifest = question_manifest("train")
    validation_manifest = question_manifest("validation")
    train_papers = {row["document_id"] for row in train_manifest}
    validation_papers = {row["document_id"] for row in validation_manifest}
    if train_papers & validation_papers:
        raise RuntimeError("Phase 5B train/validation papers overlap")
    qdrant_snapshot = None
    if require_qdrant:
        client = phase5a.qdrant_client()
        try:
            qdrant_snapshot = phase5a.collection_snapshot(client)
        finally:
            client.close()
    audit = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT_NAME,
        "status": "passed",
        "captured_at": utc_now(),
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "train_questions": len(train_manifest),
        "train_tree_examples": EXPECTED_TRAIN_TREES,
        "validation_questions": len(validation_manifest),
        "train_papers": len(train_papers),
        "validation_papers": len(validation_papers),
        "paper_overlap": [],
        "feature_counts": {"local_similarity": 173, "qwen_categorical": 3, "combined": 176},
        "qdrant_required_for_this_stage": require_qdrant,
        "qdrant_snapshot": qdrant_snapshot,
        "previous_experiments_are_read_only": True,
        "qwen_local_snapshot": str(LOCAL_QWEN_SNAPSHOT),
        "qwen_model_weight_sha256_expected": MODEL_WEIGHT_SHA256,
        "qwen_model_weight_sha256_observed": observed_model_weight_hash,
    }
    atomic_json(output_root / "integrity" / "preflight_audit.json", audit)
    return audit


def write_procedure_lock(output_root: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    lock = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT_NAME,
        "formulation_version": FORMULATION_VERSION,
        "frozen_at": utc_now(),
        "qwen": {
            "role": "question-level auxiliary context indication only",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_commit": TRANSFORMERS_COMMIT,
            "local_snapshot": str(LOCAL_QWEN_SNAPSHOT),
            "model_weight_sha256": MODEL_WEIGHT_SHA256,
            "task_fine_tuned": False,
            "prompt": FIXED_INSTRUCTION,
            "labels": list(CONTEXT_LABELS),
            "decoding": dict(DECISION_CONFIG),
            "parser": "one distinct standalone allowed label",
            "invalid_policy": "no default; abort training/evaluation",
            "oof_required": False,
            "oof_reason": "the frozen zero-shot Qwen model is never fitted on thesis train questions",
        },
        "features": {
            "phase5a_local_similarity_features": 173,
            "qwen_one_hot_features": 3,
            "total": 176,
            "qwen_value_repeated_for_each_tree_of_same_question": True,
            "gold_evidence_as_input": False,
        },
        "fixed_from_phase5a": {
            "training_examples_and_local_labels": True,
            "xgboost_hyperparameters": phase5a.FIXED_CANDIDATE,
            "class_weighting": "sqrt inverse-frequency weights",
            "cross_validation": "5-fold paper-grouped diagnostic only",
            "tree_score": "average of five within-tree level means",
            "top_n_trees": 5,
            "selected_chunks": "one most-similar chunk at predicted level per tree",
            "joined_metric": "GPT-2 token-level precision/recall/F1",
        },
        "model_selection": PROCEDURE_DEVELOPMENT_NOTE,
        "validation": {
            "used_for_training_or_selection": False,
            "predictions_saved_and_hashed_before_local_gold_or_retrieval_evaluation": True,
            "status": "development result because the split was reused in earlier phases",
        },
        "source_hashes": audit["source_hashes"],
        "vast_ai_required": False,
    }
    lock["procedure_sha256"] = stable_hash(lock)
    atomic_json(output_root / "configuration" / "procedure_lock.json", lock)
    return lock


def run_tree_experiment(output_root: Path) -> dict[str, Any]:
    import xgboost as xgb

    wall_started = time.perf_counter()
    audit = preflight(output_root, require_qdrant=True)
    lock_path = output_root / "configuration" / "procedure_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected_source_hashes = lock["source_hashes"]
        if expected_source_hashes != audit["source_hashes"]:
            raise RuntimeError("Frozen Phase 5B source hashes changed")
    else:
        lock = write_procedure_lock(output_root, audit)
    qwen_summary = json.loads((output_root / "qwen_features" / "summary.json").read_text(encoding="utf-8"))
    if qwen_summary["invalid_outputs"] or qwen_summary["valid_outputs"] != 3025:
        raise RuntimeError("Phase 5B requires 3,025 valid and zero invalid Qwen outputs")

    train_rows = combined_train_rows(output_root)
    combined_path = output_root / "features" / "train_combined_tree_features.jsonl.gz"
    phase5a.atomic_jsonl(combined_path, train_rows, gzip_output=True)
    booster, feature_names, model_metadata = phase5a.train_final(train_rows)
    model_path = output_root / "models" / "tree_local_qwen_xgboost.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    model_metadata.update(
        {
            "phase": PHASE,
            "experiment_name": EXPERIMENT_NAME,
            "feature_count": 176,
            "qwen_feature_names": [f"qwen_context_{label}" for label in CONTEXT_LABELS],
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
        }
    )
    atomic_json(output_root / "models" / "metadata.json", model_metadata)
    atomic_json(
        output_root / "cross_validation" / "paper_grouped_fixed_procedure.json",
        model_metadata["cross_validation"],
    )

    prediction_started = time.perf_counter()
    predictions = build_validation_predictions(output_root, booster, feature_names)
    prediction_seconds = time.perf_counter() - prediction_started
    prediction_path = output_root / "validation" / "predictions_pre_evaluation.jsonl"
    atomic_jsonl(prediction_path, predictions)
    prediction_distribution = {
        str(tokens): sum(
            int(tree["predicted_granularity"]) == tokens
            for row in predictions
            for tree in row["selected_trees"]
        )
        for tokens in phase5a.CLASS_TOKENS
    }
    prediction_lock = {
        "locked_at": utc_now(),
        "procedure_sha256": lock["procedure_sha256"],
        "predictions_path": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "question_predictions": len(predictions),
        "tree_predictions": sum(len(row["selected_trees"]) for row in predictions),
        "prediction_distribution": prediction_distribution,
        "validation_phase5b_local_gold_or_retrieval_evidence_requested_before_lock": False,
        "unused_legacy_question_oracle_in_raw_score_file": True,
    }
    atomic_json(output_root / "validation" / "prediction_lock.json", prediction_lock)

    client = phase5a.qdrant_client()
    try:
        evidence = phase5a.fetch_evidence_for_questions(
            client, [str(row["question_id"]) for row in predictions]
        )
        chunks = fetch_selected_chunks(client, predictions)
        raw_validation = phase5a.read_jsonl(source_paths()["phase3a_validation_scores"])
        roots = phase5a.fetch_root_payloads(client, raw_validation)
    finally:
        client.close()
    retrieval_started = time.perf_counter()
    retrieval_rows, retrieval_summary = evaluate_predictions(predictions, chunks, evidence)
    retrieval_seconds = time.perf_counter() - retrieval_started
    retrieval_path = output_root / "retrieval" / "results.jsonl"
    atomic_jsonl(retrieval_path, retrieval_rows)
    retrieval_summary.update(
        {
            "scoring_seconds": retrieval_seconds,
            "result_path": str(retrieval_path),
            "result_sha256": sha256_file(retrieval_path),
        }
    )
    baselines, baseline_summary = phase5a_baselines()
    comparisons = {
        "baseline_summary": baseline_summary,
        "versus_phase5a": paired_paper_cluster_bootstrap(
            retrieval_rows, baselines["phase5a"], "phase5a"
        ),
        "versus_same_tree_fixed_40": paired_paper_cluster_bootstrap(
            retrieval_rows, baselines["same_tree_fixed_40"], "same_tree_fixed_40"
        ),
    }
    atomic_json(output_root / "comparison" / "phase5a_and_fixed40.json", comparisons)
    atomic_json(output_root / "retrieval" / "summary.json", retrieval_summary)

    validation_gold_rows, validation_construction, validation_exclusions = phase5a.build_gold_overlap_rows(
        raw_validation, roots, evidence, "validation"
    )
    validation_qwen = qwen_lookup(output_root, "validation")
    combined_validation_rows = []
    for row in validation_gold_rows:
        combined = dict(row)
        label = str(validation_qwen[str(row["question_id"])]["parsed_context_label"])
        combined["features"] = combine_features(row["features"], label)
        combined["qwen_context_label"] = label
        combined_validation_rows.append(combined)
    validation_matrix, validation_names = phase5a.feature_matrix(combined_validation_rows)
    if validation_names != list(feature_names):
        raise RuntimeError("Phase 5B train/validation feature schema differs")
    probabilities = phase5a.phase3b.predict_booster(booster, validation_matrix, feature_names)
    targets = phase5a.target_array(combined_validation_rows)
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    classification = phase5a.phase3b.classification_metrics(targets, predicted, probabilities)
    classification.update(
        {
            "unit": "gold-overlap validation tree (secondary diagnostic)",
            "eligible_questions": validation_construction["eligible_questions"],
            "excluded_questions": validation_construction["excluded_questions"],
        }
    )
    atomic_json(output_root / "classification" / "metrics.json", classification)
    phase5a.write_confusion_csv(output_root / "classification" / "confusion_matrix.csv", classification)
    atomic_json(output_root / "integrity" / "validation_exclusions.json", validation_exclusions)

    client = phase5a.qdrant_client()
    try:
        final_snapshot = phase5a.collection_snapshot(client)
    finally:
        client.close()
    if audit["qdrant_snapshot"] != final_snapshot:
        raise RuntimeError("Qdrant collections changed during Phase 5B")
    current_hashes = {name: sha256_file(path) for name, path in source_paths().items()}
    if current_hashes != audit["source_hashes"]:
        raise RuntimeError("A source or previous-experiment artifact changed during Phase 5B")
    tree_env = tree_environment()
    atomic_json(output_root / "environment" / "tree_environment.json", tree_env)
    tree_packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    atomic_text(output_root / "environment" / "tree_package_lock.txt", tree_packages)
    qwen_environment = qwen_summary["model"]["environment"]
    qwen_execution_provider = qwen_environment.get("execution_provider")
    if qwen_execution_provider is None:
        # GPU artifacts created before provider capture was added still record
        # the dedicated remote execution profile unambiguously.
        qwen_execution_provider = (
            "Vast.ai"
            if qwen_environment.get("environment_name") == ".venv-phase5b-gpu"
            else "local"
        )
    runtime = {
        "tree_stage_wall_seconds": time.perf_counter() - wall_started,
        "validation_prediction_seconds": prediction_seconds,
        "retrieval_scoring_seconds": retrieval_seconds,
        "qwen_mean_inference_seconds": qwen_summary["mean_inference_seconds"],
        "qwen_median_inference_seconds": qwen_summary["median_inference_seconds"],
        "qwen_recorded_wall_seconds_this_invocation": qwen_summary["wall_seconds_this_invocation"],
        "qwen_execution_provider": qwen_execution_provider,
        "vast_ai_used": qwen_execution_provider == "Vast.ai",
        "gpu_used": bool(qwen_environment["cuda_available"]),
        "qwen_execution_device": qwen_environment["execution_device"],
    }
    atomic_json(output_root / "runtime" / "summary.json", runtime)
    final_summary = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT_NAME,
        "formulation_version": FORMULATION_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "qwen": qwen_summary,
        "training": {
            "questions": EXPECTED_TRAIN_QUESTIONS,
            "tree_examples": EXPECTED_TRAIN_TREES,
            "local_label_distribution": json.loads(
                (PHASE5A_ROOT / "features/train_construction_summary.json").read_text(encoding="utf-8")
            )["label_distribution"],
            "feature_count": 176,
            "cross_validation": model_metadata["cross_validation"]["oof_metrics"],
        },
        "validation": {
            "questions": EXPECTED_VALIDATION_QUESTIONS,
            "tree_predictions": EXPECTED_VALIDATION_TREES,
            "qwen_context_distribution": qwen_summary["splits"]["validation"]["distribution"],
            "granularity_prediction_distribution": prediction_distribution,
            "tree_label_classification_diagnostic": classification,
        },
        "retrieval": retrieval_summary,
        "comparisons": comparisons,
        "methodology": {
            "zero_shot_qwen_not_task_fine_tuned": True,
            "qwen_oof_not_required": True,
            "phase5a_examples_labels_and_procedure_fixed": True,
            "validation_used_for_training_or_selection": False,
            "predictions_locked_before_gold_retrieval_evaluation": True,
            "qdrant_read_only_and_unchanged": True,
            "previous_experiments_unchanged": True,
            "validation_status": "development result",
        },
        "runtime": runtime,
        "artifacts": {
            "output_root": str(output_root),
            "report": str(REPORT_PATH),
            "results_document": str(DOC_PATH),
        },
    }
    atomic_json(output_root / "final_summary.json", final_summary)
    write_documentation(final_summary)
    final_audit = {
        "status": "passed",
        "completed_at": utc_now(),
        "source_hashes_unchanged": True,
        "qdrant_collections_unchanged": True,
        "prediction_lock_sha256": sha256_file(output_root / "validation/prediction_lock.json"),
        "final_summary_sha256": sha256_file(output_root / "final_summary.json"),
        "final_qdrant_snapshot": final_snapshot,
    }
    atomic_json(output_root / "integrity" / "final_audit.json", final_audit)
    return final_summary


def write_documentation(summary: Mapping[str, Any]) -> None:
    qwen = summary["qwen"]
    validation = summary["validation"]
    retrieval = summary["retrieval"]
    comparisons = summary["comparisons"]
    classification = validation["tree_label_classification_diagnostic"]
    qwen_train = qwen["splits"]["train"]["distribution"]
    qwen_validation = qwen["splits"]["validation"]["distribution"]
    granularity = validation["granularity_prediction_distribution"]
    phase5a_mean = comparisons["baseline_summary"]["phase5a_mean_joined_f1"]
    fixed40_mean = comparisons["baseline_summary"]["same_tree_fixed_40_mean_joined_f1"]
    versus5a = comparisons["versus_phase5a"]
    versus40 = comparisons["versus_same_tree_fixed_40"]
    content = f"""# {PHASE} — {EXPERIMENT_NAME}

## Method

Frozen zero-shot `{MODEL_ID}` produces one question-level context indication:
`short`, `medium`, or `long`. The result is one-hot encoded into three features
and appended to each tree's 173 Phase 5A similarity features. The same Qwen
value is repeated across a question's trees; it is auxiliary information, not
a granularity decision. Qwen receives only the fixed instruction and original
question and is never fitted on the thesis data, so OOF Qwen inference is not
needed.

The 4,081 Phase 5A gold-overlap training trees and their local labels are
unchanged. XGBoost settings, class weights, paper-grouped diagnostic folds,
TreeScore, top-five tree selection, one-chunk-per-tree rule, and joined metric
are also unchanged. {PROCEDURE_DEVELOPMENT_NOTE}.

## Qwen feature distributions

| Split | Short | Medium | Long | Invalid |
|---|---:|---:|---:|---:|
| Train ({EXPECTED_TRAIN_QUESTIONS}) | {qwen_train['short']} | {qwen_train['medium']} | {qwen_train['long']} | {qwen['splits']['train']['invalid_outputs']} |
| Validation ({EXPECTED_VALIDATION_QUESTIONS}) | {qwen_validation['short']} | {qwen_validation['medium']} | {qwen_validation['long']} | {qwen['splits']['validation']['invalid_outputs']} |

## Results

| Method | Mean joined F1 |
|---|---:|
| {PHASE} Qwen + local-tree features | {retrieval['mean_joined_f1']:.6f} |
| Phase 5A local-tree features only | {phase5a_mean:.6f} |
| Same-tree fixed 40 | {fixed40_mean:.6f} |

{PHASE} minus Phase 5A is {versus5a['observed_mean_difference']:.6f}, with
paired paper-cluster bootstrap 95% CI
[{versus5a['confidence_interval_95'][0]:.6f}, {versus5a['confidence_interval_95'][1]:.6f}].
{PHASE} minus same-tree fixed 40 is {versus40['observed_mean_difference']:.6f},
with 95% CI [{versus40['confidence_interval_95'][0]:.6f},
{versus40['confidence_interval_95'][1]:.6f}].

{PHASE} mean precision is {retrieval['mean_joined_precision']:.6f}, mean recall
is {retrieval['mean_joined_recall']:.6f}, and median joined F1 is
{retrieval['median_joined_f1']:.6f}. Retrieval covers all 924 questions.

The 4,620 selected validation-tree predictions are: 10={granularity['10']},
20={granularity['20']}, 40={granularity['40']}, 80={granularity['80']}, and
160={granularity['160']}.

The Qwen category is strongly concentrated on `medium`: this label covers
{100 * qwen_train['medium'] / EXPECTED_TRAIN_QUESTIONS:.2f}% of training questions and
{100 * qwen_validation['medium'] / EXPECTED_VALIDATION_QUESTIONS:.2f}% of validation
questions, while validation contains no `long` prediction. Both paired confidence
intervals above include zero, so the auxiliary Qwen feature does not show a reliable
retrieval improvement over Phase 5A or the matched fixed-40 strategy.

Secondary local-tree classification accuracy is {classification['accuracy']:.6f}
and macro-F1 is {classification['macro_f1']:.6f}. This diagnostic measures the
local evidence-length label; joined retrieval F1 remains the operational metric.

## Integrity

- Validation was not used for training, model selection, prompt selection, or thresholding.
- Predictions were saved and hashed before validation evidence was requested for evaluation.
- Qdrant was read-only and its before/after snapshots matched.
- Phase 5A and all earlier source artifacts remained hash-identical.
- Qwen used `{qwen['model']['device']}` with `{qwen['model']['dtype']}`, with zero gradients,
  optimizers, backward passes, or parameter updates.
- The result is a development result because validation was reused in earlier phases.

## Reproduction

```powershell
.\\.venv-qwen\\Scripts\\python.exe {ENTRYPOINT_SCRIPT} qwen-infer
# Remote CUDA alternative: .venv-phase5b-gpu/bin/python {ENTRYPOINT_SCRIPT} qwen-infer
.\\.venv-phase5a\\Scripts\\python.exe {ENTRYPOINT_SCRIPT} train-evaluate
.\\.venv-phase5a\\Scripts\\python.exe -m pytest {TEST_COMMAND} -q
```

Artifacts: `{OUTPUT_ROOT.as_posix()}`.
"""
    atomic_text(DOC_PATH, content)
    atomic_text(REPORT_PATH, content)


def audit_command(args: argparse.Namespace) -> int:
    audit = preflight(args.output_root, require_qdrant=args.require_qdrant)
    lock_path = args.output_root / "configuration/procedure_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else write_procedure_lock(args.output_root, audit)
    print(json.dumps({"audit": audit, "procedure": lock}, indent=2))
    return 0


def qwen_command(args: argparse.Namespace) -> int:
    preflight(args.output_root, require_qdrant=False)
    lock_path = args.output_root / "configuration/procedure_lock.json"
    if not lock_path.exists():
        write_procedure_lock(args.output_root, preflight(args.output_root, require_qdrant=False))
    print(json.dumps(run_qwen_inference(args.output_root, ("train", "validation")), indent=2))
    return 0


def smoke_command(args: argparse.Namespace) -> int:
    preflight(args.output_root, require_qdrant=False)
    try:
        result = run_qwen_smoke(args.output_root, args.count)
    except Exception as exc:
        atomic_json(
            args.output_root / "smoke" / "failure.json",
            {
                "status": "failed",
                "failed_at": utc_now(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
    print(json.dumps(result, indent=2))
    return 0


def train_command(args: argparse.Namespace) -> int:
    print(json.dumps(run_tree_experiment(args.output_root), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--require-qdrant", action="store_true")
    audit.set_defaults(function=audit_command)
    subparsers.add_parser("qwen-infer").set_defaults(function=qwen_command)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--count", type=int, default=3)
    smoke.set_defaults(function=smoke_command)
    subparsers.add_parser("train-evaluate").set_defaults(function=train_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
