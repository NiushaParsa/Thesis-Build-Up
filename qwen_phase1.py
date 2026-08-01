#!/usr/bin/env python
"""Phase 1 evidence-length Oracle and pretrained Qwen router utilities.

This module never creates or modifies Qdrant collections. It reads the frozen
router example set and stored evidence, writes separate local artifacts, and
runs inference with the original post-trained Qwen3.5-0.8B model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import psutil
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from transformers import AutoModelForMultimodalLM, AutoProcessor

from chunking_utils import count_tokens
from config import (
    CHUNK_SIZES,
    EMBEDDING_DIM,
    OPENAI_EMBEDDING_MODEL,
    PAPER_QUESTION_COLLECTION,
    PAPER_EVIDENCE_COLLECTION,
    QDRANT_API_KEY,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    QDRANT_URL,
    ROUTER_DATASET_COLLECTION,
    ROUTER_RANDOM_SEED,
    TOKENIZER_NAME,
)
from fixed_sized_granularity_separate import evaluate_question


MODEL_ID = "Qwen/Qwen3.5-0.8B"
TOKENIZER_ID = MODEL_ID
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TRANSFORMERS_COMMIT = "2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7"
SOURCE_ORACLE_CONFIG_HASH = (
    "9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8"
)
ORACLE_LABEL_VERSION = "oracle-evidence-length-gpt2-smaller-midpoint-v1"
CLASS_TOKENS = (10, 20, 40, 80, 160)
FIXED_INSTRUCTION = (
    "You are a router for a retrieval-augmented generation system. Based only "
    "on the question, select the chunk size most suitable for retrieving the "
    "evidence required to answer it. Choose exactly one value from: 10, 20, "
    "40, 80, 160. Return only the number."
)
DEFAULT_EXPERIMENT_DIR = Path(
    "outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle"
)
VALID_CLASS_PATTERN = re.compile(r"(?<!\d)(10|20|40|80|160)(?!\d)")


def qdrant_client() -> QdrantClient:
    """Return the existing project client without changing server state."""
    api_key = QDRANT_API_KEY or None
    if QDRANT_URL:
        return QdrantClient(
            url=QDRANT_URL,
            api_key=api_key,
            prefer_grpc=False,
            timeout=300,
            check_compatibility=False,
        )
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_HTTP_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        api_key=api_key,
        prefer_grpc=False,
        timeout=300,
        check_compatibility=False,
    )


def _scroll(
    client: QdrantClient,
    collection: str,
    *,
    scroll_filter: Filter | None = None,
    with_vectors: bool = False,
) -> list[Any]:
    records: list[Any] = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        records.extend(batch)
        if next_offset is None:
            return records
        offset = next_offset


def clean_deduplicate_combine_evidence(texts: Iterable[str]) -> tuple[list[str], str]:
    """Apply the existing strip/exact-deduplicate/sort/newline convention."""
    cleaned = sorted({text.strip() for text in texts if text and text.strip()})
    return cleaned, "\n".join(cleaned)


def closest_chunk_size(
    evidence_token_length: int,
    candidates: Sequence[int] = CLASS_TOKENS,
) -> int:
    """Return the numerically nearest candidate; exact ties choose smaller."""
    if evidence_token_length < 0:
        raise ValueError("Evidence token length cannot be negative")
    if not candidates:
        raise ValueError("At least one chunk-size candidate is required")
    unique = sorted(set(int(value) for value in candidates))
    return min(unique, key=lambda value: (abs(value - evidence_token_length), value))


def parse_qwen_class(raw_output: str) -> tuple[int | None, str]:
    """Parse one unambiguous class using proper numeric boundaries."""
    found = {int(match) for match in VALID_CLASS_PATTERN.findall(raw_output)}
    if len(found) == 1:
        return next(iter(found)), "valid"
    if not found:
        return None, "invalid_no_valid_class"
    return None, "invalid_multiple_classes"


def build_prompt(question_text: str) -> str:
    """Return the sole semantic Qwen input: fixed instruction plus question."""
    return f"{FIXED_INSTRUCTION}\n\nQuestion: {question_text}"


def _load_frozen_router_examples(client: QdrantClient) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        query_filter = Filter(
            must=[
                FieldCondition(key="split", match=MatchValue(value=split)),
                FieldCondition(
                    key="evaluation_config_hash",
                    match=MatchValue(value=SOURCE_ORACLE_CONFIG_HASH),
                ),
            ]
        )
        for point in _scroll(
            client, ROUTER_DATASET_COLLECTION, scroll_filter=query_filter
        ):
            payload = point.payload or {}
            records.append(
                {
                    "question_id": str(payload.get("question_id") or point.id),
                    "document_id": str(payload.get("document_id", "")),
                    "split": str(payload.get("split", "")),
                    "question_text": str(payload.get("question_text", "")),
                }
            )
    counts = Counter(record["split"] for record in records)
    if counts != {"train": 2245, "validation": 924}:
        raise RuntimeError(f"Frozen router example counts changed: {dict(counts)}")
    question_ids = [record["question_id"] for record in records]
    if len(question_ids) != len(set(question_ids)):
        raise RuntimeError("Duplicate question IDs in frozen router example set")
    return records


def _load_evidence_map(client: QdrantClient) -> dict[str, list[str]]:
    by_question: dict[str, list[str]] = defaultdict(list)
    for point in _scroll(client, PAPER_EVIDENCE_COLLECTION):
        payload = point.payload or {}
        question_id = str(payload.get("question_id", ""))
        evidence_text = payload.get("evidence_text")
        if question_id and isinstance(evidence_text, str):
            by_question[question_id].append(evidence_text)
    return by_question


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append and flush one complete JSONL record for interruption safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["oracle_label"] for record in records)
    total = len(records)
    return {
        "total": total,
        "classes": {
            str(label): {
                "count": counts[label],
                "percentage": 100.0 * counts[label] / total,
            }
            for label in CLASS_TOKENS
        },
    }


def _inspection_examples(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    boundaries = (15, 30, 60, 120)

    def nearest(target: int) -> dict[str, Any]:
        return min(
            records,
            key=lambda record: (
                abs(record["evidence_token_length"] - target),
                record["question_id"],
            ),
        )

    midpoint_matches = {
        str(value): [
            record
            for record in records
            if record["evidence_token_length"] == value
        ][:3]
        for value in boundaries
    }
    multiple = [
        record for record in records if record["unique_evidence_count"] > 1
    ][:5]
    return {
        "below_10": [
            record for record in records if record["evidence_token_length"] < 10
        ][:5],
        "nearest_each_boundary": {str(value): nearest(value) for value in boundaries},
        "exact_midpoints": midpoint_matches,
        "above_160": [
            record for record in records if record["evidence_token_length"] > 160
        ][:5],
        "multiple_evidence_spans": multiple,
    }


def generate_oracle(output_dir: Path) -> dict[str, Any]:
    """Generate separate train/validation evidence-length Oracle artifacts."""
    client = qdrant_client()
    examples = _load_frozen_router_examples(client)
    evidence_map = _load_evidence_map(client)
    records: list[dict[str, Any]] = []
    for example in examples:
        evidence_items, combined = clean_deduplicate_combine_evidence(
            evidence_map.get(example["question_id"], [])
        )
        if not evidence_items:
            raise RuntimeError(f"No evidence for {example['question_id']}")
        length = count_tokens(combined)
        label = closest_chunk_size(length)
        records.append(
            {
                **example,
                "evidence_token_length": length,
                "oracle_label": label,
                "router_target_granularity": CLASS_TOKENS.index(label) + 1,
                "unique_evidence_count": len(evidence_items),
                "evidence_combination": "sorted unique stripped spans joined by newline",
                "evidence_tokenizer": "gpt2",
                "midpoint_tie_rule": "smaller_candidate",
                "label_version": ORACLE_LABEL_VERSION,
                "source_router_evaluation_config_hash": SOURCE_ORACLE_CONFIG_HASH,
            }
        )

    summary: dict[str, Any] = {
        "label_version": ORACLE_LABEL_VERSION,
        "candidate_classes": list(CLASS_TOKENS),
        "tokenizer": "gpt2",
        "tie_rule": "smaller_candidate",
        "source_router_evaluation_config_hash": SOURCE_ORACLE_CONFIG_HASH,
        "splits": {},
    }
    for split in ("train", "validation"):
        split_records = sorted(
            (record for record in records if record["split"] == split),
            key=lambda record: record["question_id"],
        )
        oracle_path = output_dir / "oracle" / f"{split}_oracle.jsonl"
        _write_jsonl(oracle_path, split_records)
        distribution = _distribution(split_records)
        _write_json(
            output_dir / "oracle" / f"{split}_distribution.json", distribution
        )
        with (
            output_dir / "oracle" / f"{split}_distribution.csv"
        ).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_tokens", "count", "percentage"])
            for label in CLASS_TOKENS:
                row = distribution["classes"][str(label)]
                writer.writerow([label, row["count"], row["percentage"]])
        summary["splits"][split] = distribution

    _write_json(output_dir / "oracle" / "oracle_summary.json", summary)
    _write_json(
        output_dir / "oracle" / "representative_inspections.json",
        _inspection_examples(records),
    )
    _write_oracle_histogram(output_dir, summary)
    return summary


def _write_oracle_histogram(output_dir: Path, summary: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path("tmp") / "matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(CLASS_TOKENS))
    width = 0.38
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for offset, split in ((-width / 2, "train"), (width / 2, "validation")):
        values = [
            summary["splits"][split]["classes"][str(label)]["percentage"]
            for label in CLASS_TOKENS
        ]
        axis.bar(x + offset, values, width, label=split)
    axis.set_xticks(x, [str(label) for label in CLASS_TOKENS])
    axis.set_xlabel("Evidence-length Oracle class (tokens)")
    axis.set_ylabel("Examples (%)")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = output_dir / "oracle" / "oracle_distribution.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def validate_oracle(output_dir: Path) -> dict[str, Any]:
    """Validate generated Oracle rows and deterministic boundary behavior."""
    expected = {
        0: 10,
        9: 10,
        10: 10,
        14: 10,
        15: 10,
        16: 20,
        30: 20,
        31: 40,
        60: 40,
        61: 80,
        120: 80,
        121: 160,
        160: 160,
        161: 160,
        1000: 160,
    }
    for length, label in expected.items():
        actual = closest_chunk_size(length)
        if actual != label:
            raise AssertionError(f"{length}: expected {label}, got {actual}")
    counts = {}
    for split, expected_count in (("train", 2245), ("validation", 924)):
        path = output_dir / "oracle" / f"{split}_oracle.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != expected_count:
            raise AssertionError(f"{split}: expected {expected_count}, got {len(rows)}")
        if any(row["oracle_label"] not in CLASS_TOKENS for row in rows):
            raise AssertionError(f"{split}: invalid Oracle class")
        counts[split] = len(rows)
    result = {"status": "passed", "record_counts": counts, "boundaries": expected}
    _write_json(output_dir / "oracle" / "validation_checks.json", result)
    return result


def _load_validation_oracle(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "oracle" / "validation_oracle.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Generate the Oracle first: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_qwen() -> tuple[Any, Any, dict[str, Any]]:
    """Load and freeze the exact post-trained Qwen model for CPU inference."""
    process = psutil.Process()
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    processor_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype="auto",
        local_files_only=True,
    )
    model_seconds = time.perf_counter() - started
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen parameters were not fully frozen")
    first_parameter = next(model.parameters())
    info = {
        "model_id": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "model_revision": MODEL_REVISION,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "processor_load_seconds": processor_seconds,
        "model_load_seconds": model_seconds,
        "dtype": str(first_parameter.dtype),
        "device": str(first_parameter.device),
        "quantization": None,
        "rss_gib_after_load": process.memory_info().rss / 2**30,
        "all_parameters_frozen": True,
    }
    return processor, model, info


def predict_one(
    processor: Any, model: Any, question_text: str
) -> tuple[str, int | None, str, float]:
    """Run deterministic inference for one instruction-plus-question prompt."""
    prompt = build_prompt(question_text)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    elapsed = time.perf_counter() - started
    raw = processor.decode(
        output[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    parsed, status = parse_qwen_class(raw)
    return raw, parsed, status, elapsed


def smoke_test(output_dir: Path, count: int) -> dict[str, Any]:
    """Run a small deterministic validation smoke test and save every output."""
    if count < 1:
        raise ValueError("Smoke-test count must be positive")
    examples = _load_validation_oracle(output_dir)
    rng = random.Random(ROUTER_RANDOM_SEED)
    selected = rng.sample(examples, min(count, len(examples)))
    processor, model, model_info = load_qwen()
    rows = []
    process = psutil.Process()
    for record in selected:
        raw, parsed, status, elapsed = predict_one(
            processor, model, record["question_text"]
        )
        rows.append(
            {
                **record,
                "raw_qwen_output": raw,
                "parsed_prediction": parsed,
                "prediction_status": status,
                "inference_seconds": elapsed,
            }
        )
        _write_jsonl(output_dir / "smoke" / "predictions.jsonl", rows)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A Qwen parameter unexpectedly requires gradients")
    timings = [row["inference_seconds"] for row in rows]
    summary = {
        **model_info,
        "smoke_examples": len(rows),
        "valid_predictions": sum(row["parsed_prediction"] is not None for row in rows),
        "invalid_predictions": sum(row["parsed_prediction"] is None for row in rows),
        "mean_inference_seconds": float(np.mean(timings)),
        "median_inference_seconds": float(np.median(timings)),
        "estimated_924_seconds": float(np.mean(timings) * 924),
        "estimated_924_hours": float(np.mean(timings) * 924 / 3600),
        "rss_gib_after_smoke": process.memory_info().rss / 2**30,
        "parameter_updates": 0,
        "optimizer_created": False,
        "backward_passes": 0,
        "prompt": FIXED_INSTRUCTION,
        "chat_template": processor.chat_template,
        "decoding": {"do_sample": False, "max_new_tokens": 8},
    }
    _write_json(output_dir / "smoke" / "summary.json", summary)
    _write_json(
        output_dir / "configuration" / "fixed_prompt.json",
        {
            "instruction": FIXED_INSTRUCTION,
            "question_format": "{instruction}\\n\\nQuestion: {original_question_text}",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "<formatted prompt>"}],
                }
            ],
        },
    )
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def infer_validation(output_dir: Path) -> dict[str, Any]:
    """Run resumable inference for all 924 preserved validation examples."""
    examples = _load_validation_oracle(output_dir)
    if len(examples) != 924:
        raise RuntimeError(f"Expected 924 validation examples, got {len(examples)}")
    prediction_path = output_dir / "validation" / "predictions.jsonl"
    existing = _read_jsonl(prediction_path)
    by_id = {row["question_id"]: row for row in existing}
    if len(by_id) != len(existing):
        raise RuntimeError("Duplicate question IDs in resumable prediction file")
    oracle_ids = {row["question_id"] for row in examples}
    if not set(by_id).issubset(oracle_ids):
        raise RuntimeError("Prediction file contains questions outside the frozen set")
    pending = [row for row in examples if row["question_id"] not in by_id]
    process = psutil.Process()
    run_started = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
    processor, model, model_info = load_qwen()
    peak_rss = process.memory_info().rss
    for index, record in enumerate(pending, start=1):
        raw, parsed, status, elapsed = predict_one(
            processor, model, record["question_text"]
        )
        result = {
            **record,
            "raw_qwen_output": raw,
            "parsed_prediction": parsed,
            "prediction_status": status,
            "inference_seconds": elapsed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(prediction_path, result)
        by_id[result["question_id"]] = result
        peak_rss = max(peak_rss, process.memory_info().rss)
        _write_json(
            output_dir / "validation" / "checkpoint.json",
            {
                "run_started_at": run_started,
                "completed": len(by_id),
                "total": len(examples),
                "remaining": len(examples) - len(by_id),
                "last_question_id": result["question_id"],
                "last_completed_at": result["completed_at"],
                "current_segment_seconds": time.perf_counter() - wall_started,
            },
        )
        if index % 10 == 0 or index == len(pending):
            print(
                f"validation progress: {len(by_id)}/924; "
                f"last={elapsed:.2f}s",
                flush=True,
            )
    rows = [by_id[row["question_id"]] for row in examples]
    if len(rows) != 924:
        raise RuntimeError("Validation inference did not produce 924 records")
    _write_jsonl(
        output_dir / "validation" / "raw_outputs.jsonl",
        (
            {
                "question_id": row["question_id"],
                "document_id": row["document_id"],
                "raw_qwen_output": row["raw_qwen_output"],
            }
            for row in rows
        ),
    )
    _write_jsonl(
        output_dir / "validation" / "parsed_predictions.jsonl",
        (
            {
                "question_id": row["question_id"],
                "document_id": row["document_id"],
                "parsed_prediction": row["parsed_prediction"],
                "prediction_status": row["prediction_status"],
            }
            for row in rows
        ),
    )
    invalid = [row for row in rows if row["parsed_prediction"] is None]
    _write_jsonl(output_dir / "validation" / "invalid_outputs.jsonl", invalid)
    timings = [float(row["inference_seconds"]) for row in rows]
    summary = {
        **model_info,
        "run_started_at": run_started,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
        "resumed_existing_records": len(existing),
        "new_records_this_segment": len(pending),
        "evaluated_examples": len(rows),
        "valid_outputs": len(rows) - len(invalid),
        "invalid_outputs": len(invalid),
        "invalid_output_percentage": 100.0 * len(invalid) / len(rows),
        "valid_output_rate": (len(rows) - len(invalid)) / len(rows),
        "total_inference_seconds": float(sum(timings)),
        "mean_inference_seconds": float(np.mean(timings)),
        "median_inference_seconds": float(np.median(timings)),
        "current_segment_wall_seconds": time.perf_counter() - wall_started,
        "approximate_peak_rss_gib": peak_rss / 2**30,
        "parameter_updates": 0,
        "optimizer_created": False,
        "backward_passes": 0,
        "decoding": {"do_sample": False, "max_new_tokens": 8},
        "fixed_instruction": FIXED_INSTRUCTION,
    }
    _write_json(output_dir / "validation" / "runtime_summary.json", summary)
    return summary


def _fixed_classification_metrics(
    rows: Sequence[dict[str, Any]], prediction_key: str
) -> dict[str, Any]:
    """Match existing five-class metrics while retaining invalid predictions."""
    class_to_index = {label: index for index, label in enumerate(CLASS_TOKENS)}
    confusion = np.zeros((5, 5), dtype=np.int64)
    support = np.zeros(5, dtype=np.int64)
    correct = 0
    valid = 0
    for row in rows:
        target = int(row["oracle_label"])
        target_index = class_to_index[target]
        support[target_index] += 1
        prediction = row[prediction_key]
        if prediction is None:
            continue
        prediction_index = class_to_index[int(prediction)]
        confusion[target_index, prediction_index] += 1
        valid += 1
        correct += int(target_index == prediction_index)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(5, dtype=float),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(5, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(5, dtype=float),
        where=(precision + recall) != 0,
    )
    total = len(rows)
    present = support > 0
    return {
        "accuracy": correct / total,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "balanced_accuracy": float(recall[present].mean()),
        "top_2_accuracy": None,
        "top_2_accuracy_status": "unavailable_no_comparable_class_scores",
        "per_class": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_TOKENS)
        },
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_note": (
            "5x5 valid-prediction matrix; invalid outputs are omitted from cells "
            "but remain false negatives in support, recall, F1, and complete-set accuracy"
        ),
        "evaluated_examples": total,
        "valid_predictions": valid,
        "invalid_predictions": total - valid,
    }


def evaluate_classification(output_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(output_dir / "validation" / "predictions.jsonl")
    if len(rows) != 924:
        raise RuntimeError(f"Expected 924 predictions, got {len(rows)}")
    metrics = _fixed_classification_metrics(rows, "parsed_prediction")
    oracle_distribution = Counter(int(row["oracle_label"]) for row in rows)
    predicted_distribution = Counter(
        int(row["parsed_prediction"])
        for row in rows
        if row["parsed_prediction"] is not None
    )
    majority_class = min(
        CLASS_TOKENS,
        key=lambda label: (-oracle_distribution[label], label),
    )
    majority_rows = [dict(row, majority_prediction=majority_class) for row in rows]
    majority_metrics = _fixed_classification_metrics(
        majority_rows, "majority_prediction"
    )
    result = {
        "classification_metrics": metrics,
        "oracle_distribution": {
            str(label): oracle_distribution[label] for label in CLASS_TOKENS
        },
        "predicted_distribution": {
            str(label): predicted_distribution[label] for label in CLASS_TOKENS
        },
        "invalid_output_count": metrics["invalid_predictions"],
        "invalid_output_percentage": 100.0 * metrics["invalid_predictions"] / 924,
        "valid_output_count": metrics["valid_predictions"],
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_metrics["accuracy"],
        "majority_baseline_macro_f1": majority_metrics["macro_f1"],
        "majority_baseline_metrics": majority_metrics,
    }
    evaluation_dir = output_dir / "classification"
    _write_json(evaluation_dir / "metrics.json", result)
    with (evaluation_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["oracle\\predicted", *CLASS_TOKENS])
        for label, values in zip(CLASS_TOKENS, metrics["confusion_matrix"]):
            writer.writerow([label, *values])
    _write_prediction_histogram(output_dir, result)
    return result


def _write_prediction_histogram(output_dir: Path, result: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((Path("tmp") / "matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(CLASS_TOKENS))
    width = 0.38
    oracle = [result["oracle_distribution"][str(label)] for label in CLASS_TOKENS]
    predicted = [
        result["predicted_distribution"][str(label)] for label in CLASS_TOKENS
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(x - width / 2, oracle, width, label="Oracle")
    axis.bar(x + width / 2, predicted, width, label="Qwen predicted")
    axis.set_xticks(x, [str(label) for label in CLASS_TOKENS])
    axis.set_xlabel("Chunk size (tokens)")
    axis.set_ylabel("Validation examples")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = output_dir / "classification" / "predicted_vs_oracle.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def evaluate_retrieval(output_dir: Path) -> dict[str, Any]:
    """Run unchanged fixed-separate retrieval at each valid predicted level."""
    predictions = _read_jsonl(output_dir / "validation" / "predictions.jsonl")
    if len(predictions) != 924:
        raise RuntimeError(f"Expected 924 predictions, got {len(predictions)}")
    valid_predictions = [
        row for row in predictions if row["parsed_prediction"] is not None
    ]
    result_path = output_dir / "retrieval" / "results.jsonl"
    existing = _read_jsonl(result_path)
    by_id = {row["question_id"]: row for row in existing}
    valid_ids = {row["question_id"] for row in valid_predictions}
    if not set(by_id).issubset(valid_ids):
        raise RuntimeError("Retrieval file contains non-valid or unknown predictions")
    client = qdrant_client()
    wall_started = time.perf_counter()
    run_id = f"qwen-phase1-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    for index, prediction in enumerate(valid_predictions, start=1):
        question_id = prediction["question_id"]
        if question_id in by_id:
            continue
        points = client.retrieve(
            collection_name=PAPER_QUESTION_COLLECTION,
            ids=[question_id],
            with_payload=True,
            with_vectors=True,
        )
        if len(points) != 1:
            raise RuntimeError(f"Question point lookup failed: {question_id}")
        point = points[0]
        payload = point.payload or {}
        predicted_tokens = int(prediction["parsed_prediction"])
        level = CLASS_TOKENS.index(predicted_tokens) + 1
        records = list(
            evaluate_question(
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
                embedding_model=OPENAI_EMBEDDING_MODEL,
                embedding_dimension=EMBEDDING_DIM,
                tokenizer_name=TOKENIZER_NAME,
                evaluation_run_id=run_id,
            )
        )
        if len(records) != 1:
            raise RuntimeError(f"Expected one retrieval record for {question_id}")
        record = records[0]
        record.update(
            {
                "method_name": "qwen-pretrained-zero-shot-router",
                "predicted_granularity_tokens": predicted_tokens,
                "predicted_granularity_level": level,
                "qwen_raw_output": prediction["raw_qwen_output"],
                "qwen_prediction_status": prediction["prediction_status"],
                "evidence_length_oracle": prediction["oracle_label"],
                "oracle_label_version": ORACLE_LABEL_VERSION,
            }
        )
        _append_jsonl(result_path, record)
        by_id[question_id] = record
        if index % 25 == 0 or len(by_id) == len(valid_predictions):
            print(
                f"retrieval progress: {len(by_id)}/{len(valid_predictions)}",
                flush=True,
            )
    ordered = [by_id[row["question_id"]] for row in valid_predictions]
    f1_values = [float(row["f1_joined_topk"]) for row in ordered]
    coverage = len(ordered) / len(predictions)
    summary = {
        "evaluated_examples": len(predictions),
        "valid_prediction_retrievals": len(ordered),
        "invalid_predictions_without_retrieval": len(predictions) - len(ordered),
        "retrieval_coverage": coverage,
        "valid_only_mean_joined_retrieval_f1": float(np.mean(f1_values)),
        "valid_only_median_joined_retrieval_f1": float(np.median(f1_values)),
        "coverage_adjusted_full_set_mean_joined_retrieval_f1": (
            float(sum(f1_values) / len(predictions))
        ),
        "full_set_note": (
            "Invalid outputs have no fabricated retrieval record. The coverage-adjusted "
            "aggregate assigns zero contribution only when summarizing the complete set; "
            "the valid-only mean is reported separately."
        ),
        "top_k": 5,
        "paper_restricted": True,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "tokenizer": TOKENIZER_NAME,
        "metric": "f1_joined_topk",
        "current_segment_wall_seconds": time.perf_counter() - wall_started,
    }
    _write_json(output_dir / "retrieval" / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate-oracle")
    subparsers.add_parser("validate-oracle")
    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--count", type=int, default=3)
    subparsers.add_parser("infer-validation")
    subparsers.add_parser("evaluate-classification")
    subparsers.add_parser("evaluate-retrieval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if tuple(CHUNK_SIZES) != CLASS_TOKENS:
        raise RuntimeError(
            f"Configured chunk sizes changed: expected {CLASS_TOKENS}, got {CHUNK_SIZES}"
        )
    if args.command == "generate-oracle":
        result = generate_oracle(args.output_dir)
    elif args.command == "validate-oracle":
        result = validate_oracle(args.output_dir)
    elif args.command == "smoke-test":
        result = smoke_test(args.output_dir, args.count)
    elif args.command == "infer-validation":
        result = infer_validation(args.output_dir)
    elif args.command == "evaluate-classification":
        result = evaluate_classification(args.output_dir)
    else:
        result = evaluate_retrieval(args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
