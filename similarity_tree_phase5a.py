#!/usr/bin/env python
"""Phase 5A tree-local similarity router with gold-overlap training.

This experiment implements Lorenzo's tree-local protocol as a new, isolated
experiment.  Each 160-token chunk is a tree root; its descendants are the
aligned 80/40/20/10-token chunks.  Training examples are only trees that
overlap gold evidence, and each tree receives a local evidence-length label.
At inference, all trees are ranked without gold evidence, the classifier is
applied independently to the top five trees, and the single most similar
chunk at each predicted level is retained for joined retrieval evaluation.

No embeddings, Qwen inference, collection writes, or collection indexing are
performed.  Frozen Phase 3A score arrays and read-only Qdrant payloads are the
only data sources.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

import similarity_tree_phase3b as phase3b


PHASE = "Phase 5A"
EXPERIMENT_NAME = "Tree-Local Similarity Router with Gold-Overlap Training"
FORMULATION_VERSION = "phase5a-tree-local-gold-overlap-v1"
FEATURE_SCHEMA_VERSION = "phase5a-local-similarity-tree-features-v1"
CLASS_TOKENS = (10, 20, 40, 80, 160)
TOP_K_TREES = 5
SEED = 42
FOLDS = 5
SOFTMAX_TEMPERATURE = 0.05
TRAIN_EXPECTED = 2245
VALIDATION_EXPECTED = 924
RAW_ROOT = Path("outputs/similarity_tree_phase3a_evidence_length_oracle/features")
DEFAULT_OUTPUT_ROOT = Path("outputs/similarity_tree_phase5a_local_gold_overlap_router")
REPORT_ROOT = Path("reports/similarity_tree_phase5a_local_gold_overlap_router")
DOC_PATH = Path("docs/SIMILARITY_TREE_PHASE5A_RESULTS.md")
PAPER_CHUNK_COLLECTION = "PaperChunk"
PAPER_EVIDENCE_COLLECTION = "PaperEvidence"
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
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
FORBIDDEN_FEATURE_TERMS = (
    "evidence",
    "oracle",
    "answer",
    "retrieval_f1",
    "joined_f1",
    "chunk_f1",
    "target",
    "label",
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
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, gzip_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if gzip_output or path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def raw_path(split: str) -> Path:
    return RAW_ROOT / f"{split}_similarity_trees.jsonl.gz"


def qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_HTTP_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=True,
        check_compatibility=False,
        timeout=120,
    )


def collection_snapshot(client: QdrantClient) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for collection in sorted(item.name for item in client.get_collections().collections):
        info = client.get_collection(collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            vector_config = {
                name: {"size": cfg.size, "distance": str(cfg.distance)}
                for name, cfg in sorted(vectors.items())
            }
        else:
            vector_config = {"size": vectors.size, "distance": str(vectors.distance)}
        result[collection] = {
            "points_count": int(info.points_count or 0),
            "indexed_vectors_count": int(info.indexed_vectors_count or 0),
            "status": str(info.status),
            "vectors": vector_config,
        }
    return result


def chunk_id(document_id: str, tokens: int, chunk_index: int) -> str:
    level = CLASS_TOKENS.index(int(tokens)) + 1
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_g{level}_c{chunk_index}"))


def _quantile(values: np.ndarray, value: float) -> float:
    return float(np.quantile(values, value))


def level_statistics(scores: Sequence[float], token_size: int) -> dict[str, float]:
    """Exact Phase 3A level-statistic definition, applied to one local tree."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid local score distribution for level {token_size}")
    descending = np.sort(values)[::-1]
    maximum = float(descending[0])
    shifted = (values - maximum) / SOFTMAX_TEMPERATURE
    weights = np.exp(shifted)
    probabilities = weights / weights.sum()
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum())
    normalized_entropy = entropy / math.log(len(values)) if len(values) > 1 else 0.0
    effective_fraction = math.exp(entropy) / len(values)
    prefix = f"level_{token_size}_"
    return {
        prefix + "log_count": math.log1p(len(values)),
        prefix + "max": maximum,
        prefix + "mean": float(values.mean()),
        prefix + "std": float(values.std()),
        prefix + "q50": _quantile(values, 0.50),
        prefix + "q75": _quantile(values, 0.75),
        prefix + "q90": _quantile(values, 0.90),
        prefix + "q95": _quantile(values, 0.95),
        prefix + "top2_mean": float(descending[: min(2, len(values))].mean()),
        prefix + "top5_mean": float(descending[: min(5, len(values))].mean()),
        prefix + "top10_mean": float(descending[: min(10, len(values))].mean()),
        prefix + "margin_top1_top2": maximum - float(descending[1]) if len(descending) > 1 else 0.0,
        prefix + "max_mean_gap": maximum - float(values.mean()),
        prefix + "near_max_002_fraction": float(np.mean(values >= maximum - 0.02)),
        prefix + "near_max_005_fraction": float(np.mean(values >= maximum - 0.05)),
        prefix + "softmax_entropy_norm_t005": normalized_entropy,
        prefix + "softmax_effective_fraction_t005": effective_fraction,
    }


def _describe(values: Sequence[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {prefix + suffix: 0.0 for suffix in ("mean", "std", "q10", "q50", "q90", "max")}
    return {
        prefix + "mean": float(array.mean()),
        prefix + "std": float(array.std()),
        prefix + "q10": _quantile(array, 0.10),
        prefix + "q50": _quantile(array, 0.50),
        prefix + "q90": _quantile(array, 0.90),
        prefix + "max": float(array.max()),
    }


def tree_edge_statistics(
    child_scores: Sequence[float],
    parent_scores: Sequence[float],
    child_tokens: int,
    parent_tokens: int,
) -> dict[str, float]:
    """Exact Phase 3A edge-statistic definition, applied to one local tree."""
    child = np.asarray(child_scores, dtype=np.float64)
    parent = np.asarray(parent_scores, dtype=np.float64)
    if len(parent) != math.ceil(len(child) / 2):
        raise ValueError(
            f"Broken local hierarchy {parent_tokens}->{child_tokens}: {len(parent)} vs {len(child)}"
        )
    max_deltas: list[float] = []
    mean_deltas: list[float] = []
    sibling_gaps: list[float] = []
    child_maxima: list[float] = []
    near_equal_002 = near_equal_005 = two_child_nodes = 0
    for parent_index, parent_score in enumerate(parent):
        children = child[parent_index * 2 : parent_index * 2 + 2]
        child_max = float(children.max())
        child_maxima.append(child_max)
        max_deltas.append(child_max - float(parent_score))
        mean_deltas.append(float(children.mean()) - float(parent_score))
        if len(children) == 2:
            gap = abs(float(children[0]) - float(children[1]))
            sibling_gaps.append(gap)
            near_equal_002 += int(gap <= 0.02)
            near_equal_005 += int(gap <= 0.05)
            two_child_nodes += 1
    prefix = f"edge_{parent_tokens}_to_{child_tokens}_"
    features: dict[str, float] = {}
    features.update(_describe(max_deltas, prefix + "child_max_minus_parent_"))
    features.update(_describe(mean_deltas, prefix + "child_mean_minus_parent_"))
    features.update(_describe(sibling_gaps, prefix + "sibling_abs_gap_"))
    features[prefix + "near_equal_002_fraction"] = near_equal_002 / two_child_nodes if two_child_nodes else 0.0
    features[prefix + "near_equal_005_fraction"] = near_equal_005 / two_child_nodes if two_child_nodes else 0.0
    if len(parent) > 1 and np.std(parent) > 0 and np.std(child_maxima) > 0:
        correlation = float(np.corrcoef(parent, child_maxima)[0, 1])
    else:
        correlation = 0.0
    features[prefix + "parent_child_max_correlation"] = correlation
    features[prefix + "argmax_alignment"] = float(int(np.argmax(child)) // 2 == int(np.argmax(parent)))
    return features


def assert_inference_safe_feature_names(names: Sequence[str]) -> None:
    bad = [name for name in names if any(term in name.lower() for term in FORBIDDEN_FEATURE_TERMS)]
    if bad:
        raise RuntimeError(f"Leakage-prone local feature names: {bad}")


def extract_local_features(scores_by_tokens: Mapping[int, Sequence[float]]) -> dict[str, float]:
    if set(scores_by_tokens) != set(CLASS_TOKENS):
        raise ValueError("All five local score levels are required")
    features: dict[str, float] = {}
    for tokens in CLASS_TOKENS:
        features.update(level_statistics(scores_by_tokens[tokens], tokens))
    for child_tokens, parent_tokens in zip(CLASS_TOKENS[:-1], CLASS_TOKENS[1:]):
        features.update(
            tree_edge_statistics(
                scores_by_tokens[child_tokens],
                scores_by_tokens[parent_tokens],
                child_tokens,
                parent_tokens,
            )
        )
    assert_inference_safe_feature_names(sorted(features))
    if len(features) != 173:
        raise RuntimeError(f"Expected 173 local features, got {len(features)}")
    return features


def local_score_tree(scores_by_tokens: Mapping[int, Sequence[float]], root_index: int) -> dict[int, np.ndarray]:
    local: dict[int, np.ndarray] = {}
    for tokens in CLASS_TOKENS:
        values = np.asarray(scores_by_tokens[tokens], dtype=np.float64)
        factor = 160 // tokens
        start = root_index * factor
        end = min((root_index + 1) * factor, len(values))
        if start >= len(values) or end <= start:
            raise ValueError(f"Root {root_index} has no descendants at level {tokens}")
        local[tokens] = values[start:end]
    for child_tokens, parent_tokens in zip(CLASS_TOKENS[:-1], CLASS_TOKENS[1:]):
        if len(local[parent_tokens]) != math.ceil(len(local[child_tokens]) / 2):
            raise ValueError(f"Unaligned local hierarchy at root {root_index}")
    return local


def tree_score(local_scores: Mapping[int, Sequence[float]]) -> float:
    """Lorenzo's frozen TreeScore: average of the five level means."""
    return float(np.mean([np.mean(np.asarray(local_scores[t], dtype=np.float64)) for t in CLASS_TOKENS]))


def rank_roots(scores_by_tokens: Mapping[int, Sequence[float]], top_n: int = TOP_K_TREES) -> list[dict[str, Any]]:
    roots = []
    for root_index in range(len(scores_by_tokens[160])):
        local = local_score_tree(scores_by_tokens, root_index)
        roots.append({"root_index": root_index, "tree_score": tree_score(local), "local_scores": local})
    roots.sort(key=lambda row: (-float(row["tree_score"]), int(row["root_index"])))
    return roots[: min(top_n, len(roots))]


def closest_chunk_size(token_length: int) -> int:
    """Nearest candidate; smaller class wins exact midpoint ties; extremes clamp."""
    if token_length < 0:
        raise ValueError("Token length cannot be negative")
    return min(CLASS_TOKENS, key=lambda tokens: (abs(tokens - token_length), tokens))


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    clean = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    merged: list[list[int]] = []
    for start, end in clean:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def clipped_overlap_intervals(
    evidence_intervals: Sequence[tuple[int, int]], root_start: int, root_end: int
) -> list[tuple[int, int]]:
    return merge_intervals(
        [(max(start, root_start), min(end, root_end)) for start, end in evidence_intervals]
    )


def normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    """NFKC/casefold text with collapsed whitespace and source-character map."""
    output: list[str] = []
    offsets: list[int] = []
    whitespace_pending = False
    whitespace_index = 0
    for source_index, source_char in enumerate(text):
        expanded = unicodedata.normalize("NFKC", source_char).casefold()
        for char in expanded:
            if char.isspace():
                if output and not whitespace_pending:
                    whitespace_pending = True
                    whitespace_index = source_index
                continue
            if whitespace_pending:
                output.append(" ")
                offsets.append(whitespace_index)
                whitespace_pending = False
            output.append(char)
            offsets.append(source_index)
    return "".join(output), offsets


def recover_span(paper_text: str, evidence_text: str) -> tuple[int, int, str] | None:
    """Recover only a unique exact or normalized evidence occurrence."""
    exact_starts = [match.start() for match in re.finditer(re.escape(evidence_text), paper_text)] if evidence_text else []
    if len(exact_starts) == 1:
        start = exact_starts[0]
        return start, start + len(evidence_text), "unique_exact_recovery"
    normalized_paper, paper_offsets = normalized_with_offsets(paper_text)
    normalized_evidence, _ = normalized_with_offsets(evidence_text)
    if not normalized_evidence:
        return None
    matches = [match.start() for match in re.finditer(re.escape(normalized_evidence), normalized_paper)]
    if len(matches) != 1:
        return None
    norm_start = matches[0]
    norm_end = norm_start + len(normalized_evidence) - 1
    return paper_offsets[norm_start], paper_offsets[norm_end] + 1, "unique_normalized_recovery"


def reconstruct_paper(root_payloads: Sequence[Mapping[str, Any]]) -> str:
    if not root_payloads:
        raise ValueError("Cannot reconstruct a paper without root chunks")
    size = max(int(row["span_end"]) for row in root_payloads)
    characters = [" "] * size
    for row in sorted(root_payloads, key=lambda item: int(item["chunk_idx"])):
        start, end = int(row["span_start"]), int(row["span_end"])
        content = str(row.get("content", ""))
        if end - start != len(content):
            raise RuntimeError(
                f"Chunk span/content mismatch for {row.get('document_id')} root {row.get('chunk_idx')}"
            )
        characters[start:end] = content
    return "".join(characters)


def fetch_root_payloads(
    client: QdrantClient, rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    required: dict[str, int] = {}
    for row in rows:
        document_id = str(row["document_id"])
        count = len(row["scores_by_tokens"]["160"])
        if document_id in required and required[document_id] != count:
            raise RuntimeError(f"Inconsistent root count for {document_id}")
        required[document_id] = count
    identities = [
        (document_id, index, chunk_id(document_id, 160, index))
        for document_id, count in sorted(required.items())
        for index in range(count)
    ]
    found: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(identities), 256):
        batch = identities[offset : offset + 256]
        points = client.retrieve(
            collection_name=PAPER_CHUNK_COLLECTION,
            ids=[identity for _, _, identity in batch],
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            found[str(point.id)] = dict(point.payload or {})
    output: dict[str, list[dict[str, Any]]] = {}
    for document_id, count in sorted(required.items()):
        payloads = []
        for index in range(count):
            identity = chunk_id(document_id, 160, index)
            if identity not in found:
                raise RuntimeError(f"Missing 160-token chunk {document_id}/{index}")
            payload = found[identity]
            if str(payload.get("document_id")) != document_id or int(payload.get("chunk_idx", -1)) != index:
                raise RuntimeError(f"Unexpected root payload identity for {document_id}/{index}")
            payloads.append(payload)
        output[document_id] = payloads
    return output


def fetch_evidence_for_questions(
    client: QdrantClient, question_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    requested = set(map(str, question_ids))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offset in range(0, len(question_ids), 100):
        batch = list(map(str, question_ids[offset : offset + 100]))
        scroll_offset = None
        while True:
            points, scroll_offset = client.scroll(
                collection_name=PAPER_EVIDENCE_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="question_id", match=models.MatchAny(any=batch)
                        )
                    ]
                ),
                limit=512,
                offset=scroll_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                question_id = str(payload.get("question_id"))
                if question_id not in requested:
                    raise RuntimeError("Qdrant returned evidence outside requested question IDs")
                payload["evidence_id"] = str(point.id)
                grouped[question_id].append(payload)
            if scroll_offset is None:
                break
    missing = requested - set(grouped)
    if missing:
        raise RuntimeError(f"Questions without evidence payloads: {len(missing)}")
    return dict(grouped)


def resolve_evidence_rows(
    evidence_rows: Sequence[Mapping[str, Any]],
    document_id: str,
    paper_text: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    resolved: list[dict[str, Any]] = []
    methods: Counter[str] = Counter()
    seen_text: set[str] = set()
    for source in evidence_rows:
        if str(source.get("document_id")) != document_id:
            raise RuntimeError("Evidence/document mismatch")
        text = str(source.get("evidence_text", "")).strip()
        normalized, _ = normalized_with_offsets(text)
        if not normalized or normalized in seen_text:
            methods["duplicate_or_empty_excluded"] += 1
            continue
        seen_text.add(normalized)
        start, end = int(source.get("span_start", -1)), int(source.get("span_end", -1))
        method = "stored_span"
        if start < 0 or end <= start or end > len(paper_text):
            recovery = recover_span(paper_text, text)
            if recovery is None:
                methods["unresolved"] += 1
                continue
            start, end, method = recovery
        resolved.append(
            {
                "evidence_id": str(source.get("evidence_id")),
                "text": text,
                "span_start": start,
                "span_end": end,
                "resolution_method": method,
            }
        )
        methods[method] += 1
    return resolved, dict(methods)


def tokenizer_count(text: str) -> int:
    from chunking_utils import count_tokens

    return int(count_tokens(text))


def build_gold_overlap_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    roots_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    output: list[dict[str, Any]] = []
    exclusions: dict[str, str] = {}
    resolution_counts: Counter[str] = Counter()
    question_tree_counts: Counter[int] = Counter()
    question_resolution_status: Counter[str] = Counter()
    papers = {doc: reconstruct_paper(payloads) for doc, payloads in roots_by_document.items()}
    for position, raw in enumerate(raw_rows, start=1):
        question_id = str(raw["question_id"])
        document_id = str(raw["document_id"])
        paper_text = papers[document_id]
        resolved, methods = resolve_evidence_rows(
            evidence_by_question[question_id], document_id, paper_text
        )
        resolution_counts.update(methods)
        unique_expected = len({
            normalized_with_offsets(str(row.get("evidence_text", "")).strip())[0]
            for row in evidence_by_question[question_id]
            if normalized_with_offsets(str(row.get("evidence_text", "")).strip())[0]
        })
        if not resolved:
            # Typical examples are QASPER table/figure evidence annotations.
            # Such content is absent from the paper text used for chunking, so
            # no chunk tree can overlap it.  It cannot define a local example.
            exclusions[question_id] = "no_gold_evidence_represented_in_chunk_text"
            question_resolution_status["no_resolved_evidence"] += 1
            continue
        if len(resolved) < unique_expected:
            question_resolution_status["partial_resolved_evidence"] += 1
        else:
            question_resolution_status["all_unique_evidence_resolved"] += 1
        intervals = merge_intervals([(row["span_start"], row["span_end"]) for row in resolved])
        scores = {int(tokens): values for tokens, values in raw["scores_by_tokens"].items()}
        created = 0
        for root in roots_by_document[document_id]:
            root_index = int(root["chunk_idx"])
            root_start, root_end = int(root["span_start"]), int(root["span_end"])
            overlap = clipped_overlap_intervals(intervals, root_start, root_end)
            if not overlap:
                continue
            local_evidence_text = "\n".join(paper_text[start:end] for start, end in overlap)
            local_length = tokenizer_count(local_evidence_text)
            if local_length <= 0:
                exclusions[question_id] = "overlap_had_zero_gpt2_tokens"
                created = 0
                break
            local_scores = local_score_tree(scores, root_index)
            features = extract_local_features(local_scores)
            output.append(
                {
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "split": split,
                    "question_id": question_id,
                    "document_id": document_id,
                    "question_text": str(raw["question_text"]),
                    "root_index": root_index,
                    "root_span_start": root_start,
                    "root_span_end": root_end,
                    "tree_score": tree_score(local_scores),
                    "local_evidence_intervals": [[start, end] for start, end in overlap],
                    "local_evidence_token_length": local_length,
                    "local_oracle_label": closest_chunk_size(local_length),
                    "features": features,
                }
            )
            created += 1
        if created == 0 and question_id not in exclusions:
            exclusions[question_id] = "no_root_overlapped_resolved_evidence"
        question_tree_counts[created] += 1
        if position % 250 == 0:
            print(json.dumps({"event": f"phase5a_{split}_gold_rows", "questions": position, "trees": len(output)}), flush=True)
    if exclusions:
        excluded = set(exclusions)
        output = [row for row in output if row["question_id"] not in excluded]
    summary = {
        "split": split,
        "source_questions": len(raw_rows),
        "eligible_questions": len({row["question_id"] for row in output}),
        "excluded_questions": len(exclusions),
        "training_or_diagnostic_tree_examples": len(output),
        "tree_examples_per_question_distribution": {
            str(count): int(frequency) for count, frequency in sorted(question_tree_counts.items())
        },
        "label_distribution": {
            str(tokens): sum(int(row["local_oracle_label"]) == tokens for row in output)
            for tokens in CLASS_TOKENS
        },
        "span_resolution_counts": dict(sorted(resolution_counts.items())),
        "question_resolution_status": dict(sorted(question_resolution_status.items())),
        "exclusion_reasons": dict(Counter(exclusions.values())),
        "unresolved_item_policy": (
            "Use every uniquely located text-evidence item. Items absent from the chunk corpus "
            "(predominantly FLOAT SELECTED table/figure annotations) cannot overlap a tree and "
            "are audited but do not invalidate other resolved evidence for that question."
        ),
    }
    return output, summary, exclusions


def inference_tree_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        scores = {int(tokens): values for tokens, values in raw["scores_by_tokens"].items()}
        for root_index in range(len(scores[160])):
            local_scores = local_score_tree(scores, root_index)
            rows.append(
                {
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "split": str(raw["split"]),
                    "question_id": str(raw["question_id"]),
                    "document_id": str(raw["document_id"]),
                    "question_text": str(raw["question_text"]),
                    "root_index": root_index,
                    "tree_score": tree_score(local_scores),
                    "features": extract_local_features(local_scores),
                }
            )
    return rows


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[str]]:
    if not rows:
        raise ValueError("No feature rows")
    names = sorted(rows[0]["features"])
    assert_inference_safe_feature_names(names)
    for row in rows:
        if sorted(row["features"]) != names:
            raise RuntimeError("Inconsistent local feature schema")
    return np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows], dtype=np.float32
    ), names


def target_array(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([CLASS_TOKENS.index(int(row["local_oracle_label"])) for row in rows], dtype=np.int64)


def predict_labels(booster: Any, rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    matrix, observed_names = feature_matrix(rows)
    if list(observed_names) != list(names):
        raise RuntimeError("Training/inference local feature schemas differ")
    probabilities = phase3b.predict_booster(booster, matrix, names)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    return predictions, probabilities


def folds_for_tree_rows(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    # Reuse the verified Phase 3B paper-grouped stratification, now over tree rows.
    compatibility_rows = [
        {
            "document_id": str(row["document_id"]),
            # Historical helper name only; this value remains a target and is
            # never included in the feature matrix.
            "oracle_label": int(row["local_oracle_label"]),
        }
        for row in rows
    ]
    return phase3b.grouped_stratified_folds(compatibility_rows, FOLDS, SEED)


def cross_validate_fixed(
    rows: Sequence[Mapping[str, Any]], matrix: np.ndarray, names: Sequence[str], targets: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    folds = folds_for_tree_rows(rows)
    oof_probabilities = np.zeros((len(rows), len(CLASS_TOKENS)), dtype=np.float64)
    fold_results = []
    for fold in range(FOLDS):
        held = folds == fold
        train = ~held
        weights, class_weights = phase3b.class_balance_weights(targets[train])
        booster = phase3b.train_booster(
            matrix[train], targets[train], weights, names, FIXED_CANDIDATE, SEED + fold
        )
        oof_probabilities[held] = phase3b.predict_booster(booster, matrix[held], names)
        predictions = np.argmax(oof_probabilities[held], axis=1)
        fold_results.append(
            {
                "fold": fold,
                "train_tree_examples": int(train.sum()),
                "held_out_tree_examples": int(held.sum()),
                "train_papers": len({str(rows[i]["document_id"]) for i in np.where(train)[0]}),
                "held_out_papers": len({str(rows[i]["document_id"]) for i in np.where(held)[0]}),
                "paper_overlap": sorted(
                    {str(rows[i]["document_id"]) for i in np.where(train)[0]}
                    & {str(rows[i]["document_id"]) for i in np.where(held)[0]}
                ),
                "class_weights": class_weights,
                "metrics": phase3b.classification_metrics(targets[held], predictions, oof_probabilities[held]),
            }
        )
    oof_predictions = np.argmax(oof_probabilities, axis=1).astype(np.int64)
    summary = {
        "folds": FOLDS,
        "grouping": "document_id/source paper",
        "fixed_hyperparameters": FIXED_CANDIDATE,
        "model_selection": "none; inherited frozen Phase 3B primary settings",
        "oof_metrics": phase3b.classification_metrics(targets, oof_predictions, oof_probabilities),
        "fold_results": fold_results,
    }
    if any(result["paper_overlap"] for result in fold_results):
        raise RuntimeError("Paper leakage detected in Phase 5A folds")
    return summary, folds, oof_probabilities


def train_final(rows: Sequence[Mapping[str, Any]]) -> tuple[Any, list[str], dict[str, Any]]:
    matrix, names = feature_matrix(rows)
    targets = target_array(rows)
    cv_summary, folds, oof_probabilities = cross_validate_fixed(rows, matrix, names, targets)
    weights, class_weights = phase3b.class_balance_weights(targets)
    started = time.perf_counter()
    booster = phase3b.train_booster(matrix, targets, weights, names, FIXED_CANDIDATE, SEED)
    fit_seconds = time.perf_counter() - started
    metadata = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT_NAME,
        "classifier": "XGBoost multiclass soft-probability classifier",
        "objective": "multi:softprob",
        "feature_count": len(names),
        "feature_names": names,
        "tree_examples": len(rows),
        "unique_questions": len({row["question_id"] for row in rows}),
        "unique_papers": len({row["document_id"] for row in rows}),
        "fixed_hyperparameters": FIXED_CANDIDATE,
        "class_weights": class_weights,
        "fit_seconds": fit_seconds,
        "cross_validation": cv_summary,
        "fold_assignments": folds.tolist(),
        "oof_probability_sha256": hashlib.sha256(oof_probabilities.tobytes()).hexdigest(),
    }
    return booster, names, metadata


def choose_single_chunk(local_scores: Mapping[int, Sequence[float]], root_index: int, tokens: int) -> dict[str, Any]:
    scores = np.asarray(local_scores[tokens], dtype=np.float64)
    local_index = int(np.argmax(scores))
    global_index = root_index * (160 // tokens) + local_index
    return {
        "tokens": int(tokens),
        "local_chunk_index": local_index,
        "global_chunk_index": global_index,
        "similarity": float(scores[local_index]),
    }


def all_chunks_at_level(
    local_scores: Mapping[int, Sequence[float]], root_index: int, tokens: int
) -> list[dict[str, Any]]:
    """Return every descendant at one predicted level in stable similarity order."""
    scores = np.asarray(local_scores[tokens], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    return [
        {
            "tokens": int(tokens),
            "local_chunk_index": int(local_index),
            "global_chunk_index": root_index * (160 // tokens) + int(local_index),
            "similarity": float(scores[int(local_index)]),
        }
        for local_index in order
    ]


def build_validation_predictions(
    raw_rows: Sequence[Mapping[str, Any]], booster: Any, names: Sequence[str]
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for raw in raw_rows:
        scores = {int(tokens): values for tokens, values in raw["scores_by_tokens"].items()}
        ranked = rank_roots(scores, TOP_K_TREES)
        local_rows = [
            {
                "features": extract_local_features(item["local_scores"]),
                "question_id": raw["question_id"],
                "document_id": raw["document_id"],
            }
            for item in ranked
        ]
        predicted_indices, probabilities = predict_labels(booster, local_rows, names)
        selected_trees = []
        fixed_choices: dict[str, list[dict[str, Any]]] = {str(tokens): [] for tokens in CLASS_TOKENS}
        for rank, (item, class_index, probability) in enumerate(
            zip(ranked, predicted_indices, probabilities), start=1
        ):
            root_index = int(item["root_index"])
            predicted_tokens = CLASS_TOKENS[int(class_index)]
            choice = choose_single_chunk(item["local_scores"], root_index, predicted_tokens)
            selected_trees.append(
                {
                    "tree_rank": rank,
                    "root_index": root_index,
                    "tree_score": float(item["tree_score"]),
                    "predicted_granularity": predicted_tokens,
                    "class_probabilities": {
                        str(tokens): float(probability[index])
                        for index, tokens in enumerate(CLASS_TOKENS)
                    },
                    "selected_chunk": choice,
                    "all_chunks_at_predicted_level": all_chunks_at_level(
                        item["local_scores"], root_index, predicted_tokens
                    ),
                }
            )
            for tokens in CLASS_TOKENS:
                fixed = choose_single_chunk(item["local_scores"], root_index, tokens)
                fixed["tree_rank"] = rank
                fixed["root_index"] = root_index
                fixed["tree_score"] = float(item["tree_score"])
                fixed_choices[str(tokens)].append(fixed)
        predictions.append(
            {
                "phase": PHASE,
                "formulation_version": FORMULATION_VERSION,
                "split": "validation",
                "question_id": str(raw["question_id"]),
                "document_id": str(raw["document_id"]),
                "question_text": str(raw["question_text"]),
                "tree_score_definition": "mean of level means at 10,20,40,80,160",
                "top_n_trees": TOP_K_TREES,
                "selected_trees": selected_trees,
                "same_tree_fixed_granularity_choices": fixed_choices,
                "gold_fields_used": False,
            }
        )
    return predictions


def fetch_selected_chunks(
    client: QdrantClient, predictions: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    identities: set[str] = set()
    for row in predictions:
        document_id = str(row["document_id"])
        for tree in row["selected_trees"]:
            selected = tree["selected_chunk"]
            identities.add(chunk_id(document_id, selected["tokens"], selected["global_chunk_index"]))
            for candidate in tree["all_chunks_at_predicted_level"]:
                identities.add(
                    chunk_id(document_id, candidate["tokens"], candidate["global_chunk_index"])
                )
        for choices in row["same_tree_fixed_granularity_choices"].values():
            for selected in choices:
                identities.add(chunk_id(document_id, selected["tokens"], selected["global_chunk_index"]))
    found: dict[str, dict[str, Any]] = {}
    ordered = sorted(identities)
    for offset in range(0, len(ordered), 256):
        points = client.retrieve(
            collection_name=PAPER_CHUNK_COLLECTION,
            ids=ordered[offset : offset + 256],
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            found[str(point.id)] = dict(point.payload or {})
    missing = set(ordered) - set(found)
    if missing:
        raise RuntimeError(f"Missing selected chunk payloads: {len(missing)}")
    return found


def deduplicated_evidence_text(rows: Sequence[Mapping[str, Any]]) -> str:
    from metrics import normalize_text

    seen: set[str] = set()
    texts: list[str] = []
    for row in rows:
        text = str(row.get("evidence_text", "")).strip()
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        texts.append(text)
    return "\n".join(texts)


def evaluate_retrieval(
    predictions: Sequence[Mapping[str, Any]],
    chunk_payloads: Mapping[str, Mapping[str, Any]],
    evidence_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from metrics import count_tokens, token_precision_recall_f1

    results: list[dict[str, Any]] = []
    methods = ["phase5a", "phase5a_all_predicted_level_chunks"] + [
        f"same_tree_fixed_{tokens}" for tokens in CLASS_TOKENS
    ]
    values: dict[str, list[float]] = {method: [] for method in methods}
    precision_values: dict[str, list[float]] = {method: [] for method in methods}
    recall_values: dict[str, list[float]] = {method: [] for method in methods}
    for row in predictions:
        question_id = str(row["question_id"])
        document_id = str(row["document_id"])
        evidence_text = deduplicated_evidence_text(evidence_by_question[question_id])
        method_choices: dict[str, list[Mapping[str, Any]]] = {
            "phase5a": [tree["selected_chunk"] for tree in row["selected_trees"]],
            "phase5a_all_predicted_level_chunks": [
                chunk
                for tree in row["selected_trees"]
                for chunk in tree["all_chunks_at_predicted_level"]
            ],
        }
        for tokens in CLASS_TOKENS:
            method_choices[f"same_tree_fixed_{tokens}"] = row["same_tree_fixed_granularity_choices"][str(tokens)]
        method_results = {}
        for method, choices in method_choices.items():
            payload_rows = []
            for choice in choices:
                identity = chunk_id(document_id, int(choice["tokens"]), int(choice["global_chunk_index"]))
                payload = chunk_payloads[identity]
                payload_rows.append(
                    {
                        "chunk_id": identity,
                        "chunk_idx": int(payload["chunk_idx"]),
                        "granularity_tokens": int(choice["tokens"]),
                        "span_start": int(payload["span_start"]),
                        "span_end": int(payload["span_end"]),
                        "similarity": float(choice["similarity"]),
                        "content": str(payload["content"]),
                    }
                )
            retrieved_text = "\n".join(item["content"] for item in payload_rows)
            precision, recall, f1 = token_precision_recall_f1(retrieved_text, evidence_text)
            values[method].append(f1)
            precision_values[method].append(precision)
            recall_values[method].append(recall)
            method_results[method] = {
                "precision_joined_top5_trees": precision,
                "recall_joined_top5_trees": recall,
                "f1_joined_top5_trees": f1,
                "retrieved_token_count": count_tokens(retrieved_text),
                "evidence_token_count": count_tokens(evidence_text),
                "selected_chunks": payload_rows,
            }
        results.append(
            {
                "phase": PHASE,
                "question_id": question_id,
                "document_id": document_id,
                "paper_restricted": True,
                "top_n_trees": TOP_K_TREES,
                "one_chunk_per_tree": True,
                "methods": method_results,
            }
        )
    summaries = {
        method: {
            "mean_joined_precision": statistics.fmean(precision_values[method]),
            "mean_joined_recall": statistics.fmean(recall_values[method]),
            "mean_joined_f1": statistics.fmean(values[method]),
            "median_joined_f1": statistics.median(values[method]),
            "mean_selected_chunk_count": statistics.fmean(
                len(row["methods"][method]["selected_chunks"]) for row in results
            ),
            "mean_retrieved_token_count": statistics.fmean(
                int(row["methods"][method]["retrieved_token_count"]) for row in results
            ),
        }
        for method in methods
    }
    return results, {
        "evaluated_questions": len(results),
        "retrieval_coverage": len(results) / VALIDATION_EXPECTED,
        "top_n_trees": TOP_K_TREES,
        "selection_per_tree": "single most similar chunk at predicted/fixed level",
        "paper_restricted": True,
        "methods": summaries,
    }


def paper_cluster_bootstrap(
    results: Sequence[Mapping[str, Any]], comparison: str, replicates: int = 10000
) -> dict[str, Any]:
    by_paper: dict[str, list[float]] = defaultdict(list)
    for row in results:
        difference = (
            float(row["methods"]["phase5a"]["f1_joined_top5_trees"])
            - float(row["methods"][comparison]["f1_joined_top5_trees"])
        )
        by_paper[str(row["document_id"])].append(difference)
    papers = sorted(by_paper)
    rng = np.random.default_rng(SEED)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(papers), size=len(papers))
        differences = [value for index in selected for value in by_paper[papers[int(index)]]]
        draws[replicate] = float(np.mean(differences))
    observed = statistics.fmean(value for values in by_paper.values() for value in values)
    return {
        "comparison": f"phase5a_minus_{comparison}",
        "observed_mean_difference": observed,
        "confidence_interval_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "resampling_unit": "source paper",
        "papers": len(papers),
        "replicates": replicates,
        "seed": SEED,
    }


def write_confusion_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["oracle\\predicted", *CLASS_TOKENS])
        for tokens, row in zip(CLASS_TOKENS, metrics["confusion_matrix"]):
            writer.writerow([tokens, *row])


def environment_summary() -> dict[str, Any]:
    import qdrant_client
    import scipy
    import transformers
    import xgboost

    return {
        "captured_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "environment_name": ".venv-phase5a",
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "xgboost": xgboost.__version__,
        "transformers": transformers.__version__,
        "qdrant_client": getattr(qdrant_client, "__version__", "1.16.1"),
        "device": "CPU",
        "gpu_required": False,
        "qwen_used": False,
        "embeddings_recomputed": False,
    }


def preflight(output_root: Path) -> dict[str, Any]:
    train_path, validation_path = raw_path("train"), raw_path("validation")
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Frozen Phase 3A score arrays are missing")
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    if len(train) != TRAIN_EXPECTED or len(validation) != VALIDATION_EXPECTED:
        raise RuntimeError("Preserved split counts differ from 2245/924")
    train_papers = {str(row["document_id"]) for row in train}
    validation_papers = {str(row["document_id"]) for row in validation}
    overlap = sorted(train_papers & validation_papers)
    if overlap:
        raise RuntimeError(f"Train/validation paper overlap: {overlap[:5]}")
    sample = {int(tokens): values[: min(len(values), 160 // int(tokens))] for tokens, values in train[0]["scores_by_tokens"].items()}
    features = extract_local_features(sample)
    client = qdrant_client()
    try:
        snapshot = collection_snapshot(client)
    finally:
        client.close()
    for required in (PAPER_CHUNK_COLLECTION, PAPER_EVIDENCE_COLLECTION):
        if required not in snapshot:
            raise RuntimeError(f"Missing Qdrant collection: {required}")
    audit = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT_NAME,
        "status": "passed",
        "captured_at": utc_now(),
        "source_score_files": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path), "rows": len(train)},
            "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path), "rows": len(validation)},
        },
        "split_isolation": {
            "train_papers": len(train_papers),
            "validation_papers": len(validation_papers),
            "paper_overlap": overlap,
        },
        "feature_count": len(features),
        "feature_names": sorted(features),
        "qdrant_endpoint": f"{QDRANT_HOST}:{QDRANT_GRPC_PORT}",
        "qdrant_access": "read-only",
        "qdrant_snapshot": snapshot,
        "environment": environment_summary(),
    }
    atomic_json(output_root / "integrity" / "preflight_audit.json", audit)
    return audit


def procedure_lock(output_root: Path, preflight_audit: Mapping[str, Any]) -> dict[str, Any]:
    procedure = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT_NAME,
        "formulation_version": FORMULATION_VERSION,
        "frozen_at": utc_now(),
        "training_unit": "one 160-token-rooted tree overlapping deduplicated gold evidence",
        "tree_definition": {
            "root": 160,
            "aligned_descendants": {"80": 2, "40": 4, "20": 8, "10": 16},
            "last_partial_tree_allowed": True,
        },
        "local_target": {
            "source": "GPT-2 token length of merged gold-evidence intervals clipped to this root",
            "classes": list(CLASS_TOKENS),
            "mapping": "nearest candidate, smaller exact-midpoint tie, clamp extremes",
            "one_label_per_gold-overlap_tree": True,
        },
        "input_features": {
            "count": 173,
            "source": "question-to-chunk cosine similarities within this tree only",
            "gold_evidence_as_input": False,
            "qwen_features": False,
        },
        "classifier": {
            "family": "XGBoost multiclass soft-probability classifier",
            "hyperparameters": FIXED_CANDIDATE,
            "class_weighting": "sqrt(max_class_count / class_count), recalculated on local-tree labels",
            "model_selection": "none; fixed settings inherited from Phase 3B",
            "cross_validation": "5-fold paper-grouped training diagnostic",
        },
        "inference": {
            "tree_score": "average of mean similarities at levels 10,20,40,80,160",
            "top_n_trees": TOP_K_TREES,
            "classifier_application": "independently to each selected tree",
            "primary_chunk_selection": "single most similar chunk at predicted level per tree",
            "exploratory_chunk_selection": (
                "all chunks at predicted level per tree; reported separately with variable chunk count"
            ),
            "negative_class": False,
            "fixed_token_budget": False,
        },
        "evaluation": {
            "primary": "joined GPT-2 token-level Precision/Recall/F1 over five selected chunks",
            "same_tree_fixed_baselines": list(CLASS_TOKENS),
            "exploratory_variant": "retain all descendants at each predicted level",
            "paper_restricted": True,
            "validation_status": "development result because validation was reused in prior phases",
        },
        "prohibited": [
            "validation use for training/model selection/thresholding",
            "Qdrant writes, collection creation, deletion, rebuilding, or re-indexing",
            "embedding or Qwen inference",
            "overwriting any previous experiment",
        ],
        "source_hashes": preflight_audit["source_score_files"],
    }
    procedure["procedure_sha256"] = stable_hash(procedure)
    atomic_json(output_root / "configuration" / "procedure_lock.json", procedure)
    return procedure


def run(output_root: Path) -> dict[str, Any]:
    wall_started = time.perf_counter()
    preflight_audit = preflight(output_root)
    procedure = procedure_lock(output_root, preflight_audit)
    train_raw = read_jsonl(raw_path("train"))
    validation_raw = read_jsonl(raw_path("validation"))
    client = qdrant_client()
    try:
        train_roots = fetch_root_payloads(client, train_raw)
        train_evidence = fetch_evidence_for_questions(client, [str(row["question_id"]) for row in train_raw])
    finally:
        client.close()
    train_rows, train_summary, train_exclusions = build_gold_overlap_rows(
        train_raw, train_roots, train_evidence, "train"
    )
    if len({row["local_oracle_label"] for row in train_rows}) != len(CLASS_TOKENS):
        raise RuntimeError("All five local classes must occur in Phase 5A training")
    train_feature_path = output_root / "features" / "train_gold_overlap_tree_features.jsonl.gz"
    atomic_jsonl(train_feature_path, train_rows, gzip_output=True)
    atomic_json(output_root / "features" / "train_construction_summary.json", train_summary)
    atomic_json(output_root / "integrity" / "train_exclusions.json", train_exclusions)

    booster, names, metadata = train_final(train_rows)
    model_path = output_root / "models" / "tree_local_xgboost.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    metadata["model_path"] = str(model_path)
    metadata["model_sha256"] = sha256_file(model_path)
    atomic_json(output_root / "models" / "metadata.json", metadata)
    atomic_json(output_root / "cross_validation" / "paper_grouped_fixed_procedure.json", metadata["cross_validation"])

    # Validation predictions are constructed only from frozen similarities and
    # the frozen classifier; no validation evidence has been requested yet.
    prediction_started = time.perf_counter()
    validation_predictions = build_validation_predictions(validation_raw, booster, names)
    prediction_seconds = time.perf_counter() - prediction_started
    if len(validation_predictions) != VALIDATION_EXPECTED:
        raise RuntimeError("Phase 5A did not produce 924 validation predictions")
    prediction_path = output_root / "validation" / "predictions_pre_evaluation.jsonl"
    atomic_jsonl(prediction_path, validation_predictions)
    prediction_lock = {
        "locked_at": utc_now(),
        "procedure_sha256": procedure["procedure_sha256"],
        "predictions_path": str(prediction_path),
        "predictions_sha256": sha256_file(prediction_path),
        "predictions": len(validation_predictions),
        "prediction_seconds": prediction_seconds,
        "validation_phase5a_local_gold_evidence_requested_before_lock": False,
        "unused_legacy_question_oracle_present_in_frozen_score_source": True,
        "selected_tree_prediction_distribution": {
            str(tokens): sum(
                int(tree["predicted_granularity"]) == tokens
                for row in validation_predictions
                for tree in row["selected_trees"]
            )
            for tokens in CLASS_TOKENS
        },
    }
    atomic_json(output_root / "validation" / "prediction_lock.json", prediction_lock)

    # Gold validation evidence is accessed only after predictions are durable.
    client = qdrant_client()
    try:
        validation_evidence = fetch_evidence_for_questions(
            client, [str(row["question_id"]) for row in validation_raw]
        )
        selected_chunks = fetch_selected_chunks(client, validation_predictions)
        validation_roots = fetch_root_payloads(client, validation_raw)
    finally:
        client.close()
    retrieval_started = time.perf_counter()
    retrieval_rows, retrieval_summary = evaluate_retrieval(
        validation_predictions, selected_chunks, validation_evidence
    )
    retrieval_seconds = time.perf_counter() - retrieval_started
    retrieval_path = output_root / "retrieval" / "results.jsonl"
    atomic_jsonl(retrieval_path, retrieval_rows)
    retrieval_summary["retrieval_scoring_seconds"] = retrieval_seconds
    retrieval_summary["result_path"] = str(retrieval_path)
    retrieval_summary["result_sha256"] = sha256_file(retrieval_path)
    retrieval_summary["paired_paper_cluster_bootstrap"] = {
        "versus_same_tree_fixed_40": paper_cluster_bootstrap(retrieval_rows, "same_tree_fixed_40"),
        "versus_same_tree_fixed_20": paper_cluster_bootstrap(retrieval_rows, "same_tree_fixed_20"),
    }
    atomic_json(output_root / "retrieval" / "summary.json", retrieval_summary)

    validation_gold_rows, validation_gold_summary, validation_exclusions = build_gold_overlap_rows(
        validation_raw, validation_roots, validation_evidence, "validation"
    )
    validation_matrix, validation_names = feature_matrix(validation_gold_rows)
    if validation_names != list(names):
        raise RuntimeError("Validation diagnostic feature schema differs")
    validation_probabilities = phase3b.predict_booster(booster, validation_matrix, names)
    validation_targets = target_array(validation_gold_rows)
    validation_predicted = np.argmax(validation_probabilities, axis=1).astype(np.int64)
    classification = phase3b.classification_metrics(
        validation_targets, validation_predicted, validation_probabilities
    )
    classification.update(
        {
            "unit": "gold-overlap validation tree (secondary diagnostic)",
            "eligible_questions": validation_gold_summary["eligible_questions"],
            "excluded_questions": validation_gold_summary["excluded_questions"],
        }
    )
    atomic_json(output_root / "classification" / "metrics.json", classification)
    write_confusion_csv(output_root / "classification" / "confusion_matrix.csv", classification)
    atomic_json(output_root / "features" / "validation_gold_overlap_construction_summary.json", validation_gold_summary)
    atomic_json(output_root / "integrity" / "validation_exclusions.json", validation_exclusions)

    client = qdrant_client()
    try:
        final_snapshot = collection_snapshot(client)
    finally:
        client.close()
    qdrant_unchanged = final_snapshot == preflight_audit["qdrant_snapshot"]
    if not qdrant_unchanged:
        raise RuntimeError("Qdrant collection snapshot changed during Phase 5A")
    package_lock = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    atomic_text(output_root / "environment" / "package_lock.txt", package_lock)
    atomic_json(output_root / "environment" / "python_environment.json", environment_summary())
    total_seconds = time.perf_counter() - wall_started
    runtime = {
        "total_wall_seconds": total_seconds,
        "validation_prediction_seconds": prediction_seconds,
        "retrieval_scoring_seconds": retrieval_seconds,
        "gpu_used": False,
        "vast_ai_required": False,
    }
    atomic_json(output_root / "runtime" / "summary.json", runtime)
    final_summary = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT_NAME,
        "formulation_version": FORMULATION_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "train": train_summary,
        "validation_tree_diagnostic": {
            "construction": validation_gold_summary,
            "classification": classification,
        },
        "retrieval": retrieval_summary,
        "model": {
            "family": metadata["classifier"],
            "feature_count": metadata["feature_count"],
            "hyperparameters": FIXED_CANDIDATE,
            "model_sha256": metadata["model_sha256"],
            "validation_selected_tree_prediction_distribution": prediction_lock[
                "selected_tree_prediction_distribution"
            ],
        },
        "methodology": {
            "validation_predictions_locked_before_phase5a_local_gold_evaluation": True,
            "legacy_question_oracle_field_in_source_scores_ignored": True,
            "validation_used_for_model_selection": False,
            "qdrant_read_only": True,
            "qdrant_collections_unchanged": qdrant_unchanged,
            "qwen_used": False,
            "embeddings_recomputed": False,
            "previous_experiments_overwritten": False,
            "validation_status": "development result",
        },
        "runtime": runtime,
        "artifacts": {
            "output_root": str(output_root),
            "report": str(REPORT_ROOT / "experiment_report.md"),
            "results_document": str(DOC_PATH),
        },
    }
    atomic_json(output_root / "final_summary.json", final_summary)
    write_documentation(final_summary, output_root)
    atomic_json(
        output_root / "integrity" / "final_audit.json",
        {
            "status": "passed",
            "completed_at": utc_now(),
            "qdrant_collections_unchanged": qdrant_unchanged,
            "final_qdrant_snapshot": final_snapshot,
            "prediction_lock_sha256": sha256_file(output_root / "validation" / "prediction_lock.json"),
            "final_summary_sha256": sha256_file(output_root / "final_summary.json"),
        },
    )
    return final_summary


def write_documentation(summary: Mapping[str, Any], output_root: Path) -> None:
    train = summary["train"]
    classification = summary["validation_tree_diagnostic"]["classification"]
    retrieval = summary["retrieval"]
    methods = retrieval["methods"]
    label_rows = "\n".join(
        f"| {tokens} | {train['label_distribution'][str(tokens)]} |"
        for tokens in CLASS_TOKENS
    )
    result_rows = "\n".join(
        f"| {name} | {values['mean_joined_precision']:.6f} | {values['mean_joined_recall']:.6f} | {values['mean_joined_f1']:.6f} | {values['median_joined_f1']:.6f} | {values['mean_selected_chunk_count']:.2f} |"
        for name, values in methods.items()
    )
    predicted_distribution = summary["model"]["validation_selected_tree_prediction_distribution"]
    predicted_rows = "\n".join(
        f"| {tokens} | {predicted_distribution[str(tokens)]} |"
        for tokens in CLASS_TOKENS
    )
    content = f"""# {PHASE} — {EXPERIMENT_NAME}

## Objective and protocol

This separate experiment implements Lorenzo's tree-local proposal. A tree is
rooted at one 160-token chunk and contains its aligned 80/40/20/10-token
descendants. Training uses only trees that overlap deduplicated gold evidence.
Each tree's label is the nearest candidate size to the GPT-2 token length of
the merged evidence portion inside that tree (smaller midpoint ties).

Inputs contain only the 173 local similarity-distribution features. Gold
evidence creates training targets but is not an input. Qwen is not used.

At inference, every paper tree is scored by the average of its five level-mean
similarities. The top five trees are classified independently, and the single
most similar chunk at the predicted level is retained from each tree.

## Data construction

- Source train questions: {train['source_questions']}
- Eligible train questions: {train['eligible_questions']}
- Excluded train questions: {train['excluded_questions']}
- Gold-overlap training tree examples: {train['training_or_diagnostic_tree_examples']}

Of the training questions, {train['question_resolution_status'].get('all_unique_evidence_resolved', 0)}
had all unique text evidence located, {train['question_resolution_status'].get('partial_resolved_evidence', 0)}
had a usable text-evidence subset plus one or more unavailable evidence items,
and {train['question_resolution_status'].get('no_resolved_evidence', 0)} were excluded
from supervised tree construction because none of their gold evidence existed
in the chunked paper text. The unresolved items are predominantly QASPER
`FLOAT SELECTED` table/figure annotations, which cannot overlap any tree in
the current text-only chunk collection. They are audited rather than assigned
fabricated spans. Inference and retrieval evaluation still cover all 924
validation questions because they do not require a local gold label.

| Local label | Training trees |
|---:|---:|
{label_rows}

## Results

| Method | Mean precision | Mean recall | Mean joined F1 | Median joined F1 | Mean chunks |
|---|---:|---:|---:|---:|---:|
{result_rows}

The Phase 5A row is the adaptive tree-local classifier. The same-tree fixed
rows use the identical top-five tree ranking and differ only in their fixed
within-tree granularity, isolating the effect of classification.

`phase5a_all_predicted_level_chunks` is the separately labelled exploratory
variant Lorenzo suggested if time allowed. It retains every descendant at the
predicted level, so its number of chunks is variable and it is not the primary
five-chunk result.

| Predicted local granularity | Top-five validation trees |
|---:|---:|
{predicted_rows}

The primary adaptive router's mean joined F1 is
{methods['phase5a']['mean_joined_f1']:.6f}. The directly matched same-tree
fixed-40 result is {methods['same_tree_fixed_40']['mean_joined_f1']:.6f}; the
adaptive-minus-fixed difference is
{retrieval['paired_paper_cluster_bootstrap']['versus_same_tree_fixed_40']['observed_mean_difference']:.6f},
with paired paper-cluster bootstrap 95% CI
[{retrieval['paired_paper_cluster_bootstrap']['versus_same_tree_fixed_40']['confidence_interval_95'][0]:.6f},
{retrieval['paired_paper_cluster_bootstrap']['versus_same_tree_fixed_40']['confidence_interval_95'][1]:.6f}].
Thus the local classifier did not improve over fixed 40 under the identical
tree-ranking and one-chunk-per-tree procedure.

The exploratory all-descendant variant has mean F1
{methods['phase5a_all_predicted_level_chunks']['mean_joined_f1']:.6f}. It raises
mean recall to {methods['phase5a_all_predicted_level_chunks']['mean_joined_recall']:.6f}
but lowers mean precision to
{methods['phase5a_all_predicted_level_chunks']['mean_joined_precision']:.6f},
while selecting {methods['phase5a_all_predicted_level_chunks']['mean_selected_chunk_count']:.2f}
chunks and {methods['phase5a_all_predicted_level_chunks']['mean_retrieved_token_count']:.2f}
tokens on average. This supports retaining only the most similar chunk per
selected tree for the primary procedure.

Secondary tree-label diagnostic: accuracy {classification['accuracy']:.6f},
macro-F1 {classification['macro_f1']:.6f}, weighted F1
{classification['weighted_f1']:.6f}, balanced accuracy
{classification['balanced_accuracy']:.6f}. Its unit is an overlapping tree,
not a question.

## Methodological safeguards

- Train/validation papers remain disjoint.
- Fixed Phase 3B XGBoost settings were reused; no Phase 5A hyperparameter search.
- Validation predictions were saved and hashed before validation evidence was
  requested for final scoring.
- The frozen Phase 3A score file contains an unused legacy question-level
  Oracle field; Phase 5A ignores it. No Phase 5A local validation target or
  validation evidence payload was requested before prediction locking.
- Qdrant was read-only and its before/after collection snapshot was unchanged.
- No embedding inference, Qwen inference, GPU training, or Vast.ai instance was used.
- This is a development result because the preserved validation set has been
  reused across earlier thesis phases.

## Reproduction

```powershell
py -3.10 -m venv .venv-phase5a
.\\.venv-phase5a\\Scripts\\python.exe -m pip install -r requirements-phase5a.txt
.\\.venv-phase5a\\Scripts\\python.exe -m pytest tests/test_similarity_tree_phase5a.py -q
.\\.venv-phase5a\\Scripts\\python.exe similarity_tree_phase5a.py run
```

Artifacts: `{output_root.as_posix()}`.
"""
    atomic_text(DOC_PATH, content)
    atomic_text(REPORT_ROOT / "experiment_report.md", content)


def audit_command(args: argparse.Namespace) -> int:
    print(json.dumps(preflight(args.output_root), indent=2))
    return 0


def run_command(args: argparse.Namespace) -> int:
    print(json.dumps(run(args.output_root), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit").set_defaults(function=audit_command)
    subparsers.add_parser("run").set_defaults(function=run_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
