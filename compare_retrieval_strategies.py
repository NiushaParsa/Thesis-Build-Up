#!/usr/bin/env python
"""Compare validation retrieval strategies without recomputing retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import (
    RETRIEVAL_EVALUATION_COLLECTION,
    ROUTER_DATASET_COLLECTION,
)
from qdrant_schema import get_qdrant_client


FROZEN_ORACLE_HASH = "9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8"
ROUTER_SELECTED_HASH = "314946f7ea8bc61f2f007e7a925cce96bd13c0459b4da60a811304a2a7ab94bb"
VALIDATION_SPLIT = "validation"
FIXED_METHOD = "fixed-sized granularity - separate"
ROUTER_SELECTED_METHOD = "router-selected granularity"
MIXED_RAW_METHOD = "mixed-raw"
MIXED_DEDUP_METHOD = "mixed-deduplicated"
LEVEL_TO_TOKENS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160}
TOKENS_TO_LEVEL = {tokens: level for level, tokens in LEVEL_TO_TOKENS.items()}
DEPLOYABLE_STRATEGIES = [
    "fixed_10",
    "fixed_20",
    "fixed_40",
    "fixed_80",
    "fixed_160",
    "mixed_raw",
    "mixed_deduplicated",
    "router_selected",
]


@dataclass(frozen=True)
class StrategySeries:
    name: str
    display_name: str
    records: Dict[str, dict]
    deployable: bool = True

    @property
    def question_ids(self) -> set[str]:
        return set(self.records)


class MissingStrategyError(RuntimeError):
    """Raised when required validation artifacts are absent or incomplete."""

    def __init__(self, missing: dict):
        super().__init__("Missing required validation strategy outputs")
        self.missing = missing


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latest_jsonl(patterns: Sequence[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(Path(".").glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _filter_rows(
    rows: Iterable[dict],
    *,
    split: str,
    config_hash: Optional[str] = None,
    method_name: Optional[str] = None,
) -> List[dict]:
    result = []
    for row in rows:
        if row.get("split") != split:
            continue
        if config_hash and row.get("evaluation_config_hash") != config_hash:
            continue
        if method_name and row.get("method_name") != method_name:
            continue
        result.append(row)
    return result


def _scroll_qdrant_payloads(
    *,
    collection: str,
    split: str,
    config_hash: Optional[str],
    method_name: Optional[str] = None,
) -> List[dict]:
    must = [FieldCondition(key="split", match=MatchValue(value=split))]
    if config_hash:
        must.append(
            FieldCondition(
                key="evaluation_config_hash", match=MatchValue(value=config_hash)
            )
        )
    if method_name:
        must.append(
            FieldCondition(key="method_name", match=MatchValue(value=method_name))
        )
    client = get_qdrant_client()
    rows = []
    offset = None
    try:
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=must),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            rows.extend(point.payload or {} for point in points)
            if next_offset is None:
                break
            offset = next_offset
    finally:
        client.close()
    return rows


def load_records(
    *,
    jsonl_path: Optional[Path],
    jsonl_patterns: Sequence[str],
    collection: str,
    split: str,
    config_hash: Optional[str],
    method_name: Optional[str],
    source: str,
) -> Tuple[List[dict], str]:
    if source in {"auto", "jsonl"}:
        path = jsonl_path or latest_jsonl(jsonl_patterns)
        if path:
            rows = _filter_rows(
                read_jsonl(path),
                split=split,
                config_hash=config_hash,
                method_name=method_name,
            )
            if rows or source == "jsonl":
                return rows, str(path)
    if source in {"auto", "qdrant"}:
        rows = _scroll_qdrant_payloads(
            collection=collection,
            split=split,
            config_hash=config_hash,
            method_name=method_name,
        )
        return rows, f"qdrant:{collection}"
    return [], "missing"


def fixed_series(records: Sequence[dict]) -> Dict[str, StrategySeries]:
    by_level: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for row in records:
        level = int(row["granularity_level"])
        by_level[level][row["question_id"]] = row
    return {
        f"fixed_{LEVEL_TO_TOKENS[level]}": StrategySeries(
            name=f"fixed_{LEVEL_TO_TOKENS[level]}",
            display_name=f"fixed separate {LEVEL_TO_TOKENS[level]} tokens",
            records=by_level.get(level, {}),
        )
        for level in sorted(LEVEL_TO_TOKENS)
    }


def oracle_series(router_rows: Sequence[dict]) -> StrategySeries:
    records = {}
    for row in router_rows:
        target_level = int(row["router_target_granularity"])
        metrics = {
            int(item["granularity_level"]): item
            for item in row.get("per_granularity_metrics", [])
        }
        if target_level not in metrics:
            continue
        selected = metrics[target_level]
        records[row["question_id"]] = {
            "question_id": row["question_id"],
            "document_id": row.get("document_id"),
            "split": row.get("split"),
            "strategy": "fixed_separate_oracle_upper_bound",
            "granularity_level": target_level,
            "granularity_tokens": LEVEL_TO_TOKENS.get(target_level),
            "f1_joined_topk": selected.get("f1_joined_topk"),
            "mean_max_evidence_similarity_topk": selected.get(
                "mean_max_evidence_similarity_topk"
            ),
            "best_query_similarity_topk": selected.get("best_query_similarity_topk"),
            "mean_query_similarity_topk": selected.get("mean_query_similarity_topk"),
            "router_target_granularity": target_level,
            "best_granularity_by_f1": row.get("best_granularity_by_f1"),
            "best_granularity_by_evidence_similarity": row.get(
                "best_granularity_by_evidence_similarity"
            ),
        }
    return StrategySeries(
        name="oracle_upper_bound",
        display_name="fixed-separate oracle upper bound",
        records=records,
        deployable=False,
    )


def single_record_series(
    records: Sequence[dict], name: str, display_name: str
) -> StrategySeries:
    return StrategySeries(
        name=name,
        display_name=display_name,
        records={row["question_id"]: row for row in records},
    )


def detect_missing(
    strategies: Dict[str, StrategySeries],
    *,
    required_names: Sequence[str],
    reference_name: str = "oracle_upper_bound",
) -> dict:
    missing = {}
    reference_ids = strategies.get(reference_name, StrategySeries("", "", {})).question_ids
    for name in required_names:
        series = strategies.get(name)
        if series is None or not series.records:
            missing[name] = {
                "status": "missing",
                "expected_question_count": len(reference_ids),
                "available_question_count": 0,
            }
            continue
        missing_ids = sorted(reference_ids - series.question_ids)
        extra_ids = sorted(series.question_ids - reference_ids)
        if missing_ids:
            missing[name] = {
                "status": "incomplete",
                "expected_question_count": len(reference_ids),
                "available_question_count": len(series.records),
                "missing_question_count": len(missing_ids),
                "missing_question_ids_sample": missing_ids[:20],
                "extra_question_count": len(extra_ids),
            }
    return missing


def common_question_ids(strategies: Dict[str, StrategySeries]) -> List[str]:
    non_empty = [series.question_ids for series in strategies.values() if series.records]
    if not non_empty:
        return []
    return sorted(set.intersection(*non_empty))


def _finite(values: Iterable[Any]) -> List[float]:
    result = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(value):
            result.append(float(value))
    return result


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 13,
    confidence: float = 0.95,
) -> dict:
    import random

    clean = list(values)
    if not clean:
        return {"mean": None, "ci_low": None, "ci_high": None, "iterations": iterations}
    rng = random.Random(seed)
    means = []
    count = len(clean)
    for _ in range(iterations):
        sample = [clean[rng.randrange(count)] for _ in range(count)]
        means.append(sum(sample) / count)
    means.sort()
    alpha = (1 - confidence) / 2
    low_index = max(0, min(len(means) - 1, int(alpha * iterations)))
    high_index = max(0, min(len(means) - 1, int((1 - alpha) * iterations) - 1))
    return {
        "mean": sum(clean) / len(clean),
        "ci_low": means[low_index],
        "ci_high": means[high_index],
        "iterations": iterations,
        "seed": seed,
    }


def paired_bootstrap_diff_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 13,
) -> dict:
    if len(left) != len(right):
        raise ValueError("Paired bootstrap inputs must have equal length")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    result = bootstrap_mean_ci(differences, iterations=iterations, seed=seed)
    result["mean_difference"] = result.pop("mean")
    return result


def summarize_strategy(
    series: StrategySeries,
    question_ids: Sequence[str],
    oracle: Optional[StrategySeries] = None,
) -> dict:
    rows = [series.records[qid] for qid in question_ids if qid in series.records]
    f1 = _finite(row.get("f1_joined_topk") for row in rows)
    evidence = _finite(row.get("mean_max_evidence_similarity_topk") for row in rows)
    query = _finite(row.get("mean_query_similarity_topk") for row in rows)
    retrieval_latency = _finite(
        row.get("retrieval_latency_ms", row.get("retrieval_time_ms")) for row in rows
    )
    total_latency = _finite(row.get("total_latency_ms") for row in rows)
    regret_values = []
    if oracle:
        for qid in question_ids:
            if qid in series.records and qid in oracle.records:
                regret_values.append(
                    float(oracle.records[qid]["f1_joined_topk"])
                    - float(series.records[qid]["f1_joined_topk"])
                )
    return {
        "strategy": series.name,
        "display_name": series.display_name,
        "deployable": series.deployable,
        "n_questions": len(rows),
        "mean_f1": statistics.mean(f1) if f1 else None,
        "median_f1": statistics.median(f1) if f1 else None,
        "std_f1": statistics.stdev(f1) if len(f1) > 1 else 0.0,
        "mean_evidence_similarity": statistics.mean(evidence) if evidence else None,
        "median_evidence_similarity": statistics.median(evidence) if evidence else None,
        "mean_query_similarity": statistics.mean(query) if query else None,
        "median_query_similarity": statistics.median(query) if query else None,
        "mean_retrieval_latency_ms": (
            statistics.mean(retrieval_latency) if retrieval_latency else None
        ),
        "median_retrieval_latency_ms": (
            statistics.median(retrieval_latency) if retrieval_latency else None
        ),
        "mean_total_latency_ms": statistics.mean(total_latency) if total_latency else None,
        "median_total_latency_ms": (
            statistics.median(total_latency) if total_latency else None
        ),
        "mean_regret_vs_oracle": (
            statistics.mean(regret_values) if regret_values else None
        ),
        "median_regret_vs_oracle": (
            statistics.median(regret_values) if regret_values else None
        ),
    }


def win_tie_loss(
    left: StrategySeries,
    right: StrategySeries,
    question_ids: Sequence[str],
    *,
    epsilon: float = 1e-12,
) -> dict:
    wins = ties = losses = 0
    for qid in question_ids:
        delta = (
            float(left.records[qid]["f1_joined_topk"])
            - float(right.records[qid]["f1_joined_topk"])
        )
        if abs(delta) <= epsilon:
            ties += 1
        elif delta > 0:
            wins += 1
        else:
            losses += 1
    return {"wins": wins, "ties": ties, "losses": losses}


def strategy_values(series: StrategySeries, question_ids: Sequence[str]) -> List[float]:
    return [float(series.records[qid]["f1_joined_topk"]) for qid in question_ids]


def per_question_rows(
    strategies: Dict[str, StrategySeries],
    question_ids: Sequence[str],
    *,
    oracle_name: str = "oracle_upper_bound",
) -> List[dict]:
    rows = []
    oracle = strategies[oracle_name]
    for qid in question_ids:
        row = {"question_id": qid}
        for name, series in strategies.items():
            record = series.records[qid]
            row[f"{name}_f1"] = record.get("f1_joined_topk")
            if name != oracle_name:
                row[f"{name}_delta_vs_oracle"] = (
                    float(record.get("f1_joined_topk"))
                    - float(oracle.records[qid].get("f1_joined_topk"))
                )
        deployable = [
            (name, strategies[name].records[qid].get("f1_joined_topk"))
            for name in DEPLOYABLE_STRATEGIES
            if name in strategies
        ]
        best_name, _ = max(deployable, key=lambda item: (item[1], item[0]))
        row["best_deployable_strategy"] = best_name
        rows.append(row)
    return rows


def router_selected_diagnostics(
    router: StrategySeries, oracle: StrategySeries, question_ids: Sequence[str]
) -> dict:
    predicted = Counter(
        router.records[qid].get("predicted_granularity_tokens") for qid in question_ids
    )
    oracle_targets = Counter(
        oracle.records[qid].get("router_target_granularity") for qid in question_ids
    )
    matches = [
        router.records[qid].get("router_oracle_match")
        for qid in question_ids
        if router.records[qid].get("router_oracle_match") is not None
    ]
    classes = [10, 20, 40, 80, 160]
    matrix = {str(row): {str(col): 0 for col in classes} for row in classes}
    for qid in question_ids:
        actual_level = oracle.records[qid].get("router_target_granularity")
        actual_tokens = LEVEL_TO_TOKENS.get(int(actual_level)) if actual_level else None
        predicted_tokens = router.records[qid].get("predicted_granularity_tokens")
        if actual_tokens in classes and predicted_tokens in classes:
            matrix[str(actual_tokens)][str(predicted_tokens)] += 1
    router_latency = _finite(
        router.records[qid].get("router_latency_ms") for qid in question_ids
    )
    return {
        "predicted_granularity_distribution": dict(predicted),
        "oracle_target_distribution": dict(oracle_targets),
        "router_oracle_match_rate": (
            sum(1 for item in matches if item) / len(matches) if matches else None
        ),
        "confusion_matrix_tokens": matrix,
        "mean_router_latency_ms": statistics.mean(router_latency)
        if router_latency
        else None,
    }


def mixed_diagnostics(series: StrategySeries, question_ids: Sequence[str]) -> dict:
    composition = Counter()
    dominant = Counter()
    for qid in question_ids:
        record = series.records[qid]
        counts = {
            int(tokens): int(count)
            for tokens, count in (record.get("granularity_counts") or {}).items()
        }
        composition.update(counts)
        if counts:
            dominant_tokens, _ = max(counts.items(), key=lambda item: (item[1], -item[0]))
            dominant[dominant_tokens] += 1
    return {
        "topk_granularity_composition": dict(composition),
        "dominant_retrieved_granularity_distribution": dict(dominant),
    }


def build_comparison(
    strategies: Dict[str, StrategySeries],
    *,
    iterations: int,
    seed: int,
) -> dict:
    qids = common_question_ids(strategies)
    oracle = strategies["oracle_upper_bound"]
    metrics = {
        name: summarize_strategy(series, qids, oracle=oracle)
        for name, series in strategies.items()
    }
    for name, series in strategies.items():
        metrics[name]["mean_f1_ci95"] = bootstrap_mean_ci(
            strategy_values(series, qids), iterations=iterations, seed=seed
        )
    for name, series in strategies.items():
        if name == "router_selected":
            continue
        if "router_selected" in strategies:
            metrics[name]["win_tie_loss_vs_router_selected"] = win_tie_loss(
                series, strategies["router_selected"], qids
            )
    if "mixed_raw" in strategies and "mixed_deduplicated" in strategies:
        metrics["mixed_raw"]["win_tie_loss_vs_mixed_deduplicated"] = win_tie_loss(
            strategies["mixed_raw"], strategies["mixed_deduplicated"], qids
        )
        metrics["mixed_deduplicated"]["win_tie_loss_vs_mixed_raw"] = win_tie_loss(
            strategies["mixed_deduplicated"], strategies["mixed_raw"], qids
        )

    deployable_metrics = {
        name: item
        for name, item in metrics.items()
        if strategies[name].deployable and item["mean_f1"] is not None
    }
    best_fixed = max(
        (metrics[name] for name in metrics if name.startswith("fixed_")),
        key=lambda item: item["mean_f1"],
    )
    best_deployable = max(deployable_metrics.values(), key=lambda item: item["mean_f1"])
    bootstrap = {
        "mean_f1_ci95": {
            name: item["mean_f1_ci95"] for name, item in metrics.items()
        },
        "paired_differences": {},
    }
    if "router_selected" in strategies:
        router_values = strategy_values(strategies["router_selected"], qids)
        bootstrap["paired_differences"][
            "router_selected_minus_best_fixed_single_level"
        ] = paired_bootstrap_diff_ci(
            router_values,
            strategy_values(strategies[best_fixed["strategy"]], qids),
            iterations=iterations,
            seed=seed,
        )
        for mixed_name in ("mixed_raw", "mixed_deduplicated"):
            if mixed_name in strategies:
                bootstrap["paired_differences"][
                    f"router_selected_minus_{mixed_name}"
                ] = paired_bootstrap_diff_ci(
                    router_values,
                    strategy_values(strategies[mixed_name], qids),
                    iterations=iterations,
                    seed=seed,
                )
    for mixed_name in ("mixed_raw", "mixed_deduplicated"):
        if mixed_name in strategies:
            bootstrap["paired_differences"][
                f"{mixed_name}_minus_best_fixed_single_level"
            ] = paired_bootstrap_diff_ci(
                strategy_values(strategies[mixed_name], qids),
                strategy_values(strategies[best_fixed["strategy"]], qids),
                iterations=iterations,
                seed=seed,
            )
    diagnostics = {
        "router_selected": (
            router_selected_diagnostics(
                strategies["router_selected"], oracle, qids
            )
            if "router_selected" in strategies
            else None
        ),
        "mixed_raw": (
            mixed_diagnostics(strategies["mixed_raw"], qids)
            if "mixed_raw" in strategies
            else None
        ),
        "mixed_deduplicated": (
            mixed_diagnostics(strategies["mixed_deduplicated"], qids)
            if "mixed_deduplicated" in strategies
            else None
        ),
        "best_deployable_strategy_per_question_distribution": dict(
            Counter(row["best_deployable_strategy"] for row in per_question_rows(strategies, qids))
        ),
    }
    return {
        "question_ids": qids,
        "strategy_metrics": metrics,
        "bootstrap_results": bootstrap,
        "diagnostics": diagnostics,
        "best_fixed_single_level": best_fixed["strategy"],
        "best_average_deployable_strategy": best_deployable["strategy"],
        "oracle_upper_bound_gap": (
            metrics["oracle_upper_bound"]["mean_f1"] - best_deployable["mean_f1"]
        ),
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    *,
    strategies: Dict[str, StrategySeries],
    comparison: dict,
    sources: dict,
    unavailable: Optional[dict] = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_missing_report = output_dir / "missing_inputs.json"
    if not unavailable and stale_missing_report.exists():
        stale_missing_report.unlink()
    qids = comparison["question_ids"]
    metrics = comparison["strategy_metrics"]
    table_rows = []
    for name, item in metrics.items():
        ci = item.get("mean_f1_ci95", {})
        row = {
            "strategy": name,
            "display_name": item["display_name"],
            "deployable": item["deployable"],
            "n_questions": item["n_questions"],
            "mean_f1": item["mean_f1"],
            "median_f1": item["median_f1"],
            "std_f1": item["std_f1"],
            "mean_f1_ci95_low": ci.get("ci_low"),
            "mean_f1_ci95_high": ci.get("ci_high"),
            "mean_evidence_similarity": item["mean_evidence_similarity"],
            "median_evidence_similarity": item["median_evidence_similarity"],
            "mean_query_similarity": item["mean_query_similarity"],
            "median_query_similarity": item["median_query_similarity"],
            "mean_retrieval_latency_ms": item["mean_retrieval_latency_ms"],
            "median_retrieval_latency_ms": item["median_retrieval_latency_ms"],
            "mean_total_latency_ms": item["mean_total_latency_ms"],
            "median_total_latency_ms": item["median_total_latency_ms"],
            "mean_regret_vs_oracle": item["mean_regret_vs_oracle"],
            "median_regret_vs_oracle": item["median_regret_vs_oracle"],
        }
        table_rows.append(row)
    write_csv(output_dir / "comparison_table.csv", table_rows)
    write_csv(output_dir / "per_question_comparison.csv", per_question_rows(strategies, qids))
    (output_dir / "strategy_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "bootstrap_results.json").write_text(
        json.dumps(comparison["bootstrap_results"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = render_summary(
        table_rows=table_rows,
        comparison=comparison,
        diagnostics=comparison["diagnostics"],
        sources=sources,
        unavailable=unavailable or {},
    )
    (output_dir / "comparison_summary.md").write_text(summary, encoding="utf-8")
    maybe_write_plots(output_dir, table_rows, comparison)
    return {
        "comparison_summary": str(output_dir / "comparison_summary.md"),
        "comparison_table": str(output_dir / "comparison_table.csv"),
        "per_question_comparison": str(output_dir / "per_question_comparison.csv"),
        "strategy_metrics": str(output_dir / "strategy_metrics.json"),
        "bootstrap_results": str(output_dir / "bootstrap_results.json"),
    }


def render_summary(
    *,
    table_rows: Sequence[dict],
    comparison: dict,
    diagnostics: dict,
    sources: dict,
    unavailable: dict,
) -> str:
    lines = [
        "# Final validation retrieval comparison",
        "",
        "Split: validation. The test split is not loaded or evaluated by this report.",
        "",
    ]
    if unavailable:
        lines.extend(
            [
                "## Unavailable or incomplete strategies",
                "",
                "The report was generated in partial mode. Missing strategies are not fabricated.",
                "",
                "```json",
                json.dumps(unavailable, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Main results",
            "",
            f"Common validation questions: {len(comparison['question_ids'])}",
            f"Best fixed single level: `{comparison['best_fixed_single_level']}`",
            f"Best average deployable strategy: `{comparison['best_average_deployable_strategy']}`",
            f"Oracle upper-bound gap for the best deployable strategy: {comparison['oracle_upper_bound_gap']:.6f}",
            "",
            "| strategy | deployable | n | mean F1 | 95% CI | median F1 | mean regret vs oracle |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in table_rows:
        ci = f"[{row['mean_f1_ci95_low']:.6f}, {row['mean_f1_ci95_high']:.6f}]"
        regret = row["mean_regret_vs_oracle"]
        lines.append(
            f"| {row['strategy']} | {row['deployable']} | {row['n_questions']} | "
            f"{row['mean_f1']:.6f} | {ci} | {row['median_f1']:.6f} | "
            f"{regret:.6f} |"
        )
    lines.extend(["", "## Router-selected diagnostics", ""])
    lines.append("```json")
    lines.append(json.dumps(diagnostics.get("router_selected"), indent=2, sort_keys=True))
    lines.append("```")
    lines.extend(["", "## Mixed diagnostics", ""])
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "mixed_raw": diagnostics.get("mixed_raw"),
                "mixed_deduplicated": diagnostics.get("mixed_deduplicated"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    lines.append("```")
    lines.extend(["", "## Sources", "", "```json", json.dumps(sources, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def maybe_write_plots(output_dir: Path, table_rows: Sequence[dict], comparison: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = [row["strategy"] for row in table_rows]
    means = [row["mean_f1"] for row in table_rows]
    lows = [row["mean_f1"] - row["mean_f1_ci95_low"] for row in table_rows]
    highs = [row["mean_f1_ci95_high"] - row["mean_f1"] for row in table_rows]
    plt.figure(figsize=(max(8, len(names) * 0.9), 5))
    plt.bar(names, means, yerr=[lows, highs], capsize=3)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean joined top-K F1")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_f1_ci.png")
    plt.close()

    # Other artifacts contain the full regret and granularity distributions.


def build_strategies_from_inputs(
    *,
    fixed_rows: Sequence[dict],
    router_dataset_rows: Sequence[dict],
    router_selected_rows: Sequence[dict],
    mixed_raw_rows: Sequence[dict],
    mixed_deduplicated_rows: Sequence[dict],
) -> Dict[str, StrategySeries]:
    strategies = fixed_series(fixed_rows)
    strategies["oracle_upper_bound"] = oracle_series(router_dataset_rows)
    strategies["router_selected"] = single_record_series(
        router_selected_rows, "router_selected", "router-selected granularity"
    )
    if mixed_raw_rows:
        strategies["mixed_raw"] = single_record_series(
            mixed_raw_rows, "mixed_raw", "mixed-granularity raw"
        )
    if mixed_deduplicated_rows:
        strategies["mixed_deduplicated"] = single_record_series(
            mixed_deduplicated_rows,
            "mixed_deduplicated",
            "mixed-granularity deduplicated",
        )
    return strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=VALIDATION_SPLIT, choices=[VALIDATION_SPLIT])
    parser.add_argument("--fixed-config-hash", default=FROZEN_ORACLE_HASH)
    parser.add_argument("--router-selected-config-hash", default=ROUTER_SELECTED_HASH)
    parser.add_argument("--source", choices=["auto", "jsonl", "qdrant"], default="auto")
    parser.add_argument("--fixed-jsonl", type=Path, default=None)
    parser.add_argument("--router-dataset-jsonl", type=Path, default=None)
    parser.add_argument("--router-selected-jsonl", type=Path, default=None)
    parser.add_argument("--mixed-raw-jsonl", type=Path, default=None)
    parser.add_argument("--mixed-deduplicated-jsonl", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/final_validation_comparison"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument(
        "--allow-missing-strategies",
        action="store_true",
        help="Write a partial report instead of failing when mixed validation outputs are absent.",
    )
    return parser.parse_args()


def load_all_inputs(args: argparse.Namespace) -> Tuple[Dict[str, StrategySeries], dict]:
    fixed_rows, fixed_source = load_records(
        jsonl_path=args.fixed_jsonl,
        jsonl_patterns=["outputs/oracle_frozen/validation/RetrievalEvalFixedSeparate_*.jsonl"],
        collection=RETRIEVAL_EVALUATION_COLLECTION,
        split=args.split,
        config_hash=args.fixed_config_hash,
        method_name=FIXED_METHOD,
        source=args.source,
    )
    router_dataset_rows, router_dataset_source = load_records(
        jsonl_path=args.router_dataset_jsonl,
        jsonl_patterns=["outputs/oracle_frozen/validation/RouterDataset_*.jsonl"],
        collection=ROUTER_DATASET_COLLECTION,
        split=args.split,
        config_hash=args.fixed_config_hash,
        method_name=None,
        source=args.source,
    )
    router_selected_rows, router_selected_source = load_records(
        jsonl_path=args.router_selected_jsonl,
        jsonl_patterns=[
            "outputs/router_selected/validation/RetrievalEvalRouterSelected_*.jsonl"
        ],
        collection=RETRIEVAL_EVALUATION_COLLECTION,
        split=args.split,
        config_hash=args.router_selected_config_hash,
        method_name=ROUTER_SELECTED_METHOD,
        source=args.source,
    )
    mixed_raw_rows, mixed_raw_source = load_records(
        jsonl_path=args.mixed_raw_jsonl,
        jsonl_patterns=["outputs/**/RetrievalEvalMixedRaw_*.jsonl"],
        collection=RETRIEVAL_EVALUATION_COLLECTION,
        split=args.split,
        config_hash=None,
        method_name=MIXED_RAW_METHOD,
        source=args.source,
    )
    mixed_dedup_rows, mixed_dedup_source = load_records(
        jsonl_path=args.mixed_deduplicated_jsonl,
        jsonl_patterns=["outputs/**/RetrievalEvalMixedDeduplicated_*.jsonl"],
        collection=RETRIEVAL_EVALUATION_COLLECTION,
        split=args.split,
        config_hash=None,
        method_name=MIXED_DEDUP_METHOD,
        source=args.source,
    )
    strategies = build_strategies_from_inputs(
        fixed_rows=fixed_rows,
        router_dataset_rows=router_dataset_rows,
        router_selected_rows=router_selected_rows,
        mixed_raw_rows=mixed_raw_rows,
        mixed_deduplicated_rows=mixed_dedup_rows,
    )
    sources = {
        "fixed_separate": fixed_source,
        "router_dataset_oracle": router_dataset_source,
        "router_selected": router_selected_source,
        "mixed_raw": mixed_raw_source,
        "mixed_deduplicated": mixed_dedup_source,
    }
    return strategies, sources


def write_missing_report(output_dir: Path, missing: dict, sources: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "missing_required_validation_outputs",
        "missing": missing,
        "sources_checked": sources,
        "note": "No expensive retrieval was recomputed. Mixed strategy numbers are not fabricated.",
    }
    (output_dir / "missing_inputs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = [
        "# Missing validation comparison inputs",
        "",
        "The final comparison was stopped because required validation outputs are absent or incomplete.",
        "No test split was read, and no retrieval was recomputed.",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path = output_dir / "comparison_summary.md"
    path.write_text("\n".join(markdown), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    strategies, sources = load_all_inputs(args)
    required = [
        "fixed_10",
        "fixed_20",
        "fixed_40",
        "fixed_80",
        "fixed_160",
        "oracle_upper_bound",
        "router_selected",
        "mixed_raw",
        "mixed_deduplicated",
    ]
    missing = detect_missing(strategies, required_names=required)
    if missing and not args.allow_missing_strategies:
        path = write_missing_report(args.output_dir, missing, sources)
        print(
            json.dumps(
                {
                    "status": "missing_required_validation_outputs",
                    "summary": str(path),
                    "missing": missing,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if missing:
        strategies = {
            name: series
            for name, series in strategies.items()
            if name not in missing and series.records
        }
    comparison = build_comparison(
        strategies,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    artifacts = write_outputs(
        args.output_dir,
        strategies=strategies,
        comparison=comparison,
        sources=sources,
        unavailable=missing,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "common_validation_questions": len(comparison["question_ids"]),
                "included_strategies": list(strategies),
                "unavailable": missing,
                "best_average_deployable_strategy": comparison[
                    "best_average_deployable_strategy"
                ],
                "oracle_upper_bound_gap": comparison["oracle_upper_bound_gap"],
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
