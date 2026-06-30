from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compare_retrieval_strategies as compare


def fixed_record(question_id: str, level: int, f1: float) -> dict:
    return {
        "question_id": question_id,
        "split": "validation",
        "method_name": compare.FIXED_METHOD,
        "evaluation_config_hash": "fixed-hash",
        "granularity_level": level,
        "granularity_tokens": compare.LEVEL_TO_TOKENS[level],
        "f1_joined_topk": f1,
        "mean_max_evidence_similarity_topk": f1 + 0.1,
        "mean_query_similarity_topk": 0.5,
        "retrieval_latency_ms": 10.0,
    }


def router_dataset_record(question_id: str, target_level: int = 3) -> dict:
    return {
        "question_id": question_id,
        "document_id": f"doc-{question_id}",
        "split": "validation",
        "evaluation_config_hash": "fixed-hash",
        "router_target_granularity": target_level,
        "best_granularity_by_f1": target_level,
        "best_granularity_by_evidence_similarity": 4,
        "per_granularity_metrics": [
            {
                "granularity_level": level,
                "f1_joined_topk": 0.1 * level,
                "mean_max_evidence_similarity_topk": 0.2 * level,
                "mean_query_similarity_topk": 0.3,
                "best_query_similarity_topk": 0.4,
            }
            for level in range(1, 6)
        ],
    }


def single_record(question_id: str, method: str, f1: float) -> dict:
    return {
        "question_id": question_id,
        "split": "validation",
        "method_name": method,
        "evaluation_config_hash": "hash",
        "f1_joined_topk": f1,
        "mean_max_evidence_similarity_topk": f1,
        "mean_query_similarity_topk": 0.1,
        "retrieval_latency_ms": 5.0,
        "total_latency_ms": 6.0,
        "granularity_counts": {"10": 2, "20": 3},
        "predicted_granularity_tokens": 40,
        "router_oracle_match": True,
    }


class ComparisonUtilityTests(unittest.TestCase):
    def test_common_question_alignment_uses_intersection(self):
        strategies = {
            "a": compare.StrategySeries("a", "A", {"q1": {}, "q2": {}}),
            "b": compare.StrategySeries("b", "B", {"q2": {}, "q3": {}}),
            "c": compare.StrategySeries("c", "C", {"q2": {}, "q4": {}}),
        }
        self.assertEqual(compare.common_question_ids(strategies), ["q2"])

    def test_oracle_upper_bound_extraction_uses_router_target_metrics(self):
        series = compare.oracle_series([router_dataset_record("q1", target_level=4)])
        self.assertIn("q1", series.records)
        self.assertFalse(series.deployable)
        self.assertEqual(series.records["q1"]["granularity_level"], 4)
        self.assertEqual(series.records["q1"]["granularity_tokens"], 80)
        self.assertAlmostEqual(series.records["q1"]["f1_joined_topk"], 0.4)

    def test_missing_strategy_detection_reports_absent_and_incomplete(self):
        strategies = {
            "oracle_upper_bound": compare.StrategySeries(
                "oracle_upper_bound", "oracle", {"q1": {}, "q2": {}}
            ),
            "mixed_raw": compare.StrategySeries("mixed_raw", "mixed", {}),
            "router_selected": compare.StrategySeries(
                "router_selected", "router", {"q1": {}}
            ),
        }
        missing = compare.detect_missing(
            strategies, required_names=["mixed_raw", "router_selected"]
        )
        self.assertEqual(missing["mixed_raw"]["status"], "missing")
        self.assertEqual(missing["router_selected"]["status"], "incomplete")
        self.assertEqual(missing["router_selected"]["missing_question_count"], 1)

    def test_bootstrap_output_is_reproducible_with_fixed_seed(self):
        first = compare.bootstrap_mean_ci([0.1, 0.2, 0.3], iterations=100, seed=7)
        second = compare.bootstrap_mean_ci([0.1, 0.2, 0.3], iterations=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"mean", "ci_low", "ci_high", "iterations", "seed"})

    def test_parser_rejects_test_split(self):
        with patch("sys.argv", ["compare_retrieval_strategies.py", "--split", "test"]):
            with self.assertRaises(SystemExit):
                compare.parse_args()

    def test_jsonl_loader_filters_validation_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            rows = [
                {
                    "question_id": "q1",
                    "split": "validation",
                    "evaluation_config_hash": "wanted",
                    "method_name": "method",
                },
                {
                    "question_id": "q2",
                    "split": "test",
                    "evaluation_config_hash": "wanted",
                    "method_name": "method",
                },
                {
                    "question_id": "q3",
                    "split": "validation",
                    "evaluation_config_hash": "other",
                    "method_name": "method",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            loaded, source = compare.load_records(
                jsonl_path=path,
                jsonl_patterns=[],
                collection="unused",
                split="validation",
                config_hash="wanted",
                method_name="method",
                source="jsonl",
            )
        self.assertEqual(source, str(path))
        self.assertEqual([row["question_id"] for row in loaded], ["q1"])


class ComparisonBuildTests(unittest.TestCase):
    def test_build_comparison_outputs_expected_shapes(self):
        qids = ["q1", "q2"]
        fixed_rows = [
            fixed_record(qid, level, 0.1 * level)
            for qid in qids
            for level in range(1, 6)
        ]
        router_rows = [router_dataset_record(qid, 5) for qid in qids]
        router_selected = [
            single_record(qid, compare.ROUTER_SELECTED_METHOD, 0.35) for qid in qids
        ]
        mixed_raw = [single_record(qid, compare.MIXED_RAW_METHOD, 0.4) for qid in qids]
        mixed_dedup = [single_record(qid, compare.MIXED_DEDUP_METHOD, 0.45) for qid in qids]
        strategies = compare.build_strategies_from_inputs(
            fixed_rows=fixed_rows,
            router_dataset_rows=router_rows,
            router_selected_rows=router_selected,
            mixed_raw_rows=mixed_raw,
            mixed_deduplicated_rows=mixed_dedup,
        )
        comparison = compare.build_comparison(strategies, iterations=50, seed=3)
        self.assertEqual(len(comparison["question_ids"]), 2)
        self.assertIn("router_selected_minus_mixed_raw", comparison["bootstrap_results"]["paired_differences"])
        self.assertEqual(comparison["best_fixed_single_level"], "fixed_160")
        self.assertEqual(
            comparison["diagnostics"]["mixed_raw"]["topk_granularity_composition"],
            {10: 4, 20: 6},
        )


if __name__ == "__main__":
    unittest.main()
