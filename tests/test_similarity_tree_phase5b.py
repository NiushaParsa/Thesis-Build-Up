from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import similarity_tree_phase5b as phase5b


def local_features() -> dict[str, float]:
    scores = {
        tokens: np.linspace(0.1, 0.9, 160 // tokens, dtype=np.float64)
        for tokens in phase5b.phase5a.CLASS_TOKENS
    }
    return phase5b.phase5a.extract_local_features(scores)


@pytest.mark.parametrize(
    ("raw", "expected", "status"),
    [
        ("short", "short", "valid"),
        ("MEDIUM", "medium", "valid"),
        ("The answer is long.", "long", "valid"),
        ("short-term", "short", "valid"),
        ("", None, "invalid_no_context_label"),
        ("brief", None, "invalid_no_context_label"),
        ("short or long", None, "invalid_multiple_context_labels"),
    ],
)
def test_context_parser(raw: str, expected: str | None, status: str) -> None:
    assert phase5b.parse_context_label(raw) == (expected, status)


def test_prompt_contains_only_fixed_instruction_and_original_question() -> None:
    question = "What did the authors conclude?"
    prompt = phase5b.build_prompt(question)
    assert prompt == f"{phase5b.FIXED_INSTRUCTION}\n\nQuestion: {question}"
    assert "evidence" not in prompt.casefold()
    assert "answer:" not in prompt.casefold()


def test_one_hot_encoding_has_exactly_three_states() -> None:
    assert phase5b.categorical_features("short") == {
        "qwen_context_short": 1.0,
        "qwen_context_medium": 0.0,
        "qwen_context_long": 0.0,
    }
    for label in phase5b.CONTEXT_LABELS:
        values = phase5b.categorical_features(label)
        assert len(values) == 3
        assert sum(values.values()) == 1.0
    with pytest.raises(ValueError):
        phase5b.categorical_features("unknown")


def test_combined_feature_schema_is_176_and_inference_safe() -> None:
    combined = phase5b.combine_features(local_features(), "medium")
    assert len(combined) == 176
    assert sum(combined[f"qwen_context_{label}"] for label in phase5b.CONTEXT_LABELS) == 1.0
    phase5b.phase5a.assert_inference_safe_feature_names(sorted(combined))


def test_same_question_category_can_be_repeated_across_independent_trees() -> None:
    first = phase5b.combine_features(local_features(), "long")
    second_source = local_features()
    second_source["level_10_max"] += 0.1
    second = phase5b.combine_features(second_source, "long")
    for label in phase5b.CONTEXT_LABELS:
        name = f"qwen_context_{label}"
        assert first[name] == second[name]
    assert first["level_10_max"] != second["level_10_max"]


def test_resumable_output_validation_reparses_raw_text(tmp_path: Path) -> None:
    manifest = [
        {
            "split": "train",
            "question_id": "q1",
            "document_id": "p1",
            "question_text": "A question?",
        }
    ]
    output = {
        **manifest[0],
        "model_id": phase5b.MODEL_ID,
        "model_revision": phase5b.MODEL_REVISION,
        "prompt_sha256": phase5b.stable_hash(phase5b.build_prompt("A question?")),
        "raw_qwen_output": "short",
        "parsed_context_label": "short",
        "prediction_status": "valid",
    }
    summary = phase5b.validate_qwen_outputs([output], manifest, "train")
    assert summary["valid_outputs"] == 1
    assert summary["distribution"] == {"short": 1, "medium": 0, "long": 0}
    corrupt = json.loads(json.dumps(output))
    corrupt["parsed_context_label"] = "long"
    with pytest.raises(RuntimeError, match="not reproducible"):
        phase5b.validate_qwen_outputs([corrupt], manifest, "train")


def test_paired_bootstrap_uses_papers_as_clusters() -> None:
    rows = [
        {"question_id": "q1", "document_id": "p1", "f1_joined_top5_trees": 0.4},
        {"question_id": "q2", "document_id": "p1", "f1_joined_top5_trees": 0.2},
        {"question_id": "q3", "document_id": "p2", "f1_joined_top5_trees": 0.5},
    ]
    baseline = {"q1": 0.3, "q2": 0.1, "q3": 0.4}
    result = phase5b.paired_paper_cluster_bootstrap(rows, baseline, "baseline", replicates=100)
    assert result["resampling_unit"] == "source paper"
    assert result["papers"] == 2
    assert result["observed_mean_difference"] == pytest.approx(0.1)


def test_invalid_output_policy_has_no_default_class() -> None:
    parsed, status = phase5b.parse_context_label("I cannot determine this")
    assert parsed is None
    assert status == "invalid_no_context_label"


def test_qwen_execution_profiles_preserve_exact_cpu_and_cuda_builds() -> None:
    assert phase5b.qwen_execution_profile(False) == {
        "environment_name": ".venv-qwen",
        "torch": "2.8.0+cpu",
        "device": "cpu",
    }
    assert phase5b.qwen_execution_profile(True) == {
        "environment_name": ".venv-phase5b-gpu",
        "torch": "2.8.0+cu128",
        "device": "cuda:0",
    }
