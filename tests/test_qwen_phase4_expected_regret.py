from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import qwen_phase4_expected_regret as phase4


def test_regret_matrix_preserves_all_tied_optimal_actions() -> None:
    utilities = np.asarray(
        [
            [0.1, 0.4, 0.4, 0.2, 0.0],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    regrets = phase4.regret_matrix(utilities)
    assert regrets[0].tolist() == pytest.approx([0.3, 0.0, 0.0, 0.2, 0.4])
    assert regrets[1].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_choose_actions_uses_smaller_granularity_for_exact_tie() -> None:
    predicted = np.asarray(
        [
            [0.2, 0.1, 0.1, 0.3, 0.4],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    assert phase4.choose_actions(predicted).tolist() == [1, 4]


def test_action_summary_uses_observed_selected_utility() -> None:
    utilities = np.asarray(
        [
            [0.1, 0.4, 0.4, 0.2, 0.0],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    actions = np.asarray([2, 1])
    summary = phase4.action_summary(utilities, actions)
    assert summary["mean_joined_retrieval_f1"] == pytest.approx(0.4)
    assert summary["mean_regret"] == pytest.approx(0.05)
    assert summary["retrieval_optimal_any_tie_accuracy"] == pytest.approx(0.5)
    assert summary["retrieval_optimal_smaller_tie_accuracy"] == pytest.approx(0.0)
    assert summary["selected_distribution"] == {
        "10": 0,
        "20": 1,
        "40": 1,
        "80": 0,
        "160": 0,
    }


def test_utility_matrix_aligns_by_question_id_not_file_order() -> None:
    features = [
        {"question_id": "q1", "document_id": "p1"},
        {"question_id": "q2", "document_id": "p2"},
    ]

    def row(question: str, document: str, offset: float) -> dict:
        return {
            "question_id": question,
            "document_id": document,
            "per_granularity_metrics": [
                {"f1_joined_topk": offset + index / 10}
                for index in range(len(phase4.CLASS_TOKENS))
            ],
        }

    matrix = phase4.utility_matrix(
        features, [row("q2", "p2", 0.2), row("q1", "p1", 0.1)]
    )
    assert matrix[0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert matrix[1].tolist() == pytest.approx([0.2, 0.3, 0.4, 0.5, 0.6])


def test_paper_cluster_bootstrap_is_reproducible_and_paper_grouped() -> None:
    values = np.asarray([0.0, 1.0, 0.5, 0.25])
    documents = ["paper-a", "paper-a", "paper-b", "paper-c"]
    first = phase4.paper_cluster_bootstrap(
        values, documents, iterations=200, seed=7
    )
    second = phase4.paper_cluster_bootstrap(
        values, documents, iterations=200, seed=7
    )
    assert first == second
    assert first["resampling_unit"] == "paper"
    assert first["paper_count"] == 3
    assert first["question_count"] == 4
    assert first["point_estimate"] == pytest.approx(values.mean())


def test_pre_evaluation_prediction_rows_contain_no_gold_utility() -> None:
    rows = [
        {
            "question_id": "q1",
            "document_id": "p1",
            "question_text": "Question?",
            "oracle_label": 160,
        }
    ]
    predicted = np.asarray([[0.5, 0.4, 0.3, 0.2, 0.1]])
    output = phase4.prediction_rows(rows, predicted, phase4.choose_actions(predicted))
    keys = {key.lower() for key in output[0]}
    assert not any("oracle" in key for key in keys)
    assert not any("utility" in key for key in keys)
    assert "selected_joined_retrieval_f1" not in keys
    assert "retrieval_regret" not in keys
    assert "gold" in output[0]["prediction_status"]


def test_complete_saved_training_utility_matrix_matches_frozen_features() -> None:
    if not all(path.exists() for path in phase4.TRAIN_UTILITY_PATHS):
        pytest.skip("Canonical Phase 4 inputs are unavailable")
    rows = phase4.phase3c.source_rows("train")
    utilities = phase4.load_train_utility_rows()
    matrix = phase4.utility_matrix(rows, utilities)
    assert matrix.shape == (2245, 5)
    assert np.isfinite(matrix).all()


def test_clean_phase3c_qwen_inputs_cover_both_preserved_splits() -> None:
    paths = {
        "train": (
            phase4.phase3c.DEFAULT_OUTPUT_ROOT
            / "qwen_features"
            / "train_oof_logits.npz"
        ),
        "validation": (
            phase4.phase3c.DEFAULT_OUTPUT_ROOT
            / "qwen_features"
            / "validation_logits.npz"
        ),
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("Canonical Phase 3C-OOF inputs are unavailable")
    with np.load(paths["train"], allow_pickle=False) as archive:
        assert archive["logits"].shape == (2245, 5)
        assert len(set(archive["question_ids"].tolist())) == 2245
    with np.load(paths["validation"], allow_pickle=False) as archive:
        assert archive["logits"].shape == (924, 5)
        assert len(set(archive["question_ids"].tolist())) == 924
