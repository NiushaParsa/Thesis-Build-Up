from __future__ import annotations

import numpy as np
import pytest

import similarity_tree_phase3b as phase3b


def synthetic_rows(papers: int = 20) -> list[dict]:
    rows = []
    for paper in range(papers):
        for question in range(2):
            rows.append(
                {
                    "document_id": f"paper-{paper}",
                    "oracle_label": phase3b.CLASS_TOKENS[(paper + question) % 5],
                }
            )
    return rows


def test_predeclared_grid_has_twelve_shallow_candidates():
    assert len(phase3b.GRID) == 12
    assert {candidate["max_depth"] for candidate in phase3b.GRID} == {2, 3, 4}
    assert {candidate["learning_rate"] for candidate in phase3b.GRID} == {0.03, 0.05}
    assert {candidate["n_estimators"] for candidate in phase3b.GRID} == {200, 400}


def test_square_root_class_weights_are_moderate_and_exact():
    targets = np.repeat(np.arange(5), [2, 8, 18, 20, 16]).astype(np.int64)
    weights, mapping = phase3b.class_balance_weights(targets)
    assert mapping["10"] == pytest.approx(np.sqrt(10.0))
    assert mapping["80"] == pytest.approx(1.0)
    assert weights[0] == pytest.approx(np.sqrt(10.0))
    assert np.isfinite(weights).all()


def test_paper_grouped_folds_never_split_a_paper():
    rows = synthetic_rows()
    folds = phase3b.grouped_stratified_folds(rows, 5, 42)
    assert set(folds.tolist()) == set(range(5))
    for paper in range(20):
        assigned = {
            int(folds[index])
            for index, row in enumerate(rows)
            if row["document_id"] == f"paper-{paper}"
        }
        assert len(assigned) == 1


def test_metrics_include_multiclass_and_ordinal_diagnostics():
    targets = np.asarray([0, 1, 2, 3, 4])
    predictions = np.asarray([1, 1, 3, 1, 4])
    probabilities = np.eye(5, dtype=np.float64)[predictions]
    metrics = phase3b.classification_metrics(targets, predictions, probabilities)
    assert metrics["accuracy"] == pytest.approx(0.4)
    assert metrics["mean_absolute_class_distance"] == pytest.approx(0.8)
    assert metrics["within_one_level_accuracy"] == pytest.approx(0.8)
    assert metrics["top_2_accuracy_status"] == "available"


def test_leakage_prone_feature_names_are_rejected():
    phase3b.assert_no_leakage_feature_names(["level_10_max", "edge_alignment"])
    with pytest.raises(RuntimeError, match="Leakage-prone"):
        phase3b.assert_no_leakage_feature_names(["evidence_similarity"])


def test_candidate_tie_break_prefers_simpler_tree():
    metrics = {"macro_f1": 0.2, "balanced_accuracy": 0.2, "accuracy": 0.3}
    shallow = {
        "oof_metrics": metrics,
        "parameters": {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.03},
    }
    deep = {
        "oof_metrics": metrics,
        "parameters": {"max_depth": 4, "n_estimators": 400, "learning_rate": 0.05},
    }
    assert phase3b.candidate_key(shallow) > phase3b.candidate_key(deep)


def test_xgboost_training_produces_five_probabilities():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(100, 4)).astype(np.float32)
    targets = np.repeat(np.arange(5), 20).astype(np.int64)
    features[:, 0] += targets
    weights, _ = phase3b.class_balance_weights(targets)
    candidate = dict(phase3b.GRID[0])
    candidate["n_estimators"] = 10
    booster = phase3b.train_booster(
        features, targets, weights, [f"f{index}" for index in range(4)], candidate, 7
    )
    probabilities = phase3b.predict_booster(
        booster, features, [f"f{index}" for index in range(4)]
    )
    assert probabilities.shape == (100, 5)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
