from __future__ import annotations

import math

import numpy as np
import pytest

import similarity_tree_phase3a as phase3a


def synthetic_scores() -> dict[int, np.ndarray]:
    return {
        10: np.asarray([0.9, 0.8, 0.2, 0.1, 0.3, 0.2, 0.1, 0.0]),
        20: np.asarray([0.85, 0.15, 0.25, 0.05]),
        40: np.asarray([0.60, 0.20]),
        80: np.asarray([0.40]),
        160: np.asarray([0.30]),
    }


def test_hierarchy_counts_require_exact_binary_parent_counts():
    assert phase3a.hierarchy_counts_are_valid({10: 9, 20: 5, 40: 3, 80: 2, 160: 1})
    assert not phase3a.hierarchy_counts_are_valid({10: 9, 20: 4, 40: 2, 80: 1, 160: 1})


def test_tree_features_are_finite_and_contain_no_leakage_names():
    level, tree = phase3a.extract_features(synthetic_scores())
    assert len(tree) > len(level)
    assert all(math.isfinite(value) for value in tree.values())
    phase3a.assert_no_leakage_feature_names(tree)
    with pytest.raises(RuntimeError, match="Forbidden"):
        phase3a.assert_no_leakage_feature_names(["evidence_similarity"])


def test_parent_child_features_follow_chunk_index_hierarchy():
    features = phase3a.tree_edge_statistics(
        [0.8, 0.6, 0.4, 0.2], [0.7, 0.3], 10, 20
    )
    assert features["edge_20_to_10_child_max_minus_parent_mean"] == pytest.approx(0.1)
    assert features["edge_20_to_10_argmax_alignment"] == 1.0
    with pytest.raises(ValueError, match="Broken hierarchy"):
        phase3a.tree_edge_statistics([0.8, 0.6, 0.4], [0.7], 10, 20)


def test_grouped_folds_never_split_a_paper():
    rows = []
    for document in range(20):
        for question in range(2):
            rows.append(
                {
                    "document_id": f"d{document}",
                    "oracle_label": phase3a.CLASS_TOKENS[(document + question) % 5],
                }
            )
    folds = phase3a.grouped_stratified_folds(rows, 5, 42)
    assert set(folds.tolist()) == set(range(5))
    for document in range(20):
        assigned = {
            int(folds[index])
            for index, row in enumerate(rows)
            if row["document_id"] == f"d{document}"
        }
        assert len(assigned) == 1


def test_leaf_breadth_maps_concentrated_and_distributed_scores_differently():
    base = {
        "scores_by_tokens": {
            "10": [0.9, 0.89, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        }
    }
    concentrated, probabilities = phase3a.leaf_breadth_predictions([base], 0.02)
    assert probabilities is None
    assert phase3a.CLASS_TOKENS[int(concentrated[0])] == 20
    distributed = {
        "scores_by_tokens": {
            "10": [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.89],
        }
    }
    prediction, _ = phase3a.leaf_breadth_predictions([distributed], 0.02)
    assert phase3a.CLASS_TOKENS[int(prediction[0])] == 80


def test_extended_metrics_include_ordinal_diagnostics():
    targets = np.asarray([0, 1, 2, 3, 4])
    predictions = np.asarray([1, 1, 3, 1, 4])
    probabilities = np.eye(5, dtype=np.float32)[predictions]
    metrics = phase3a.extended_metrics(targets, predictions, probabilities)
    assert metrics["mean_absolute_class_distance"] == pytest.approx(0.8)
    assert metrics["within_one_level_accuracy"] == pytest.approx(0.8)
    assert metrics["top_2_accuracy_status"] == "available"


def test_linear_classifier_learns_simple_separable_features():
    features = np.repeat(np.eye(5, dtype=np.float32), 10, axis=0)
    targets = np.repeat(np.arange(5, dtype=np.int64), 10)
    standardizer = phase3a.fit_standardizer(features)
    model, _ = phase3a.fit_linear_classifier(
        phase3a.transform(features, standardizer),
        targets,
        learning_rate=0.1,
        weight_decay=0.0,
        epochs=100,
        seed=7,
    )
    predictions, _ = phase3a.predict_linear(model, phase3a.transform(features, standardizer))
    assert predictions.tolist() == targets.tolist()
