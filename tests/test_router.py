from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from granularity_router import (
    CLASS_TOKENS,
    classification_metrics,
    class_balance_warnings,
    examples_to_arrays,
    fit_preprocessor,
    mlp_is_justified,
    load_router_examples,
    predict_with_artifact,
    target_to_tokens,
    transform_features,
    tune_models,
    validate_split_isolation,
)


class RouterDatasetTests(unittest.TestCase):
    def test_training_loader_does_not_read_test_split(self):
        class FakeClient:
            def __init__(self):
                self.requested_splits = []

            def scroll(self, **kwargs):
                conditions = kwargs["scroll_filter"].must
                split = next(item.match.value for item in conditions if item.key == "split")
                self.requested_splits.append(split)
                point = SimpleNamespace(
                    id=f"q-{split}",
                    vector=[1.0, 0.0],
                    payload={
                        "question_id": f"q-{split}",
                        "document_id": f"d-{split}",
                        "split": split,
                        "router_target_granularity": 1,
                        "evaluation_config_hash": "hash",
                        "embedding_model": "embedding",
                        "label_version": "label-v1",
                        "per_granularity_metrics": [
                            {"granularity_level": level} for level in range(1, 6)
                        ],
                    },
                )
                return [point], None

        client = FakeClient()
        examples, config_hash = load_router_examples(
            client,
            collection="router",
            splits=["train", "validation"],
            expected_dimension=2,
        )
        self.assertEqual(client.requested_splits, ["train", "validation"])
        self.assertEqual({item["split"] for item in examples}, {"train", "validation"})
        self.assertEqual(config_hash, "hash")

    def test_level_targets_map_to_fixed_token_classes(self):
        self.assertEqual([target_to_tokens(level) for level in range(1, 6)], list(CLASS_TOKENS))
        self.assertEqual(target_to_tokens(80), 80)
        with self.assertRaises(ValueError):
            target_to_tokens(6)

    def test_question_and_document_leakage_are_rejected(self):
        base = {
            "question_id": "q1",
            "document_id": "d1",
            "split": "train",
        }
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            validate_split_isolation(
                [base, {**base, "question_id": "q2", "split": "validation"}]
            )
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            validate_split_isolation(
                [base, {**base, "document_id": "d2", "split": "validation"}]
            )

    def test_features_are_question_vectors_only(self):
        examples = [
            {
                "vector": [1.0, 2.0],
                "target_tokens": 10,
                "per_granularity_metrics": [{"f1_joined_topk": 1.0}],
                "evidence_text": "must not be a feature",
            },
            {
                "vector": [3.0, 4.0],
                "target_tokens": 20,
                "per_granularity_metrics": [{"f1_joined_topk": 0.0}],
            },
        ]
        features, targets = examples_to_arrays(examples)
        np.testing.assert_array_equal(features, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(targets, [0, 1])

    def test_class_balance_warnings_cover_absent_and_small_classes(self):
        examples = [
            {"target_tokens": 10},
            {"target_tokens": 10},
            {"target_tokens": 20},
        ]
        warnings = class_balance_warnings(
            examples, min_count=2, min_fraction=0.1
        )
        self.assertTrue(any("class 20 is underrepresented" in item for item in warnings))
        self.assertTrue(any("class 160 is absent" in item for item in warnings))


class RouterMetricTests(unittest.TestCase):
    def test_multiclass_report_contains_required_metrics(self):
        targets = np.asarray([0, 1, 2, 3, 4])
        predictions = np.asarray([0, 1, 1, 3, 3])
        probabilities = np.eye(5, dtype=np.float32)[predictions]
        report = classification_metrics(targets, predictions, probabilities)
        self.assertEqual(report["accuracy"], 0.6)
        self.assertEqual(report["top_2_accuracy"], 0.6)
        self.assertEqual(report["confusion_matrix"][2][1], 1)
        self.assertEqual(report["per_class"]["10"]["recall"], 1.0)
        self.assertEqual(report["per_class"]["160"]["recall"], 0.0)
        for field in ("macro_f1", "weighted_f1", "balanced_accuracy"):
            self.assertIn(field, report)


class RouterTrainingTests(unittest.TestCase):
    def test_mlp_requires_configured_validation_improvement(self):
        logistic = {"macro_f1": 0.50}
        selected, improvement = mlp_is_justified(
            logistic, {"macro_f1": 0.505}, 0.01
        )
        self.assertFalse(selected)
        self.assertAlmostEqual(improvement, 0.005)
        selected, _ = mlp_is_justified(logistic, {"macro_f1": 0.52}, 0.01)
        self.assertTrue(selected)

    def test_logistic_router_beats_majority_and_artifact_round_trips(self):
        train_features = np.repeat(np.eye(5, dtype=np.float32), 8, axis=0)
        train_targets = np.repeat(np.arange(5, dtype=np.int64), 8)
        validation_features = np.repeat(np.eye(5, dtype=np.float32), 2, axis=0)
        validation_targets = np.repeat(np.arange(5, dtype=np.int64), 2)
        preprocessing = fit_preprocessor(train_features, standardize=True)
        train_scaled = transform_features(train_features, preprocessing)
        validation_scaled = transform_features(validation_features, preprocessing)
        tuned = tune_models(
            train_features=train_scaled,
            train_targets=train_targets,
            validation_features=validation_scaled,
            validation_targets=validation_targets,
            logistic_learning_rates=[0.1],
            weight_decays=[0.0],
            epochs=30,
            batch_size=40,
            seed=7,
            enable_mlp=True,
            mlp_hidden_sizes=[8],
            mlp_dropouts=[0.0],
            mlp_learning_rates=[0.01],
            mlp_min_improvement=0.5,
        )
        self.assertEqual(tuned["selected_model_type"], "logistic_regression")
        self.assertIsNotNone(tuned["best_mlp"])
        self.assertFalse(tuned["mlp_justified"])
        self.assertGreater(
            tuned["best_logistic"]["validation_metrics"]["accuracy"],
            tuned["majority_validation_metrics"]["accuracy"],
        )
        artifact = {
            "selected_model_type": "logistic_regression",
            "class_tokens": list(CLASS_TOKENS),
            "embedding_dimension": 5,
            "preprocessing": preprocessing,
            "majority_class_index": tuned["majority_class_index"],
            "majority_class_probabilities": tuned["majority_class_probabilities"],
            "logistic_state_dict": tuned["best_logistic"]["state_dict"],
            "mlp_state_dict": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.pt"
            torch.save(artifact, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        predictions, probabilities = predict_with_artifact(
            loaded, validation_features, "primary"
        )
        self.assertEqual(predictions.tolist(), validation_targets.tolist())
        self.assertEqual(probabilities.shape, (10, 5))


if __name__ == "__main__":
    unittest.main()
