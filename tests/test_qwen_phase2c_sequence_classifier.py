import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import qwen_phase2 as phase2
import qwen_phase2c_sequence_classifier as phase2c


class FakeTokenizer:
    pad_token_id = 99
    pad_token = "<pad>"
    padding_side = "right"

    def __init__(self, ids=None):
        self.ids = ids
        self.seen_text = None
        self.seen_kwargs = None

    def __call__(self, text, **kwargs):
        self.seen_text = text
        self.seen_kwargs = kwargs
        if self.ids is not None:
            return {"input_ids": list(self.ids)}
        return {"input_ids": list(range(1, len(text.split()) + 1))}


class FakeConfig:
    def __init__(self):
        self.num_labels = 2
        self.problem_type = None
        self.id2label = {}
        self.label2id = {}
        self.pad_token_id = None
        self.use_cache = True
        self.text_config = SimpleNamespace(
            hidden_size=8,
            pad_token_id=None,
            use_cache=True,
        )

    def get_text_config(self):
        return self.text_config


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = phase2c.configure_classifier_config(FakeConfig(), 99)
        self.score = torch.nn.Linear(8, 5, bias=False)


def oracle_record(label=40, question="What was measured?"):
    return {
        "question_id": "q-1",
        "document_id": "d-1",
        "question_text": question,
        "oracle_label": label,
        "split": "validation",
        "label_version": phase2.ORACLE_VERSION,
        "evidence": "must never be an input",
        "evidence_token_length": 999,
    }


def metric_row(gold, predicted, top2):
    return {
        "oracle_label": gold,
        "parsed_prediction": predicted,
        "top_2_predictions": top2,
    }


def test_exact_model_revision_prompt_and_label_mapping():
    assert phase2c.MODEL_ID == "Qwen/Qwen3.5-0.8B-Base"
    assert (
        phase2c.MODEL_REVISION
        == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    )
    assert phase2c.CHUNK_SIZES == (10, 20, 40, 80, 160)
    assert phase2c.LABEL_TO_ID == {10: 0, 20: 1, 40: 2, 80: 3, 160: 4}
    assert phase2c.ID_TO_LABEL == {0: 10, 1: 20, 2: 40, 3: 80, 4: 160}
    assert phase2c.SUPERVISOR_INSTRUCTION.endswith("Return only the number")
    assert "1 = very short context" in phase2c.SUPERVISOR_INSTRUCTION
    assert "5 = very long context" in phase2c.SUPERVISOR_INSTRUCTION


def test_config_sets_nested_and_top_level_padding_and_exact_head_labels():
    config = phase2c.configure_classifier_config(FakeConfig(), 99)
    audit = phase2c.verify_model_config(config, 99)
    assert audit["status"] == "passed"
    assert config.num_labels == 5
    assert config.problem_type == "single_label_classification"
    assert config.id2label == {0: "10", 1: "20", 2: "40", 3: "80", 4: "160"}
    assert config.label2id == {"10": 0, "20": 1, "40": 2, "80": 3, "160": 4}
    assert config.pad_token_id == 99
    assert config.get_text_config().pad_token_id == 99
    assert config.use_cache is False
    assert config.get_text_config().use_cache is False


def test_formatting_is_plain_prompt_plus_original_question_only():
    tokenizer = FakeTokenizer(ids=[4, 5, 6])
    formatted = phase2c.format_classification_example(
        tokenizer, oracle_record(), max_sequence_length=128
    )
    assert tokenizer.seen_text == (
        phase2c.SUPERVISOR_INSTRUCTION + "\n\nQuestion: What was measured?"
    )
    assert tokenizer.seen_kwargs == {
        "add_special_tokens": True,
        "truncation": False,
        "return_attention_mask": False,
    }
    assert formatted["target_class_id"] == 2
    assert formatted["oracle_label"] == 40
    assert formatted["input_ids"] == [4, 5, 6]
    assert "evidence" not in formatted
    assert "evidence_token_length" not in formatted


def test_formatting_rejects_truncation_need_and_pooling_ambiguity():
    with pytest.raises(RuntimeError, match="never silently truncates"):
        phase2c.format_classification_example(
            FakeTokenizer(ids=list(range(129))),
            oracle_record(),
            max_sequence_length=128,
        )
    with pytest.raises(RuntimeError, match="pooling would become ambiguous"):
        phase2c.format_classification_example(
            FakeTokenizer(ids=[1, 99, 2]), oracle_record()
        )


def test_right_padding_and_labels_preserve_last_non_pad_positions():
    tokenizer = FakeTokenizer(ids=[7, 8])
    short = phase2c.format_classification_example(tokenizer, oracle_record(10))
    tokenizer.ids = [1, 2, 3, 4]
    long = phase2c.format_classification_example(tokenizer, oracle_record(160))
    batch = phase2c.collate_classification_batch([short, long], 99)
    assert batch["input_ids"].tolist() == [[7, 8, 99, 99], [1, 2, 3, 4]]
    assert batch["attention_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert batch["last_non_pad_positions"].tolist() == [1, 3]
    assert batch["labels"].tolist() == [0, 4]


def test_uniform_ce_is_standard_float_cross_entropy_and_accumulates_exactly():
    logits_a = torch.tensor([[1.0, 0.0, -1.0, 2.0, 0.5]], requires_grad=True)
    logits_b = torch.tensor(
        [[0.0, 2.0, 1.0, -1.0, 0.5], [1.0, 0.5, 2.0, 0.0, -1.0]],
        requires_grad=True,
    )
    targets_a = torch.tensor([3])
    targets_b = torch.tensor([1, 2])
    first = phase2c.uniform_ce_components(logits_a, targets_a)
    second = phase2c.uniform_ce_components(logits_b, targets_b)
    assert torch.allclose(first["mean"], F.cross_entropy(logits_a, targets_a))
    assert torch.allclose(second["mean"], F.cross_entropy(logits_b, targets_b))
    accumulated = (first["loss_sum"] + second["loss_sum"]) / 3
    direct = F.cross_entropy(
        torch.cat([logits_a, logits_b]), torch.cat([targets_a, targets_b])
    )
    assert torch.allclose(accumulated, direct)


def test_head_audit_requires_only_expected_randomly_initialized_score_weight():
    torch.manual_seed(42)
    model = FakeModel()
    audit = phase2c.audit_classifier_head(
        model,
        {
            "missing_keys": ["score.weight"],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
        initial_base_load=True,
        seed=42,
    )
    assert audit["head_shape"] == [5, 8]
    assert audit["head_bias"] is False
    assert len(audit["head_weight_sha256_float32"]) == 64
    with pytest.raises(RuntimeError, match="model-loading audit failed"):
        phase2c.audit_classifier_head(
            model,
            {
                "missing_keys": ["score.weight", "model.bad.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
            initial_base_load=True,
            seed=42,
        )


def test_checkpoint_head_audit_requires_no_missing_keys():
    audit = phase2c.audit_classifier_head(
        FakeModel(),
        {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
        initial_base_load=False,
        seed=42,
    )
    assert audit["initial_base_load"] is False


def test_deterministic_ranking_and_top2_use_exact_five_head_scores():
    assert phase2c.deterministic_ranking([1.0, 3.0, 3.0, 2.0, 0.0]) == [1, 2, 3, 0, 4]
    rows = [
        metric_row(10, 20, [20, 10]),
        metric_row(20, 20, [20, 40]),
        metric_row(40, 80, [80, 160]),
        metric_row(80, 80, [80, 40]),
        metric_row(160, 80, [80, 160]),
    ]
    metrics = phase2c.metrics_with_top2(rows)
    assert math.isclose(metrics["accuracy"], 2 / 5)
    assert math.isclose(metrics["top_2_accuracy"], 4 / 5)
    assert metrics["top_2_accuracy_status"] == (
        "available_from_comparable_five_class_head_logits"
    )


def test_prediction_identity_rejects_mapping_or_score_drift():
    frozen = [oracle_record(40)]
    prediction = {
        **frozen[0],
        "predicted_class_id": 2,
        "predicted_label": 40,
        "parsed_prediction": 40,
        "prediction_status": phase2c.PREDICTION_STATUS,
        "class_logits_by_label": {
            "10": 0.0,
            "20": 1.0,
            "40": 4.0,
            "80": 3.0,
            "160": 2.0,
        },
        "class_probabilities_by_label": {
            "10": 0.01,
            "20": 0.02,
            "40": 0.7,
            "80": 0.2,
            "160": 0.07,
        },
        "ranked_class_ids": [2, 3, 4, 1, 0],
        "ranked_predictions": [40, 80, 160, 20, 10],
        "top_2_class_ids": [2, 3],
        "top_2_predictions": [40, 80],
        "selected_checkpoint": "step-000213",
    }
    phase2c.validate_prediction_identity(
        [prediction], frozen, "step-000213"
    )
    broken = {**prediction, "predicted_label": 80}
    with pytest.raises(RuntimeError, match="mapping mismatch"):
        phase2c.validate_prediction_identity(
            [broken], frozen, "step-000213"
        )


def test_output_root_is_isolated_and_fingerprinted(tmp_path):
    marker = phase2c.ensure_output_root(tmp_path / "phase2c")
    assert marker["phase"] == "Phase 2C"
    assert marker["model_id"] == phase2c.MODEL_ID
    assert marker["model_revision"] == phase2c.MODEL_REVISION
    args = argparse.Namespace(output_root=phase2.DEFAULT_OUTPUT_ROOT)
    with pytest.raises(RuntimeError, match="must not reuse"):
        phase2c.ensure_output_root(args.output_root)


def test_cli_exposes_inspect_train_and_final_validation():
    parser = phase2c.build_parser()
    assert parser.parse_args(["inspect"]).command == "inspect"
    train = parser.parse_args(["train", "--mode", "smoke", "--max-steps", "1"])
    assert train.command == "train"
    assert train.mode == "smoke"
    final = parser.parse_args(["final-validation", "--run-id", "run-1"])
    assert final.command == "final-validation"
    assert final.run_id == "run-1"


def test_installed_transformers_maps_composite_qwen35_to_sequence_classifier():
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    )

    assert MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES["qwen3_5"] == (
        "Qwen3_5ForSequenceClassification"
    )

