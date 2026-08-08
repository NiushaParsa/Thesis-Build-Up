import argparse
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import qwen_phase2 as phase2
import qwen_phase2b as phase2b


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        alias = int(text)
        return [phase2b.EXPECTED_ALIAS_TOKEN_IDS[alias]]


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.observed_messages = []

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        **kwargs,
    ):
        assert tokenize is True
        self.observed_messages.append(messages)
        user_text = messages[0]["content"][0]["text"]
        prompt = [101, 102, 103]
        if "a deliberately longer question" in user_text:
            prompt += [104, 105]
        if add_generation_prompt:
            return [prompt]
        alias = int(messages[-1]["content"][0]["text"])
        return [
            prompt
            + [phase2b.EXPECTED_ALIAS_TOKEN_IDS[alias], 999, 998]
        ]


def record(label=40, question="What was measured?"):
    return {
        "question_id": f"question-{label}",
        "document_id": "document",
        "question_text": question,
        "oracle_label": label,
        "ground_truth_evidence": "EVIDENCE_MUST_NOT_ENTER_THE_PROMPT",
        "evidence_length_gpt2_tokens": 999,
    }


def metric_row(gold, prediction, top2):
    return {
        "oracle_label": gold,
        "parsed_prediction": prediction,
        "top_2_predictions": top2,
    }


def test_alias_mapping_is_bijective_and_in_canonical_order():
    assert phase2b.ALIASES == (1, 2, 3, 4, 5)
    assert tuple(phase2b.ALIAS_TO_CHUNK.values()) == phase2.CLASS_TOKENS
    assert phase2b.CHUNK_TO_ALIAS == {10: 1, 20: 2, 40: 3, 80: 4, 160: 5}
    assert phase2b.EXPECTED_ALIAS_TOKEN_IDS == {
        1: 16,
        2: 17,
        3: 18,
        4: 19,
        5: 20,
    }


def test_effective_number_weights_match_predeclared_values():
    observed = phase2b.effective_number_class_weights()
    expected = (
        3.1872088653568436,
        0.7279213406697836,
        0.38467220811977887,
        0.34329010532422555,
        0.3569074805293684,
    )
    assert observed == pytest.approx(expected, rel=1e-14, abs=1e-14)
    assert sum(observed) / 5 == pytest.approx(1.0)


def test_weight_manifest_uses_only_preserved_training_counts():
    manifest = phase2b.class_weight_manifest(phase2b.VARIANT_CLASSBALANCED)
    assert manifest["training_counts"] == {
        "10": 55,
        "20": 267,
        "40": 586,
        "80": 687,
        "160": 650,
    }
    assert manifest["training_counts"] != {
        str(key): value
        for key, value in phase2.EXPECTED_DISTRIBUTIONS["validation"].items()
    }
    assert manifest["beta"] == 0.999


def test_runtime_alias_tokenization_verifies_chat_template_extensions():
    result = phase2b.verify_alias_tokenization(
        FakeProcessor(), ["first question", "second question"]
    )
    assert result["status"] == "passed"
    assert result["standalone_alias_token_ids"] == {
        "1": [16],
        "2": [17],
        "3": [18],
        "4": [19],
        "5": [20],
    }
    for check in result["chat_template_checks"]:
        assert check["assistant_template_suffix_token_ids"] == [999, 998]


def test_runtime_alias_tokenization_rejects_wrong_token_id():
    processor = FakeProcessor()
    processor.tokenizer.encode = lambda text, *, add_special_tokens: [777]
    with pytest.raises(RuntimeError, match="expected exactly"):
        phase2b.verify_alias_tokenization(processor)


def test_formatting_is_prompt_only_and_excludes_evidence_fields():
    processor = FakeProcessor()
    formatted = phase2b.format_classification_example(
        processor, record(160), "fixed instruction"
    )
    assert formatted["input_ids"] == [101, 102, 103]
    assert "labels" not in formatted
    assert formatted["target_alias"] == 5
    assert formatted["target_class_index"] == 4
    assert formatted["target_alias_token_id"] == 20
    rendered = json.dumps(processor.observed_messages)
    assert "EVIDENCE_MUST_NOT_ENTER_THE_PROMPT" not in rendered
    assert "999" not in rendered


def test_collator_right_padding_preserves_each_last_prompt_position():
    processor = FakeProcessor()
    short = phase2b.format_classification_example(processor, record(10))
    long = phase2b.format_classification_example(
        processor,
        record(160, "a deliberately longer question"),
    )
    batch = phase2b.collate_classification_batch([short, long], pad_token_id=0)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["last_prompt_positions"].tolist() == [2, 4]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    assert batch["class_indices"].tolist() == [0, 4]


def test_restricted_logits_use_each_prompt_end_and_ignore_other_vocabulary():
    logits = torch.zeros(2, 4, 30)
    logits[0, 1, 16:21] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    logits[1, 3, 16:21] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    logits[:, :, 29] = 10_000.0
    observed = phase2b.restricted_alias_logits(logits, torch.tensor([1, 3]))
    assert observed.tolist() == [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
    ]
    assert observed.argmax(dim=-1).tolist() == [4, 0]


def test_uniform_weighted_loss_equals_ordinary_cross_entropy():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0], [0.0] * 5])
    targets = torch.tensor([0, 4])
    components = phase2b.weighted_loss_components(
        logits, targets, torch.ones(5)
    )
    expected = F.cross_entropy(logits, targets)
    assert components["weighted_mean"] == pytest.approx(expected)
    assert components["unweighted_mean"] == pytest.approx(expected)
    assert components["weight_denominator"].item() == 2.0


def test_accumulated_weight_normalization_matches_one_weighted_mean_backward():
    weights = torch.tensor(phase2b.effective_number_class_weights())
    targets = torch.tensor([0, 4, 1])
    first = torch.nn.Parameter(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    expected = torch.nn.Parameter(first.detach().clone())

    accumulated_weight = 0.0
    for row_slice in (slice(0, 1), slice(1, 3)):
        components = phase2b.weighted_loss_components(
            first[row_slice], targets[row_slice], weights
        )
        components["weighted_numerator"].backward()
        accumulated_weight += components["weight_denominator"].item()
    phase2b.normalize_accumulated_gradients([first], accumulated_weight)

    expected_components = phase2b.weighted_loss_components(
        expected, targets, weights
    )
    expected_components["weighted_mean"].backward()
    assert torch.allclose(first.grad, expected.grad, rtol=1e-6, atol=1e-7)


def test_deterministic_score_ranking_breaks_ties_by_smaller_alias():
    assert phase2b.deterministic_ranking([1.0, 2.0, 2.0, 0.0, -1.0]) == [
        1,
        2,
        0,
        3,
        4,
    ]


def test_top2_accuracy_is_available_from_restricted_scores():
    rows = [
        metric_row(10, 20, [20, 10]),
        metric_row(40, 40, [40, 80]),
        metric_row(160, 80, [80, 40]),
    ]
    metrics = phase2b.metrics_with_top2(rows)
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["top_2_accuracy"] == pytest.approx(2 / 3)
    assert metrics["top_2_accuracy_status"].startswith("available")


def test_default_output_roots_are_distinct_and_exclude_completed_phase2():
    first = argparse.Namespace(variant=phase2b.VARIANT_UNWEIGHTED, output_root=None)
    second = argparse.Namespace(
        variant=phase2b.VARIANT_CLASSBALANCED, output_root=None
    )
    assert phase2b.resolve_output_root(first) != phase2b.resolve_output_root(second)
    forbidden = argparse.Namespace(
        variant=phase2b.VARIANT_UNWEIGHTED,
        output_root=phase2.DEFAULT_OUTPUT_ROOT,
    )
    with pytest.raises(RuntimeError, match="must not be"):
        phase2b.resolve_output_root(forbidden)


def test_experiment_root_rejects_cross_variant_reuse(tmp_path):
    phase2b.ensure_experiment_root(tmp_path, phase2b.VARIANT_UNWEIGHTED)
    with pytest.raises(RuntimeError, match="mismatch"):
        phase2b.ensure_experiment_root(
            tmp_path, phase2b.VARIANT_CLASSBALANCED
        )


def test_checkpoint_pruning_is_narrow_and_keeps_requested_ids(tmp_path):
    checkpoint_root = tmp_path / "run" / "checkpoints"
    keep = checkpoint_root / "step-000071"
    stale = checkpoint_root / "step-000142"
    unrelated = checkpoint_root / "notes"
    for path in (keep, stale, unrelated):
        path.mkdir(parents=True)
        (path / "marker").write_text(path.name, encoding="utf-8")
    removed = phase2b.prune_checkpoints(checkpoint_root, {keep.name})
    assert removed == [stale.name]
    assert keep.is_dir()
    assert unrelated.is_dir()
    assert not stale.exists()
    with pytest.raises(RuntimeError, match="named checkpoints"):
        phase2b.prune_checkpoints(tmp_path / "run", set())


def test_experiment_fingerprints_separate_loss_variants():
    data = {
        "train_oracle_sha256": "train",
        "validation_oracle_sha256": "validation",
    }
    tokenization = {
        "expected_alias_token_ids": {
            str(alias): token_id
            for alias, token_id in phase2b.EXPECTED_ALIAS_TOKEN_IDS.items()
        }
    }
    assert phase2b.experiment_fingerprint(
        phase2b.VARIANT_UNWEIGHTED, data, tokenization
    ) != phase2b.experiment_fingerprint(
        phase2b.VARIANT_CLASSBALANCED, data, tokenization
    )


def test_cli_exposes_inspect_train_and_final_validation():
    parser = phase2b.build_parser()
    inspected = parser.parse_args(
        ["inspect", "--variant", phase2b.VARIANT_UNWEIGHTED]
    )
    trained = parser.parse_args(
        ["train", "--variant", phase2b.VARIANT_CLASSBALANCED]
    )
    final = parser.parse_args(
        [
            "final-validation",
            "--variant",
            phase2b.VARIANT_UNWEIGHTED,
            "--run-id",
            "run",
        ]
    )
    assert inspected.command == "inspect"
    assert trained.command == "train" and trained.mode == "full"
    assert final.command == "final-validation"
