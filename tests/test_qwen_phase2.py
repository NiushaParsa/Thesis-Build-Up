import json
from pathlib import Path

import pytest
import torch

import qwen_phase2 as phase2


PHASE1 = Path("outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle")


class FakeProcessor:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert tokenize is True
        prompt = [101, 102, 103]
        if add_generation_prompt:
            return [prompt]
        label = int(messages[-1]["content"][0]["text"])
        return [prompt + [1000 + phase2.CLASS_TO_INDEX[label], 999]]


def record(label=40):
    return {
        "question_id": "question",
        "document_id": "document",
        "question_text": "What was measured?",
        "oracle_label": label,
    }


def test_preserved_oracle_counts_distributions_and_no_overlap():
    manifest = phase2.validate_frozen_data(PHASE1)
    assert manifest["train_examples"] == 2245
    assert manifest["validation_examples"] == 924
    assert manifest["train_documents"] == 845
    assert manifest["validation_documents"] == 277
    assert manifest["train_distribution"] == phase2.EXPECTED_DISTRIBUTIONS["train"]
    assert manifest["validation_distribution"] == phase2.EXPECTED_DISTRIBUTIONS["validation"]


@pytest.mark.parametrize("label", phase2.CLASS_TOKENS)
def test_all_five_targets_have_target_only_loss(label):
    result = phase2.format_training_example(FakeProcessor(), record(label), "instruction")
    assert result["labels"][:3] == [-100, -100, -100]
    assert result["labels"][3:] == [1000 + phase2.CLASS_TO_INDEX[label], 999]
    assert result["target_token_count"] == 2


def test_padding_and_prompt_tokens_are_masked():
    short = phase2.format_training_example(FakeProcessor(), record(10), "instruction")
    long = dict(short)
    long["input_ids"] = long["input_ids"] + [777]
    long["labels"] = long["labels"] + [777]
    batch = phase2.collate_training_batch([short, long], pad_token_id=0)
    assert batch["labels"][0, -1].item() == -100
    assert batch["attention_mask"][0, -1].item() == 0
    assert torch.all(batch["labels"][:, :3] == -100)


def test_phase1_parser_compatibility_against_saved_outputs():
    path = PHASE1 / "validation" / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        parsed, status = phase2.parse_qwen_class(row["raw_qwen_output"])
        assert parsed == row["parsed_prediction"]
        assert status == row["prediction_status"]


def test_phase1_metrics_are_reproduced_exactly():
    prediction_path = PHASE1 / "validation" / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    metrics = phase2.fixed_classification_metrics(rows)
    summary = json.loads((PHASE1 / "final_summary.json").read_text(encoding="utf-8"))
    for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
        assert metrics[key] == summary["classification"][key]


def test_parser_invalid_outputs_never_get_a_default():
    assert phase2.parse_qwen_class("none")[0] is None
    assert phase2.parse_qwen_class("10 or 160")[0] is None
    assert phase2.parse_qwen_class("1160")[0] is None


def test_deterministic_seed_repeats_torch_values():
    phase2.set_deterministic_seed(42)
    first = torch.rand(4)
    phase2.set_deterministic_seed(42)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_restored_optimizer_state_moves_to_requested_device():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    phase2.move_optimizer_state(optimizer, torch.device("cpu"))
    assert all(
        not isinstance(value, torch.Tensor) or value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
    )
