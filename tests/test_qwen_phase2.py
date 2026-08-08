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


def test_full_split_accumulation_flushes_once_at_each_epoch_end():
    batches = 562
    assert phase2.optimizer_steps_for_batches(batches, 8) == 71
    assert 3 * phase2.optimizer_steps_for_batches(batches, 8) == 213
    assert phase2.partial_window_gradient_scale(8, 4, 32) == 1.0
    assert phase2.partial_window_gradient_scale(8, 4, 5) == 6.4


def test_checkpoint_selection_uses_declared_order_and_earlier_step_tie():
    def candidate(step, macro, accuracy=0.5, weighted=0.4, balanced=0.3, loss=1.0):
        return {
            "global_step": step,
            "validation_loss": loss,
            "classification_metrics": {
                "macro_f1": macro,
                "accuracy": accuracy,
                "weighted_f1": weighted,
                "balanced_accuracy": balanced,
            },
        }

    records = [candidate(71, 0.2), candidate(142, 0.3), candidate(213, 0.3)]
    assert phase2.select_best_evaluation(records)["global_step"] == 142


def test_resume_log_truncation_removes_only_post_checkpoint_rows(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text(
        "".join(json.dumps({"global_step": step}) + "\n" for step in range(1, 6)),
        encoding="utf-8",
    )
    assert phase2.truncate_jsonl_after_step(path, 3) == 2
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["global_step"] for row in rows] == [1, 2, 3]


def create_completed_run(tmp_path, *, duplicate_first_id=False):
    output_root = tmp_path / "phase2"
    run_id = "phase2-test-run"
    run_dir = output_root / "runs" / run_id
    selected_id = "step-000071"
    selected_path = run_dir / "validation" / f"predictions_{selected_id}.jsonl"
    phase1_rows = phase2.read_jsonl(PHASE1 / "validation" / "predictions.jsonl")
    rows = [
        {**row, "selected_checkpoint": selected_id}
        for row in phase1_rows
    ]
    if duplicate_first_id:
        rows[1] = {**rows[1], "question_id": rows[0]["question_id"]}
    phase2.atomic_jsonl(selected_path, rows)
    metrics = phase2.fixed_classification_metrics(rows)
    best = {
        "checkpoint": f"remote/path/{selected_id}",
        "checkpoint_id": selected_id,
        "global_step": 71,
        "epoch": 1,
        "validation_loss": 0.5,
        "classification_metrics": metrics,
        "predicted_distribution": {},
        "predictions": f"remote/path/predictions_{selected_id}.jsonl",
        "validation_wall_seconds": 12.5,
        "selection_metric": "validation_macro_f1",
        "tie_break": "accuracy, weighted_f1, balanced_accuracy, lower_validation_loss, earlier_step",
    }
    phase2.atomic_json(run_dir / "best_checkpoint.json", best)
    phase2.atomic_json(
        run_dir / "summary.json",
        {
            "global_step": 213,
            "initial_loss": 1.0,
            "final_loss": 0.1,
            "validation_metrics": metrics,
            "validation_events": 3,
            "elapsed_seconds": 100.0,
            "peak_gpu_allocated_gib": 8.0,
            "peak_gpu_reserved_gib": 12.0,
            "rss_gib": 2.0,
            "total_parameters": 100,
            "trainable_parameters": 100,
        },
    )
    phase2.atomic_jsonl(
        run_dir / "training_history.jsonl",
        [{"global_step": 1, "cpu_ram_gib": 1.5}],
    )
    phase2.atomic_json(
        run_dir / "training_config.json",
        {
            "python_version": "3.10.7",
            "python_executable": "/test/python",
            "torch_version": "2.8.0+cu128",
            "torch_cuda_version": "12.8",
            "transformers_version": "5.15.0.dev0",
            "transformers_commit": phase2.TRANSFORMERS_COMMIT,
            "tensorboard_version": "2.20.0",
            "gpu": "A100",
            "device": "cuda",
            "dtype": "torch.bfloat16",
            "quantization": None,
            "training_script_sha256": "script-sha",
            "repository_commit": "commit-sha",
        },
    )
    phase2.atomic_json(
        run_dir / "dataset_manifest.json",
        {
            "train_examples": 2245,
            "validation_examples": 924,
            "train_documents": 845,
            "validation_documents": 277,
            "train_distribution": phase2.EXPECTED_DISTRIBUTIONS["train"],
            "validation_distribution": phase2.EXPECTED_DISTRIBUTIONS["validation"],
            "train_oracle_sha256": phase2.sha256_file(
                PHASE1 / "oracle" / "train_oracle.jsonl"
            ),
            "validation_oracle_sha256": phase2.sha256_file(
                PHASE1 / "oracle" / "validation_oracle.jsonl"
            ),
        },
    )
    return output_root, run_id, selected_id


def test_final_prediction_validation_and_materialization(tmp_path):
    output_root, run_id, selected_id = create_completed_run(tmp_path)
    run_dir = output_root / "runs" / run_id
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    predictions = phase2.read_jsonl(
        run_dir / "validation" / f"predictions_{selected_id}.jsonl"
    )
    frozen = phase2.load_oracle_records("validation", PHASE1)
    canonical = phase2.validate_and_canonicalize_final_predictions(
        predictions, frozen, best, run_dir
    )
    phase2.materialize_final_classification(
        output_root,
        PHASE1,
        run_id,
        canonical,
        {
            "source": "post_training_selected_checkpoint_reload",
            "new_inference_performed": True,
        },
    )
    final = json.loads((output_root / "final_summary.json").read_text(encoding="utf-8"))
    canonical = phase2.read_jsonl(output_root / "validation" / "predictions.jsonl")
    assert final["run_id"] == run_id
    assert final["valid_output_rate"] == 1.0
    assert final["majority_class"] == 160
    assert final["majority_baseline_accuracy"] == 420 / 924
    assert final["runtime"]["selected_validation"]["new_inference_performed"] is True
    assert canonical[0]["selected_checkpoint_id"] == selected_id
    assert "selected_checkpoint" not in canonical[0]


def test_final_prediction_validation_rejects_duplicate_ids(tmp_path):
    output_root, run_id, selected_id = create_completed_run(
        tmp_path, duplicate_first_id=True
    )
    run_dir = output_root / "runs" / run_id
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    predictions = phase2.read_jsonl(
        run_dir / "validation" / f"predictions_{selected_id}.jsonl"
    )
    with pytest.raises(RuntimeError, match="duplicate IDs"):
        phase2.validate_and_canonicalize_final_predictions(
            predictions,
            phase2.load_oracle_records("validation", PHASE1),
            best,
            run_dir,
        )


def test_all_invalid_retrieval_summary_is_json_safe_and_has_zero_coverage():
    summary = phase2.build_phase2_retrieval_summary(
        3, [], 1.0, 1.0, "embedding", "tokenizer"
    )
    assert summary["retrieval_coverage"] == 0.0
    assert summary["valid_only_mean_joined_retrieval_f1"] is None
    assert summary["valid_only_median_joined_retrieval_f1"] is None
    assert summary["coverage_adjusted_full_set_mean_joined_retrieval_f1"] == 0.0


def test_stale_phase2_retrieval_record_is_rejected():
    prediction = {
        "question_id": "question",
        "document_id": "document",
        "parsed_prediction": 40,
        "oracle_label": 80,
        "selected_checkpoint_path": "checkpoint",
    }
    stale = {
        "method_name": "qwen-pretrained-zero-shot-router",
        "question_id": "question",
        "evaluation_config_hash": "hash",
    }
    with pytest.raises(RuntimeError, match="Stale or incompatible"):
        phase2.validate_phase2_retrieval_record(
            stale, prediction, "run", "step-000071", "embedding", 1536, "gpt2"
        )
