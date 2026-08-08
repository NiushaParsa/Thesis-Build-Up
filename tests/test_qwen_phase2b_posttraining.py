import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import qwen_phase2 as phase2
import qwen_phase2b as phase2b
import qwen_phase2b_posttraining as post
import qwen_phase1 as phase1


class FakeClient:
    def __init__(self):
        self.requested_ids = []

    def retrieve(self, *, ids, **kwargs):
        del kwargs
        self.requested_ids.extend(ids)
        return [SimpleNamespace(vector=[0.1, 0.2, 0.3])]


def fake_phase1(client=None):
    client = client or FakeClient()

    def evaluate_question(**kwargs):
        level = kwargs["granularity_levels"][0]
        tokens = kwargs["chunk_sizes"][level - 1]
        question_id = kwargs["question_point_id"]
        score = 0.2 if question_id == "q1" else 0.4
        return [
            {
                "question_id": question_id,
                "document_id": kwargs["document_id"],
                "split": kwargs["split"],
                "granularity_tokens": tokens,
                "granularity_level": level,
                "k_requested": kwargs["top_k"],
                "embedding_model": kwargs["embedding_model"],
                "embedding_dimension": kwargs["embedding_dimension"],
                "tokenizer_identity": kwargs["tokenizer_name"],
                "evaluation_run_id": kwargs["evaluation_run_id"],
                "evaluation_config_hash": kwargs["evaluation_config_hash"],
                "schema_version": post.RETRIEVAL_SCHEMA_VERSION,
                "metric_version": post.RETRIEVAL_METRIC_VERSION,
                "normalization_version": post.RETRIEVAL_NORMALIZATION_VERSION,
                "f1_joined_topk": score,
            }
        ]

    return SimpleNamespace(
        OPENAI_EMBEDDING_MODEL="embedding",
        EMBEDDING_DIM=3,
        TOKENIZER_NAME="gpt2",
        PAPER_QUESTION_COLLECTION="PaperQuestion",
        qdrant_client=lambda: client,
        evaluate_question=evaluate_question,
        client=client,
    )


def prediction(question_id, document_id, alias, checkpoint="step-000213"):
    return {
        "question_id": question_id,
        "document_id": document_id,
        "question_text": f"Question {question_id}?",
        "oracle_label": phase2b.ALIAS_TO_CHUNK[alias],
        "raw_qwen_output": str(alias),
        "predicted_alias": alias,
        "parsed_prediction": phase2b.ALIAS_TO_CHUNK[alias],
        "prediction_status": phase2b.PREDICTION_STATUS,
        "phase2b_variant": phase2b.VARIANT_UNWEIGHTED,
        "experiment_fingerprint": "fingerprint",
        "selected_checkpoint_id": checkpoint,
        "selected_checkpoint_path": f"checkpoints/{checkpoint}",
    }


def test_phase1_evaluate_question_accepts_the_complete_frozen_call_contract():
    parameters = inspect.signature(phase1.evaluate_question).parameters
    assert {
        "client",
        "question_point_id",
        "question_vector",
        "document_id",
        "question_text",
        "split",
        "top_k",
        "granularity_levels",
        "store_retrieved_text",
        "chunk_sizes",
        "embedding_model",
        "embedding_dimension",
        "tokenizer_name",
        "evaluation_run_id",
        "evaluation_config_hash",
    }.issubset(parameters)


def create_phase2b_root(tmp_path, *, variant=phase2b.VARIANT_UNWEIGHTED):
    root = tmp_path / variant
    run_id = f"run-{variant}"
    rows = [prediction("q1", "d1", 1), prediction("q2", "d2", 5)]
    for row in rows:
        row["phase2b_variant"] = variant
    phase2.atomic_jsonl(root / "validation" / "predictions.jsonl", rows)
    phase2.atomic_json(
        root / "final_summary.json",
        {
            "status": "classification_complete_retrieval_pending",
            "phase": "Phase 2B",
            "variant": variant,
            "run_id": run_id,
            "model_id": phase2b.MODEL_ID,
            "model_revision": phase2b.MODEL_REVISION,
            "evaluated_examples": 2,
            "selected_checkpoint_id": "step-000213",
            "classification": {
                "accuracy": 1.0,
                "macro_f1": 0.4,
                "weighted_f1": 1.0,
                "balanced_accuracy": 1.0,
                "top_2_accuracy": 1.0,
                "top_2_accuracy_status": "available_restricted_class_scores",
            },
            "oracle_distribution": {"10": 1, "20": 0, "40": 0, "80": 0, "160": 1},
            "predicted_distribution": {"10": 1, "20": 0, "40": 0, "80": 0, "160": 1},
            "valid_output_rate": 1.0,
            "training": {
                "elapsed_seconds": 10.0,
                "experiment_fingerprint": "fingerprint",
            },
            "runtime": {
                "model_load_seconds": 2.0,
                "isolated_inference_wall_seconds": 3.0,
            },
            "retrieval": None,
            "artifacts": {"classification_metrics": "classification/metrics.json"},
        },
    )
    return root, run_id, rows


def test_resolve_output_root_rejects_legacy_roots():
    with pytest.raises(RuntimeError, match="isolated root"):
        post.resolve_output_root(
            phase2b.VARIANT_UNWEIGHTED, phase2.DEFAULT_OUTPUT_ROOT
        )
    with pytest.raises(RuntimeError, match="isolated root"):
        post.resolve_output_root(phase2b.VARIANT_UNWEIGHTED, phase2.PHASE1_ROOT)


def test_context_rejects_alias_mapping_mismatch(tmp_path):
    root, run_id, rows = create_phase2b_root(tmp_path)
    rows[0]["parsed_prediction"] = 160
    phase2.atomic_jsonl(root / "validation" / "predictions.jsonl", rows)
    with pytest.raises(RuntimeError, match="Alias-to-chunk mismatch"):
        post.load_retrieval_context(root, phase2b.VARIANT_UNWEIGHTED, run_id, expected_examples=2)


def test_retrieval_is_resumable_and_updates_only_selected_root(tmp_path):
    root, run_id, _ = create_phase2b_root(tmp_path)
    fake = fake_phase1()
    summary = post.evaluate_retrieval(
        root,
        phase2b.VARIANT_UNWEIGHTED,
        run_id,
        phase1_module=fake,
        expected_examples=2,
    )
    assert fake.client.requested_ids == ["q1", "q2"]
    assert summary["retrieval_coverage"] == 1.0
    assert summary["valid_only_mean_joined_retrieval_f1"] == pytest.approx(0.3)
    assert summary["valid_only_median_joined_retrieval_f1"] == pytest.approx(0.3)
    assert summary["top_k"] == 5
    assert summary["paper_restricted"] is True
    final = json.loads((root / "final_summary.json").read_text(encoding="utf-8"))
    assert final["status"] == "complete"
    assert final["classification"]["accuracy"] == 1.0
    assert final["runtime"]["known_training_plus_final_validation_wall_seconds"] == 15.0
    assert final["retrieval"] == summary

    second_fake = fake_phase1()
    repeated = post.evaluate_retrieval(
        root,
        phase2b.VARIANT_UNWEIGHTED,
        run_id,
        phase1_module=second_fake,
        expected_examples=2,
    )
    assert repeated == summary
    assert second_fake.client.requested_ids == []


def test_partial_jsonl_resume_retrieves_only_missing_question(tmp_path):
    root, run_id, rows = create_phase2b_root(tmp_path)
    fake = fake_phase1()
    context = post.load_retrieval_context(
        root, phase2b.VARIANT_UNWEIGHTED, run_id, expected_examples=2
    )
    first = fake.evaluate_question(
        client=fake.client,
        question_point_id="q1",
        question_vector=[0.1, 0.2, 0.3],
        document_id="d1",
        question_text="Question q1?",
        split="validation",
        top_k=5,
        granularity_levels=[1],
        store_retrieved_text=False,
        chunk_sizes=list(phase2b.CHUNK_SIZES),
        embedding_model=fake.OPENAI_EMBEDDING_MODEL,
        embedding_dimension=fake.EMBEDDING_DIM,
        tokenizer_name=fake.TOKENIZER_NAME,
        evaluation_run_id=post.retrieval_run_id(run_id),
        evaluation_config_hash=post.RETRIEVAL_CONFIG_HASH,
    )[0]
    first.update(
        {
            "method_name": post.METHOD_NAMES[context.variant],
            "phase2b_run_id": run_id,
            "phase2b_variant": context.variant,
            "formulation_version": phase2b.FORMULATION_VERSION,
            "experiment_fingerprint": "fingerprint",
            "predicted_alias": 1,
            "predicted_granularity_tokens": 10,
            "predicted_granularity_level": 1,
            "qwen_raw_output": "1",
            "qwen_prediction_status": phase2b.PREDICTION_STATUS,
            "evidence_length_oracle": 10,
            "oracle_label_version": phase2.ORACLE_VERSION,
            "selected_checkpoint_id": "step-000213",
            "selected_checkpoint_path": "checkpoints/step-000213",
            "top_k": 5,
            "paper_restricted": True,
            "phase2b_retrieval_wall_seconds": 0.25,
        }
    )
    post.validate_retrieval_record(first, rows[0], context, fake)
    phase2.atomic_jsonl(root / "retrieval" / "results.jsonl", [first])

    summary = post.evaluate_retrieval(
        root,
        phase2b.VARIANT_UNWEIGHTED,
        run_id,
        phase1_module=fake,
        expected_examples=2,
    )
    assert fake.client.requested_ids == ["q2"]
    saved = post.read_jsonl(root / "retrieval" / "results.jsonl")
    assert [row["question_id"] for row in saved] == ["q1", "q2"]
    assert summary["complete_uninterrupted_run_wall_seconds"] is None
    assert summary["reported_retrieval_wall_basis"].startswith("durable_sum")


def test_stale_cross_variant_record_is_rejected(tmp_path):
    root, run_id, rows = create_phase2b_root(tmp_path)
    context = post.load_retrieval_context(
        root, phase2b.VARIANT_UNWEIGHTED, run_id, expected_examples=2
    )
    fake = fake_phase1()
    stale = {
        "question_id": "q1",
        "phase2b_variant": phase2b.VARIANT_CLASSBALANCED,
        "phase2b_retrieval_wall_seconds": 1.0,
    }
    with pytest.raises(RuntimeError, match="Stale or incompatible"):
        post.validate_retrieval_record(stale, rows[0], context, fake)


def test_completed_summary_is_recomputed_before_reuse(tmp_path):
    root, run_id, _ = create_phase2b_root(tmp_path)
    fake = fake_phase1()
    post.evaluate_retrieval(
        root,
        phase2b.VARIANT_UNWEIGHTED,
        run_id,
        phase1_module=fake,
        expected_examples=2,
    )
    path = root / "retrieval" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["valid_only_mean_joined_retrieval_f1"] = 0.99
    phase2.atomic_json(path, summary)
    with pytest.raises(RuntimeError, match="does not recompute"):
        post.evaluate_retrieval(
            root,
            phase2b.VARIANT_UNWEIGHTED,
            run_id,
            phase1_module=fake_phase1(),
            expected_examples=2,
        )


def summary_file(path, *, phase, variant=None, retrieval_schema="phase2"):
    retrieval = {
        "top_k": 5,
        "paper_restricted": True,
        "coverage": 1.0,
        "valid_only_mean_joined_f1": 0.2,
        "valid_only_median_joined_f1": 0.19,
    }
    if retrieval_schema == "phase2":
        retrieval = {
            "top_k": 5,
            "paper_restricted": True,
            "retrieval_coverage": 1.0,
            "valid_only_mean_joined_retrieval_f1": 0.3,
            "valid_only_median_joined_retrieval_f1": 0.29,
        }
    value = {
        "status": "complete",
        "phase": phase,
        "variant": variant,
        "run_id": f"run-{phase}",
        "model_id": phase2b.MODEL_ID,
        "model_revision": phase2b.MODEL_REVISION,
        "evaluated_examples": 2,
        "classification": {
            "accuracy": 0.4,
            "macro_f1": 0.3,
            "weighted_f1": 0.35,
            "balanced_accuracy": 0.31,
            "top_2_accuracy": 0.7 if variant else None,
            "top_2_accuracy_status": "available" if variant else "unavailable",
        },
        "oracle_distribution": {"10": 1, "160": 1},
        "predicted_distribution": {"10": 1, "160": 1},
        "valid_output_rate": 1.0,
        "retrieval": retrieval,
        "runtime": {},
    }
    phase2.atomic_json(path, value)
    return path


def test_compare_normalizes_all_four_summary_schemas(tmp_path):
    phase1 = summary_file(tmp_path / "p1.json", phase="phase1", retrieval_schema="phase1")
    phase2_path = summary_file(tmp_path / "p2.json", phase="phase2")
    unweighted = summary_file(
        tmp_path / "u.json",
        phase="phase2b",
        variant=phase2b.VARIANT_UNWEIGHTED,
    )
    balanced = summary_file(
        tmp_path / "b.json",
        phase="phase2b",
        variant=phase2b.VARIANT_CLASSBALANCED,
    )
    result = post.compare_summaries(phase1, phase2_path, unweighted, balanced)
    assert [row["name"] for row in result["rows"]] == [
        "phase1_zero_shot",
        "phase2_numeric_sft",
        "phase2b_alias_unweighted",
        "phase2b_alias_classbalanced",
    ]
    assert result["rows"][0]["mean_joined_retrieval_f1"] == 0.2
    assert result["rows"][1]["mean_joined_retrieval_f1"] == 0.3
    assert result["rows"][2]["top_2_accuracy"] == 0.7


def test_compare_rejects_different_oracle_distribution(tmp_path):
    paths = [
        summary_file(tmp_path / f"{index}.json", phase=str(index))
        for index in range(4)
    ]
    changed = json.loads(paths[-1].read_text(encoding="utf-8"))
    changed["variant"] = phase2b.VARIANT_CLASSBALANCED
    changed["oracle_distribution"] = {"10": 2, "160": 0}
    phase2.atomic_json(paths[-1], changed)
    first = json.loads(paths[2].read_text(encoding="utf-8"))
    first["variant"] = phase2b.VARIANT_UNWEIGHTED
    phase2.atomic_json(paths[2], first)
    with pytest.raises(RuntimeError, match="same validation set"):
        post.compare_summaries(*paths)
