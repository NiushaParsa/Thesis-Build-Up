import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import qwen_phase1 as phase1
import qwen_phase2 as phase2
import qwen_phase2b as phase2b
import qwen_phase2c_posttraining as phase2c_post
import qwen_phase2c_sequence_classifier as phase2c
import qwen_phase2d_posttraining as post
import qwen_phase2d_sequence_classifier as phase2d


class FakeClient:
    def __init__(self):
        self.requested_ids = []
        self.retrieve_calls = []

    def retrieve(self, *, ids, **kwargs):
        self.retrieve_calls.append({"ids": list(ids), **kwargs})
        self.requested_ids.extend(ids)
        return [SimpleNamespace(vector=[0.1, 0.2, 0.3])]


def fake_phase1(client=None):
    client = client or FakeClient()
    evaluate_calls = []

    def evaluate_question(**kwargs):
        evaluate_calls.append(dict(kwargs))
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
        evaluate_calls=evaluate_calls,
        client=client,
    )


def prediction(question_id, document_id, class_id, checkpoint="step-000213"):
    label = phase2d.CHUNK_SIZES[class_id]
    return {
        "question_id": question_id,
        "document_id": document_id,
        "question_text": f"Question {question_id}?",
        "oracle_label": label,
        "predicted_class_id": class_id,
        "predicted_label": label,
        "parsed_prediction": label,
        "prediction_status": getattr(
            phase2d, "PREDICTION_STATUS", "valid_classifier_argmax"
        ),
        "formulation_version": phase2d.FORMULATION_VERSION,
        "experiment_fingerprint": "fingerprint",
        "class_logits": [0.1, 0.2, 0.3, 0.4, 0.5],
        "class_probabilities": [0.1, 0.1, 0.2, 0.2, 0.4],
        "selected_checkpoint_id": checkpoint,
        "selected_checkpoint_path": f"checkpoints/{checkpoint}",
    }


def create_phase2d_root(tmp_path):
    root = tmp_path / "phase2d"
    run_id = "run-phase2d"
    rows = [prediction("q1", "d1", 0), prediction("q2", "d2", 4)]
    phase2.atomic_json(
        root / "configuration" / "experiment.json",
        {
            "phase": "Phase 2D",
            "formulation_version": phase2d.FORMULATION_VERSION,
            "model_id": phase2d.MODEL_ID,
            "model_revision": phase2d.MODEL_REVISION,
            "architecture": "AutoModelForSequenceClassification",
            "instruction": phase2d.SUPERVISOR_INSTRUCTION,
            "instruction_sha256": post.PHASE2D_INSTRUCTION_SHA256,
            "input_template": "{instruction}\\n\\nQuestion: {original_question_text}",
            "id2label": {
                str(index): str(label)
                for index, label in phase2d.ID_TO_LABEL.items()
            },
            "label2id": {
                str(label): index
                for label, index in phase2d.LABEL_TO_ID.items()
            },
            "objective": "uniform_five_class_cross_entropy",
        },
    )
    phase2.atomic_jsonl(root / "validation" / "predictions.jsonl", rows)
    phase2.atomic_json(
        root / "final_summary.json",
        {
            "status": "classification_complete_retrieval_pending",
            "experiment_status": "classification_complete_retrieval_pending",
            "phase": "Phase 2D",
            "formulation_version": phase2d.FORMULATION_VERSION,
            "run_id": run_id,
            "model_id": phase2d.MODEL_ID,
            "model_revision": phase2d.MODEL_REVISION,
            "evaluated_examples": 2,
            "valid_output_count": 2,
            "invalid_output_count": 0,
            "valid_output_rate": 1.0,
            "selected_checkpoint_id": "step-000213",
            "selected_checkpoint": "checkpoints/step-000213",
            "classification": {
                "accuracy": 1.0,
                "macro_f1": 0.4,
                "weighted_f1": 1.0,
                "balanced_accuracy": 1.0,
                "top_2_accuracy": 1.0,
                "top_2_accuracy_status": "available_classifier_logits",
            },
            "oracle_distribution": {
                "10": 1,
                "20": 0,
                "40": 0,
                "80": 0,
                "160": 1,
            },
            "predicted_distribution": {
                "10": 1,
                "20": 0,
                "40": 0,
                "80": 0,
                "160": 1,
            },
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


def test_phase1_evaluate_question_accepts_frozen_retrieval_contract():
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


def test_posttrainer_contains_no_qdrant_mutation_calls():
    source = inspect.getsource(post)
    for forbidden in (
        ".upsert(",
        ".delete(",
        ".create_collection(",
        ".recreate_collection(",
        ".update_collection(",
    ):
        assert forbidden not in source


def test_retrieval_contract_constants_match_phase2c_exactly():
    assert post.TOP_K == phase2c_post.TOP_K == 5
    assert post.RETRIEVAL_CONFIG_HASH == phase2c_post.RETRIEVAL_CONFIG_HASH
    assert post.RETRIEVAL_SCHEMA_VERSION == phase2c_post.RETRIEVAL_SCHEMA_VERSION
    assert post.RETRIEVAL_METRIC_VERSION == phase2c_post.RETRIEVAL_METRIC_VERSION
    assert (
        post.RETRIEVAL_NORMALIZATION_VERSION
        == phase2c_post.RETRIEVAL_NORMALIZATION_VERSION
    )
    assert post.FROZEN_RETRIEVAL_IDENTITY == {
        "top_k": 5,
        "paper_restricted": True,
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
        "tokenizer": "gpt2",
        "metric": "f1_joined_topk",
        "evaluation_config_hash": (
            "9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8"
        ),
        "schema_version": 2,
        "metric_version": "qasper-token-prf-v2",
        "normalization_version": (
            "lowercase-remove-punctuation-collapse-whitespace-v1"
        ),
    }


def test_retrieval_computational_bodies_match_phase2c():
    names = (
        "validate_retrieval_record",
        "build_retrieval_summary",
        "_known_pre_retrieval_wall",
        "update_final_summary",
        "_validate_completed_summary",
        "evaluate_retrieval",
    )
    for name in names:
        phase2d_source = inspect.getsource(getattr(post, name))
        normalized = (
            phase2d_source.replace("Phase 2D", "Phase 2C")
            .replace("phase2d", "phase2c")
            .replace("PHASE2D", "PHASE2C")
        )
        assert normalized == inspect.getsource(getattr(phase2c_post, name)), name


def test_resolve_output_root_rejects_prior_experiment_roots():
    forbidden = [
        phase2.PHASE1_ROOT,
        phase2.DEFAULT_OUTPUT_ROOT,
        *phase2b.DEFAULT_OUTPUT_ROOTS.values(),
        phase2c.DEFAULT_OUTPUT_ROOT,
    ]
    for root in forbidden:
        with pytest.raises(RuntimeError, match="isolated root"):
            post.resolve_output_root(root)
        with pytest.raises(RuntimeError, match="isolated root"):
            post.resolve_output_root(root / "phase2d")
    with pytest.raises(RuntimeError, match="isolated root"):
        post.resolve_output_root(phase2.PHASE1_ROOT.parent)


def test_resolve_output_root_requires_exact_phase2d_marker(tmp_path):
    root = tmp_path / "phase2d"
    with pytest.raises(RuntimeError, match="no configuration marker"):
        post.resolve_output_root(root)
    root, _, _ = create_phase2d_root(tmp_path)
    assert post.resolve_output_root(root) == root
    marker_path = root / "configuration" / "experiment.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["instruction_sha256"] = "wrong"
    phase2.atomic_json(marker_path, marker)
    with pytest.raises(RuntimeError, match="marker mismatch"):
        post.resolve_output_root(root)


def test_comparison_output_is_fail_closed_and_source_safe():
    source = phase2c.DEFAULT_OUTPUT_ROOT / "final_summary.json"
    assert post.resolve_comparison_output(
        post.DEFAULT_COMPARISON_OUTPUT, [source]
    ) == post.DEFAULT_COMPARISON_OUTPUT
    for unsafe in (
        source,
        Path("outputs/unsafe-comparison.json"),
        post.DEFAULT_COMPARISON_ROOT,
        post.DEFAULT_COMPARISON_ROOT / "not-json.txt",
    ):
        with pytest.raises(RuntimeError, match="comparison output"):
            post.resolve_comparison_output(unsafe, [source])


def test_context_rejects_class_id_mapping_mismatch(tmp_path):
    root, run_id, rows = create_phase2d_root(tmp_path)
    rows[0]["predicted_label"] = 160
    rows[0]["parsed_prediction"] = 160
    phase2.atomic_jsonl(root / "validation" / "predictions.jsonl", rows)
    with pytest.raises(RuntimeError, match="ID-to-chunk mapping mismatch"):
        post.load_retrieval_context(root, run_id, expected_examples=2)


def test_context_rejects_any_default_or_fallback_prediction(tmp_path):
    root, run_id, rows = create_phase2d_root(tmp_path)
    rows[0]["default_applied"] = True
    phase2.atomic_jsonl(root / "validation" / "predictions.jsonl", rows)
    with pytest.raises(RuntimeError, match="default/fallback"):
        post.load_retrieval_context(root, run_id, expected_examples=2)


def test_retrieval_is_resumable_and_updates_only_phase2d_root(tmp_path):
    root, run_id, _ = create_phase2d_root(tmp_path)
    fake = fake_phase1()
    summary = post.evaluate_retrieval(
        root, run_id, phase1_module=fake, expected_examples=2
    )
    assert fake.client.requested_ids == ["q1", "q2"]
    assert summary["retrieval_coverage"] == 1.0
    assert summary["valid_only_mean_joined_retrieval_f1"] == pytest.approx(0.3)
    assert summary["valid_only_median_joined_retrieval_f1"] == pytest.approx(0.3)
    assert summary["coverage_adjusted_full_set_mean_joined_retrieval_f1"] == pytest.approx(
        0.3
    )
    assert summary["top_k"] == 5
    assert summary["paper_restricted"] is True
    assert summary["method_name"] == post.METHOD_NAME
    assert fake.client.retrieve_calls == [
        {
            "ids": ["q1"],
            "collection_name": "PaperQuestion",
            "with_payload": True,
            "with_vectors": True,
        },
        {
            "ids": ["q2"],
            "collection_name": "PaperQuestion",
            "with_payload": True,
            "with_vectors": True,
        },
    ]
    assert len(fake.evaluate_calls) == 2
    for call, expected_question, expected_document, expected_level in zip(
        fake.evaluate_calls,
        ("q1", "q2"),
        ("d1", "d2"),
        (1, 5),
    ):
        assert call["client"] is fake.client
        assert call["question_point_id"] == expected_question
        assert call["document_id"] == expected_document
        assert call["split"] == "validation"
        assert call["top_k"] == 5
        assert call["granularity_levels"] == [expected_level]
        assert call["store_retrieved_text"] is False
        assert call["chunk_sizes"] == [10, 20, 40, 80, 160]
        assert call["embedding_model"] == "embedding"
        assert call["embedding_dimension"] == 3
        assert call["tokenizer_name"] == "gpt2"
        assert call["evaluation_config_hash"] == post.RETRIEVAL_CONFIG_HASH
        assert call["evaluation_run_id"] == post.retrieval_run_id(run_id)

    final = json.loads((root / "final_summary.json").read_text(encoding="utf-8"))
    assert final["status"] == "complete"
    assert final["experiment_status"] == "complete"
    assert final["classification"]["accuracy"] == 1.0
    assert final["runtime"]["known_training_plus_final_validation_wall_seconds"] == 15.0
    assert final["retrieval"] == summary

    second_fake = fake_phase1()
    repeated = post.evaluate_retrieval(
        root, run_id, phase1_module=second_fake, expected_examples=2
    )
    assert repeated == summary
    assert second_fake.client.requested_ids == []


def _first_durable_record(root, run_id, rows, fake):
    context = post.load_retrieval_context(root, run_id, expected_examples=2)
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
        chunk_sizes=list(phase2d.CHUNK_SIZES),
        embedding_model=fake.OPENAI_EMBEDDING_MODEL,
        embedding_dimension=fake.EMBEDDING_DIM,
        tokenizer_name=fake.TOKENIZER_NAME,
        evaluation_run_id=post.retrieval_run_id(run_id),
        evaluation_config_hash=post.RETRIEVAL_CONFIG_HASH,
    )[0]
    first.update(
        {
            "method_name": post.METHOD_NAME,
            "phase2d_run_id": run_id,
            "formulation_version": phase2d.FORMULATION_VERSION,
            "experiment_fingerprint": "fingerprint",
            "predicted_class_id": 0,
            "predicted_granularity_tokens": 10,
            "predicted_granularity_level": 1,
            "classifier_prediction_status": rows[0]["prediction_status"],
            "evidence_length_oracle": 10,
            "oracle_label_version": phase2d.ORACLE_VERSION,
            "selected_checkpoint_id": "step-000213",
            "selected_checkpoint_path": "checkpoints/step-000213",
            "model_id": phase2d.MODEL_ID,
            "model_revision": phase2d.MODEL_REVISION,
            "top_k": 5,
            "paper_restricted": True,
            "phase2d_retrieval_wall_seconds": 0.25,
        }
    )
    post.validate_retrieval_record(first, rows[0], context, fake)
    return first


def test_partial_jsonl_resume_retrieves_only_missing_question(tmp_path):
    root, run_id, rows = create_phase2d_root(tmp_path)
    fake = fake_phase1()
    first = _first_durable_record(root, run_id, rows, fake)
    phase2.atomic_jsonl(root / "retrieval" / "results.jsonl", [first])

    summary = post.evaluate_retrieval(
        root, run_id, phase1_module=fake, expected_examples=2
    )
    assert fake.client.requested_ids == ["q2"]
    saved = post.read_jsonl(root / "retrieval" / "results.jsonl")
    assert [row["question_id"] for row in saved] == ["q1", "q2"]
    assert summary["complete_uninterrupted_run_wall_seconds"] is None
    assert summary["reported_retrieval_wall_basis"].startswith("durable_sum")


def test_stale_cross_run_record_is_rejected(tmp_path):
    root, run_id, rows = create_phase2d_root(tmp_path)
    context = post.load_retrieval_context(root, run_id, expected_examples=2)
    fake = fake_phase1()
    stale = {
        "question_id": "q1",
        "phase2d_run_id": "different-run",
        "phase2d_retrieval_wall_seconds": 1.0,
    }
    with pytest.raises(RuntimeError, match="Stale or incompatible"):
        post.validate_retrieval_record(stale, rows[0], context, fake)


def test_completed_summary_is_recomputed_before_reuse(tmp_path):
    root, run_id, _ = create_phase2d_root(tmp_path)
    fake = fake_phase1()
    post.evaluate_retrieval(root, run_id, phase1_module=fake, expected_examples=2)
    path = root / "retrieval" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["valid_only_mean_joined_retrieval_f1"] = 0.99
    phase2.atomic_json(path, summary)
    with pytest.raises(RuntimeError, match="does not recompute"):
        post.evaluate_retrieval(
            root, run_id, phase1_module=fake_phase1(), expected_examples=2
        )


def summary_file(
    path,
    *,
    phase,
    variant=None,
    formulation=None,
    retrieval_schema="full",
):
    retrieval = {
        "top_k": 5,
        "paper_restricted": True,
        "retrieval_coverage": 1.0,
        "valid_only_mean_joined_retrieval_f1": 0.3,
        "valid_only_median_joined_retrieval_f1": 0.29,
    }
    if retrieval_schema == "legacy_minimal":
        retrieval = {
            "top_k": 5,
            "paper_restricted": True,
            "coverage": 1.0,
            "valid_only_mean_joined_f1": 0.2,
            "valid_only_median_joined_f1": 0.19,
        }
    elif retrieval_schema == "legacy_partial":
        retrieval.update(
            {
                "embedding_model": "text-embedding-3-small",
                "tokenizer": "gpt2",
                "metric": "f1_joined_topk",
            }
        )
    elif retrieval_schema == "full":
        retrieval.update(post.FROZEN_RETRIEVAL_IDENTITY)
    else:
        raise ValueError(retrieval_schema)
    run_id = f"run-{Path(path).stem}"
    value = {
        "status": "complete",
        "phase": phase,
        "variant": variant,
        "formulation_version": formulation,
        "run_id": run_id,
        "model_id": phase2d.MODEL_ID,
        "model_revision": phase2d.MODEL_REVISION,
        "evaluated_examples": 924,
        "classification": {
            "accuracy": 0.4,
            "macro_f1": 0.3,
            "weighted_f1": 0.35,
            "balanced_accuracy": 0.31,
            "top_2_accuracy": 0.7 if (variant or formulation) else None,
            "top_2_accuracy_status": (
                "available" if (variant or formulation) else "unavailable"
            ),
        },
        "oracle_distribution": dict(post.EXPECTED_ORACLE_DISTRIBUTION),
        "predicted_distribution": dict(post.EXPECTED_ORACLE_DISTRIBUTION),
        "valid_output_rate": 1.0,
        "id2label": {
            str(index): label for index, label in phase2d.ID_TO_LABEL.items()
        },
        "retrieval": retrieval,
        "runtime": {},
        "artifacts": {},
    }
    module = None
    if formulation == phase2c.FORMULATION_VERSION:
        module = phase2c
    elif formulation == phase2d.FORMULATION_VERSION:
        module = phase2d
    if module is not None:
        run_dir = Path(path).parent / "runs" / run_id
        training_config = {
            **vars(module.TrainingConfig()),
            "run_mode": "full",
            "active_per_device_batch_size": 4,
            "active_gradient_accumulation_steps": 8,
            "active_effective_batch_size": 32,
            "maximum_optimizer_steps": None,
            "total_optimizer_steps": 213,
            "warmup_steps": 11,
            "run_id": run_id,
            "output_root": str(Path(path).parent),
            "repository_commit": f"commit-{phase}",
            "training_script_sha256": f"script-{phase}",
            "python_version": "3.10.7",
            "python_executable": "/workspace/.venv-qwen/bin/python",
            "torch_version": "2.8.0+cu128",
            "torch_cuda_version": "12.8",
            "transformers_version": "5.15.0.dev0",
            "transformers_commit": phase2.TRANSFORMERS_COMMIT,
            "tensorboard_version": "2.20.0",
            "gpu": "NVIDIA A100-SXM4-40GB",
            "instruction": module.SUPERVISOR_INSTRUCTION,
            "instruction_sha256": module.text_sha256(
                module.SUPERVISOR_INSTRUCTION
            ),
            "pad_token_id": 248044,
            "id2label": {
                str(index): str(label)
                for index, label in module.ID_TO_LABEL.items()
            },
            "label2id": {
                str(label): index
                for label, index in module.LABEL_TO_ID.items()
            },
            "initial_model_loading_audit": {
                "configuration": {"status": "passed"},
                "classifier_head": {
                    "status": "passed",
                    "head_weight_sha256_float32": "initial-head-hash",
                },
            },
            "experiment_fingerprint": f"fingerprint-{phase}",
            "resume_contract_sha256": f"resume-{phase}",
            "created_at": f"created-{phase}",
        }
        train_distribution = {
            str(key): count
            for key, count in phase2.EXPECTED_DISTRIBUTIONS["train"].items()
        }
        validation_distribution = {
            str(key): count
            for key, count in phase2.EXPECTED_DISTRIBUTIONS["validation"].items()
        }
        dataset_manifest = {
            "train_examples": 2245,
            "validation_examples": 924,
            "train_documents": 845,
            "validation_documents": 277,
            "train_distribution": train_distribution,
            "validation_distribution": validation_distribution,
            "train_oracle_sha256": post.TRAIN_ORACLE_SHA256,
            "validation_oracle_sha256": post.VALIDATION_ORACLE_SHA256,
            "verified_at": f"verified-{phase}",
            "active_train_examples": 2245,
            "active_validation_examples": 924,
            "active_train_distribution": train_distribution,
            "active_validation_distribution": validation_distribution,
            "sequence_length": {
                "train_minimum": 80 if module is phase2d else 86,
                "train_maximum": 106 if module is phase2d else 112,
                "validation_minimum": 81 if module is phase2d else 87,
                "validation_maximum": 109 if module is phase2d else 115,
            },
            "model_inputs": [
                "fixed_supervisor_instruction",
                "original_question_text",
            ],
            "experiment_fingerprint": f"fingerprint-{phase}",
            "created_at": f"created-{phase}",
        }
        phase2.atomic_json(run_dir / "training_config.json", training_config)
        phase2.atomic_json(run_dir / "dataset_manifest.json", dataset_manifest)
        value["artifacts"] = {
            "training_config": str(run_dir / "training_config.json"),
            "dataset_manifest": str(run_dir / "dataset_manifest.json"),
        }
    phase2.atomic_json(path, value)
    return path


def test_compare_normalizes_all_six_summary_schemas(tmp_path):
    phase1_path = summary_file(
        tmp_path / "p1.json", phase="phase1", retrieval_schema="legacy_minimal"
    )
    phase2_path = summary_file(
        tmp_path / "p2.json", phase="phase2", retrieval_schema="legacy_partial"
    )
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
    phase2c_classifier = summary_file(
        tmp_path / "c.json",
        phase="Phase 2C",
        formulation=phase2c.FORMULATION_VERSION,
    )
    phase2d_classifier = summary_file(
        tmp_path / "d.json",
        phase="Phase 2D",
        formulation=phase2d.FORMULATION_VERSION,
    )
    result = post.compare_summaries(
        phase1_path,
        phase2_path,
        unweighted,
        balanced,
        phase2c_classifier,
        phase2d_classifier,
    )
    assert [row["name"] for row in result["rows"]] == [
        "phase1_zero_shot",
        "phase2_numeric_sft",
        "phase2b_alias_unweighted",
        "phase2b_alias_classbalanced",
        "phase2c_base_sequence_classifier",
        "phase2d_base_sequence_classifier_token_count_prompt",
    ]
    assert result["rows"][0]["mean_joined_retrieval_f1"] == 0.2
    assert result["rows"][1]["mean_joined_retrieval_f1"] == 0.3
    assert result["rows"][4]["top_2_accuracy"] == 0.7
    assert result["rows"][5]["top_2_accuracy"] == 0.7
    assert result["prompt_only_protocol_audit"]["status"] == "passed"
    assert result["prompt_only_protocol_audit"]["relationship"] == (
        "prompt_only_single_seed_ablation"
    )
    assert result["rows"][0]["retrieval_identity"]["status"] == (
        "accepted_explicit_legacy_schema"
    )
    assert result["rows"][1]["retrieval_identity"]["status"] == (
        "accepted_explicit_legacy_schema"
    )
    assert result["rows"][4]["retrieval_identity"]["status"] == (
        "complete_frozen_identity"
    )
    assert result["rows"][5]["retrieval_identity"]["status"] == (
        "complete_frozen_identity"
    )
    assert "prompt-only single-seed ablation" in result["comparability_note"]


def test_prompt_only_protocol_audit_rejects_non_prompt_training_drift(tmp_path):
    c_path = summary_file(
        tmp_path / "c.json",
        phase="Phase 2C",
        formulation=phase2c.FORMULATION_VERSION,
    )
    d_path = summary_file(
        tmp_path / "d.json",
        phase="Phase 2D",
        formulation=phase2d.FORMULATION_VERSION,
    )
    d_summary = json.loads(d_path.read_text(encoding="utf-8"))
    config_path = (
        d_path.parent
        / "runs"
        / d_summary["run_id"]
        / "training_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["learning_rate"] = 3e-5
    phase2.atomic_json(config_path, config)
    with pytest.raises(RuntimeError, match="training configuration mismatch"):
        post.audit_phase2c_phase2d_protocol(c_path, d_path)


def test_prompt_only_protocol_audit_rejects_dataset_sha_drift(tmp_path):
    c_path = summary_file(
        tmp_path / "c.json",
        phase="Phase 2C",
        formulation=phase2c.FORMULATION_VERSION,
    )
    d_path = summary_file(
        tmp_path / "d.json",
        phase="Phase 2D",
        formulation=phase2d.FORMULATION_VERSION,
    )
    d_summary = json.loads(d_path.read_text(encoding="utf-8"))
    manifest_path = (
        d_path.parent
        / "runs"
        / d_summary["run_id"]
        / "dataset_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_oracle_sha256"] = "wrong"
    phase2.atomic_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="dataset mismatch"):
        post.audit_phase2c_phase2d_protocol(c_path, d_path)


def test_full_retrieval_identity_rejects_coordinated_metric_drift():
    retrieval = dict(post.FROZEN_RETRIEVAL_IDENTITY)
    retrieval["evaluation_config_hash"] = "different"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        post.validate_retrieval_identity(
            retrieval, post.RETRIEVAL_IDENTITY_FULL
        )


def test_compare_rejects_different_oracle_distribution(tmp_path):
    paths = [
        summary_file(tmp_path / "p1.json", phase="p1"),
        summary_file(tmp_path / "p2.json", phase="p2"),
        summary_file(
            tmp_path / "a.json",
            phase="p2b",
            variant=phase2b.VARIANT_UNWEIGHTED,
        ),
        summary_file(
            tmp_path / "b.json",
            phase="p2b",
            variant=phase2b.VARIANT_CLASSBALANCED,
        ),
        summary_file(
            tmp_path / "c.json",
            phase="p2c",
            formulation=phase2c.FORMULATION_VERSION,
        ),
        summary_file(
            tmp_path / "d.json",
            phase="p2d",
            formulation=phase2d.FORMULATION_VERSION,
        ),
    ]
    changed = json.loads(paths[-1].read_text(encoding="utf-8"))
    changed["oracle_distribution"] = {"10": 2, "160": 0}
    phase2.atomic_json(paths[-1], changed)
    with pytest.raises(RuntimeError, match="same validation set"):
        post.compare_summaries(*paths)
