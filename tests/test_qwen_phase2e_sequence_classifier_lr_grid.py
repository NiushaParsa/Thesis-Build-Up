import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

import qwen_phase2d_sequence_classifier as phase2d
import qwen_phase2e_sequence_classifier_lr_grid as phase2e


def _preserve_phase2d_runtime(monkeypatch):
    for name in (
        "FORMULATION_VERSION",
        "DEFAULT_OUTPUT_ROOT",
        "TrainingConfig",
        "ensure_output_root",
        "experiment_fingerprint",
        "_run_id",
    ):
        monkeypatch.setattr(phase2d, name, getattr(phase2d, name))


def _write_resume_fixture(root, variant="lr1e-5", step=71):
    run_dir = root / "runs" / phase2e.RUN_IDS[variant]
    checkpoint = run_dir / "checkpoints" / f"step-{step:06d}"
    (checkpoint / "model").mkdir(parents=True)
    for name in ("optimizer.pt", "scheduler.pt", "random_states.pt"):
        (checkpoint / name).write_bytes(b"fixture")
    phase2e.phase2.atomic_json(
        checkpoint / "training_state.json",
        {"global_step": step, "resume_contract_sha256": "contract"},
    )
    phase2e.phase2.atomic_json(
        run_dir / "training_config.json",
        {
            "resume_contract_sha256": "contract",
            "experiment_fingerprint": "fingerprint",
        },
    )
    phase2e.phase2.atomic_json(
        run_dir / "checkpoint_manifest.json",
        [
            {
                "checkpoint_id": checkpoint.name,
                "global_step": step,
                "experiment_fingerprint": "fingerprint",
            }
        ],
    )
    return checkpoint


def _write_completed_run(study, variant="lr1e-5", **config_overrides):
    output_root = phase2e.trial_root(study, variant)
    run_id = phase2e.RUN_IDS[variant]
    run_dir = output_root / "runs" / run_id
    config = asdict(phase2e.config_for_variant(variant))
    config.update(config_overrides)
    data = {
        "train_oracle_sha256": phase2e.TRAIN_ORACLE_SHA256,
        "validation_oracle_sha256": phase2e.VALIDATION_ORACLE_SHA256,
        "active_train_examples": 2245,
        "active_validation_examples": 924,
    }
    fingerprint = phase2e.trial_fingerprint(data, 248044, variant)
    training = {
        **config,
        "run_id": run_id,
        "run_mode": "full",
        "active_per_device_batch_size": 4,
        "active_gradient_accumulation_steps": 8,
        "active_effective_batch_size": 32,
        "maximum_optimizer_steps": None,
        "total_optimizer_steps": 355,
        "warmup_steps": 18,
        "pad_token_id": 248044,
        "experiment_fingerprint": fingerprint,
        "initial_model_loading_audit": {
            "classifier_head": {
                "head_weight_sha256_float32": phase2e.EXPECTED_INITIAL_HEAD_SHA256
            }
        },
    }
    best = {
        "checkpoint_id": "step-000355",
        "global_step": 355,
        "experiment_fingerprint": fingerprint,
    }
    summary = {
        "status": "complete",
        "phase": "Phase 2D",
        "run_id": run_id,
        "global_step": 355,
        "validation_events": 5,
        "selected_checkpoint_id": "step-000355",
        "experiment_fingerprint": fingerprint,
    }
    data["experiment_fingerprint"] = fingerprint
    for name, value in {
        "training_config.json": training,
        "dataset_manifest.json": data,
        "best_checkpoint.json": best,
        "summary.json": summary,
    }.items():
        phase2e.phase2.atomic_json(run_dir / name, value)
    return run_dir


def _candidate(
    *,
    variant="lr1e-5",
    macro=0.2,
    accuracy=0.3,
    weighted=0.25,
    balanced=0.22,
    loss=1.4,
    step=355,
):
    return {
        "variant": variant,
        "learning_rate": phase2e.LEARNING_RATES[variant],
        "global_step": step,
        "validation_loss": loss,
        "classification_metrics": {
            "macro_f1": macro,
            "accuracy": accuracy,
            "weighted_f1": weighted,
            "balanced_accuracy": balanced,
        },
    }


def test_exact_predeclared_grid_schedule_and_run_ids():
    assert phase2e.VARIANT_ORDER == ("lr5e-6", "lr1e-5", "lr2e-5")
    assert phase2e.LEARNING_RATES == {
        "lr5e-6": 5e-6,
        "lr1e-5": 1e-5,
        "lr2e-5": 2e-5,
    }
    assert phase2e.EXPECTED_STEPS == (71, 142, 213, 284, 355)
    assert phase2e.EXPECTED_TOTAL_STEPS == 355
    assert phase2e.EXPECTED_WARMUP_STEPS == 18
    assert set(phase2e.RUN_IDS) == set(phase2e.VARIANT_ORDER)
    assert len(set(phase2e.RUN_IDS.values())) == 3
    for variant, run_id in phase2e.RUN_IDS.items():
        assert variant in run_id
        assert "5epochs" in run_id
        assert "seed42" in run_id


def test_configs_differ_from_phase2d_only_by_declared_fields():
    phase2d_config = asdict(phase2d.TrainingConfig())
    for variant in phase2e.VARIANT_ORDER:
        config = asdict(phase2e.config_for_variant(variant))
        differences = {
            key for key in config if config[key] != phase2d_config[key]
        }
        expected = {"formulation_version", "epochs", "early_stopping"}
        if phase2e.LEARNING_RATES[variant] != phase2d_config["learning_rate"]:
            expected.add("learning_rate")
        assert differences == expected
        assert config["learning_rate"] == phase2e.LEARNING_RATES[variant]
        assert config["epochs"] == 5
        assert config["early_stopping"] == "none_fixed_five_epochs"
        assert config["class_weights"] == "uniform"


def test_model_prompt_mapping_and_base_source_are_frozen():
    assert phase2e.PROMPT_SHA256 == hashlib.sha256(
        phase2d.SUPERVISOR_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    assert phase2d.MODEL_ID == "Qwen/Qwen3.5-0.8B-Base"
    assert phase2d.MODEL_REVISION == (
        "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    )
    assert phase2d.ID_TO_LABEL == {0: 10, 1: 20, 2: 40, 3: 80, 4: 160}
    assert phase2d.LABEL_TO_ID == {10: 0, 20: 1, 40: 2, 80: 3, 160: 4}
    assert phase2e.EXPECTED_INITIAL_HEAD_SHA256 == (
        "09826669f451891218742ea86926e0b484d1696e57999276889d97b5ccdcbda5"
    )
    assert hashlib.sha256(Path(phase2d.__file__).read_bytes()).hexdigest() == (
        "99ba4f9a57b5547e776d81d2c7c94aece2670a9b5ccaf348c8a1fbeb0baa5efc"
    )


def test_grid_and_trial_fingerprints_are_exact_and_variant_specific():
    manifest = {
        "train_oracle_sha256": phase2e.TRAIN_ORACLE_SHA256,
        "validation_oracle_sha256": phase2e.VALIDATION_ORACLE_SHA256,
    }
    grid = phase2e.grid_fingerprint()
    assert len(grid) == 64
    trials = {
        variant: phase2e.trial_fingerprint(manifest, 248044, variant)
        for variant in phase2e.VARIANT_ORDER
    }
    assert len(set(trials.values())) == 3
    assert all(len(value) == 64 and value != grid for value in trials.values())
    with pytest.raises(RuntimeError, match="train Oracle hash drifted"):
        phase2e.trial_fingerprint(
            {**manifest, "train_oracle_sha256": "bad"}, 248044, "lr1e-5"
        )


def test_rejects_unknown_variant_and_prior_output_overlap(tmp_path):
    with pytest.raises(ValueError, match="Unknown Phase 2E variant"):
        phase2e.config_for_variant("lr3e-5")
    with pytest.raises(RuntimeError, match="prior experiment root"):
        phase2e._validate_study_root(phase2d.DEFAULT_OUTPUT_ROOT)
    with pytest.raises(RuntimeError, match="prior experiment root"):
        phase2e._validate_study_root(
            phase2d.DEFAULT_OUTPUT_ROOT / "phase2e"
        )
    root = phase2e._validate_study_root(tmp_path / "phase2e")
    assert root == tmp_path / "phase2e"


def test_prepare_and_trial_markers_are_immutable(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASE2E_REPOSITORY_COMMIT", "test-commit")
    root = tmp_path / "phase2e"
    grid = phase2e.prepare_study(root)
    assert grid["status"] == "predeclared_before_training"
    assert grid["ordered_variants"] == list(phase2e.VARIANT_ORDER)
    assert grid["retrieval_selection_rule"].startswith(
        "lock the classification winner"
    )
    ensure = phase2e._ensure_trial_root_factory(root, "lr1e-5")
    marker = ensure(phase2e.trial_root(root, "lr1e-5"))
    assert marker["phase"] == "Phase 2E"
    assert marker["learning_rate"] == 1e-5
    assert marker["epochs"] == 5
    with pytest.raises(RuntimeError, match="must write exactly"):
        ensure(root / "trials" / "wrong")


def test_all_variants_activate_sequentially_without_mutating_protected_root(
    tmp_path, monkeypatch
):
    protected = phase2e._PHASE2D_DEFAULT_OUTPUT_ROOT
    _preserve_phase2d_runtime(monkeypatch)
    monkeypatch.setenv("PHASE2D_REPOSITORY_COMMIT", "restore-after-test")

    study = tmp_path / "phase2e"
    for variant in phase2e.VARIANT_ORDER:
        config, root = phase2e.activate_variant(study, variant)
        assert root == study / "trials" / variant
        assert config.learning_rate == phase2e.LEARNING_RATES[variant]
        assert phase2e._PHASE2D_DEFAULT_OUTPUT_ROOT == protected
        assert protected in phase2e._prior_output_roots()


def test_selection_uses_frozen_lexicographic_order_and_lower_lr_final_tie():
    base = _candidate(variant="lr1e-5")
    better_macro = _candidate(variant="lr2e-5", macro=0.21, accuracy=0.1)
    assert phase2e.select_candidate_rows([base, better_macro])[
        "variant"
    ] == "lr2e-5"

    better_accuracy = _candidate(variant="lr2e-5", accuracy=0.31)
    assert phase2e.select_candidate_rows([base, better_accuracy])[
        "variant"
    ] == "lr2e-5"

    lower_loss = _candidate(variant="lr2e-5", loss=1.3)
    assert phase2e.select_candidate_rows([base, lower_loss])[
        "variant"
    ] == "lr2e-5"

    earlier = _candidate(variant="lr2e-5", step=284)
    assert phase2e.select_candidate_rows([base, earlier])[
        "variant"
    ] == "lr2e-5"

    exact_low = _candidate(variant="lr5e-6")
    exact_mid = _candidate(variant="lr1e-5")
    exact_high = _candidate(variant="lr2e-5")
    assert phase2e.select_candidate_rows([exact_high, exact_mid, exact_low])[
        "variant"
    ] == "lr5e-6"

    invalid = _candidate(macro=float("nan"))
    with pytest.raises(RuntimeError, match="Invalid Phase 2E selection candidate"):
        phase2e.select_candidate_rows([invalid])


def test_cli_exposes_only_predeclared_variants():
    parser = phase2e.build_parser()
    assert parser.parse_args(["prepare"]).command == "prepare"
    train = parser.parse_args(["train", "--variant", "lr1e-5"])
    assert train.variant == "lr1e-5"
    assert train.resume is None
    resume = parser.parse_args(["resume-latest", "--variant", "lr1e-5"])
    assert resume.command == "resume-latest"
    audit = parser.parse_args(["audit-completed", "--variant", "lr2e-5"])
    assert audit.command == "audit-completed"
    assert audit.variant == "lr2e-5"
    assert parser.parse_args(["select"]).command == "select"
    assert parser.parse_args(["final-selected"]).command == "final-selected"
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--variant", "lr3e-5"])


def test_resume_checkpoint_must_belong_to_exact_trial(tmp_path):
    root = phase2e.trial_root(tmp_path / "study", "lr1e-5")
    expected = _write_resume_fixture(root)
    assert phase2e.validate_resume_checkpoint(
        root, "lr1e-5", expected
    ) == expected.resolve()

    other = tmp_path / "other" / "checkpoints" / "step-000071"
    other.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="inside .*checkpoints"):
        phase2e.validate_resume_checkpoint(root, "lr1e-5", other)


def test_latest_resume_ignores_and_quarantines_orphan_checkpoint(tmp_path):
    root = phase2e.trial_root(tmp_path / "study", "lr1e-5")
    expected = _write_resume_fixture(root)
    orphan = expected.parent / "step-000142"
    (orphan / "model").mkdir(parents=True)
    for name in ("optimizer.pt", "scheduler.pt", "random_states.pt"):
        (orphan / name).write_bytes(b"orphan")
    phase2e.phase2.atomic_json(
        orphan / "training_state.json",
        {"global_step": 142, "resume_contract_sha256": "contract"},
    )
    assert phase2e.latest_manifest_backed_checkpoint(root, "lr1e-5") == expected
    with pytest.raises(RuntimeError, match="not uniquely manifest-backed"):
        phase2e.validate_resume_checkpoint(root, "lr1e-5", orphan)
    moved = phase2e._quarantine_unmanifested_checkpoints(root, "lr1e-5")
    assert len(moved) == 1
    assert not orphan.exists()
    assert Path(moved[0]).name == "step-000142"


@pytest.mark.parametrize(
    ("field", "wrong"), (("learning_rate", 9e-5), ("epochs", 99))
)
def test_completed_run_audit_rejects_core_config_without_rewriting(
    tmp_path, monkeypatch, field, wrong
):
    _preserve_phase2d_runtime(monkeypatch)
    study = tmp_path / "phase2e"
    run_dir = _write_completed_run(study, **{field: wrong})
    path = run_dir / "training_config.json"
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="completed-run audit failed"):
        phase2e._augment_run_artifacts(study, "lr1e-5")
    assert path.read_bytes() == before


def test_completed_run_audit_adds_namespaced_metadata_after_validation(
    tmp_path, monkeypatch
):
    _preserve_phase2d_runtime(monkeypatch)
    study = tmp_path / "phase2e"
    run_dir = _write_completed_run(study)
    result = phase2e._augment_run_artifacts(study, "lr1e-5")
    training = json.loads((run_dir / "training_config.json").read_text())
    assert training["learning_rate"] == 1e-5
    assert training["epochs"] == 5
    assert training["phase2e_metadata"]["variant"] == "lr1e-5"
    assert result["phase"] == "Phase 2E"
    assert result["execution_implementation_phase"] == "Phase 2D"


def test_learning_rates_are_positive_finite_and_ordered():
    values = [phase2e.LEARNING_RATES[name] for name in phase2e.VARIANT_ORDER]
    assert values == sorted(values)
    assert all(math.isfinite(value) and value > 0 for value in values)
