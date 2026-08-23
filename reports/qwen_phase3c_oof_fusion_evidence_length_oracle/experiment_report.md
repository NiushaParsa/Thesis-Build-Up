# Phase 3C-OOF experiment report

## Objective

Rerun the primary Phase 3C Qwen/tree fusion as a methodologically clean
stacking experiment. The key correction is that every training-row Qwen logit
must be produced out of fold at the paper level. No question or paper may be
seen by the Qwen model that generates that row's five logits, and the preserved
924-question validation split must remain untouched until the entire procedure
is frozen using training data only.

## Artifact preflight

The following prerequisites were present and passed their frozen hashes or
saved-configuration checks:

- 2,245 Phase 3A/3B training feature rows and 924 validation rows;
- 845 training papers and 277 validation papers;
- 173 inference-safe similarity-tree features per row;
- the deterministic five-fold paper grouping used by Phase 3B;
- Phase 2D Qwen model, prompt, tokenizer, optimization configuration, and
  preserved summaries;
- original Phase 3C primary XGBoost candidate;
- Phase 2D, Phase 3B, original Phase 3C, and fixed-granularity reference
  summaries;
- existing Qdrant container and unchanged paper, question, evidence, and chunk
  collections.

Frozen Phase 3A/3B feature hashes:

- train: `6d55e1d10872c8db24cf9af9becfb8e2e6570e13a7697151febc7f44ecebdd9c`
- validation: `548e3cccab3b19dee644eb9858081ff380b6375765433f1d2369c6d7d2ecb893`

Fold assignment hash:
`892a44f57a11c6fa9be7ec708df3355f4247555bde319c61325c0d201492ee62`.

## Method

### OOF Qwen features

The same five paper-grouped folds from Phase 3B were reconstructed
deterministically. Five fresh `Qwen/Qwen3.5-0.8B-Base` sequence classifiers
were trained. For fold *k*, Qwen was trained on every training paper outside
fold *k* and produced logits only for questions in fold *k*. Each completed
fold stored its train/held-out paper lists, question IDs, SHA-256 provenance,
training history, and raw five-logit matrix.

| Fold | Qwen train examples | Qwen train papers | Held-out examples | Held-out papers | Steps | Paper overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,796 | 676 | 449 | 169 | 171 | 0 |
| 1 | 1,796 | 678 | 449 | 167 | 171 | 0 |
| 2 | 1,796 | 676 | 449 | 169 | 171 | 0 |
| 3 | 1,795 | 675 | 450 | 170 | 171 | 0 |
| 4 | 1,797 | 675 | 448 | 170 | 171 | 0 |

Assembly confirmed 2,245 unique question IDs with exact once-only coverage and
correct row, document, label, and fold alignment. The assembled OOF matrix hash
is `ee73c962d45b86034da7dfa903f9ff9ce92abd5707c6cec3b49a6ca7d1f4115d`.

### Qwen configuration

- model: `Qwen/Qwen3.5-0.8B-Base`
- revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Transformers: `5.15.0.dev0`
- Transformers commit: `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`
- Python: 3.10.7
- PyTorch: 2.8.0+cu128
- XGBoost: 3.0.2
- hardware: one NVIDIA A100-SXM4-40GB
- dtype: `torch.bfloat16`
- quantization: none
- objective: uniform five-class cross-entropy
- maximum sequence length: 128
- per-device batch size: 4
- gradient accumulation: 8, effective batch size 32
- learning rate: 2e-5
- weight decay: 0.01
- scheduler: cosine, warm-up ratio 0.05
- epochs: 3
- gradient clipping: 1.0
- seed: 42
- input: unchanged Phase 2D token-count routing instruction plus the original
  question text only

There was no validation evaluation, optimizer decision, early stopping, or
checkpoint selection during any OOF or full-refit Qwen training command.

### Full Qwen refit and validation lock

A sixth Qwen classifier was trained from the pristine base revision on all
2,245 training questions for the fixed three epochs/213 optimizer steps. It
reproduced the preserved Phase 2D model hash exactly:
`020af0a83af773239e7e60e9983afad29cae3f31493c7073e9162e040b732814`.

Before validation rows were opened for inference, `selection_lock.json` froze:

- all five fold and assembled-logit hashes;
- the full-refit model hash;
- the 178-feature schema;
- raw concatenation with no scaling or thresholding;
- square-root inverse-frequency class weighting;
- seed 42;
- the original primary Phase 3C XGBoost parameters.

The lock timestamp is `2026-08-23T19:06:59.732303+00:00`; validation feature
extraction completed later at `2026-08-23T19:07:17.745010+00:00`.

### Fusion model

The primary fusion input was five clean Qwen logits plus the unchanged 173
Phase 3B tree features. The fixed XGBoost candidate was depth 2, learning rate
0.05, 200 trees, minimum child weight 5, subsample 0.8, column subsample 0.8,
L2 1, and L1 0. No new search or variant selection was performed.

A grouped train-only meta-model diagnostic was retained, but it did not select
or alter any setting. Because base fits used to create other meta-training rows
can have seen the current meta-held-out papers, this diagnostic is not described
as a fully nested performance estimate; the untouched 924-question validation
result is definitive.

## Classification results

| Metric | Phase 3C-OOF |
|---|---:|
| Accuracy | 0.3225108225108225 |
| Macro-F1 | 0.20998522506102316 |
| Weighted F1 | 0.32674104553954336 |
| Balanced accuracy | 0.21785914500689135 |
| Top-2 accuracy | 0.6103896103896104 |
| Correct | 298/924 |

| Class | Precision | Recall | F1 | Support | Predicted |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.000000 | 0.000000 | 0.000000 | 13 | 9 |
| 20 | 0.105263 | 0.049383 | 0.067227 | 81 | 38 |
| 40 | 0.209091 | 0.258427 | 0.231156 | 178 | 220 |
| 80 | 0.257143 | 0.426724 | 0.320908 | 232 | 385 |
| 160 | 0.547794 | 0.354762 | 0.430636 | 420 | 272 |

Confusion matrix, rows Oracle and columns prediction in order
10/20/40/80/160:

```text
  0   1   5   6   1
  2   4  27  28  20
  4  10  46  77  41
  2  11  59  99  61
  1  12  83 175 149
```

The fixed-candidate train-only diagnostic is accuracy 0.304232 and macro-F1
0.231679. The original Phase 3C train OOF diagnostic was 0.471715/0.383835;
its much higher value depended on in-sample Qwen logits and was optimistic for
stacking assessment.

## Retrieval results

- evaluated questions: 924/924
- coverage: 100%
- top-k: 5
- source-paper restriction: true
- mean joined retrieval F1: 0.27830322619047615
- median joined retrieval F1: 0.255051
- embedding model: `text-embedding-3-small`, 1,536 dimensions
- similarity: cosine
- Qdrant collection state: unchanged before/after

## Comparison

| Method | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Mean joined retrieval F1 |
|---|---:|---:|---:|---:|---:|
| Phase 3C-OOF | 0.322511 | 0.209985 | 0.326741 | 0.217859 | 0.278303 |
| Original Phase 3C | 0.319264 | 0.230677 | 0.333987 | 0.235850 | 0.287202 |
| Phase 2D | 0.369048 | 0.229945 | 0.364466 | 0.239181 | 0.276717 |
| Phase 3B | 0.329004 | 0.224670 | 0.329620 | 0.232723 | 0.271721 |

The clean fusion is 0.008898 below original Phase 3C retrieval F1, 0.001587
above Phase 2D, and 0.006582 above Phase 3B. It remains 0.037553 below fixed
40, the strongest fixed operational baseline.

| Fixed level | Mean joined retrieval F1 |
|---:|---:|
| 10 | 0.22111944047619048 |
| 20 | 0.28874969588744587 |
| 40 | 0.3158559274891775 |
| 80 | 0.2857245205627706 |
| 160 | 0.21679366125541125 |

## Conclusion

The methodological concern was valid. Replacing in-sample Phase 2D training
logits with strict paper-grouped OOF logits substantially lowers the apparent
train-side stacking performance. On untouched validation, the clean fusion
does not improve macro-F1 or downstream retrieval over original Phase 3C. It
does preserve a small retrieval gain over Phase 2D and Phase 3B individually,
but fixed 40 remains clearly stronger. Consequently, the evidence does not
support adopting this fusion as an operational adaptive router.

This result does not imply that the component models are unusable for every
formulation. It shows that, under the evidence-length five-class target and the
frozen Phase 3C architecture, hyperparameters, and inference setup, leakage-safe
stacking does not resolve the target-alignment and learnability limitations.

## Runtime and resources

| Stage | Seconds |
|---|---:|
| Five outer Qwen fits | 5,888.457 |
| Full Qwen refit | 1,433.618 |
| Validation Qwen extraction | 8.111 |
| XGBoost diagnostic/fit/evaluation | 9.938 |
| Retrieval | 372.007 |
| Known sequential total | 7,712.131 |

Known total is approximately 2 h 8 m 32 s. Maximum CUDA allocation was about
9.03 GiB, maximum CUDA reservation about 9.60 GiB, and approximate peak process
RSS about 2.09 GiB.

## Reproduction commands

GPU host:

```bash
bash scripts/run_phase3c_oof_remote.sh /workspace/thesis-granularity-router
```

Qdrant host after transferring the output directory:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase3c_oof.py retrieve
.\.venv-qwen\Scripts\python.exe qwen_phase3c_oof.py finalize
python -m unittest tests.test_qwen_phase3c_oof -v
```

## Artifact index

The canonical index with hashes is
`outputs/qwen_phase3c_oof_fusion_evidence_length_oracle/final_summary.json`.
The main supporting artifacts are the selection lock, fold manifests and
histories, OOF and validation logit arrays, XGBoost model and metadata,
validation predictions, classification metrics/confusion matrix/histogram,
retrieval records and summary, comparison table, runtime summary, and final
integrity audit under the same output root.

The new clean-protocol tests and the preserved Phase 3B/Phase 3C regression
tests were run in the remote experiment environment: 15 passed.
