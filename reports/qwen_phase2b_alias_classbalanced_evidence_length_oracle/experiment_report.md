# Qwen3.5-0.8B Phase 2B-B: class-balanced restricted-alias router

## Status and objective

Phase 2B-B is complete. It evaluates whether effective-number class weighting
improves the restricted-alias formulation of the full-parameter
`Qwen/Qwen3.5-0.8B` router. The authoritative variant and run are:

- Variant: `alias-classbalanced`
- Run ID: `qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1`
- Formulation: `qwen-phase2b-restricted-five-alias-next-token-v1`
- Status in `final_summary.json`: `complete`

The run used all 2,245 preserved training questions, evaluated every epoch on
all 924 preserved validation questions, selected the checkpoint using the
predeclared validation macro-F1 rule, reloaded that checkpoint for a separate
final validation pass, and completed the unchanged downstream retrieval
evaluation for all 924 valid predictions. Phase 1, numeric-target Phase 2, and
Phase 2B-A artifacts were not overwritten.

## Restricted-alias formulation

The Phase 2B instruction is:

> You are a router for a retrieval-augmented generation system. Based only on
> the question, select the chunk size most suitable for retrieving the
> evidence required to answer it. Return only its alias: 1=10, 2=20, 3=40,
> 4=80, 5=160.

Its SHA-256 is
`d4a59dcd26b01c5bd81981e43c5d69fc1c8db14e9160e0791412dbe6af7067ac`.
The fixed alias mapping and verified tokenizer IDs are:

| Alias | Chunk size | Token ID |
|---:|---:|---:|
| 1 | 10 | 16 |
| 2 | 20 | 17 |
| 3 | 40 | 18 |
| 4 | 80 | 19 |
| 5 | 160 | 20 |

Each alias is exactly one token both in isolation and at the first assistant
answer position under the saved Qwen chat template. Training gathers the
language-model vocabulary logits at that position and restricts the objective
to those five token logits. Inference ranks the same five directly comparable
logits and deterministically selects their argmax, resolving an exact score tie
by the smaller alias. It is not unrestricted text generation, and the legacy
free-text output parser is not part of this formulation. Aliases are mapped
back to chunk sizes before Oracle-label scoring and retrieval.

Only the fixed instruction and original question text are model inputs. The
model did not receive evidence, evidence length, the Oracle label as an input,
answers, paper text, retrieved chunks, retrieval scores, embeddings, metadata,
or handcrafted features. Saved prompt lengths across the training split are
89--115 tokens, with mean `95.12026726057907`, below the maximum sequence
length of 128.

## Frozen data and evidence-length Oracle

Phase 2B-B reuses the frozen evidence-length Oracle introduced for the Qwen
experiments. For each question, the Oracle counts GPT-2 tokens in the complete
deduplicated ground-truth evidence and chooses the nearest candidate from 10,
20, 40, 80, and 160. Exact midpoint ties choose the smaller candidate; values
below 10 map to 10 and values above 160 map to 160. This Oracle is independent
of retrieval F1, embedding quality, cosine similarity, retrieved chunks, and
router performance.

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

The frozen Oracle hashes are:

- Train:
  `64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`
- Validation:
  `ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`

The validation labels are strongly imbalanced: class 160 contains 420/924
questions (45.45%), whereas class 10 contains only 13/924 (1.41%). Class
weights were computed from the preserved training counts only; validation
labels never contributed to the weighting calculation or gradient updates.

## Effective-number objective

Phase 2B-B is the class-balanced member of the two-variant Phase 2B design. For
training class count \(n_c\) and fixed \(\beta=0.999\), its unnormalized
effective-number weight is

\[
w_c = \frac{1-\beta}{1-\beta^{n_c}}.
\]

The five weights were normalized to arithmetic class mean one. Training then
used the following weighted mean over every complete gradient-accumulation
window, including the five-example tail at the end of each epoch:

\[
\mathcal{L} =
\frac{\sum_i w_{y_i}\,\mathrm{CE}(z_i,y_i)}
     {\sum_i w_{y_i}}.
\]

| Chunk class | Training count | Saved normalized weight |
|---:|---:|---:|
| 10 | 55 | 3.1872088653568436 |
| 20 | 267 | 0.7279213406697836 |
| 40 | 586 | 0.38467220811977887 |
| 80 | 687 | 0.34329010532422555 |
| 160 | 650 | 0.3569074805293684 |

The saved reduction is exactly
`sum(weight[target] * per_example_ce) / sum(weight[target])`. Normalizing all
five weights by one common constant changes their reported scale but not that
weighted-mean gradient. No over- or undersampling was used.

## Model, environment, and training configuration

- Model: `Qwen/Qwen3.5-0.8B`
- Revision: `2fc06364715b967f1860aea9cf38778875588b17`
- Total/trainable parameters: 852,985,920 / 852,985,920 (100%)
- Training method: full-parameter restricted-alias classification
- Environment: `.venv-qwen`
- Python: `3.10.7 (main, Oct 3 2022, 02:19:58) [Clang 14.0.3]`
- Python executable: `/workspace/thesis-granularity-router/.venv-qwen/bin/python`
- Transformers: `5.15.0.dev0`, commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`
- PyTorch: `2.8.0+cu128`; CUDA 12.8
- TensorBoard: `2.20.0`
- Device/dtype: one NVIDIA A100-SXM4-40GB, CUDA, `torch.bfloat16`
- Quantization: none
- Repository commit recorded at launch:
  `55af1bcbc4d7a089adaafd4da539581b2dbbed67`
- Training-script SHA-256:
  `60572d8c3054e7ef76055b2c40cf65c2999ef18000930f5a6967fd2ae673041c`
- Experiment fingerprint:
  `91021dd5de4bb5186ce98ac873dcfae9200f721a62f5a3163f7958cceb240fe6`

The protected legacy Python 3.9 `.venv` remained unchanged. This experiment
used only the separate remote CUDA `.venv-qwen` and did not modify system
Python or the frozen Phase 1/Phase 2 environments.

| Setting | Saved value |
|---|---|
| Objective | Restricted five-logit cross-entropy with effective-number weights |
| Optimizer | AdamW over all parameters |
| Epochs | 3 fixed; no early stopping |
| Optimizer/parameter-update steps | 71 per epoch; 213 total |
| Per-device batch size | 4 |
| Gradient accumulation | 8 microbatches |
| Effective batch size | 32 examples |
| Maximum sequence length | 128 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Scheduler | Cosine |
| Warmup ratio | `0.05` |
| Gradient clipping | `1.0` |
| Seed | 42 |
| Logging | Every optimizer step |
| Validation/checkpoint | End of every epoch |
| Selection metric | Validation macro-F1 |
| Tie-break | Accuracy, weighted F1, balanced accuracy, lower unweighted validation CE, earlier step |
| Retention | Current and best during training; selected checkpoint only at completion |

The first and final recorded weighted training objectives were
`1.816834687228543` and `1.602007440099642`. Training performed 213 optimizer
steps and 213 parameter updates. It used no LoRA, QLoRA, adapters, prompt
tuning, classification head, partial-layer freezing, or quantization.

## Epoch validation and checkpoint selection

Every epoch event evaluated all 924 preserved validation questions. The table
reports both unweighted validation cross-entropy and the class-weighted
training objective evaluated on validation. Predicted distributions are in
chunk-size order `10/20/40/80/160`.

| Epoch | Step | Unweighted CE | Weighted-objective CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 accuracy | Predicted distribution | Invalid | Wall (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 71 | 1.5536545303476836 | 1.5932705140456938 | 0.19264069264069264 | 0.0646098003629764 | 0.06223238346650324 | 0.2 | 0.4945887445887446 | 0/0/924/0/0 | 0 | 35.651938546448946 |
| 2 | 142 | 1.4620294988929452 | 1.550350315119168 | 0.37012987012987014 | **0.16836616836616836** | 0.3142183142183142 | 0.20607553366174058 | 0.7056277056277056 | 0/0/0/434/490 | 0 | 35.897498374804854 |
| 3 | 213 | 1.4841528165908087 | 1.5570664912477334 | 0.2683982683982684 | 0.11060887911611213 | 0.1660641330693655 | 0.20119628789136734 | 0.7088744588744589 | 0/0/7/851/66 | 0 | 35.34139230288565 |

`step-000142` (epoch 2) was selected because its macro-F1,
`0.16836616836616836`, is higher than the epoch-1 and epoch-3 values. No
tie-break was needed. Epoch 3's slightly higher top-2 accuracy did not override
the predeclared primary selection metric. Retrieval F1 was not used to choose
the checkpoint.

The selected checkpoint is:

`outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/checkpoints/step-000142`

The selected checkpoint was loaded afresh after training. Its restricted-logit
scores for all 924 questions exactly matched the selected epoch's saved scores
(`selected_epoch_exact_score_match: true`; 924 outputs compared).

## Final classification results

| Metric | Phase 2B-B class-balanced | Evidence-length majority baseline |
|---|---:|---:|
| Accuracy | 0.37012987012987014 | 0.45454545454545453 |
| Macro-F1 | 0.16836616836616836 | 0.125 |
| Weighted F1 | 0.3142183142183142 | Not recorded in the Phase 2B summary |
| Balanced accuracy | 0.20607553366174058 | Not recorded in the Phase 2B summary |
| Top-2 accuracy | 0.7056277056277056 | Not applicable |
| Valid outputs | 924/924 (100%) | 924/924 |
| Invalid outputs | 0 (0.0%) | 0 |

Top-2 accuracy is available here because each example stores five directly
comparable restricted next-token logits. It is the fraction for which the
Oracle class appears among the two highest-scoring mapped chunk classes. This
must not be approximated for Phase 1 or numeric-target Phase 2, whose
generation procedures did not produce comparable five-class scores.

### Per-class results and distributions

| Class | Precision | Recall | F1 | Oracle support | Predictions |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 | 0 |
| 20 | 0.0 | 0.0 | 0.0 | 81 | 0 |
| 40 | 0.0 | 0.0 | 0.0 | 178 | 0 |
| 80 | 0.25806451612903225 | 0.4827586206896552 | 0.3363363363363363 | 232 | 434 |
| 160 | 0.46938775510204084 | 0.5476190476190477 | 0.5054945054945055 | 420 | 490 |

The final Oracle distribution is `13/81/178/232/420`; the final predicted
distribution is `0/0/0/434/490`, both ordered `10/20/40/80/160`. Thus the
class-balanced run emitted only classes 80 and 160. It did not recover any
class-10, class-20, or class-40 predictions.

### Confusion matrix

Rows are Oracle labels and columns are predictions, ordered 10, 20, 40, 80,
160.

| Oracle ↓ / predicted → | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 0 | 9 | 4 |
| 20 | 0 | 0 | 0 | 39 | 42 |
| 40 | 0 | 0 | 0 | 84 | 94 |
| 80 | 0 | 0 | 0 | 112 | 120 |
| 160 | 0 | 0 | 0 | 190 | 230 |

## Controlled comparison with Phase 2B-A

The saved Phase 2B-A and Phase 2B-B training configurations agree on the
model/revision, repository and training-script identities, frozen Oracle
counts and hashes, instruction and instruction hash, alias mapping and token
IDs, input exclusions, optimization hyperparameters, seed, epochs, evaluation
schedule, checkpoint-selection rule, and retrieval configuration. Their
intended methodological difference is the loss weighting: uniform weights in
Phase 2B-A and the effective-number weights above in Phase 2B-B. Run/output
identities, weighting fields, resume-contract hashes, and experiment
fingerprints necessarily differ.

| Metric | Phase 2B-A unweighted | Phase 2B-B class-balanced |
|---|---:|---:|
| Selected checkpoint | step 213 | step 142 |
| Accuracy | 0.35064935064935066 | 0.37012987012987014 |
| Macro-F1 | **0.20922603632601472** | 0.16836616836616836 |
| Weighted F1 | **0.3406050804511769** | 0.3142183142183142 |
| Balanced accuracy | **0.2383201416948027** | 0.20607553366174058 |
| Top-2 accuracy | 0.6071428571428571 | **0.7056277056277056** |
| Predicted 10/20/40/80/160 | 0/0/427/189/308 | 0/0/0/434/490 |
| Mean joined retrieval F1 | **0.28646775432900434** | 0.24962774025974027 |
| Median joined retrieval F1 | **0.2748425** | 0.223194 |

Class balancing raised accuracy and top-2 accuracy in this single run, but it
reduced macro-F1, weighted F1, balanced accuracy, and joined retrieval F1
relative to the unweighted alias run. It also removed the unweighted run's 427
class-40 predictions without producing class-10 or class-20 predictions. The
result therefore does not support the hypothesis that this particular
effective-number setting improves minority-sensitive top-1 routing under the
selected metric. That conclusion is limited to this seed, schedule, and
validation-selected checkpoint; it is not a general rejection of class
weighting.

## Unchanged end-to-end retrieval evaluation

The selected aliases were mapped to chunk sizes before retrieval. The existing
Qdrant collections were reused read-only with the unchanged evaluation path:
retrieval was restricted to the source paper, used the predicted granularity,
`top-k=5`, existing `text-embedding-3-small` 1,536-dimensional embeddings,
the unchanged similarity ranking, chunk ordering and concatenation, and GPT-2
joined token-level retrieval F1 (`qasper-token-prf-v2`).

| Retrieval metric | Value |
|---|---:|
| Coverage | 924/924 (1.0) |
| Valid-prediction retrievals | 924 |
| Invalid predictions without retrieval | 0 |
| Valid-only mean joined retrieval F1 | 0.24962774025974027 |
| Valid-only median joined retrieval F1 | 0.223194 |
| Coverage-adjusted full-set mean joined retrieval F1 | 0.24962774025974016 |
| Reported retrieval wall time | 377.0999227000284 s |

Because every restricted prediction was valid, valid-only coverage is 100%
and the valid-only and coverage-adjusted means differ only by floating-point
aggregation. Had an invalid prediction occurred, no default granularity would
have been assigned and no retrieval record would have been created for it.

Classification accuracy, macro-F1, weighted F1, balanced accuracy, and top-2
accuracy measure prediction of the evidence-length Oracle label. Joined
retrieval F1 measures downstream token overlap after same-paper retrieval.
They are different metrics; neither should be reported as the other.

## Runtime and resource use

- Full training elapsed time: `1306.7509042322636` seconds (about 21 min 47 s).
- Selected-checkpoint model load: `3.260306715965271` seconds.
- Isolated final-validation inference: `39.01004763878882` seconds.
- Mean inference time: `0.04196543970197697` seconds/question.
- Median inference time: `0.04170184303075075` seconds/question.
- Known training plus final-validation wall time:
  `1349.0212585870177` seconds.
- Retrieval wall time: `377.0999227000284` seconds.
- Known training, final validation, and retrieval wall time:
  `1726.121181287046` seconds (about 28 min 46 s).
- Training peak allocated/reserved GPU memory:
  `10.053670883178711` / `11.99609375` GiB.
- Training process RSS sample: `1.9857444763183594` GiB.
- Final-validation peak allocated/reserved GPU memory:
  `2.2182140350341797` / `3.6640625` GiB.
- Final-validation process RSS sample: `1.7090835571289062` GiB.

The saved local retrieval ran while the weighted selected-checkpoint archive
was being transferred. Its exact wall time and the derived known-combined
duration are retained for provenance, but they are not clean model- or
method-speed measurements and must not be compared with Phase 2B-A retrieval
as if only the routing method changed. The training and final-validation
timings above are isolated as saved.

## Manual post-transfer integrity gates

No standalone Phase 2B hash-inventory file was saved. A manual `rsync`
checksum dry-run compared every copied remote training, classification,
validation, TensorBoard, and checkpoint file, excluding `final_summary.json`
because local retrieval legitimately extends it. The first balanced-tree check
identified only two stale local preflight copies,
`configuration/experiment.json` and
`configuration/preflight_manifest.json`. Both were replaced from the GPU
source, and a targeted checksum rerun returned no differences; their semantic
configuration was unchanged and only generated timestamps differed. The
selected `step-000142` checkpoint contains 11 files totaling 4,735,895,530
bytes. Completed-summary replay through the retrieval evaluator revalidated
all 924 records and returned `complete`.

Qdrant counts were unchanged before and after the two Phase 2B retrieval/QA
passes: `PaperChunk` 1,701,822; `PaperEvidence` 9,522; `PaperQuestion` 4,526;
`RouterDataset` 3,170; `RetrievalEvaluation` 18,622;
`Stage4VerifyRetrievalEval` 10; `Stage4VerifyRouterDataset` 2; and
`Stage5MixedEvaluation` 2. Frozen Phase 1/2 artifact and script hashes were
also rechecked unchanged. These are manual post-transfer gates, not a claim
that a separate Phase 2B integrity artifact exists.

The configured TensorBoard event directory is
`outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/tensorboard/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1`.
Structured JSON/JSONL/CSV artifacts, rather than visually estimated or
smoothed TensorBoard curves, are the source of the metrics in this report.

## Four-way Qwen comparison and comparability limits

All four rows below use the same preserved 924 validation questions, frozen
evidence-length Oracle, five chunk classes, and downstream retrieval pipeline.

| Run | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 accuracy | Mean joined retrieval F1 | Predicted 10/20/40/80/160 |
|---|---:|---:|---:|---:|---:|---:|---|
| Phase 1 zero-shot | 0.04004329004329004 | 0.049045932422555796 | 0.032612933907418644 | 0.23369399361908724 | Unavailable | 0.23910868506493507 | 767/40/116/0/1 |
| Phase 2 numeric-target SFT | **0.4318181818181818** | 0.16502267760462996 | 0.32805741427623947 | 0.20697865353037764 | Unavailable | 0.22658488852813854 | 0/0/0/149/775 |
| Phase 2B-A alias-unweighted | 0.35064935064935066 | **0.20922603632601472** | **0.3406050804511769** | **0.2383201416948027** | 0.6071428571428571 | **0.28646775432900434** | 0/0/427/189/308 |
| Phase 2B-B alias-classbalanced | 0.37012987012987014 | 0.16836616836616836 | 0.3142183142183142 | 0.20607553366174058 | **0.7056277056277056** | 0.24962774025974027 | 0/0/0/434/490 |

This is a same-Oracle method comparison, not an identical-prompt replication.
Relative to Phase 2, both Phase 2B variants change the instruction's output
schema, encode numeric chunk-size targets as aliases, and replace unrestricted
generated text plus parser inference with restricted five-token next-token
argmax. Phase 2B-B additionally changes the objective through class weights.
Consequently, a difference between Phase 2 and Phase 2B cannot be attributed
solely to alias encoding. The A/B comparison is cleaner because the saved
variants align on the intended controls and differ methodologically in class
weighting.

Earlier Logistic Regression and MLP classification results used the old
retrieval-F1 Oracle. They are not directly comparable with any row in this
table. A fair cross-model router comparison requires the same evidence-length
Oracle labels and preserved splits for every model.

## Interpretation and limitations

The class-balanced run exceeded the evidence-length majority baseline's
macro-F1 (`0.16836616836616836` versus `0.125`) but remained below its accuracy
(`0.37012987012987014` versus `0.45454545454545453`). It made no top-1
prediction for the three smallest classes. Effective-number weighting at
`beta=0.999` therefore did not prevent top-1 class collapse in this run, and
the unweighted alias variant remains the strongest of the two Phase 2B runs on
the predeclared macro-F1 metric and downstream joined retrieval F1.

These conclusions have important limits:

- Only one seed (`42`) and one fixed three-epoch schedule were evaluated.
- The reported checkpoint was selected on the validation split, so the result
  does not establish test-set generalization.
- The imbalanced 924-example validation set contains only 13 class-10 and 81
  class-20 examples.
- No multi-seed confidence interval or significance test is available.
- Top-2 recovery does not compensate for zero top-1 recall in classes 10, 20,
  and 40 when the deployed router chooses one granularity.
- The Phase 2B-versus-Phase 2 change combines alias targets with a different
  decision rule; only the within-Phase-2B weighting comparison is narrowly
  controlled.

## Artifact inventory

The authoritative structured result is:

- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/final_summary.json`

Configuration and preflight:

- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/configuration/experiment.json`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/configuration/preflight_manifest.json`

Authoritative run records:

- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/training_config.json`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/dataset_manifest.json`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/formatted_example_inspection.json`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/training_history.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/validation_history.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/checkpoint_manifest.json`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/best_checkpoint.json`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/summary.json`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/validation/predictions_step-000071.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/validation/predictions_step-000142.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/validation/predictions_step-000213.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/runs/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/checkpoints/step-000142`

Final selected-checkpoint validation:

- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/validation/predictions.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/validation/raw_outputs.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/validation/parsed_predictions.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/validation/invalid_outputs.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/validation/runtime_summary.json`

Classification and retrieval:

- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/classification/metrics.json`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/classification/confusion_matrix.csv`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/classification/predicted_vs_oracle.svg`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/retrieval/results.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/retrieval/runtime_segments.jsonl`
- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/retrieval/summary.json`

Monitoring and comparison:

- `outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/tensorboard/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1/events.out.tfevents.1785776901.0f964f6e9b80.3121.0`
- `outputs/qwen_phase2b_comparison_evidence_length_oracle/four_way_comparison.json`

Frozen Oracle sources:

- `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/train_oracle.jsonl`
- `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/validation_oracle.jsonl`

Checkpoint binaries and TensorBoard event trees are intentionally Git-ignored
and must be preserved in the experiment archive separately from the
versionable summaries and manifests.

## Exact reproduction commands

Run preflight, full training, and selected-checkpoint validation in the saved
CUDA `.venv-qwen` environment from the repository root:

```bash
.venv-qwen/bin/python qwen_phase2b.py inspect --variant alias-classbalanced
.venv-qwen/bin/python qwen_phase2b.py train --variant alias-classbalanced --mode full --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py final-validation --variant alias-classbalanced --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
```

With the already verified local Qdrant service and collections available, run
the unchanged retrieval evaluation from the Windows project environment:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py evaluate-retrieval --variant alias-classbalanced --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
```

Regenerate the structured four-way comparison after all four source summaries
are complete:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py compare --output outputs/qwen_phase2b_comparison_evidence_length_oracle/four_way_comparison.json
```

TensorBoard can be inspected without changing the structured results:

```bash
.venv-qwen/bin/tensorboard --logdir outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/tensorboard/qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
```
