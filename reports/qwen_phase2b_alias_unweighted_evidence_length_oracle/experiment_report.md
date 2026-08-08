# Qwen3.5-0.8B Phase 2B-A restricted-alias router

## Status and objective

Phase 2B-A is complete. It tests whether a five-token restricted
classification formulation reduces the class-collapse behavior seen in the
earlier numeric-target Qwen router. The authoritative run is:

`qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1`

The run fine-tuned all parameters of the exact `Qwen/Qwen3.5-0.8B` checkpoint
at revision `2fc06364715b967f1860aea9cf38778875588b17`. It used the 2,245
preserved training questions and evaluated the validation-selected checkpoint
on all 924 preserved validation questions. Phase 2B-A used uniform loss
weights; the separate Phase 2B-B run changes only the intended class-weighting
treatment within the shared restricted-alias formulation.

All outputs are isolated under
`outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/`. Phase 1,
numeric-target Phase 2, and Phase 2B-B artifacts were not overwritten.

## Experimental formulation

### Input and fixed instruction

The model received only the fixed routing instruction and original question
text. The exact instruction was:

> You are a router for a retrieval-augmented generation system. Based only on
> the question, select the chunk size most suitable for retrieving the
> evidence required to answer it. Return only its alias: 1=10, 2=20, 3=40,
> 4=80, 5=160.

Its SHA-256 is
`d4a59dcd26b01c5bd81981e43c5d69fc1c8db14e9160e0791412dbe6af7067ac`.
The input is formatted with the official Qwen chat template and an assistant
generation prompt. The model was not supplied evidence, evidence length,
answers, paper text, retrieved chunks, retrieval scores, metadata, or
handcrafted features. Those fields occur only in saved evaluation records.

Across the preserved data, prompt lengths range from 89 to 115 tokens, with a
mean of `95.12026726057907`; all fit within the configured maximum length of
128.

### Alias classes and objective

The five aliases are single tokenizer tokens and map back to the canonical
chunk sizes as follows:

| Alias | Token ID | Chunk size | Class index |
|---:|---:|---:|---:|
| 1 | 16 | 10 | 0 |
| 2 | 17 | 20 | 1 |
| 3 | 18 | 40 | 2 |
| 4 | 19 | 80 | 3 |
| 5 | 20 | 160 | 4 |

The preflight verified each alias as one standalone token and as the first
assistant token under the chat template. At the final prompt position, the
training code gathers the vocabulary logits for exactly these five token IDs
and applies five-class cross-entropy to the Oracle class. Thus this is a
prompt-only next-token classification objective, not unrestricted text
generation and not parser-mediated classification.

The saved formulation identifier is
`qwen-phase2b-restricted-five-alias-next-token-v1`; the objective is
`restricted_five_logit_cross_entropy`. Inference takes deterministic argmax
over the same five logits. The saved `raw_qwen_output` is therefore the
restricted winning alias, and aliases are mapped back to 10, 20, 40, 80, or
160 before Oracle-label scoring and retrieval.

Because all five logits are directly comparable, top-2 accuracy is available
for Phase 2B. It remains unavailable for Phase 1 and numeric-target Phase 2,
whose deterministic generated text did not provide comparable scores for all
five classes.

### Unweighted loss

Phase 2B-A uses uniform class weights computed from no validation information:

| Chunk class | Training count | Weight |
|---:|---:|---:|
| 10 | 55 | `1.0` |
| 20 | 267 | `1.0` |
| 40 | 586 | `1.0` |
| 80 | 687 | `1.0` |
| 160 | 650 | `1.0` |

The saved reduction is
`sum(weight[target] * per_example_ce) / sum(weight[target])`. With all weights
equal to one, objective-weighted and unweighted cross-entropy are identical.
The class-weight source is recorded as `preserved_training_split_only`, and
`beta` is `null`.

## Preserved dataset and evidence-length Oracle

The experiment reuses the frozen evidence-length Oracle created for Phase 1.
For each question, evidence spans are stripped, empty spans are dropped, exact
stripped evidence is deduplicated across annotators, the unique spans are
lexicographically sorted and joined with newline separators, and the complete
result is tokenized with GPT-2 without special tokens. The nearest candidate
among 10, 20, 40, 80, and 160 is chosen. Exact midpoint ties use the smaller
candidate; lengths below 10 map to 10 and lengths above 160 map to 160. This
Oracle is independent of retrieval F1, embedding quality, cosine similarity,
retrieved chunks, and router performance.

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

Train Oracle SHA-256:
`64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`.

Validation Oracle SHA-256:
`ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`.

The validation set is strongly imbalanced: class 160 contains 420 of 924
examples (45.45%), while class 10 contains only 13 (1.41%). The majority-class
baseline therefore predicts 160 for every question, obtaining accuracy
`0.45454545454545453` and macro-F1 `0.125`.

## Model, environment, and training configuration

- Model: `Qwen/Qwen3.5-0.8B`.
- Model revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Training method: full-parameter restricted-alias classification.
- Total parameters: `852985920`.
- Trainable parameters: `852985920` (100%).
- Environment: `.venv-qwen`; the original legacy `.venv` remained unchanged.
- Python: `3.10.7 (main, Oct 3 2022, 02:19:58) [Clang 14.0.3]`.
- Python executable:
  `/workspace/thesis-granularity-router/.venv-qwen/bin/python`.
- Transformers: `5.15.0.dev0` at commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`.
- PyTorch: `2.8.0+cu128`; CUDA build `12.8`.
- TensorBoard: `2.20.0`.
- Hardware: `NVIDIA A100-SXM4-40GB`.
- Device and dtype: CUDA and `torch.bfloat16`.
- Quantization: none (`null`).
- Optimizer: AdamW over all parameters.
- Per-device batch size: 4.
- Gradient accumulation: 8.
- Effective batch size: 32.
- Epochs: 3.
- Optimizer/parameter-update steps: 213, 71 per epoch.
- Learning rate: `2e-05`.
- Weight decay: `0.01`.
- Scheduler: cosine.
- Warmup ratio: `0.05`.
- Gradient clipping: `1.0`.
- Seed: 42.
- Evaluation and checkpointing: end of every epoch.
- Early stopping: none; the run used the fixed three epochs.
- Checkpoint retention: current and best during training, selected checkpoint
  only at completion.

There was no LoRA, QLoRA, adapter, prompt tuning, quantization, frozen
backbone, separate classification head, or partial-layer training. All
`852985920` parameters were trainable. Validation and final inference used no
gradient or optimizer update.

The repository commit recorded at launch is
`55af1bcbc4d7a089adaafd4da539581b2dbbed67`. The training script SHA-256 is
`60572d8c3054e7ef76055b2c40cf65c2999ef18000930f5a6967fd2ae673041c`.
The experiment fingerprint is
`961786801daa066b40cd2ad9325cb48b08a45942dacb7516470891aa715bc61d`,
and the resume-contract SHA-256 is
`19455b217f51e7549a174cec913829340d75e2e2b5c95719ab4164f38694f8d4`.

## Epoch validation and checkpoint selection

Each epoch was evaluated on all 924 preserved validation questions. The
primary checkpoint-selection metric was validation macro-F1. Ties would be
resolved by accuracy, weighted F1, balanced accuracy, lower unweighted
validation cross-entropy, then earlier step.

| Epoch | Step | Validation CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 accuracy | Predicted 10/20/40/80/160 | Validation wall time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 71 | `1.3775848428924362` | `0.2510822510822511` | `0.08027681660899653` | `0.10078041911951946` | `0.2` | `0.44372294372294374` | 0 / 0 / 0 / 924 / 0 | `36.127269204705954` |
| 2 | 142 | `1.3699120345053735` | `0.3008658008658009` | `0.19043978783245397` | `0.2890888439338361` | `0.2243431855500821` | `0.5584415584415584` | 0 / 0 / 374 / 379 / 171 | `36.21613594703376` |
| 3 | 213 | `1.3402932242397623` | `0.35064935064935066` | `0.20922603632601472` | `0.3406050804511769` | `0.2383201416948027` | `0.6071428571428571` | 0 / 0 / 427 / 189 / 308 | `36.16520829498768` |

Macro-F1 increased at every epoch, so no tie-break was needed. Epoch 3,
`step-000213`, was selected and retained:

`outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/checkpoints/step-000213`

The earlier step-71 and step-142 model checkpoints were pruned under the saved
retention policy, while their validation predictions and checkpoint-manifest
records remain. A post-training reload of the retained checkpoint reproduced
the selected epoch's scores for all 924 outputs exactly
(`selected_epoch_exact_score_match: true`). No separate checkpoint-hash
inventory was saved in this output tree.

The logged initial loss `1.835081309080124` and final loss
`1.541044330596924` are the first and last optimizer-step values, not epoch
means. The last optimizer step contains the five-example epoch tail. The
selected validation unweighted and objective-weighted cross-entropies are both
`1.3402932242397623`.

## Final classification results

The post-training selected-checkpoint evaluation completed all 924 examples.
All predictions were valid restricted aliases; none required a fallback or
default chunk size.

| Metric | Value |
|---|---:|
| Accuracy | `0.35064935064935066` |
| Macro-F1 | `0.20922603632601472` |
| Weighted F1 | `0.3406050804511769` |
| Balanced accuracy | `0.2383201416948027` |
| Top-2 accuracy | `0.6071428571428571` |
| Valid outputs | `924/924` |
| Invalid outputs | `0` |
| Invalid-output percentage | `0.0%` |
| Majority-class baseline accuracy | `0.45454545454545453` |
| Majority-class baseline macro-F1 | `0.125` |

Top-2 status is
`available_from_comparable_restricted_five_class_logits`. It is calculated
from the two highest of the same five restricted next-token logits, not from
an approximation or text-parser heuristic.

### Per-class metrics

| Oracle class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | `0.0` | `0.0` | `0.0` | 13 |
| 20 | `0.0` | `0.0` | `0.0` | 81 |
| 40 | `0.234192037470726` | `0.5617977528089888` | `0.33057851239669417` | 178 |
| 80 | `0.26455026455026454` | `0.21551724137931033` | `0.2375296912114014` | 232 |
| 160 | `0.564935064935065` | `0.4142857142857143` | `0.47802197802197804` | 420 |

### Oracle and predicted distributions

| Chunk class | Oracle | Predicted |
|---:|---:|---:|
| 10 | 13 | 0 |
| 20 | 81 | 0 |
| 40 | 178 | 427 |
| 80 | 232 | 189 |
| 160 | 420 | 308 |
| **Total** | **924** | **924** |

The unweighted model did not predict classes 10 or 20. It spread its outputs
across 40, 80, and 160 instead of collapsing entirely to one class. Its
accuracy remained below the majority-class baseline, while its macro-F1 was
higher than the majority baseline's macro-F1.

### Confusion matrix

Rows are Oracle labels and columns are predicted chunk sizes, both ordered
10, 20, 40, 80, 160.

| Oracle \\ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 10 | 2 | 1 |
| 20 | 0 | 0 | 48 | 13 | 20 |
| 40 | 0 | 0 | 100 | 35 | 43 |
| 80 | 0 | 0 | 112 | 50 | 70 |
| 160 | 0 | 0 | 157 | 89 | 174 |

## End-to-end retrieval

Aliases were mapped back to canonical chunk sizes before retrieval. Retrieval
used the unchanged paper-restricted pipeline: `text-embedding-3-small`, 1,536
dimensions, cosine similarity against the existing Qdrant collections, the
predicted granularity, top-k 5, the same source-paper restriction, chunk
ordering and concatenation, and `f1_joined_topk` token overlap with GPT-2
tokenization.

| Retrieval metric | Value |
|---|---:|
| Evaluated examples | `924` |
| Valid-prediction retrievals | `924` |
| Retrieval coverage | `1.0` (100%) |
| Valid-only mean joined retrieval F1 | `0.28646775432900434` |
| Valid-only median joined retrieval F1 | `0.2748425` |
| Coverage-adjusted full-set mean joined retrieval F1 | `0.2864677543290044` |
| Top-k | `5` |
| Paper restricted | `true` |

Because every prediction was valid, valid-only and full-set coverage-adjusted
means differ only by floating-point accumulation. No invalid output was mapped
to a default granularity. The retrieval evaluation configuration SHA-256 is
`9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8`;
metric version is `qasper-token-prf-v2`, schema version is 2, and normalization
is `lowercase-remove-punctuation-collapse-whitespace-v1`.

Classification accuracy, macro-F1, weighted F1, and balanced accuracy measure
prediction of the evidence-length Oracle label. Joined retrieval F1 measures
token overlap after downstream retrieval. These metrics are not
interchangeable.

## Runtime and resource use

### Training

- Training wall time: `1308.664808139205` seconds.
- Peak GPU allocated: `10.058899402618408` GiB.
- Peak GPU reserved: `11.99609375` GiB.
- Recorded RSS: `1.9831161499023438` GiB.

### Selected-checkpoint reload and final inference

- Model-load time: `2.6078288350254297` seconds.
- Isolated inference wall time: `35.81447528861463` seconds.
- Mean inference time: `0.0385481188279371` seconds/question.
- Median inference time: `0.03886844031512737` seconds/question.
- Total synchronized batch-forward allocation: `35.61846179701388` seconds.
- Peak GPU allocated: `2.2182140350341797` GiB.
- Peak GPU reserved: `3.6640625` GiB.
- Recorded RSS: `1.708984375` GiB.

### Retrieval and known combined runtime

- Retrieval wall time: `178.27286399999866` seconds.
- Durable per-question retrieval-processing sum:
  `174.24649079941446` seconds.
- Training plus selected-checkpoint loading/final inference:
  `1347.087112262845` seconds.
- Training, selected-checkpoint loading/final inference, and retrieval:
  `1525.3599762628437` seconds.

The retrieval completed in one uninterrupted runtime segment.

## Manual post-transfer integrity gates

No standalone Phase 2B hash-inventory file was saved. A manual `rsync`
checksum dry-run compared every copied remote training, classification,
validation, TensorBoard, and checkpoint file, excluding `final_summary.json`
because local retrieval legitimately extends it. The unweighted tree had no
content differences. Its selected `step-000213` checkpoint contains 11 files
totaling 4,735,895,574 bytes. Completed-summary replay through the retrieval
evaluator revalidated all 924 records and returned `complete`.

Qdrant counts were unchanged before and after the two Phase 2B retrieval/QA
passes: `PaperChunk` 1,701,822; `PaperEvidence` 9,522; `PaperQuestion` 4,526;
`RouterDataset` 3,170; `RetrievalEvaluation` 18,622;
`Stage4VerifyRetrievalEval` 10; `Stage4VerifyRouterDataset` 2; and
`Stage5MixedEvaluation` 2. Frozen Phase 1/2 artifact and script hashes were
also rechecked unchanged. These are manual post-transfer gates, not a claim
that a separate Phase 2B integrity artifact exists.

## Four-way same-Oracle comparison

The saved comparison artifact confirms that Phase 1, numeric-target Phase 2,
Phase 2B-A, and Phase 2B-B all use the same 924 validation questions,
evidence-length Oracle distribution, five canonical chunk sizes, paper-level
retrieval restriction, and top-k 5.

| Run | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 | Mean joined retrieval F1 |
|---|---:|---:|---:|---:|---:|---:|
| Phase 1 pretrained zero-shot | `0.04004329004329004` | `0.049045932422555796` | `0.032612933907418644` | `0.23369399361908724` | unavailable | `0.23910868506493507` |
| Phase 2 numeric-target SFT | `0.4318181818181818` | `0.16502267760462996` | `0.32805741427623947` | `0.20697865353037764` | unavailable | `0.22658488852813854` |
| **Phase 2B-A alias-unweighted** | **`0.35064935064935066`** | **`0.20922603632601472`** | **`0.3406050804511769`** | **`0.2383201416948027`** | **`0.6071428571428571`** | **`0.28646775432900434`** |
| Phase 2B-B alias-classbalanced | `0.37012987012987014` | `0.16836616836616836` | `0.3142183142183142` | `0.20607553366174058` | `0.7056277056277056` | `0.24962774025974027` |

The bold row identifies this report's run, not universal statistical
superiority. In this single-seed comparison, Phase 2B-A has the highest
macro-F1, weighted F1, balanced accuracy, and mean joined retrieval F1 among
the four saved Qwen runs. Numeric-target Phase 2 has the highest accuracy, and
Phase 2B-B has the higher Phase 2B top-2 accuracy.

These rows are comparable at the preserved data, Oracle, and retrieval levels,
but they are not identical-prompt replications. Relative to numeric-target
Phase 2, Phase 2B changes the instruction's output schema, replaces numeric
chunk targets with aliases, replaces generated-text/parser inference with
restricted five-token argmax, and changes the supervised objective to
five-logit next-token classification. Consequently, a Phase 2B improvement
cannot be attributed solely to aliases. Phase 2B-B additionally changes the
loss weighting.

Phase 2B-A and Phase 2B-B are the cleanest controlled pair: their saved model,
revision, instruction and instruction hash, alias mapping, training script,
data and Oracle hashes, optimizer schedule, seed, environment, epoch count,
checkpoint rule, and retrieval setup agree. Their intended methodological
difference is uniform versus effective-number class weighting. Their distinct
experiment fingerprints correctly reflect that difference.

Earlier Logistic Regression and MLP classification results used the old
retrieval-F1 Oracle. They are not directly comparable with any row in this
same-evidence-length-Oracle Qwen table unless retrained and reevaluated on the
new Oracle and preserved splits.

## Interpretation and limitations

Restricted-alias Phase 2B-A materially changed the prediction distribution
relative to the numeric-target Phase 2 collapse and achieved the highest saved
macro-F1 and downstream mean joined retrieval F1 in the four-way single-seed
comparison. It nevertheless failed to predict either minority class 10 or 20,
and its accuracy remained below the 160-majority baseline. The result therefore
does not establish balanced five-class routing.

The following limitations must accompany the result:

- This is one seed and one predeclared training configuration; no multi-seed
  uncertainty estimate is available.
- The selected checkpoint was chosen using validation macro-F1 and reported on
  that same validation split. The result is not an untouched-test-set estimate
  and may benefit from validation selection.
- The class distribution is strongly imbalanced, especially for classes 10
  and 160.
- Phase 2B changes both target representation and decision rule relative to
  numeric-target Phase 2, preventing single-factor causal attribution.
- Phase 2B-A versus Phase 2B-B isolates the intended weighting change more
  cleanly, but one seed remains insufficient for statistical conclusions.
- Top-2 is available only for the restricted-logit Phase 2B runs and must not
  be invented for Phase 1 or Phase 2.
- Classification metrics and joined retrieval F1 measure different stages;
  an increase in one does not guarantee an increase in the other.

## Artifacts

### Configuration and final summary

- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/configuration/experiment.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/configuration/preflight_manifest.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/final_summary.json`

### Training run

- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/training_config.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/dataset_manifest.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/formatted_example_inspection.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/training_history.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/validation_history.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/checkpoint_manifest.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/best_checkpoint.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/summary.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/validation/predictions_step-000071.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/validation/predictions_step-000142.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/validation/predictions_step-000213.jsonl`

The retained checkpoint is:

- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/runs/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/checkpoints/step-000213/`

It contains the model, processor/tokenizer, optimizer, scheduler, random states,
and training state.

### TensorBoard

- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/tensorboard/qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1/events.out.tfevents.1785775484.0f964f6e9b80.2198.0`

### Final validation and classification

- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/validation/predictions.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/validation/raw_outputs.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/validation/parsed_predictions.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/validation/invalid_outputs.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/validation/runtime_summary.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/classification/metrics.json`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/classification/confusion_matrix.csv`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/classification/predicted_vs_oracle.svg`

### Retrieval and comparison

- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/retrieval/results.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/retrieval/runtime_segments.jsonl`
- `outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/retrieval/summary.json`
- `outputs/qwen_phase2b_comparison_evidence_length_oracle/four_way_comparison.json`

### Frozen Oracle sources

- `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/train_oracle.jsonl`
- `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/validation_oracle.jsonl`

## Reproduction commands

The GPU-side preflight, training, and selected-checkpoint evaluation commands
are:

```bash
.venv-qwen/bin/python qwen_phase2b.py \
  --output-root outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle \
  inspect --variant alias-unweighted

.venv-qwen/bin/python qwen_phase2b.py \
  --output-root outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle \
  train \
  --variant alias-unweighted \
  --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1

.venv-qwen/bin/python qwen_phase2b.py \
  --output-root outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle \
  final-validation \
  --variant alias-unweighted \
  --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
```

With the existing verified Qdrant service and collections available at the
project's unchanged configured endpoint, reproduce retrieval and the four-way
comparison on Windows with:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py `
  evaluate-retrieval `
  --variant alias-unweighted `
  --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1 `
  --output-root outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle

.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py `
  compare `
  --output outputs/qwen_phase2b_comparison_evidence_length_oracle/four_way_comparison.json
```

These commands reuse the preserved Oracle files, splits, and existing Qdrant
collections. They do not regenerate the Oracle, rebuild or re-index any
collection, or overwrite Phase 1, Phase 2, or Phase 2B-B artifacts.
The literal paths and run ID above record the completed authoritative run; use
a clean workspace or a distinct isolated output root when testing a fresh
reproduction so the retained Phase 2B-A artifacts remain frozen.
