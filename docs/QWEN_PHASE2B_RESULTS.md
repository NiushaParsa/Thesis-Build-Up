# Qwen Phase 2B restricted-alias results

Phase 2B completed two isolated full-parameter experiments with the same
`Qwen/Qwen3.5-0.8B` model, revision
`2fc06364715b967f1860aea9cf38778875588b17`, and frozen evidence-length
Oracle used by Qwen Phases 1 and 2:

- **Phase 2B-A, alias-unweighted:** uniform restricted five-class
  cross-entropy.
- **Phase 2B-B, alias-classbalanced:** the same formulation with
  effective-number class weights computed only from the preserved training
  split, using `beta=0.999`.

Both runs trained all 852,985,920 parameters for three epochs and 213 optimizer
steps. They are separate experiments and do not overwrite the frozen Phase 1
zero-shot or Phase 2 numeric-target baselines.

## What changed and what remained fixed

The preserved train/validation sets, Oracle, model revision, optimizer
configuration, question-only leakage boundary, and downstream retrieval setup
remained fixed. Phase 2B deliberately changed the output formulation.

The instruction asks for a single alias with the mapping `1→10`, `2→20`,
`3→40`, `4→80`, and `5→160`. Each alias is one verified tokenizer token with
token IDs 16--20. Training and inference use cross-entropy and deterministic
argmax over exactly those five vocabulary logits at the first assistant answer
position. Phase 2B therefore does not use unrestricted text generation or the
Phase 1/2 generated-text parser. The aliases are mapped back to chunk sizes
before classification and retrieval evaluation.

The only semantic model input remains the fixed Phase 2B routing instruction
and original question text. Evidence, evidence length, answers, paper text,
retrieved chunks, retrieval scores, embeddings, metadata, and handcrafted
features are excluded. Formatted prompt lengths are 89--115 tokens, mean
95.12026726057907, under the fixed maximum of 128.

The Phase 2B instruction SHA-256 is
`d4a59dcd26b01c5bd81981e43c5d69fc1c8db14e9160e0791412dbe6af7067ac`.
This differs from the Phase 1/2 numeric-output prompt, so Phase 2B is a
controlled method comparison under the same data and Oracle rather than an
identical-prompt replication.

## Frozen data and Oracle

The Oracle counts GPT-2 tokens in complete stripped, exact-deduplicated,
deterministically ordered ground-truth evidence; chooses the closest of 10,
20, 40, 80, and 160; resolves exact midpoint ties toward the smaller class;
maps values below 10 to 10; and maps values above 160 to 160. It is independent
of retrieval F1, embeddings, retrieved chunks, and router performance.

| Split | Questions | Papers | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2,245 | 845 | 55 | 267 | 586 | 687 | 650 |
| Validation | 924 | 277 | 13 | 81 | 178 | 232 | 420 |

Train Oracle SHA-256:
`64999b9f29c07f01566c478c70fa87d860b397af457b6c0f5fca214bea6beb88`.
Validation Oracle SHA-256:
`ad68655209b258908e90db11cdd54a6e5db49329132912dc4bd8e71c73422a8d`.
Class 160 is 420/924 (45.45%) of validation, while class 10 has only 13
examples.

## Weighting and fixed training configuration

Phase 2B-A uses weights of 1.0 for every class. Phase 2B-B uses the following
effective-number weights, normalized to arithmetic class mean one:

| Chunk class | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Phase 2B-B weight | 3.1872088653568436 | 0.7279213406697836 | 0.38467220811977887 | 0.34329010532422555 | 0.3569074805293684 |

For each gradient-accumulation window, both variants use
`sum(weight[target] * per_example_ce) / sum(weight[target])`. Phase 2B-B is
the only run that changes the weights; it does not resample or relabel data.

Shared configuration:

- full-parameter restricted-alias classification; 852,985,920/852,985,920
  trainable parameters;
- Python 3.10.7 in `.venv-qwen`;
- Transformers `5.15.0.dev0`, commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`;
- PyTorch `2.8.0+cu128`, CUDA 12.8, one NVIDIA A100-SXM4-40GB;
- `torch.bfloat16`, no quantization, seed 42, strict deterministic CUDA;
- per-device batch 4, gradient accumulation 8, effective batch 32;
- AdamW, learning rate `2e-5`, weight decay `0.01`, cosine schedule, 5%
  warmup, gradient clipping `1.0`;
- three fixed epochs, evaluation/checkpointing after each epoch, no early
  stopping;
- checkpoint selection by validation macro-F1, followed by accuracy, weighted
  F1, balanced accuracy, lower unweighted validation CE, and earlier step;
- selected checkpoint only retained at completion.

The shared training-script SHA-256 is
`60572d8c3054e7ef76055b2c40cf65c2999ef18000930f5a6967fd2ae673041c`.
The protected legacy Python 3.9 `.venv` remained unchanged; Phase 2B used only
the separate remote CUDA `.venv-qwen` and did not modify system Python.

## Epoch validation and checkpoint selection

### Phase 2B-A — alias-unweighted

| Epoch/checkpoint | Unweighted CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 accuracy | Predicted 10/20/40/80/160 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 / `step-000071` | 1.3775848428924362 | 0.2510822510822511 | 0.08027681660899653 | 0.10078041911951946 | 0.2 | 0.44372294372294374 | 0/0/0/924/0 |
| 2 / `step-000142` | 1.3699120345053735 | 0.3008658008658009 | 0.19043978783245397 | 0.2890888439338361 | 0.2243431855500821 | 0.5584415584415584 | 0/0/374/379/171 |
| 3 / `step-000213` | 1.3402932242397623 | 0.35064935064935066 | **0.20922603632601472** | 0.3406050804511769 | 0.2383201416948027 | 0.6071428571428571 | 0/0/427/189/308 |

`step-000213` is selected by the highest validation macro-F1.

### Phase 2B-B — alias-classbalanced

| Epoch/checkpoint | Unweighted CE | Weighted objective CE | Accuracy | Macro-F1 | Weighted F1 | Balanced accuracy | Top-2 accuracy | Predicted 10/20/40/80/160 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 / `step-000071` | 1.5536545303476836 | 1.5932705140456938 | 0.19264069264069264 | 0.0646098003629764 | 0.06223238346650324 | 0.2 | 0.4945887445887446 | 0/0/924/0/0 |
| 2 / `step-000142` | 1.4620294988929452 | 1.550350315119168 | 0.37012987012987014 | **0.16836616836616836** | 0.3142183142183142 | 0.20607553366174058 | 0.7056277056277056 | 0/0/0/434/490 |
| 3 / `step-000213` | 1.4841528165908087 | 1.5570664912477334 | 0.2683982683982684 | 0.11060887911611213 | 0.1660641330693655 | 0.20119628789136734 | 0.7088744588744589 | 0/0/7/851/66 |

`step-000142` is selected by the highest validation macro-F1. The epoch-3
top-2 score is higher, but top-2 is not the checkpoint-selection metric.

## Final four-way classification comparison

All rows use the same 924 validation questions, evidence-length Oracle, and
chunk classes. Phase 2B changes target encoding and decision rule; Phase 2B-B
also changes loss weighting.

| Metric | Phase 1 zero-shot | Phase 2 numeric SFT | Phase 2B-A alias-unweighted | Phase 2B-B alias-classbalanced |
|---|---:|---:|---:|---:|
| Accuracy | 0.04004329004329004 | **0.4318181818181818** | 0.35064935064935066 | 0.37012987012987014 |
| Macro-F1 | 0.049045932422555796 | 0.16502267760462996 | **0.20922603632601472** | 0.16836616836616836 |
| Weighted F1 | 0.032612933907418644 | 0.32805741427623947 | **0.3406050804511769** | 0.3142183142183142 |
| Balanced accuracy | 0.23369399361908724 | 0.20697865353037764 | **0.2383201416948027** | 0.20607553366174058 |
| Top-2 accuracy | unavailable | unavailable | 0.6071428571428571 | 0.7056277056277056 |
| Valid-output rate | 1.0 | 1.0 | 1.0 | 1.0 |

Phase 2 numeric SFT has the highest accuracy. Phase 2B-A has the highest
macro-F1, weighted F1, and balanced accuracy. Phase 2B-B has the highest
available top-2 accuracy, but the balanced weighting does not improve the
primary macro-F1 over unweighted alias training. It also does not recover
classes 10 or 20 and predicts only 80 and 160 at its selected checkpoint.
This is a negative result for the tested weighting configuration, not evidence
that all imbalance treatments are ineffective.

The evidence-length majority baseline always predicts 160 and has accuracy
0.45454545454545453 and macro-F1 0.125. Neither Phase 2B variant exceeds its
accuracy; both exceed its macro-F1.

## Per-class results and distributions

| Class | Oracle support | 2B-A precision | 2B-A recall | 2B-A F1 | 2B-B precision | 2B-B recall | 2B-B F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 13 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 20 | 81 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 40 | 178 | 0.234192037470726 | 0.5617977528089888 | 0.33057851239669417 | 0.0 | 0.0 | 0.0 |
| 80 | 232 | 0.26455026455026454 | 0.21551724137931033 | 0.2375296912114014 | 0.25806451612903225 | 0.4827586206896552 | 0.3363363363363363 |
| 160 | 420 | 0.564935064935065 | 0.4142857142857143 | 0.47802197802197804 | 0.46938775510204084 | 0.5476190476190477 | 0.5054945054945055 |

| Source | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Oracle | 13 | 81 | 178 | 232 | 420 |
| Phase 1 predictions | 767 | 40 | 116 | 0 | 1 |
| Phase 2 predictions | 0 | 0 | 0 | 149 | 775 |
| Phase 2B-A predictions | 0 | 0 | 427 | 189 | 308 |
| Phase 2B-B predictions | 0 | 0 | 0 | 434 | 490 |

Phase 2B-A predicts three classes and substantially reduces the Phase 2
concentration on 160, but still completely misses 10 and 20. Phase 2B-B
returns to a two-class solution.

Phase 2B-A confusion matrix, with Oracle rows and predicted columns ordered
10, 20, 40, 80, 160:

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 10 | 2 | 1 |
| 20 | 0 | 0 | 48 | 13 | 20 |
| 40 | 0 | 0 | 100 | 35 | 43 |
| 80 | 0 | 0 | 112 | 50 | 70 |
| 160 | 0 | 0 | 157 | 89 | 174 |

Phase 2B-B confusion matrix:

| Oracle \ predicted | 10 | 20 | 40 | 80 | 160 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 0 | 9 | 4 |
| 20 | 0 | 0 | 0 | 39 | 42 |
| 40 | 0 | 0 | 0 | 84 | 94 |
| 80 | 0 | 0 | 0 | 112 | 120 |
| 160 | 0 | 0 | 0 | 190 | 230 |

## Unchanged downstream retrieval

The aliases were mapped back to chunk sizes before using the unchanged local
Qdrant retrieval pipeline: source-paper filtering, predicted granularity,
`top_k=5`, existing `text-embedding-3-small` embeddings, cosine ranking,
unchanged chunk ordering/concatenation, GPT-2 tokenization, and joined
token-level F1. No collection, index, schema, port, or stored record was
changed.

| Retrieval metric | Phase 1 | Phase 2 | Phase 2B-A | Phase 2B-B |
|---|---:|---:|---:|---:|
| Coverage | 1.0 | 1.0 | 1.0 | 1.0 |
| Mean joined retrieval F1 | 0.23910868506493507 | 0.22658488852813854 | **0.28646775432900434** | 0.24962774025974027 |
| Median joined retrieval F1 | 0.2210845 | 0.19615549999999998 | **0.2748425** | 0.223194 |
| Retrieval wall seconds | 367.7590293000012 | 178.12831589998677 | 178.27286399999866 | 377.0999227000284 |

Phase 2B-A has the highest saved mean joined retrieval F1 of the four Qwen
runs. Phase 2B-B is below Phase 2B-A on both mean and median retrieval F1.
Classification metrics measure Oracle-label prediction; joined retrieval F1
measures downstream evidence-token overlap and is a different outcome.

## Runtime and resources

| Measurement | Phase 2B-A | Phase 2B-B |
|---|---:|---:|
| Training wall seconds | 1308.664808139205 | 1306.7509042322636 |
| Initial/final training loss | 1.835081309080124 / 1.541044330596924 | 1.816834687228543 / 1.602007440099642 |
| Selected-checkpoint model load seconds | 2.6078288350254297 | 3.260306715965271 |
| Isolated final inference seconds | 35.81447528861463 | 39.01004763878882 |
| Mean inference seconds/question | 0.0385481188279371 | 0.04196543970197697 |
| Median inference seconds/question | 0.03886844031512737 | 0.04170184303075075 |
| Training peak allocated GPU GiB | 10.058899402618408 | 10.053670883178711 |
| Training peak reserved GPU GiB | 11.99609375 | 11.99609375 |
| Retrieval wall seconds | 178.27286399999866 | 377.0999227000284 |
| Known training + final validation + retrieval seconds | 1525.3599762628437 | 1726.121181287046 |

The observed retrieval-wall difference is recorded but should not be
interpreted as a model- or method-speed effect. Both retrieval evaluations ran
separately against the local service, and Phase 2B-B retrieval overlapped the
weighted selected-checkpoint archive transfer. Training and final-validation
timings are the isolated measurements saved by each run.

## Manual post-transfer integrity gates

The final copy-back used manual integrity gates; no standalone Phase 2B hash
inventory file was saved, so none is cited. An `rsync` checksum dry-run
compared every remote training, classification, validation, TensorBoard, and
checkpoint file while excluding `final_summary.json`, which was legitimately
extended locally by retrieval evaluation. Phase 2B-A had no content
differences. Phase 2B-B initially exposed only two stale local preflight
copies, `configuration/experiment.json` and
`configuration/preflight_manifest.json`; both were replaced from the GPU
source, and the targeted checksum rerun returned no differences.

The selected Phase 2B-A checkpoint contains 11 files totaling 4,735,895,574
bytes. The selected Phase 2B-B checkpoint contains 11 files totaling
4,735,895,530 bytes. Completed-summary replay through the retrieval evaluator
revalidated all 924 records for each run and returned `complete`.

Qdrant collection counts were identical before and after this read/evaluation
workflow:

| Collection | Before | After |
|---|---:|---:|
| `PaperChunk` | 1,701,822 | 1,701,822 |
| `PaperEvidence` | 9,522 | 9,522 |
| `PaperQuestion` | 4,526 | 4,526 |
| `RouterDataset` | 3,170 | 3,170 |
| `RetrievalEvaluation` | 18,622 | 18,622 |
| `Stage4VerifyRetrievalEval` | 10 | 10 |
| `Stage4VerifyRouterDataset` | 2 | 2 |
| `Stage5MixedEvaluation` | 2 | 2 |

Frozen baseline and script hashes also remained unchanged: Phase 1
`final_summary.json`
`d421d57342331b2d6418d9fe3a10a0886a5fa4f24bbf146fadb7e41a050500c1`;
Phase 2 `final_summary.json`
`73f9ffb773aedcc47ba7ebe3850d28e372038ae795e3f6cb69f888bfcfb87d04`;
Phase 2 `integrity_audit.json`
`1acd3336161f1508a65b7118b138267b51489593ed3ac96064c4192db4c63ff8`;
`qwen_phase2.py`
`c9a6f2a277bd841d6bf0ede9e948b18e91e1a8f5a298f7d704d0b4279c99ed39`;
and `qwen_phase2b.py`
`60572d8c3054e7ef76055b2c40cf65c2999ef18000930f5a6967fd2ae673041c`.

## Interpretation and comparability limits

The four-way table is valid as a same-data, same-Oracle, same-chunk-class, and
same-retrieval comparison. It is not an identical-prompt or identical-decision
replication: Phase 2B changes the prompt's output schema, target encoding, and
decision rule; Phase 2B-B additionally changes the loss. Consequently, gains
from Phase 2 to Phase 2B-A cannot be attributed solely to the alias symbols.

Phase 2B-A and Phase 2B-B are the closest controlled pair: their saved model,
data, seed, optimizer schedule, alias formulation, and evaluation settings are
the same, while the predeclared class weights differ. For this single seed,
effective-number weighting is worse than unweighted alias training on the
primary macro-F1, weighted F1, balanced accuracy, and mean joined retrieval
F1. The weighted run's higher top-2 accuracy does not reverse that conclusion
under the predeclared primary metric.

These are single-seed results selected using the same validation split later
reported as final validation. No QASPER test split was loaded or evaluated.
The results therefore do not quantify run-to-run variance or unbiased test
generalization.

Earlier Logistic Regression and MLP classification results use the old
retrieval-F1 Oracle and are not directly comparable with any of these
evidence-length-Oracle Qwen classification results.

## Artifacts and reproduction

Authoritative Phase 2B-A artifacts are under
`outputs/qwen_phase2b_alias_unweighted_evidence_length_oracle/`; Phase 2B-B
artifacts are under
`outputs/qwen_phase2b_alias_classbalanced_evidence_length_oracle/`. Each root
contains its configuration/preflight manifest, run configuration and
histories, selected-checkpoint metadata, canonical validation predictions,
classification metrics/confusion matrix/histogram, retrieval records/summary,
runtime, and `final_summary.json`.

The authoritative four-way comparison is
`outputs/qwen_phase2b_comparison_evidence_length_oracle/four_way_comparison.json`.
Standalone reports are:

- `reports/qwen_phase2b_alias_unweighted_evidence_length_oracle/experiment_report.md`
- `reports/qwen_phase2b_alias_classbalanced_evidence_length_oracle/experiment_report.md`

Recorded commands, from a compatible CUDA project checkout:

```bash
.venv-qwen/bin/python qwen_phase2b.py inspect --variant alias-unweighted
.venv-qwen/bin/python qwen_phase2b.py inspect --variant alias-classbalanced
.venv-qwen/bin/python qwen_phase2b.py train --variant alias-unweighted --mode full --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py train --variant alias-classbalanced --mode full --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py final-validation --variant alias-unweighted --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
.venv-qwen/bin/python qwen_phase2b.py final-validation --variant alias-classbalanced --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
```

Against the unchanged local Qdrant service:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py evaluate-retrieval --variant alias-unweighted --run-id qwen-phase2b-alias-unweighted-full-parameter-20260803-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py evaluate-retrieval --variant alias-classbalanced --run-id qwen-phase2b-alias-classbalanced-full-parameter-20260803-seed42-v1
.\.venv-qwen\Scripts\python.exe qwen_phase2b_posttraining.py compare --output outputs\qwen_phase2b_comparison_evidence_length_oracle\four_way_comparison.json
```
