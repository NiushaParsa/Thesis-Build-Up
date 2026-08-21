# Phase 3C Results: Qwen Phase 2D and Similarity-Tree Phase 3B Fusion

## Status and objective

Phase 3C is complete. It implements Lorenzo's suggested final combination of
the question-based Qwen router and the paper-specific similarity-tree router.
The experiment uses the exact selected Phase 2D checkpoint and the exact saved
173-dimensional Phase 3A/3B tree rows. No earlier model, feature, prediction,
Oracle, Qdrant collection, or retrieval artifact was overwritten.

The authoritative result is
`outputs/qwen_phase3c_fusion_evidence_length_oracle/final_summary.json`.

## Inputs and fusion models

For every question, the frozen Phase 2D sequence classifier receives its exact
token-count prompt and original question. The runner extracts:

- the five Phase 2D class logits; and
- the 1,024-dimensional final non-padding Qwen hidden state.

These are combined with the 173 Phase 3B same-paper similarity-tree features:

| Variant | Qwen features | Tree features | Total |
|---|---:|---:|---:|
| `qwen_logits_tree` | 5 logits | 173 | 178 |
| `qwen_hidden_tree` | 1,024 hidden values | 173 | 1,197 |

Qwen is frozen: trainable parameters are zero, gradients are disabled, and
parameter updates are zero. The fusion classifier is class-weighted XGBoost
with `multi:softprob`. It uses the same 12-candidate grid and five
paper-grouped folds as Phase 3B. Selection uses training-fold macro-F1,
followed by balanced accuracy and accuracy; hidden fusion is preferred only on
an exact tie.

The features do not contain the answer, gold evidence, evidence length,
retrieval F1, or Oracle label. The Oracle is used only as the supervised target.

## Integrity and environment

- Phase 3A train feature SHA-256:
  `6d55e1d10872c8db24cf9af9becfb8e2e6570e13a7697151febc7f44ecebdd9c`.
- Phase 3A validation feature SHA-256:
  `548e3cccab3b19dee644eb9858081ff380b6375765433f1d2369c6d7d2ecb893`.
- Phase 2D checkpoint SHA-256:
  `020af0a83af773239e7e60e9983afad29cae3f31493c7073e9162e040b732814`.
- Dataset: 2,245 train questions from 845 papers and 924 validation
  questions from 277 disjoint papers.
- Remote environment: `.venv-fusion`, Python 3.10.7.
- GPU: NVIDIA A100-SXM4-40GB.
- PyTorch: 2.8.0+cu128; Qwen dtype: `torch.bfloat16`.
- Transformers: 5.15.0.dev0 from commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`.
- XGBoost: 3.0.2; NumPy: 2.2.6; SciPy: 1.15.3; scikit-learn: 1.6.1.

Feature extraction at the exact Phase 2D evaluation batch size reproduced all
924 saved Phase 2D argmax predictions. The two final XGBoost models were
reloaded and reproduced every saved label and metric. Maximum probability
difference was 0.0 for logits fusion and 1.4901161193847656e-08 for hidden
fusion.

## Training-fold selection

| Variant | Selected candidate | OOF accuracy | OOF macro-F1 | OOF balanced accuracy |
|---|---|---:|---:|---:|
| Logits + tree | depth 2, LR 0.05, 200 trees | 0.4717149220489978 | 0.38383459364081357 | 0.38118555379226826 |
| Hidden + tree | depth 2, LR 0.03, 200 trees | 0.44097995545657015 | 0.35436172573956737 | 0.3532414108900802 |

The compact logits fusion was locked as primary. The OOF figures describe the
fusion classifier's grouped folds, but they are not a fully nested estimate:
the frozen Phase 2D checkpoint had already been trained on all 2,245 training
questions, including each fusion held-out fold. They must not be reported as
an unbiased end-to-end generalization estimate.

## Validation results

| Metric | Logits + tree, primary | Hidden + tree |
|---|---:|---:|
| Accuracy | 0.31926406926406925 | 0.3203463203463203 |
| Macro-F1 | 0.2306766979333351 | 0.2298412019678988 |
| Weighted F1 | 0.33398724658349993 | 0.3337214717716569 |
| Balanced accuracy | 0.2358503567311523 | 0.2326428445012973 |
| Top-2 accuracy | 0.5768398268398268 | 0.5735930735930735 |
| Quadratic weighted kappa | 0.16358230799067508 | 0.15704266933375122 |

The primary predicted 10/20/40/80/160 counts are 27/88/251/300/258. Oracle
counts are 13/81/178/232/420.

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | 0.0 | 0.0 | 13 |
| 20 | 0.14772727272727273 | 0.16049382716049382 | 0.15384615384615383 | 81 |
| 40 | 0.24701195219123506 | 0.34831460674157305 | 0.2890442890442891 | 178 |
| 80 | 0.25333333333333335 | 0.3275862068965517 | 0.28571428571428575 | 232 |
| 160 | 0.5581395348837209 | 0.34285714285714286 | 0.4247787610619469 | 420 |

Confusion matrix, with Oracle rows and predictions in 10/20/40/80/160 order:

```text
[[  0,   1,   5,   4,   3],
 [  6,  13,  20,  28,  14],
 [  9,  20,  62,  53,  34],
 [  5,  20,  68,  76,  63],
 [  7,  34,  96, 139, 144]]
```

## Comparison with the components

| Metric | Phase 2D Qwen | Phase 3B tree | Phase 3C fusion |
|---|---:|---:|---:|
| Accuracy | 0.36904761904761907 | 0.329004329004329 | 0.31926406926406925 |
| Macro-F1 | 0.22994524079282935 | 0.2246699873714014 | 0.2306766979333351 |
| Weighted F1 | 0.3644656337102369 | 0.329619858512113 | 0.33398724658349993 |
| Balanced accuracy | 0.2391812745015638 | 0.2327228358766522 | 0.2358503567311523 |
| Top-2 accuracy | 0.6341991341991342 | 0.5898268398268398 | 0.5768398268398268 |
| Mean joined retrieval F1 | 0.2767166677489178 | 0.27172125974025974 | 0.2872016341991342 |
| Median joined retrieval F1 | 0.2558975 | 0.2487165 | 0.271346 |

Phase 3C improves macro-F1 over Phase 2D by 0.00073145714050575 and over
Phase 3B by 0.0060067105619337. It lowers accuracy relative to both. The result
is therefore not a general classification improvement.

The downstream result is more favorable: mean joined retrieval F1 improves by
0.0104849664502164 over Phase 2D and 0.01548037445887446 over Phase 3B. This
shows why Oracle-label classification and downstream retrieval must remain
separate metrics. The fusion's less majority-heavy prediction distribution
can improve retrieval overlap even when exact Oracle-label accuracy declines.

## Retrieval and Qdrant

The unchanged retrieval pipeline evaluated 924/924 questions at 100% coverage:

- same-paper filtering: true;
- top-k: 5;
- embedding: `text-embedding-3-small`, 1,536 dimensions;
- similarity: cosine;
- mean joined retrieval F1: 0.2872016341991342;
- median joined retrieval F1: 0.271346.

Qdrant was read-only. Its complete snapshots are identical before and after;
`PaperChunk` remains at 1,701,822 points and `PaperQuestion` at 4,526 points.

## Runtime

- Post-model-load Qwen extraction and verification: 93.55617220513523 seconds.
- Logits-fusion candidate time: 200.63359110243618 seconds.
- Hidden-fusion candidate time: 1036.5514699425548 seconds.
- Complete train-evaluate command, including both searches and final fits:
  1247.1628678720444 seconds.
- Retrieval: 335.15109509998 seconds.
- Known sequential stage time: 1675.8701351771597 seconds, approximately
  27 minutes 56 seconds.

The CV candidate times are already inside the train-evaluate command time and
are not added twice. Model-loading time before the extraction timer was not
recorded and is not fabricated.

## Comparability limitation

The Phase 2D checkpoint was previously selected using this same 924-example
validation split. Phase 3C is therefore a development-set result, not an
unbiased final test-set estimate. In addition, previous phases and this fusion
have repeatedly exposed the same validation outcomes during research. Any
claim about final generalization requires confirmation on an untouched test
set or a fully nested retraining protocol.

## Artifacts

The output directory contains the complete configuration, environment lock,
Qwen feature archives, both grouped searches, selection lock, both XGBoost
models, feature importance, predictions, metrics, confusion matrix, histogram,
retrieval records, runtime summary, and integrity records. See
`outputs/qwen_phase3c_fusion_evidence_length_oracle/final_summary.json` for the
complete SHA-256 inventory.
The final integrity decision is also recorded in `integrity/final_audit.json`.

## Reproduction

On a CUDA machine from the project root:

```bash
uv python install 3.10.7
uv venv --python 3.10.7 .venv-fusion
source .venv-fusion/bin/activate
uv pip install -r requirements-phase3c-fusion.txt
python qwen_phase3c_fusion.py audit
python qwen_phase3c_fusion.py extract-qwen --batch-size 4
python qwen_phase3c_fusion.py train-evaluate
```

After copying the output directory to the machine hosting the preserved Qdrant
service:

```powershell
.\.venv-qwen\Scripts\python.exe qwen_phase3c_fusion.py retrieve
.\.venv-phase3b\Scripts\python.exe qwen_phase3c_fusion.py finalize
.\.venv-phase3b\Scripts\python.exe -m pytest -q tests\test_qwen_phase3c_fusion.py tests\test_similarity_tree_phase3b.py
```
