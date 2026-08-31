# Phase 5A — Tree-Local Similarity Router with Gold-Overlap Training

## Objective and protocol

This separate experiment implements Lorenzo's tree-local proposal. A tree is
rooted at one 160-token chunk and contains its aligned 80/40/20/10-token
descendants. Training uses only trees that overlap deduplicated gold evidence.
Each tree's label is the nearest candidate size to the GPT-2 token length of
the merged evidence portion inside that tree (smaller midpoint ties).

Inputs contain only the 173 local similarity-distribution features. Gold
evidence creates training targets but is not an input. Qwen is not used.

At inference, every paper tree is scored by the average of its five level-mean
similarities. The top five trees are classified independently, and the single
most similar chunk at the predicted level is retained from each tree.

## Data construction

- Source train questions: 2245
- Eligible train questions: 2101
- Excluded train questions: 144
- Gold-overlap training tree examples: 4081

Of the training questions, 1,915 had all unique text evidence located, 186
had a usable text-evidence subset plus one or more unavailable evidence items,
and 144 were excluded from supervised tree construction because none of their
gold evidence existed in the chunked paper text. The unresolved items are
predominantly QASPER `FLOAT SELECTED` table/figure annotations, which cannot
overlap any tree in the current text-only chunk collection. They are audited
rather than assigned fabricated spans. Inference and retrieval evaluation
still cover all 924 validation questions because they do not require a local
gold label.

| Local label | Training trees |
|---:|---:|
| 10 | 557 |
| 20 | 876 |
| 40 | 1307 |
| 80 | 958 |
| 160 | 383 |

## Results

| Method | Mean precision | Mean recall | Mean joined F1 | Median joined F1 | Mean chunks |
|---|---:|---:|---:|---:|---:|
| phase5a | 0.275434 | 0.511375 | 0.305266 | 0.303030 | 5.00 |
| phase5a_all_predicted_level_chunks | 0.138196 | 0.782622 | 0.213142 | 0.182302 | 22.44 |
| same_tree_fixed_10 | 0.382916 | 0.193173 | 0.219495 | 0.212121 | 5.00 |
| same_tree_fixed_20 | 0.344019 | 0.323690 | 0.281981 | 0.277414 | 5.00 |
| same_tree_fixed_40 | 0.284637 | 0.486212 | 0.307462 | 0.306590 | 5.00 |
| same_tree_fixed_80 | 0.210347 | 0.646902 | 0.279123 | 0.260656 | 5.00 |
| same_tree_fixed_160 | 0.138467 | 0.783262 | 0.213518 | 0.183269 | 5.00 |

The Phase 5A row is the adaptive tree-local classifier. The same-tree fixed
rows use the identical top-five tree ranking and differ only in their fixed
within-tree granularity, isolating the effect of classification.

`phase5a_all_predicted_level_chunks` is the separately labelled exploratory
variant Lorenzo suggested if time allowed. It retains every descendant at the
predicted level, so its number of chunks is variable and it is not the primary
five-chunk result.

| Predicted local granularity | Top-five validation trees |
|---:|---:|
| 10 | 246 |
| 20 | 529 |
| 40 | 2823 |
| 80 | 965 |
| 160 | 57 |

The primary adaptive router's mean joined F1 is 0.305266. The directly matched
same-tree fixed-40 result is 0.307462; the adaptive-minus-fixed difference is
-0.002195, with paired paper-cluster bootstrap 95% CI [-0.005797, 0.001451].
Thus the local classifier did not improve over fixed 40 under the identical
tree-ranking and one-chunk-per-tree procedure.

The exploratory all-descendant variant has mean F1 0.213142. It raises mean
recall to 0.782622 but lowers mean precision to 0.138196, while selecting 22.44
chunks and 809.50 tokens on average. This supports retaining only the most
similar chunk per selected tree for the primary procedure.

Secondary tree-label diagnostic: accuracy 0.284957,
macro-F1 0.214401, weighted F1
0.256329, balanced accuracy
0.222537. Its unit is an overlapping tree,
not a question.

## Methodological safeguards

- Train/validation papers remain disjoint.
- Fixed Phase 3B XGBoost settings were reused; no Phase 5A hyperparameter search.
- Validation predictions were saved and hashed before validation evidence was
  requested for final scoring.
- The frozen Phase 3A score file contains an unused legacy question-level
  Oracle field; Phase 5A ignores it. No Phase 5A local validation target or
  validation evidence payload was requested before prediction locking.
- Qdrant was read-only and its before/after collection snapshot was unchanged.
- No embedding inference, Qwen inference, GPU training, or Vast.ai instance was used.
- This is a development result because the preserved validation set has been
  reused across earlier thesis phases.

## Reproduction

```powershell
py -3.10 -m venv .venv-phase5a
.\.venv-phase5a\Scripts\python.exe -m pip install -r requirements-phase5a.txt
.\.venv-phase5a\Scripts\python.exe -m pytest tests/test_similarity_tree_phase5a.py -q
.\.venv-phase5a\Scripts\python.exe similarity_tree_phase5a.py run
```

Artifacts: `outputs/similarity_tree_phase5a_local_gold_overlap_router`.
