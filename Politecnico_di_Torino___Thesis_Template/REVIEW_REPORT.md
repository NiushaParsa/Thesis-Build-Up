# Thesis review: experimental coverage, results, and narrative

Reviewed on 19 September 2026 against the preserved repository history,
phase reports, structured metrics, and prediction/retrieval records.

## Outcome

The main reported retrieval scores agree with the saved experiments.
The review found omissions and imprecise interpretations, rather than an
incorrect headline ranking. The thesis has been revised to make the early
work, secondary candidates, selection rules, and unsuccessful runs clearer.
The abstract, summary, results, and conclusion now give the same bounded
finding: the tested learned routers do not establish an advantage over
their strongest corresponding fixed-size baselines.

This is a review of the preserved evidence, not a new model evaluation or
a claim that every implementation detail is independently proven correct.

## Coverage from the first prototype to the final experiment

| Stage | Where covered | Important distinction retained |
| --- | --- | --- |
| Initial Weaviate ingestion, JSON exports, sample retrieval | Chapter 3; Appendix A | Engineering prototype, not a measured vector-store comparison |
| Evidence filtering, parallel processing, Qdrant migration and checkpoints | Chapter 3; Appendix A | Pipeline development precedes the frozen benchmark |
| Empty and three-question smoke outputs | Chapter 3; Appendix A | Not full validation experiments |
| Offsets, integrity audits, JSONL locking, completion counters, metric schema and persistence | Chapter 3; Appendices A/C | Historical collection issues are not silently declared repaired |
| Five fixed sizes and offline best-action oracle | Chapters 3-5 | Oracle uses gold outcomes and is not deployable |
| Four embedding-logistic candidates and one MLP | Chapters 4-5; Appendix B | MLP classification gain misses its adoption threshold; no invented MLP retrieval result |
| Raw and overlap-deduplicated mixed retrieval | Chapters 4-5 | Same chunk count does not imply equal text budgets |
| Phase 1 frozen Qwen | Chapters 4-5 | Valid outputs do not imply good size decisions |
| Phase 2 smoke/tiny-overfit checks, interrupted v1, corrected full v2 | Chapters 4-5; Appendices A/B | Interrupted run has no final validation result |
| Phases 2B-A/B aliases and class weighting | Chapters 4-5; Appendix B | All three checkpoints retained; balanced variant selects epoch 2 |
| Phases 2C/D head and token-count instruction | Chapters 4-5; Appendix B | Macro F1, accuracy, and retrieval can move differently |
| Phase 2E learning-rate/duration grid | Chapters 4-5; Appendix B | All 15 candidates; only locked winner has final retrieval evaluation |
| Phase 3A linear level/tree models and four heuristics | Chapters 4-5; Appendix B | Tree representation predesignated; heuristic results are classification diagnostics |
| Phase 3B level/tree XGBoost | Chapters 4-5; Appendix B | Training-fold macro F1 selects the tree model; weighting and capacity both change |
| Phase 3C logits and hidden-state fusion | Chapters 4-5; Appendix B | In-sample base-feature contamination explicitly retained |
| Phase 3C-OOF correction and full-data refit | Chapters 4-5 | Corrected stacking interface, not a fully nested or untouched test evaluation |
| Phase 4 utility-target recovery and expected-regret regression | Chapters 3-5 | All 2,245 training targets recovered; evaluation uses frozen actions |
| Phase 5A local decisions, five same-tree controls, all-descendants probe | Chapters 3-5 | Primary and larger-budget exploratory outputs separated |
| Phase 5B failed free generation | Chapters 4-5 | 231 invalid outputs; no downstream retrieval score |
| Training-only repair probe and Phase 5B-v2 constrained feature | Chapters 4-5 | 173/175 still fails the probe gate; valid final feature is nearly constant |

The chronology uses the order of preserved development, not commit dates
as a substitute for exact experiment execution timestamps. Copies, hash
audits, transfers, and resumed extraction are not counted as new models.

## Numerical reconciliation

`scripts/audit_results.py` is a read-only, reproducible audit. Its detailed
output is written to ignored `build/qa/results_audit.json`. It checks:

- Frozen question and paper populations, training/validation separation,
  evidence-length counts, legacy validation targets, and exact utility
  argmax counts.
- The complete 4,620-row original fixed-action grid against the Phase 4
  utility vectors, and the completed 2,245-question training utility set.
- Accuracy, macro F1, weighted F1, balanced accuracy, confusion matrices,
  available top-two results, and prediction counts for all 11 main
  evidence-length question-level phases.
- Per-question retrieval means and medians. Every routed score in those
  11 phases agrees exactly with the corresponding saved fixed-size action.
- Early logistic, raw-mixed, and deduplicated-mixed retrieval aggregates.
- Fixed/global means and the Phase 4 paired paper-cluster intervals,
  independently reconstructed from question scores and paper membership.
- Local precision, recall, F1 and token means, medians, 4,620 local size
  decisions per method, diagnostic confusion matrices, and Phase 5A/5B-v2
  paired intervals.
- Free-generation failures, valid context-category distributions, and the
  training-only prompt-repair probe.
- Source hashes used by the generated tables and figures.

The final audit passes 12,398 record/aggregate assertions across 70 read
source files. Many assertions are per-record checks, not independent
statistical tests. The result generator separately records 41 input-file
hashes in `content/generated/source_manifest.json`.

The headline mean retrieval F1 values remain:

| Method | Mean F1 | Appropriate reference |
| --- | ---: | --- |
| Global fixed 40 | 0.3159 | Best observed global fixed-size setting |
| Offline best-of-five oracle | 0.3821 | Gold-informed opportunity within the saved action set |
| Selected embedding-logistic router | 0.2853 | Global fixed retrieval |
| Raw / deduplicated mixed retrieval | 0.2562 / 0.2618 | Global fixed retrieval |
| Best Qwen-only retrieval: uniform aliases | 0.2865 | Global fixed retrieval |
| Corrected OOF fusion | 0.2783 | Global fixed retrieval |
| Expected-regret router | 0.3075 | Global fixed 40: 0.3159 |
| Local Phase 5A | 0.3053 | Same-tree fixed 40: 0.3075 |
| Local Phase 5B-v2 | 0.3045 | Phase 5A and same-tree fixed 40 |

Phase 4 and same-tree fixed 40 both round to 0.3075 but use different
retrieval protocols. They are not the same system or the same exact score.

## Corrections and additions

1. Expanded the early infrastructure and evaluation history, including
   prototype storage, filtering, parallel processing, schema/persistence
   work, and the explicit frozen-data eligibility rule.
2. Separated JSONL-write locking from the double-counted chunk counter;
   clarified that valid empty evidence stages are not necessarily errors.
3. Named the exact evidence-similarity aggregate and distinguished it from
   all-pairs similarity and joined lexical F1.
4. Corrected Phase 3B's primary-model selection to training-fold macro F1.
   Its mean retrieval improvement is now accompanied by its lower median;
   the comparison does not isolate nonlinearity from class weighting.
5. Distinguished early 500-replicate question bootstrapping from later
   10,000-replicate paper-cluster intervals.
6. Added generated tables for the five early candidates, fifteen
   intermediate three-epoch checkpoints, and seven secondary variants.
   The existing fifteen-candidate Phase 2E table remains.
7. Added retrieval medians to the Qwen and global tree/fusion tables.
8. Clarified the still-failing 173/175 repair probe; corrected GPU memory
   units from GB to GiB for the recorded fusion measurements.
9. Revised the abstract and summary to avoid unsupported claims that one
   design factor matters more than another. The final context-feature
   score and the limits of the retrieval-only study remain explicit.
10. Improved chapter transitions, removed a repetitive methodology ending,
    and corrected stale statements that acknowledgements were omitted.
11. Rebuilt the chronology as two explicitly continued table blocks,
    removing the previous longtable glue error. Strengthened the document
    checker to fail on ignored errors and layout warnings.

## Remaining scientific limits

The official validation split was repeatedly used during development;
bootstrap intervals do not include training-seed or model-selection
uncertainty. Lexical evidence F1 is not answer correctness, and a constant
chunk count is not a constant token budget. The local training population
excludes questions without mappable text evidence. No untouched test run,
new retrieval call, GPU training, or database mutation was performed for
this review. Unrecorded experiments cannot be reconstructed from absent
artifacts. Final scientific interpretation and institutional requirements
still need the author's and supervisors' approval.

## Repeating the checks

From the thesis directory:

```powershell
python scripts/generate_results.py
python scripts/audit_results.py
powershell -ExecutionPolicy Bypass -File scripts/build_thesis.ps1
python scripts/check_document.py
```

The first two commands read saved experimental artifacts. The build and
document check compile, extract, and render the thesis without using the
experimental services. The final rendered pages were inspected after the
content changes; build products and review images remain ignored by Git.

The final PDF has 74 A4 pages, sixteen numbered tables, four figures,
and eighteen cited references. The document checks find no missing
citations or cross-references, duplicate labels, unresolved markers,
overfull boxes, ignored errors, or LaTeX/package warnings. MiKTeX's
separate update-check reminder remains; it does not block compilation.
