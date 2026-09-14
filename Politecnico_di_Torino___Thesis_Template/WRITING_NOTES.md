# Thesis draft

Title: Adaptive Granularity Retrieval for Retrieval-Augmented Generation  
Author: Niusha Parsa  
Programme: Data Science and Engineering  
Supervisors: Professor Giuseppe Rizzo; Dr. Lorenzo Bongiovanni  
Session: September 2026 — academic year 2025–2026

## Structure

The main document contains the abstract, extended summary, notation, six
chapters, three appendices, and an experiment-specific bibliography.
The six chapters are Introduction; Background and Literature Review;
Dataset and Preprocessing; Methodology; Evaluation, Results, and Discussion;
and Conclusion and Future Work.

## Build

From this directory, run:

    powershell -ExecutionPolicy Bypass -File scripts/build_thesis.ps1

The PDF is build/thesis.pdf. Existing LaTeX Workshop recipes also work.
All four scientific figures and generated tables are retained as source
assets; a normal build does not need Python, Qdrant, model weights, or API keys.
Use the optional -RefreshResults switch only to regenerate the derived
assets from the existing repository artifacts. This reads experiment files
without rerunning any experiments.

## Scientific review points

- The official validation split was repeatedly used for development; the
  document does not claim an untouched test evaluation.
- Retrieval F1 measures joined lexical evidence overlap, not generated-answer
  quality or the official QASPER answer metric.
- The early embedding router uses retrieval-derived labels. Most later
  classifiers use evidence-length labels. Their classification F1 values are
  not interchangeable.
- Phase 4's 0.3075 and same-tree fixed 40's 0.3075 are rounded scores from
  different protocols, not identical systems or duplicated results.
- The original fusion's in-sample feature problem and the first Phase 5B
  version's invalid-output failure are retained explicitly.
- Top-five comparisons do not enforce an equal total token budget.
- Personal acknowledgements are omitted, not invented. The old template
  acknowledgement, glossary, and bibliography examples are not compiled.

## Evidence and editing

The file content/generated/source_manifest.json records hashes of the structured
inputs used for generated results. The script scripts/generate_results.py checks
question counts, paper counts, and fixed-action averages. Hand-written
chapter prose should be reviewed alongside the phase reports in ../docs/.
No model training, retrieval run, remote write, or database mutation was
performed for this draft.

Before submission, review wording and interpretation with the supervisors,
check institutional formatting requirements, and add any personal
acknowledgements. This is a complete draft for review, not a claim of formal
approval or a substitute for the author's scientific responsibility.

## Verification on 12 September 2026

- The final PDF contains 70 A4 pages, six main chapters, three appendices,
  thirteen numbered tables, four figures, and eighteen cited references.
- The pdfLaTeX/Biber build completed successfully.
- Automated checks found no missing citations, missing cross-references,
  duplicate labels, unresolved markers, overfull boxes, or LaTeX warnings.
- All pages were rendered for visual review. Pagination, table placement,
  bibliography overflow, and split diagnostic examples were corrected.
- Build products and QA renders remain ignored by Git.
- MiKTeX still prints its separate update-check reminder; it does not prevent
  compilation and is not an error in the thesis source.
