# Pretrained Qwen3.5-0.8B router with evidence-length Oracle

## Status and objective

This is Phase 1: evaluation of the original post-trained `Qwen/Qwen3.5-0.8B`
as a zero-shot granularity router before any fine-tuning. Phase 2 will
fine-tune the same model separately. The implementation, Oracle generation,
focused tests, one-prompt compatibility check, three-question smoke test, full
924-question validation inference, classification evaluation, and unchanged
end-to-end retrieval evaluation are complete.

Repository revision at implementation time was
`dfb49b6ef04bcdbb488d35302bb890cae5f84183`.

## Environment and model

- Environment: `.venv-qwen`, Python 3.10.7.
- Executable: `C:\Users\behno\Repos\Thesis Build Up\.venv-qwen\Scripts\python.exe`.
- Transformers: `5.15.0.dev0`, source commit
  `2ef79f87a02111f8b49a72fb7d0c86b5b0bf10b7`.
- PyTorch: `2.8.0+cpu`; device `cpu`; dtype `torch.bfloat16`.
- Model and tokenizer/processor: `Qwen/Qwen3.5-0.8B`.
- Pinned model revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Quantization: none.
- Official chat template: used with one user text-content item.
- Decoding: `do_sample=False`, `max_new_tokens=8`.
- Model state: `eval()`, all parameters frozen with
  `requires_grad_(False)`, and `torch.inference_mode()`.
- No optimizer, gradients, backward pass, adapters, LoRA/QLoRA, prompt tuning,
  classification-head training, or parameter update is present.

The original Python 3.9 `.venv` was not modified and remains the environment
for reproducing the previous Logistic Regression, MLP, retrieval-F1 Oracle,
and retrieval experiments. `.venv-qwen` is reserved for Qwen Phase 1 and the
later Phase 2.

The interpreter audit, minimal direct dependency manifest, version differences,
and exact environment creation commands are documented in
`docs/QWEN_ENVIRONMENT.md`. `requirements-qwen.txt` contains only direct
Qwen-pipeline dependencies; the complete resolved package set is retained in
the environment lock.

## Dataset and Oracle

The frozen old-Oracle configuration
`9a3022fd1c808f72ccbf3265fe6020593bb58bdd28aeb9025b8c4b735d669de8`
defines the preserved examples: 2,245 official-train questions and 924
official-validation questions. Each example is an answerable question with
non-empty highlighted evidence and a complete five-granularity old-Oracle
record. No example was added or removed.

The previous Oracle selected the chunk size with the best joined top-five
retrieval F1, with its existing tie procedure. The new Oracle is independent
of embeddings and retrieval: strip evidence strings, discard empty strings,
deduplicate exact stripped strings across annotators, lexicographically sort
them, join with newline separators, count tokens using the project GPT-2
tokenizer without special tokens, and select the numerically closest class
from 10, 20, 40, 80, and 160. Exact midpoint ties choose the smaller class.
Lengths below 10 map to 10 and lengths above 160 map to 160.

| Split | 10 | 20 | 40 | 80 | 160 |
|---|---:|---:|---:|---:|---:|
| Train (n=2,245) | 55 (2.45%) | 267 (11.89%) | 586 (26.10%) | 687 (30.60%) | 650 (28.95%) |
| Validation (n=924) | 13 (1.41%) | 81 (8.77%) | 178 (19.26%) | 232 (25.11%) | 420 (45.45%) |

Representative checks include evidence lengths below 10, exact midpoints at
15/30/60/120, values above 160, and questions with multiple evidence spans.
All passed. The focused automated suite contains 33 passing tests.

The new validation Oracle is strongly imbalanced: 160 tokens is the majority
class at 420/924 (45.45%), while 10 tokens has only 13 examples (1.41%). This
is the opposite of Qwen's prediction tendency and makes accuracy, macro-F1,
and weighted F1 convey different aspects of performance.

## Qwen input

Logistic Regression and MLP consume 1,536-dimensional
`text-embedding-3-small` question embeddings. Qwen instead consumes original
question text because it is a text-generating model. Its only semantic input is:

> You are a router for a retrieval-augmented generation system. Based only on
> the question, select the chunk size most suitable for retrieving the evidence
> required to answer it. Choose exactly one value from: 10, 20, 40, 80, 160.
> Return only the number.

followed by `Question: {original_question_text}`. No evidence, evidence length,
answer, paper text/title, retrieval output/score, embedding, metadata,
handcrafted feature, demonstration, or label is provided.

## Parser, inference, and classification results

The parser uses numeric boundaries and accepts only one distinct candidate
class. Repeated occurrences of the same class are valid; no class or multiple
different classes is invalid. Invalid output is never defaulted.

The first compatibility prompt loaded in 86.49 seconds from download state;
the three-example smoke test was only a technical check. The approved full run
processed all 924 examples without resumption. Total model inference time was
5,745.05 seconds; full inference-stage wall time was 5,774.17 seconds
(1 h 36 min 14 s). Mean/median generation time was 6.2176/4.9827 seconds.
Approximate peak process RSS was 2.84 GiB. All 924 outputs were valid, giving
100% valid-output rate and zero invalid or ambiguous outputs.

| Metric | Qwen zero-shot | Evidence-length majority baseline |
|---|---:|---:|
| Accuracy | 0.040043 | 0.454545 |
| Macro-F1 | 0.049046 | 0.125000 |
| Weighted F1 | 0.032613 | 0.284091 |
| Balanced accuracy | 0.233694 | 0.200000 |
| Top-2 accuracy | unavailable | unavailable |

Top-2 accuracy is unavailable because deterministic text generation does not
produce comparable five-class scores. Qwen predicted 10: 767, 20: 40, 40:
116, 80: 0, and 160: 1, versus Oracle counts 13/81/178/232/420. Thus the
pretrained model collapses heavily toward 10 and completely misses class 80;
it predicts the majority Oracle class 160 only once. Per-class metrics and the
5×5 confusion matrix are stored in `classification/metrics.json`.

Classification accuracy is the fraction of exact class matches. Macro-F1
weights all five classes equally; weighted F1 weights class F1 by Oracle
support. These classification quantities are not retrieval F1.

## End-to-end retrieval results

All 924 valid predictions were evaluated with the unchanged paper-restricted,
top-k=5 retrieval pipeline using `text-embedding-3-small`, the existing Qdrant
collections/ranking, newline concatenation, GPT-2 tokenization, and joined
token-level retrieval F1. Retrieval coverage is 1.0. Mean joined retrieval F1
is 0.239109 and median joined retrieval F1 is 0.221085. The retrieval stage
took 367.76 seconds. Because no output was invalid, valid-only and
coverage-adjusted full-set mean retrieval F1 are identical. If invalid outputs
had existed, they would have no fabricated retrieval record; coverage and the
valid-only aggregate would be reported separately.

Old Logistic Regression/MLP classification results use the retrieval-F1
Oracle and are not directly comparable with this evidence-length-Oracle
classification result. Existing retrieval scores can be identified by their
retrieval method, but differing Oracle definitions must not be presented as
equivalent router-label experiments.

## Artifacts

- Oracle JSONL and distributions:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/`
- Histogram:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/oracle_distribution.svg`
- Representative inspections:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/representative_inspections.json`
- Smoke predictions and timings:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/smoke/`
- Complete predictions/raw/parsed/invalid/runtime artifacts:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/validation/`
- Classification metrics/confusion matrix/histogram:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/classification/`
- Per-question retrieval records and summary:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/retrieval/`
- Prompt and environment:
  `outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/configuration/`
- Environment lock:
  `reports/qwen_pretrained_zero_shot_router_evidence_length_oracle/qwen_environment_lock.txt`
- Implementation and tests: `qwen_phase1.py`, `tests/test_qwen_phase1.py`.

## Reproduction

```powershell
# Focused Oracle and parser tests
.\.venv-qwen\Scripts\python.exe -m pytest tests\test_qwen_phase1.py -q

# Generate and validate the separate evidence-length Oracle
.\.venv-qwen\Scripts\python.exe qwen_phase1.py generate-oracle
.\.venv-qwen\Scripts\python.exe qwen_phase1.py validate-oracle

# Small smoke test (does not run full validation)
$env:HF_HOME="$PWD\tmp\huggingface_qwen_cache"
.\.venv-qwen\Scripts\python.exe qwen_phase1.py smoke-test --count 3

# Resumable complete validation inference and evaluations
.\.venv-qwen\Scripts\python.exe qwen_phase1.py infer-validation
.\.venv-qwen\Scripts\python.exe qwen_phase1.py evaluate-classification
.\.venv-qwen\Scripts\python.exe qwen_phase1.py evaluate-retrieval
```

Phase 2 must use the same evidence-length Oracle and otherwise comparable
evaluation conditions. Its fine-tuned result must remain separate from this
pretrained zero-shot baseline.
