# Qwen Phase 1 validation results

The pretrained `Qwen/Qwen3.5-0.8B` Phase 1 run evaluated all 924 preserved
official-QASPER validation questions against the GPT-2 evidence-length Oracle.
No fine-tuning, optimizer, gradients, adapters, LoRA/QLoRA, or parameter
updates were used. Qwen received only the fixed instruction and original
question text.

The Oracle distribution is strongly imbalanced: 10/20/40/80/160 has
13/81/178/232/420 examples. Class 160 alone represents 45.45% of validation.
Qwen instead predicted 767/40/116/0/1, collapsing toward class 10 and almost
never selecting the Oracle-majority class 160.

| Metric | Value |
|---|---:|
| Classification accuracy | 0.040043 |
| Classification macro-F1 | 0.049046 |
| Classification weighted F1 | 0.032613 |
| Balanced accuracy | 0.233694 |
| Valid outputs | 924/924 (100%) |
| Invalid outputs | 0 (0.00%) |
| Majority-class baseline accuracy | 0.454545 |
| Majority-class baseline macro-F1 | 0.125000 |
| Mean joined retrieval F1 | 0.239109 |
| Median joined retrieval F1 | 0.221085 |
| Retrieval coverage | 924/924 (100%) |

Top-2 accuracy is unavailable because deterministic generated text provides no
comparable five-class scores. Classification accuracy, macro-F1, and weighted
F1 measure Oracle-label prediction. Joined retrieval F1 measures token overlap
after unchanged same-paper top-five retrieval; it is a different metric.

Inference wall time was 5,774.17 seconds (1 h 36 min 14 s), mean/median
generation time was 6.2176/4.9827 seconds, and approximate peak process RSS
was 2.84 GiB. Retrieval took 367.76 seconds.

Previous Logistic Regression and MLP results use the old retrieval-F1 Oracle
and are not directly comparable with this evidence-length-Oracle
classification result. Full configuration, per-example records, confusion
matrix, histogram, and reproduction commands are in
`reports/qwen_pretrained_zero_shot_router_evidence_length_oracle/experiment_report.md`.
