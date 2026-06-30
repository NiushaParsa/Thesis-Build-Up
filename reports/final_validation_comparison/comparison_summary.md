# Final validation retrieval comparison

Split: validation. The test split is not loaded or evaluated by this report.

## Main results

Common validation questions: 924
Best fixed single level: `fixed_40`
Best average deployable strategy: `fixed_40`
Oracle upper-bound gap for the best deployable strategy: 0.066217

| strategy | deployable | n | mean F1 | 95% CI | median F1 | mean regret vs oracle |
|---|---:|---:|---:|---:|---:|---:|
| fixed_10 | True | 924 | 0.221119 | [0.214280, 0.227792] | 0.207362 | 0.160954 |
| fixed_20 | True | 924 | 0.288750 | [0.281308, 0.295729] | 0.280918 | 0.093324 |
| fixed_40 | True | 924 | 0.315856 | [0.307486, 0.324308] | 0.309167 | 0.066217 |
| fixed_80 | True | 924 | 0.285725 | [0.277073, 0.295539] | 0.266568 | 0.096349 |
| fixed_160 | True | 924 | 0.216794 | [0.209045, 0.226020] | 0.183777 | 0.165280 |
| oracle_upper_bound | False | 924 | 0.382073 | [0.373228, 0.390128] | 0.383553 | 0.000000 |
| router_selected | True | 924 | 0.285307 | [0.277256, 0.293646] | 0.273309 | 0.096767 |
| mixed_raw | True | 924 | 0.256175 | [0.247691, 0.264146] | 0.227824 | 0.125898 |
| mixed_deduplicated | True | 924 | 0.261784 | [0.253113, 0.270304] | 0.239164 | 0.120290 |

## Router-selected diagnostics

```json
{
  "confusion_matrix_tokens": {
    "10": {
      "10": 26,
      "160": 1,
      "20": 47,
      "40": 39,
      "80": 24
    },
    "160": {
      "10": 15,
      "160": 6,
      "20": 24,
      "40": 21,
      "80": 16
    },
    "20": {
      "10": 42,
      "160": 7,
      "20": 67,
      "40": 78,
      "80": 41
    },
    "40": {
      "10": 52,
      "160": 11,
      "20": 79,
      "40": 94,
      "80": 47
    },
    "80": {
      "10": 36,
      "160": 6,
      "20": 43,
      "40": 66,
      "80": 36
    }
  },
  "mean_router_latency_ms": 1.4131709956709957,
  "oracle_target_distribution": {
    "1": 137,
    "2": 235,
    "3": 283,
    "4": 187,
    "5": 82
  },
  "predicted_granularity_distribution": {
    "10": 171,
    "20": 260,
    "40": 298,
    "80": 164,
    "160": 31
  },
  "router_oracle_match_rate": 0.24783549783549783
}
```

## Mixed diagnostics

```json
{
  "mixed_deduplicated": {
    "dominant_retrieved_granularity_distribution": {
      "10": 806,
      "20": 84,
      "40": 22,
      "80": 4,
      "160": 8
    },
    "topk_granularity_composition": {
      "10": 3257,
      "20": 792,
      "40": 322,
      "80": 133,
      "160": 116
    }
  },
  "mixed_raw": {
    "dominant_retrieved_granularity_distribution": {
      "10": 801,
      "20": 86,
      "40": 23,
      "80": 8,
      "160": 6
    },
    "topk_granularity_composition": {
      "10": 2708,
      "20": 1148,
      "40": 460,
      "80": 185,
      "160": 119
    }
  }
}
```

## Sources

```json
{
  "fixed_separate": "outputs\\oracle_frozen\\validation\\RetrievalEvalFixedSeparate_20260623_171712.jsonl",
  "mixed_deduplicated": "outputs\\mixed_granularity\\validation\\deduplicated\\RetrievalEvalMixedDeduplicated_20260623_194703.jsonl",
  "mixed_raw": "outputs\\mixed_granularity\\validation\\raw\\RetrievalEvalMixedRaw_20260623_194035.jsonl",
  "router_dataset_oracle": "outputs\\oracle_frozen\\validation\\RouterDataset_20260623_171712.jsonl",
  "router_selected": "outputs\\router_selected\\validation\\RetrievalEvalRouterSelected_20260623_191427.jsonl"
}
```
