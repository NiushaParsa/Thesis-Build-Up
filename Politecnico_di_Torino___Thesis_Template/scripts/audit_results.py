"""Reconcile thesis results with saved records, without running experiments.

Uses the Python standard library and NumPy. Output is an audit in build/qa;
source artifacts, models, database state, and thesis prose are never changed.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics

import numpy as np

THESIS = Path(__file__).resolve().parents[1]
REPO = THESIS.parent
SIZES = [10, 20, 40, 80, 160]
SOURCES = {}
CHECKS = []
RESULTS = {}


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    raw = path.read_bytes()
    SOURCES[path.relative_to(REPO).as_posix()] = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8-sig')
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def check(name, condition):
    CHECKS.append({'name': name, 'passed': bool(condition)})


def close(name, actual, expected, tolerance=1e-10):
    check(name, math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0))


def population(name, rows, expected_ids):
    ids = [row['question_id'] for row in rows]
    check(name + ': unique and complete question IDs', len(ids) == len(set(ids)) and set(ids) == expected_ids)
    check(name + ': paper identities', all(row['document_id'] == VALIDATION[row['question_id']]['document_id'] for row in rows))


def classification(name, rows, saved):
    matrix = [[0] * 5 for _ in SIZES]
    for row in rows:
        target = int(row['oracle_label'])
        predicted = int(row.get('parsed_prediction', row.get('predicted_label')))
        matrix[SIZES.index(target)][SIZES.index(predicted)] += 1
        check(name + ': label ' + row['question_id'], target == int(VALIDATION[row['question_id']]['oracle_label']))
    support = [sum(row) for row in matrix]
    predicted = [sum(row[index] for row in matrix) for index in range(5)]
    f1 = [2 * matrix[i][i] / (support[i] + predicted[i]) if support[i] + predicted[i] else 0 for i in range(5)]
    recall = [matrix[i][i] / support[i] if support[i] else 0 for i in range(5)]
    n = len(rows)
    values = {
        'accuracy': sum(matrix[i][i] for i in range(5)) / n,
        'macro_f1': statistics.mean(f1),
        'weighted_f1': sum(f1[i] * support[i] for i in range(5)) / n,
        'balanced_accuracy': statistics.mean(recall),
    }
    cm = saved.get('classification_metrics', saved)
    for metric, value in values.items():
        close(name + ': ' + metric, value, cm[metric])
    check(name + ': confusion matrix', matrix == cm['confusion_matrix'])
    check(name + ': prediction distribution', predicted == [saved['predicted_distribution'][str(g)] for g in SIZES])
    if all('top_2_predictions' in row for row in rows) and isinstance(cm.get('top_2_accuracy'), (int, float)):
        close(name + ': top-two accuracy', sum(int(row['oracle_label']) in row['top_2_predictions'] for row in rows) / n, cm['top_2_accuracy'])
    return values


def bootstrap(values, paper_ids, saved, name):
    if 'observed_mean_difference' in saved:
        saved = dict(saved, point_estimate=saved['observed_mean_difference'],
                     ci95_lower=saved['confidence_interval_95'][0],
                     ci95_upper=saved['confidence_interval_95'][1],
                     iterations=saved['replicates'])
    # Aggregate by paper, then resample paper clusters with their question counts.
    groups = defaultdict(list)
    for value, paper in zip(values, paper_ids):
        groups[paper].append(value)
    ordered = sorted(groups)
    sums = np.array([sum(groups[paper]) for paper in ordered])
    counts = np.array([len(groups[paper]) for paper in ordered])
    rng = np.random.default_rng(saved.get('seed', 42))
    draws = rng.integers(0, len(ordered), size=(saved.get('iterations', 10000), len(ordered)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    for key, actual in [('point_estimate', statistics.mean(values)), ('ci95_lower', low), ('ci95_upper', high)]:
        close(name + ': ' + key, actual, saved[key])


oracle_root = 'outputs/qwen_pretrained_zero_shot_router_evidence_length_oracle/oracle/'
train = read(oracle_root + 'train_oracle.jsonl')
validation = read(oracle_root + 'validation_oracle.jsonl')
VALIDATION = {row['question_id']: row for row in validation}
IDS = set(VALIDATION)
check('Frozen populations', len(train) == 2245 and len(validation) == len(IDS) == 924)
check('Frozen paper counts', len({r['document_id'] for r in train}) == 845 and len({r['document_id'] for r in validation}) == 277)
check('Question split separation', not ({r['question_id'] for r in train} & IDS))
check('Paper split separation', not ({r['document_id'] for r in train} & {r['document_id'] for r in validation}))
for split, rows, expected in [('train', train, [55,267,586,687,650]), ('validation', validation, [13,81,178,232,420])]:
    check(split + ': evidence-length distribution', [Counter(r['oracle_label'] for r in rows)[g] for g in SIZES] == expected)

manifest = read(THESIS / 'content/generated/source_manifest.json')
for relative, expected in manifest['source_sha256'].items():
    check('Generated source hash: ' + relative, hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == expected)

base = read('outputs/qwen_phase4_expected_regret_retrieval_utility/comparison/baselines.json')
phase4 = read('outputs/qwen_phase4_expected_regret_retrieval_utility/retrieval/results.jsonl')
population('Phase 4', phase4, IDS)
utilities = {row['question_id']: row['utility_by_granularity'] for row in phase4}
original_fixed = read('outputs/oracle_frozen/validation/RetrievalEvalFixedSeparate_20260623_171712.jsonl')
fixed_keys = {(row['question_id'], row['granularity_tokens']) for row in original_fixed}
check('Original fixed records: complete unique action grid',
      len(original_fixed) == len(fixed_keys) == 4620 and fixed_keys == {(qid, g) for qid in IDS for g in SIZES})
check('Original fixed records: paper identity and eligible output',
      all(row['document_id'] == VALIDATION[row['question_id']]['document_id']
          and row['valid_evidence_count'] > 0 and row['returned_k'] > 0 for row in original_fixed))
close('Original fixed records: exact agreement with Phase 4 utilities',
      max(abs(row['f1_joined_topk'] - utilities[row['question_id']][str(row['granularity_tokens'])])
          for row in original_fixed), 0)
legacy = read('outputs/oracle_frozen/validation/RouterDataset_20260623_171712.jsonl')
population('Legacy router target records', legacy, IDS)
check('Legacy validation target distribution',
      [Counter(row['router_target_granularity'] for row in legacy)[level] for level in range(1, 6)] == [137,235,283,187,82])
check('Exact validation utility-argmax distribution',
      [Counter(min(SIZES, key=lambda g: (-u[str(g)], g)) for u in utilities.values())[g] for g in SIZES]
      == [138,235,282,187,82])
training_utilities = read('outputs/qwen_phase4_expected_regret_retrieval_utility/targets/train_utility_targets.jsonl')
check('Completed training utility population', len(training_utilities) == 2245
      and {r['question_id'] for r in training_utilities} == {r['question_id'] for r in train})
check('Training utilities: all five actions and valid regret',
      all(set(row['utility_by_granularity']) == {str(g) for g in SIZES}
          and all(math.isclose(max(row['utility_by_granularity'].values()) - row['utility_by_granularity'][str(g)],
                               row['regret_by_granularity'][str(g)], abs_tol=1e-12)
                  for g in SIZES) for row in training_utilities))
strategy_vectors = {}
for size in SIZES:
    vector = [row['utility_by_granularity'][str(size)] for row in phase4]
    strategy_vectors[f'fixed_{size}'] = vector
strategy_vectors['retrieval_oracle_upper_bound'] = [max(row['utility_by_granularity'].values()) for row in phase4]
strategy_vectors['phase4'] = [row['selected_joined_retrieval_f1'] for row in phase4]
for row in phase4:
    utility = row['utility_by_granularity'][str(row['predicted_granularity'])]
    close('Phase 4 selected utility: ' + row['question_id'], utility, row['selected_joined_retrieval_f1'])
    close('Phase 4 regret: ' + row['question_id'], max(row['utility_by_granularity'].values()) - utility, row['retrieval_regret'])

phases = [
    ('1', 'qwen_pretrained_zero_shot_router_evidence_length_oracle'),
    ('2', 'qwen_finetuned_router_evidence_length_oracle'),
    ('2B-A', 'qwen_phase2b_alias_unweighted_evidence_length_oracle'),
    ('2B-B', 'qwen_phase2b_alias_classbalanced_evidence_length_oracle'),
    ('2C', 'qwen_phase2c_sequence_classifier_evidence_length_oracle'),
    ('2D', 'qwen_phase2d_sequence_classifier_token_count_prompt_evidence_length_oracle'),
    ('2E', 'qwen_phase2e_lr_grid_token_count_prompt_5epochs_evidence_length_oracle/trials/lr5e-6'),
    ('3A', 'similarity_tree_phase3a_evidence_length_oracle'),
    ('3B', 'similarity_tree_phase3b_xgboost_evidence_length_oracle'),
    ('3C', 'qwen_phase3c_fusion_evidence_length_oracle'),
    ('3C-OOF', 'qwen_phase3c_oof_fusion_evidence_length_oracle'),
]
for phase, directory in phases:
    prefix = 'outputs/' + directory + '/'
    predictions = read(prefix + 'validation/predictions.jsonl')
    retrieval = read(prefix + 'retrieval/results.jsonl')
    population('Phase ' + phase + ' predictions', predictions, IDS)
    population('Phase ' + phase + ' retrieval', retrieval, IDS)
    values = classification('Phase ' + phase, predictions, read(prefix + 'classification/metrics.json'))
    summary = read(prefix + 'retrieval/summary.json')
    scores = [row['f1_joined_topk'] for row in retrieval]
    mean = summary.get('mean_joined_retrieval_f1', summary.get('valid_only_mean_joined_retrieval_f1'))
    median = summary.get('median_joined_retrieval_f1', summary.get('valid_only_median_joined_retrieval_f1'))
    close('Phase ' + phase + ': retrieval mean', statistics.mean(scores), mean)
    close('Phase ' + phase + ': retrieval median', statistics.median(scores), median)
    prediction_map = {row['question_id']: int(row.get('parsed_prediction', row.get('predicted_label'))) for row in predictions}
    discrepancies = [abs(row['f1_joined_topk'] - utilities[row['question_id']][str(prediction_map[row['question_id']])]) for row in retrieval]
    close('Phase ' + phase + ': every result matches its frozen action', max(discrepancies), 0)
    values.update(retrieval_f1=mean, retrieval_median=median, max_difference_from_saved_action=max(discrepancies))
    RESULTS[phase] = values
    alias = {'2D':'phase2d','3B':'phase3b','3C':'original_phase3c','3C-OOF':'phase3c_oof'}.get(phase)
    if alias:
        strategy_vectors[alias] = [utilities[row['question_id']][str(prediction_map[row['question_id']])] for row in phase4]

papers = [row['document_id'] for row in phase4]
for name, vector in strategy_vectors.items():
    metric = base['strategy_metrics'][name]
    close(name + ': mean against Phase 4 comparison', statistics.mean(vector), metric['mean_joined_retrieval_f1'])
    close(name + ': median against Phase 4 comparison', statistics.median(vector), metric['median_joined_retrieval_f1'])
    bootstrap(vector, papers, metric['mean_retrieval_f1_ci95'], name + ' mean interval')
for name, interval in base['paired_differences'].items():
    comparator = name.removeprefix('phase4_minus_')
    delta = [a-b for a,b in zip(strategy_vectors['phase4'], strategy_vectors[comparator])]
    bootstrap(delta, papers, interval, name)

early = read('reports/final_validation_comparison/strategy_metrics.json')
for name, pattern in [
    ('router_selected','outputs/router_selected/validation/RetrievalEval*.jsonl'),
    ('mixed_raw','outputs/mixed_granularity/validation/raw/RetrievalEval*.jsonl'),
    ('mixed_deduplicated','outputs/mixed_granularity/validation/deduplicated/RetrievalEval*.jsonl'),
]:
    paths = sorted(REPO.glob(pattern))
    check(name + ': one result file', len(paths) == 1)
    rows = read(paths[0])
    population(name, rows, IDS)
    scores = [row['f1_joined_topk'] for row in rows]
    RESULTS[name] = {'retrieval_f1':statistics.mean(scores),'retrieval_median':statistics.median(scores)}
    metric = early.get('strategy_metrics', early)[name]
    close(name + ': mean', statistics.mean(scores), metric.get('mean_f1', metric.get('mean_joined_retrieval_f1')))

local_root = 'outputs/similarity_tree_phase5a_local_gold_overlap_router/'
local = read(local_root + 'retrieval/results.jsonl')
population('Local protocols', local, IDS)
local_summary = read(local_root + 'retrieval/summary.json')
local_vectors = {}
for method, summary in local_summary['methods'].items():
    scores = [row['methods'][method]['f1_joined_top5_trees'] for row in local]
    local_vectors[method] = scores
    for saved_key, record_key in [('mean_joined_f1','f1_joined_top5_trees'), ('mean_joined_precision','precision_joined_top5_trees'), ('mean_joined_recall','recall_joined_top5_trees'), ('mean_retrieved_token_count','retrieved_token_count')]:
        close(method + ': ' + saved_key, statistics.mean(row['methods'][method][record_key] for row in local), summary[saved_key])
    close(method + ': median F1', statistics.median(scores), summary['median_joined_f1'])
    RESULTS[method] = {'retrieval_f1':statistics.mean(scores),'retrieval_median':statistics.median(scores)}
for name, interval in local_summary['paired_paper_cluster_bootstrap'].items():
    other = name.removeprefix('versus_')
    bootstrap([a-b for a,b in zip(local_vectors['phase5a'],local_vectors[other])], [r['document_id'] for r in local], interval, 'Phase 5A ' + name)

prefix = 'outputs/similarity_tree_phase5b_v2_zero_shot_qwen_context_feature/'
localb = read(prefix + 'retrieval/results.jsonl')
population('Phase 5B-v2', localb, IDS)
summary = read(prefix + 'retrieval/summary.json')
for saved_key, record_key in [('mean_joined_f1','f1_joined_top5_trees'), ('mean_joined_precision','precision_joined_top5_trees'), ('mean_joined_recall','recall_joined_top5_trees'), ('mean_retrieved_token_count','retrieved_token_count')]:
    close('Phase 5B-v2: ' + saved_key, statistics.mean(r[record_key] for r in localb), summary[saved_key])
RESULTS['phase5b_v2'] = {'retrieval_f1':statistics.mean(r['f1_joined_top5_trees'] for r in localb)}
local_map = {r['question_id']:r for r in local}
comparisons = read(prefix + 'comparison/phase5a_and_fixed40.json')
for name in ['versus_phase5a', 'versus_same_tree_fixed_40']:
    method = name.removeprefix('versus_')
    deltas = [r['f1_joined_top5_trees'] - local_map[r['question_id']]['methods'][method]['f1_joined_top5_trees'] for r in localb]
    bootstrap(deltas, [r['document_id'] for r in localb], comparisons[name], 'Phase 5B-v2 ' + name)

for directory, expected in [
    ('similarity_tree_phase5a_local_gold_overlap_router', [246,529,2823,965,57]),
    ('similarity_tree_phase5b_v2_zero_shot_qwen_context_feature', [247,506,2804,1008,55]),
]:
    predictions = read('outputs/' + directory + '/validation/predictions_pre_evaluation.jsonl')
    population(directory + ' locked predictions', predictions, IDS)
    check(directory + ': five trees per question', all(len(r['selected_trees']) == 5 for r in predictions))
    counts = Counter(tree['predicted_granularity'] for r in predictions for tree in r['selected_trees'])
    check(directory + ': local decision distribution', [counts[g] for g in SIZES] == expected)
    cm = read('outputs/' + directory + '/classification/metrics.json')
    matrix = cm['confusion_matrix']
    support = [sum(row) for row in matrix]
    predicted = [sum(row[i] for row in matrix) for i in range(5)]
    close(directory + ': diagnostic accuracy from confusion matrix',
          sum(matrix[i][i] for i in range(5)) / sum(support), cm['accuracy'])
    close(directory + ': diagnostic macro F1 from confusion matrix',
          statistics.mean(2*matrix[i][i]/(support[i]+predicted[i]) for i in range(5)), cm['macro_f1'])
    check(directory + ': diagnostic population', sum(support) == 2074
          and cm['eligible_questions'] == 889 and cm['excluded_questions'] == 35)

for version, root, expectations in [
    ('5B-v1', 'similarity_tree_phase5b_zero_shot_qwen_context_feature',
     {'train':(2101,175,[54,1867,5]), 'validation':(924,56,[31,836,1])}),
    ('5B-v2', 'similarity_tree_phase5b_v2_zero_shot_qwen_context_feature',
     {'train':(2101,0,[24,2076,1]), 'validation':(924,0,[12,912,0])}),
]:
    for split, (count, invalid, distribution) in expectations.items():
        rows = read(f'outputs/{root}/qwen_features/{split}_outputs.jsonl')
        check(version + ' ' + split + ': population', len(rows) == count
              and len({r['question_id'] for r in rows}) == count)
        check(version + ' ' + split + ': invalid outputs',
              sum(r['prediction_status'] != 'valid' for r in rows) == invalid)
        actual = Counter(r['parsed_context_label'] for r in rows if r['prediction_status'] == 'valid')
        check(version + ' ' + split + ': valid label distribution',
              [actual[label] for label in ['short','medium','long']] == distribution)
probe = read(prefix + 'development/training_only_prompt_repair_probe.jsonl')
check('Training-only repair probe: still fails full-validity gate',
      len(probe) == 175 and sum(r['prediction_status'] == 'valid' for r in probe) == 173)
check('Training-only repair probe: no validation questions',
      not ({r['question_id'] for r in probe} & IDS))

output = THESIS / 'build/qa/results_audit.json'
output.parent.mkdir(parents=True, exist_ok=True)
failures = [item for item in CHECKS if not item['passed']]
report = {'checks':len(CHECKS), 'failures':failures, 'sources':SOURCES, 'results':RESULTS,
          'scope':'Saved prediction/score reconciliation, split identities, source hashes, local decisions, context-output validity, and independently regenerated Phase 4/5A/5B intervals. No new model or retrieval runs.'}
output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'checks':len(CHECKS),'source_files':len(SOURCES),'failures':failures,'results':RESULTS},indent=2))
raise SystemExit(bool(failures))
