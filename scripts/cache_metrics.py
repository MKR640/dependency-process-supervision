"""Recompute historical metrics and labelled post-hoc diagnostics from caches.

No model loads, training, threshold fitting, or external requests. New analyses
use fixed 0.5 decisions and the manuscript's density strata. Bootstrap units
are normalized questions, with paired resampling across model/decoder outputs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np

ROLES = ['correct', 'new_error', 'propagated_error']
TYPES = ['calculation', 'derivation_or_logic', 'method_or_formula', 'problem_understanding',
         'assumption_or_condition', 'answer_mismatch', 'missing_evidence', 'unit']
MODELS = ['Binary_PRM', 'Three_class_v2', 'Full_hierarchy']
PB_MODELS = ['Qwen3-4B-base-PRM800K-final-fixed', 'Qwen3-4B-base-PRM800K-final-three-class-v2',
             'Qwen3-4B-hierarchical-PRM800K-phase2-from-3class-8gpu']
SUBSETS = ['gsm8k', 'math', 'olympiadbench', 'omnimath']


def read_rows(z, name):
    return [json.loads(line) for line in z.read(name).splitlines() if line.strip()]


def ap_plan(y, scores):
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(order)-1]
    def evaluate(weights=None):
        w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
        positive = np.sum(w*y)
        if positive <= 0 or np.sum(w*(1-y)) <= 0:
            return None
        tp = np.cumsum((w*y)[order])[ends]
        total = np.cumsum(w[order])[ends]
        hits = np.diff(np.r_[0., tp])
        precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
        return float(np.sum(hits * precision)/positive)
    return evaluate


def class_metrics(truth, guess, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (np.asarray(truth, dtype=int), np.asarray(guess, dtype=int)), 1)
    support, predicted = cm.sum(1), cm.sum(0)
    diag = np.diag(cm)
    recall = np.divide(diag, support, out=np.zeros(n), where=support > 0)
    f1 = np.divide(2*diag, support+predicted, out=np.zeros(n), where=support+predicted > 0)
    return cm, recall, f1


def summarize(ref, predictions):
    y = np.array([r['y_error'] for r in ref])
    score = np.array([p['p_error'] for p in predictions])
    cm, recall, f1 = class_metrics(y, score >= .5, 2)
    groups = defaultdict(list)
    for i, r in enumerate(ref):
        groups[r['trace_id']].append(i)
    ok, has_error = [], []
    for inds in groups.values():
        inds.sort(key=lambda i: ref[i]['step_index'])
        gold = next((ref[i]['step_index'] for i in inds if y[i]), -1)
        guess = next((ref[i]['step_index'] for i in inds if score[i] >= .5), -1)
        ok.append(gold == guess)
        has_error.append(gold != -1)
    ok, has_error = np.array(ok), np.array(has_error)
    result = {'error_ap': ap_plan(y, score)(), 'error_f1': float(f1[1]),
              'balanced_accuracy': float(recall.mean()), 'error_recall': float(recall[1]),
              'false_positive_rate': float(1-recall[0]),
              'first_error_accuracy': float(ok.mean()),
              'first_error_accuracy_error_traces': float(ok[has_error].mean()) if has_error.any() else None,
              'no_error_trace_accuracy': float(ok[~has_error].mean()) if (~has_error).any() else None}
    diagnostic = {}
    if predictions[0].get('role_probs') is not None:
        truth = np.array([ROLES.index(r['role']) for r in ref])
        probs = np.array([p['role_probs'] for p in predictions])
        rcm, rr, rf = class_metrics(truth, probs.argmax(1), 3)
        result.update(new_error_ap=ap_plan(truth == 1, probs[:, 1])(),
                      new_error_recall=float(rr[1]), role_macro_f1=float(rf.mean()))
        diagnostic['role'] = {'labels': ROLES, 'confusion_matrix': rcm.tolist(),
                              'recall': rr.tolist(), 'f1': rf.tolist()}
    type_field = 'type_probs' if predictions[0].get('type_probs') is not None else 'error_type_probs'
    if predictions[0].get(type_field) is not None:
        inds = [i for i, r in enumerate(ref) if r['error_type'] in TYPES]
        truth = np.array([TYPES.index(ref[i]['error_type']) for i in inds])
        guess = np.array([np.argmax(predictions[i][type_field]) for i in inds])
        tcm, tr, tf = class_metrics(truth, guess, 8)
        supported = tcm.sum(1) > 0
        result.update(type_macro_f1_supported=float(tf[supported].mean()), type_macro_f1_fixed8=float(tf.mean()))
        diagnostic['type'] = {'labels': TYPES, 'support': tcm.sum(1).tolist(), 'recall': tr.tolist(), 'f1': tf.tolist()}
    return result, diagnostic


def interval(values):
    a = [x for x in values if x is not None and np.isfinite(x)]
    return {'low': float(np.percentile(a, 2.5)), 'high': float(np.percentile(a, 97.5)),
            'valid_replicates': len(a)} if a else None


def pb_metrics(rows, guesses, weights=None):
    labels = np.array([r['label'] for r in rows])
    weights = np.ones(len(rows)) if weights is None else np.asarray(weights)
    correct = labels == -1
    hit = np.asarray(guesses) == labels
    if weights[correct].sum() == 0 or weights[~correct].sum() == 0:
        return None
    e = float(np.sum(weights[~correct]*hit[~correct])/weights[~correct].sum())
    c = float(np.sum(weights[correct]*hit[correct])/weights[correct].sum())
    return {'error_acc': e, 'correct_acc': c, 'f1': 2*e*c/(e+c) if e+c else 0,
            'overall_acc': float(np.sum(weights*hit)/weights.sum())}


def predict_first(scores):
    return next((i for i, value in enumerate(scores) if value < .5), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--materialized-math', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261102)
    args = parser.parse_args()
    checks, differences = {}, {}
    rng = np.random.default_rng(args.seed)
    result = {'checked_on': '2026-11-02', 'historical_recomputed': {}, 'posthoc_density': {},
              'posthoc_processbench_decoder': {}, 'mixed_error_diagnostic': {}}
    with zipfile.ZipFile(args.archive) as z, zipfile.ZipFile(args.records) as record_zip:
        manifest = json.loads(z.read('recovery_manifest.json'))
        for item in manifest['files']:
            data = z.read(item['name'])
            checks['archive:' + item['name']] = len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
        for scope, split in [('math_existing', 'test'), ('physics_fixed', 'all')]:
            ref = read_rows(z, f'{scope}/data/{split}.jsonl')
            meta = json.loads(record_zip.read(f'{scope}/data/{split}.jsonl.meta.json'))
            checks[scope + ':reference_hash'] = hashlib.sha256(z.read(f'{scope}/data/{split}.jsonl')).hexdigest() == meta['reference_sha256']
            keys = [(r['trace_id'], r['step_index']) for r in ref]
            checks[scope + ':unique_reference_keys'] = len(set(keys)) == len(keys)
            checks[scope + ':reference_label_consistency'] = all(
                r['y_error'] in (0, 1) and r['role'] in ROLES and
                bool(r['y_error']) == (r['role'] != 'correct') for r in ref)
            tr = defaultdict(list)
            for i, row in enumerate(ref):
                tr[row['trace_id']].append(i)
            density = {tid: sum(ref[i]['y_error'] for i in inds)/len(inds) for tid, inds in tr.items()}
            group = {tid: 'no_error' if d == 0 else 'sparse' if d <= .25 else 'medium' if d <= .5 else 'dense'
                     for tid, d in density.items()}
            all_predictions = {}
            stored = json.loads(record_zip.read('math_existing/results/metrics.json' if scope == 'math_existing'
                                                 else 'physics_fixed/overall_evaluation/metrics.json'))
            summary = {}
            for model in MODELS:
                name = f'{scope}/predictions/{model}.{split}.jsonl'
                raw_predictions = read_rows(z, name)
                mapping = {(p['trace_id'], p['step_index']): p for p in raw_predictions}
                checks[scope + ':' + model + ':keys'] = set(mapping) == set(keys) and len(mapping) == len(raw_predictions)
                predictions = [mapping[key] for key in keys]
                all_predictions[model] = predictions
                checks[scope + ':' + model + ':valid_probabilities'] = all(
                    np.isfinite(p['p_error']) and 0 <= p['p_error'] <= 1 and all(
                        len(p[field]) == dim and np.all(np.isfinite(p[field])) and
                        np.all(np.asarray(p[field]) >= 0) and np.all(np.asarray(p[field]) <= 1) and
                        abs(sum(p[field])-1) < 1e-5
                        for field, dim in [('role_probs', 3), ('type_probs', 8), ('error_type_probs', 8)]
                        if p.get(field) is not None) for p in predictions)
                predmeta = json.loads(record_zip.read(name + '.meta.json'))
                checks[scope + ':' + model + ':hash'] = hashlib.sha256(z.read(name)).hexdigest() == predmeta['predictions_sha256']
                metrics, diagnostics = summarize(ref, predictions)
                expected = stored['models'][model]['metrics'] if scope == 'math_existing' else {
                    k: v['value'] for row in stored['results'] if row['model'] == model for k, v in row['metrics'].items()}
                for key, val in expected.items():
                    if key in metrics and val is not None:
                        delta = abs(metrics[key] - val)
                        if delta > 1e-10:
                            differences[f'{scope}:{model}:{key}'] = [val, metrics[key]]
                summary[model] = {'metrics': metrics, 'diagnostics': diagnostics}
            hashes = [[p['input_sha256'] for p in all_predictions[m]] for m in MODELS]
            checks[scope + ':identical_model_input_hashes'] = hashes[0] == hashes[1] == hashes[2]
            result['historical_recomputed'][scope] = summary
            density_results = {}
            for label in ['sparse', 'medium', 'dense']:
                inds = [i for i, row in enumerate(ref) if group[row['trace_id']] == label]
                rr = [ref[i] for i in inds]
                problems = sorted({row['problem_id'] for row in rr})
                lookup = {p: i for i, p in enumerate(problems)}
                pi = np.array([lookup[row['problem_id']] for row in rr])
                y = np.array([row['y_error'] for row in rr])
                funcs = {m: ap_plan(y, [all_predictions[m][i]['p_error'] for i in inds])
                         for m in ['Three_class_v2', 'Full_hierarchy']}
                values = {m: f() for m, f in funcs.items()}
                samples = []
                for _ in range(args.bootstrap):
                    pw = np.bincount(rng.integers(len(problems), size=len(problems)), minlength=len(problems))
                    a, b = [funcs[m](pw[pi]) for m in ['Three_class_v2', 'Full_hierarchy']]
                    samples.append(b-a if a is not None and b is not None else None)
                density_results[label] = {
                    'problems': len(problems), 'traces': len({row['trace_id'] for row in rr}),
                    'steps': len(rr), 'error_steps': int(y.sum()), 'error_prevalence': float(y.mean()),
                    'error_ap': values, 'hierarchy_minus_three_class': values['Full_hierarchy']-values['Three_class_v2'],
                    'paired_problem_bootstrap_95ci': interval(samples)}
            result['posthoc_density'][scope] = density_results
            if scope == 'math_existing':
                source = [json.loads(l) for l in args.materialized_math.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
                semantic = {(r['id'], s['index']-1): s for r in source for s in r['steps']}
                checks['math:mixed_source_key_alignment'] = set(semantic) == set(keys)
                for label, select in [('mixed', lambda s: s['mixed_error']),
                                      ('pure_new', lambda s: s['role'] == 'new_error' and not s['mixed_error'])]:
                    inds = [i for i, key in enumerate(keys) if select(semantic[key])]
                    predictions = [int(np.argmax(all_predictions['Full_hierarchy'][i]['role_probs'])) for i in inds]
                    counts = np.bincount(predictions, minlength=3)
                    result['mixed_error_diagnostic'][label] = {
                        'steps': len(inds), 'predicted_roles': dict(zip(ROLES, counts.tolist())),
                        'new_error_recall': float(counts[1]/len(inds))}
        pb_stored = json.loads(record_zip.read('processbench/processbench_3models_detailed.json'))
        paired_bootstrap_inputs = {}
        result['historical_recomputed']['processbench'] = {}
        for model in PB_MODELS:
            result['historical_recomputed']['processbench'][model] = {}
            for subset in SUBSETS:
                rows = read_rows(z, f'processbench/{model}/{subset}/all_results.jsonl')
                checks[f'pb:{model}:{subset}:score_shapes_and_labels'] = all(
                    len(r['step_scores']) == len(r['steps']) and
                    (-1 <= r['label'] < len(r['steps'])) and
                    all(np.isfinite(s) and 0 <= s <= 1 for s in r['step_scores']) for r in rows)
                guesses = [predict_first(row['step_scores']) for row in rows]
                checks[f'pb:{model}:{subset}:predictions'] = guesses == [row['prediction'] for row in rows]
                checks[f'pb:{model}:{subset}:matches'] = all(row['match'] == (g == row['label']) for row, g in zip(rows, guesses))
                metrics = pb_metrics(rows, guesses)
                for k, val in pb_stored[model]['fixed'][subset].items():
                    if abs(metrics[k]*100-val) > 1e-9:
                        differences[f'pb:{model}:{subset}:{k}'] = [val, metrics[k]*100]
                result['historical_recomputed']['processbench'][model][subset] = metrics
                if model == PB_MODELS[-1]:
                    baseline = read_rows(z, f'processbench/{PB_MODELS[1]}/{subset}/all_results.jsonl')
                    binary = read_rows(z, f'processbench/{PB_MODELS[0]}/{subset}/all_results.jsonl')
                    fields = ['id', 'problem', 'steps', 'label']
                    checks[f'pb:{subset}:paired_reference_fields'] = len(rows) == len(baseline) == len(binary) and all(
                        all(r[k] == b[k] == c[k] for k in fields) for r, b, c in zip(rows, baseline, binary))
                    close = lambda a, b: abs(a-b) < 2e-7
                    checks[f'pb:{subset}:joint_score_matches_aux'] = all(
                        len(r['step_scores']) == len(r['steps']) == len(r['step_aux']) and all(
                            close(s, 1-a['p_joint_first_error']) for s, a in zip(r['step_scores'], r['step_aux'])) for r in rows)
                    checks[f'pb:{subset}:joint_formula_matches_aux'] = all(all(
                        close(a['p_joint_first_error'], a['p_error']*a['p_new_error']/max(a['p_new_error']+a['p_propagated_error'], 1e-8))
                        for a in r['step_aux']) for r in rows)
                    checks[f'pb:{subset}:correctness_probability_complement'] = all(all(
                        close(a['p_error'] + a['p_non_error'], 1.) for a in r['step_aux']) for r in rows)
                    corr = [predict_first([a['p_non_error'] for a in row['step_aux']]) for row in rows]
                    metrics_corr = pb_metrics(rows, corr)
                    problems = [hashlib.sha256(re.sub(r'\s+', ' ', row['problem']).strip().encode()).hexdigest() for row in rows]
                    unique = sorted(set(problems)); lut = {p: i for i, p in enumerate(unique)}
                    pi = np.array([lut[p] for p in problems])
                    boot = []
                    for _ in range(args.bootstrap):
                        weights = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))[pi]
                        a, b = pb_metrics(rows, guesses, weights), pb_metrics(rows, corr, weights)
                        boot.append(a['f1']-b['f1'] if a and b else None)
                    paired_bootstrap_inputs[subset] = boot
                    result['posthoc_processbench_decoder'][subset] = {
                        'traces': len(rows), 'normalized_questions': len(unique),
                        'correctness_only': metrics_corr, 'joint_first_error': metrics,
                        'joint_minus_correctness_f1': metrics['f1']-metrics_corr['f1'],
                        'paired_problem_bootstrap_95ci': interval(boot),
                        'changed_predictions': sum(a != b for a, b in zip(guesses, corr))}
        avg_joint = sum(v['joint_first_error']['f1'] for v in result['posthoc_processbench_decoder'].values())/4
        avg_corr = sum(v['correctness_only']['f1'] for v in result['posthoc_processbench_decoder'].values())/4
        means = [float(np.mean([paired_bootstrap_inputs[s][i] for s in SUBSETS])) for i in range(args.bootstrap)
                 if all(paired_bootstrap_inputs[s][i] is not None for s in SUBSETS)]
        result['posthoc_processbench_decoder']['average'] = {
            'joint_first_error_f1': avg_joint, 'correctness_only_f1': avg_corr,
            'joint_minus_correctness_f1': avg_joint-avg_corr,
            'paired_stratified_problem_bootstrap_95ci': interval(means)}
    result.update({
        'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        'checks_total': len(checks), 'checks_failed': {k: v for k, v in checks.items() if not v},
        'historical_metric_mismatches': differences,
        'code_entry_candidates_reported': manifest.get('code_entry_candidates', []),
        'posthoc_protocol': {'seed': args.seed, 'replicates': args.bootstrap, 'interval': 'paired percentile 95%',
                             'density_unit': 'Normalized problem ID, sampled within each density stratum',
                             'processbench_unit': 'Whitespace-normalized question text, resampled within each subset; equal-weight mean of four subsets',
                             'threshold': .5, 'training_runs': 'Frozen single historical model per condition; no training-seed uncertainty',
                             'selection': 'Density cuts and two decoder definitions fixed from existing manuscript/reviewer questions; no threshold fitting',
                             'multiplicity': 'Exploratory unadjusted intervals'},
        'scope': 'Historical metrics reproduced from raw caches; density/decoder/mixed slices are new retrospective analyses, not new training or independent attribution validation.',
        'role_interpretation': 'Mixed slice reports reference-new recall, not independent annotation accuracy or repair success.',
    })
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['checks_total', 'checks_failed', 'historical_metric_mismatches',
                                          'posthoc_density', 'posthoc_processbench_decoder', 'mixed_error_diagnostic']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
