"""Recompute Table 2 from text-free frozen probabilities; no model inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from metrics import metrics, threshold
from role_metrics import sequential_roles, confusion, macro_from_cm, score

ROOT = Path(__file__).resolve().parent
ARMS = ['M0', 'M1', 'M3', 'M4', 'M5']
NAMES = {'M0': 'First-error mask', 'M1': 'BFE-style binary',
         'M3': 'Correctness + type', 'M4': 'Correctness + role', 'M5': 'Full hierarchy'}
CONTRASTS = {'type_added_to_binary': {'M3': 1, 'M1': -1},
             'role_added_to_binary': {'M4': 1, 'M1': -1},
             'both_added_to_binary': {'M5': 1, 'M1': -1},
             'role_added_given_type': {'M5': 1, 'M3': -1},
             'type_added_given_role': {'M5': 1, 'M4': -1},
             'all_steps_vs_first_error': {'M1': 1, 'M0': -1},
             'role_type_interaction': {'M5': 1, 'M4': -1, 'M3': -1, 'M1': 1}}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files():
    manifest = read(ROOT / 'manifest.json')
    for name, want in manifest['files'].items():
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or digest(p) != want:
            raise ValueError('Package file missing or changed: ' + name)


def load(split):
    common = {'trace', 'step', 'correctness', 'role'}
    expected = common | {a + '_p_error' for a in ARMS}
    if split == 'test':
        expected |= {'detection_group', 'role_group', 'type_target',
                     'M3_type_probabilities', 'M5_type_probabilities',
                     'M4_role_probabilities', 'M5_role_probabilities'}
    with np.load(ROOT / 'data' / (split + '.npz'), allow_pickle=False) as z:
        if set(z.files) != expected:
            raise ValueError('Unexpected data fields: ' + split)
        data = {key: z[key] for key in z.files}
    n = len(data['trace'])
    for key, value in data.items():
        if value.dtype.kind not in 'iu f'.replace(' ', '') or len(value) != n or not np.isfinite(value).all():
            raise ValueError('Non-numeric, malformed or nonfinite data: ' + key)
        if key.endswith('_p_error') or key.endswith('_probabilities'):
            if (value < 0).any() or (value > 1).any():
                raise ValueError('Invalid probability: ' + key)
        if key.endswith('_probabilities'):
            width = 3 if '_role_' in key else 8
            if value.shape != (n, width) or not np.allclose(value.sum(1), 1, atol=1e-5):
                raise ValueError('Invalid probability vector: ' + key)
    if set(data['correctness']) != {0, 1} or not set(data['role']) <= {0, 1, 2}:
        raise ValueError('Invalid reference labels')
    if not np.array_equal(np.lexsort((data['step'], data['trace'])), np.arange(n)):
        raise ValueError('Rows must be ordered by trace then step')
    for tr in np.unique(data['trace']):
        sel = data['trace'] == tr
        if not np.array_equal(data['step'][sel], np.arange(sel.sum())):
            raise ValueError('Incomplete trajectory')
        for group in ['detection_group', 'role_group']:
            if group in data and len(np.unique(data[group][sel])) != 1:
                raise ValueError('Inconsistent trajectory group')
    if split == 'test':
        if not set(data['type_target']) <= set(range(8)) | {-100}:
            raise ValueError('Invalid type target')
        if np.any((data['type_target'] >= 0) & (data['role'] != 1)):
            raise ValueError('Type target outside new-error steps')
        for name in ['detection_group', 'role_group']:
            if not np.array_equal(np.unique(data[name]), np.arange(data[name].max() + 1)):
                raise ValueError('Non-contiguous group codes')
    return data


def head_metrics(target, probabilities, classes):
    mask = target >= 0
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (target[mask], probabilities[mask].argmax(1)), 1)
    support = cm.sum(1)
    denominator = support + cm.sum(0)
    f1 = np.divide(2 * np.diag(cm), denominator, out=np.zeros(classes), where=denominator > 0)
    return {'confusion_matrix': cm.tolist(), 'support': support.tolist(),
            'f1_per_class': f1.tolist(), 'macro_f1_fixed_taxonomy': float(f1.mean()),
            'macro_f1_supported': float(f1[support > 0].mean())}


def detection(validation, test):
    y = 1 - test['correctness']
    gi = test['detection_group']
    groups = int(gi.max()) + 1
    traces = [np.flatnonzero(test['trace'] == tr) for tr in np.unique(test['trace'])]
    models, evaluators = {}, {}
    for arm in ARMS:
        t = threshold(1 - validation['correctness'], validation[arm + '_p_error'])
        scores = test[arm + '_p_error']
        first_ok, trace_group, has_error = [], [], []
        for inds in traces:
            gold = next((i for i in inds if y[i]), None)
            guess = next((i for i in inds if scores[i] >= t), None)
            first_ok.append(gold == guess)
            trace_group.append(gi[inds[0]])
            has_error.append(gold is not None)
        tg, ok, has = np.array(trace_group), np.array(first_ok), np.array(has_error)

        def evaluate(w, scores=scores, t=t, tg=tg, ok=ok, has=has):
            out = metrics(y, scores, t, w[gi])
            tw = w[tg]
            out['first_error_accuracy'] = float(np.average(ok, weights=tw))
            out['first_error_accuracy_error_traces'] = float(np.average(ok[has], weights=tw[has])) if tw[has].sum() else None
            return out

        evaluators[arm] = evaluate
        heads = {}
        if arm in ['M4', 'M5']:
            heads['role'] = head_metrics(test['role'], test[arm + '_role_probabilities'], 3)
        if arm in ['M3', 'M5']:
            heads['type'] = head_metrics(test['type_target'], test[arm + '_type_probabilities'], 8)
        models[arm] = {'threshold': t, 'metrics': evaluate(np.ones(groups)), 'heads': heads}
    draws = {arm: {} for arm in ARMS}
    contrasts = {name: {} for name in CONTRASTS}
    rng = np.random.default_rng(20260923)
    for b in range(2000):
        w = rng.multinomial(groups, np.full(groups, 1 / groups))
        current = {arm: evaluate(w) for arm, evaluate in evaluators.items()}
        for arm, values in current.items():
            for key, value in values.items():
                if value is not None:
                    draws[arm].setdefault(key, []).append(value)
        for name, terms in CONTRASTS.items():
            for key in current['M0']:
                if all(current[arm][key] is not None for arm in terms):
                    contrasts[name].setdefault(key, []).append(sum(c * current[arm][key] for arm, c in terms.items()))
        if (b + 1) % 500 == 0:
            print(f'Detection bootstrap {b + 1}/2000', flush=True)
    def ci(xs):
        return {'low': float(np.quantile(xs, .025)), 'high': float(np.quantile(xs, .975)), 'valid_replicates': len(xs)}
    for arm in ARMS:
        models[arm]['ci95'] = {key: ci(xs) for key, xs in draws[arm].items()}
    contrasts = {name: {key: {'delta': sum(c * models[arm]['metrics'][key] for arm, c in CONTRASTS[name].items()),
                                     'ci95': ci(xs)} for key, xs in fields.items()} for name, fields in contrasts.items()}
    return {'models': models, 'contrasts': contrasts}


def roles(validation, test, binary_detection_threshold):
    def rows(data):
        return [{'trace_id': int(t), 'p_error': float(p)} for t, p in zip(data['trace'], data['M1_p_error'])]
    vr, tr = rows(validation), rows(test)
    truth = test['role'].tolist()
    result = {}
    for name, t in [('fixed_0.5', .5), ('existing_validation_balanced_accuracy', binary_detection_threshold)]:
        result[name] = {'threshold': t, **score(truth, sequential_roles(tr, t).tolist())}
    hp = test['M5_role_probabilities'].argmax(1)
    result['hierarchical_role_argmax'] = score(truth, hp.tolist())
    grid = np.linspace(0, 1, 1001)
    values = np.array([macro_from_cm(confusion(validation['role'], sequential_roles(vr, t))) for t in grid])
    index = min(np.flatnonzero(values == values.max()), key=lambda i: (abs(grid[i] - .5), grid[i]))
    t = float(grid[index])
    bp = sequential_roles(tr, t)
    result['validation_role_macro_f1_calibrated'] = {'threshold': t, 'validation_role_macro_f1': float(values[index]), **score(truth, bp.tolist())}
    groups = int(test['role_group'].max()) + 1
    cmh = np.zeros((groups, 3, 3), dtype=np.int64)
    cmb = cmh.copy()
    for g, y, a, b in zip(test['role_group'], test['role'], hp, bp):
        cmh[g, y, a] += 1
        cmb[g, y, b] += 1
    rng = np.random.default_rng(20260925)
    weights = rng.multinomial(groups, np.full(groups, 1 / groups), size=2000)
    hm, bm = np.einsum('rg,gij->rij', weights, cmh), np.einsum('rg,gij->rij', weights, cmb)
    delta = 100 * (macro_from_cm(hm) - macro_from_cm(bm))
    result['hierarchy_minus_role_calibrated_heuristic'] = {
        'macro_f1_delta_pp': 100 * (float(macro_from_cm(cmh.sum(0))) - float(macro_from_cm(cmb.sum(0)))),
        'paired_ci95_pp': np.quantile(delta, [.025, .975]).tolist(), 'bootstrap_replicates': 2000,
        'bootstrap_unit': 'normalized problem ID', 'groups': groups}
    return result


def compare(actual, expected, prefix=''):
    """Expected values validate results only; they never generate predictions or metrics."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise ValueError('Missing result ' + prefix + '/' + key)
            compare(actual[key], value, prefix + '/' + key)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Length mismatch ' + prefix)
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, prefix + '/' + str(i))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isfinite(actual) or not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError(f'Historical mismatch {prefix}: {actual} != {expected}')
    elif actual != expected:
        raise ValueError('Historical mismatch ' + prefix)


def write_tables(out, result):
    keys = ['error_ap', 'error_f1', 'balanced_accuracy', 'first_error_accuracy', 'first_error_accuracy_error_traces']
    headers = ['Supervision', 'Error AP', 'Error F1', 'Balanced accuracy', 'FE (all)', 'FE (err.)', 'Role F1', 'Type F1']
    rows = []
    for arm, model in result['table2']['models'].items():
        heads = model['heads']
        rows.append([NAMES[arm]] + [100 * model['metrics'][k] for k in keys] +
                    [100 * heads['role']['macro_f1_fixed_taxonomy'] if 'role' in heads else None,
                     100 * heads['type']['macro_f1_supported'] if 'type' in heads else None])
    r = result['roles']
    role_rows = []
    for name, field in [('BFE-style binary + sequential rule', 'validation_role_macro_f1_calibrated'),
                        ('Hierarchical role head', 'hierarchical_role_argmax')]:
        v = r[field]
        role_rows.append([name, 100*v['macro_f1'], 100*v['classes'][1]['f1'], 100*v['classes'][2]['f1']])
    role_headers = ['Predictor', 'Role macro-F1', 'New F1', 'Propagated F1']
    lines = ['# Table 2: shared-label detection and role diagnosis', '', 'Percentages; computed from bundled frozen predictions.', '']
    def table(title, columns, values, file):
        lines.extend([title, '', '| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---']*len(columns)) + '|'])
        for row in values:
            lines.append('| ' + ' | '.join('N/A' if v is None else f'{v:.2f}' if isinstance(v, float) else v for v in row) + ' |')
        lines.append('')
        with (out/file).open('w', encoding='utf-8', newline='') as f:
            w=csv.writer(f); w.writerow(columns); w.writerows(values)
    table('## (a) Supervision objectives', headers, rows, 'table2a.csv')
    table('## (b) Role diagnosis', role_headers, role_rows, 'table2b.csv')
    d=r['hierarchy_minus_role_calibrated_heuristic']
    lines.append(f"Role macro-F1 difference: {d['macro_f1_delta_pp']:+.2f} points; paired 95% CI [{d['paired_ci95_pp'][0]:.2f}, {d['paired_ci95_pp'][1]:.2f}].")
    lines.extend(['', 'Type F1 averages the six reference-supported types; role F1 averages all three classes.',
                  'Shared dependency-derived labels are fixed across conditions. Intervals reflect test-problem sampling, not retraining variation.',
                  '', 'All checked values match the archived reference within absolute tolerance 1e-12.'])
    (out/'table2.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (out/'contrasts.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f);w.writerow(['contrast','metric','difference_pp','ci_low_pp','ci_high_pp'])
        for name, fields in result['table2']['contrasts'].items():
            for key, value in fields.items():
                w.writerow([name,key,100*value['delta'],100*value['ci95']['low'],100*value['ci95']['high']])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'results', help='New output directory (default: results next to this script)')
    a=p.parse_args()
    if a.output.exists():
        p.error('Output already exists; choose a new directory with --output')
    verify_files()
    validation, test = load('validation'), load('test')
    if set(validation['trace']) & set(test['trace']):
        raise ValueError('Validation/test trace overlap')
    print('Package hashes verified. Recomputing five shared-label conditions...', flush=True)
    table2 = detection(validation, test)
    role_result = roles(validation, test, table2['models']['M1']['threshold'])
    result={'table2':table2,'roles':role_result}
    compare(result, read(ROOT/'expected.json'))
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (a.output/'verification.json').write_text(json.dumps({'status':'passed','absolute_tolerance':1e-12,
         'python':sys.version,'numpy':np.__version__,'data_sha256':{s:digest(ROOT/'data'/(s+'.npz')) for s in ['validation','test']},
         'scope':'Frozen prediction metric replay; no annotation inference, model inference or training.'},indent=2)+'\n',encoding='utf-8')
    write_tables(a.output, result)
    print((a.output/'table2.md').read_text(encoding='utf-8'))
    print('Verified results written to:', a.output.resolve())


if __name__ == '__main__':
    main()
