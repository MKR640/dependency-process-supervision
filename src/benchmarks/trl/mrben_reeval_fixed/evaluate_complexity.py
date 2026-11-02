#!/usr/bin/env python3
"""Offline, paired problem-cluster evaluation of complete PRM trajectories.
Requires numpy only. Compatible with merged Table-2 prediction caches.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from reference_io import (ROLES, file_hash, key, read_jsonl, validate_rows,
                          problem_key, problem_text, write_json)

METRICS = ['error_ap', 'error_f1', 'balanced_accuracy', 'error_recall',
           'false_positive_rate', 'first_error_accuracy',
           'first_error_accuracy_error_traces', 'no_error_trace_accuracy',
           'new_error_ap', 'new_error_recall', 'propagated_error_recall', 'role_macro_f1']
TRUNC = ('problem_tokens_dropped', 'current_tokens_dropped', 'prefix_steps_dropped')


def average_precision(y, score, weights):
    """Non-interpolated AP, simultaneous handling of equal scores."""
    order = np.argsort(-np.asarray(score), kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(np.asarray(score)[order])), len(order)-1]
    return sorted_ap(np.asarray(y)[order], np.asarray(weights)[order], ends)


def sorted_ap(y, w, ends):
    pos, neg = np.sum(w*y), np.sum(w*(1-y))
    if pos <= 0 or neg <= 0:
        return None  # AP intentionally undefined for single-class subgroups.
    tp = np.cumsum(w*y)[ends]
    total = np.cumsum(w)[ends]
    precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
    return float(np.sum(np.diff(np.r_[0., tp]) * precision) / pos)


def mean_or_none(x, w):
    return float(np.dot(x, w)/w.sum()) if w.sum() > 0 else None


def train_boundaries(path):
    """Stream flat training rows; retain only IDs, step-index sets and problem IDs."""
    traces, problems = {}, set()
    for r in read_jsonl(path):
        tid = str(r.get('trace_id') or r.get('id') or '')
        if not tid:
            raise ValueError('Training row lacks trace_id/id')
        pid = problem_key(problem_text(r))
        problems.add(pid)
        if 'steps' in r:
            if tid in traces or not r['steps']:
                raise ValueError('Duplicate/empty training trajectory: '+tid)
            traces[tid] = (pid, set(range(len(r['steps']))))
        else:
            i = r.get('step_index')
            if type(i) is not int or i < 0:
                raise ValueError('Training flat rows require nonnegative integer step_index')
            if tid not in traces:
                traces[tid] = (pid, set())
            oldpid, indices = traces[tid]
            if oldpid != pid or i in indices:
                raise ValueError('Duplicate step or multiple problems for training trace '+tid)
            indices.add(i)
    lengths = []
    for tid, (_, inds) in traces.items():
        if inds != set(range(len(inds))):
            raise ValueError('Incomplete training trajectory '+tid)
        lengths.append(len(inds))
    if not lengths:
        raise ValueError('Empty training file')
    # Inverse empirical CDF, compatible with all supported NumPy versions.
    seq = sorted(lengths)
    cuts = [seq[max(0, int(np.ceil(len(seq)*q))-1)] for q in (1/3, 2/3)]
    return cuts, problems, dict(path=str(Path(path).resolve()), sha256=file_hash(path),
        traces=len(lengths), problems=len(problems), length_cuts=cuts,
        quantile_rule='inverse empirical CDF at 1/3 and 2/3; ties kept together')


def load_reference(path):
    rows = list(read_jsonl(path))
    validate_rows(rows)
    meta = json.loads(Path(str(path)+'.meta.json').read_text())
    if meta.get('split') != 'test' or meta.get('reference_sha256') != file_hash(path):
        raise ValueError('Expected unchanged Table-2 test reference and .meta.json')
    return rows, meta


def load_prediction(path, rows, ref_meta, name):
    meta = json.loads(Path(str(path)+'.meta.json').read_text())
    if (meta.get('name') != name or meta.get('split') != ref_meta['split'] or
        meta.get('reference_sha256') != ref_meta['reference_sha256'] or
        meta.get('predictions_sha256') != file_hash(path)):
        raise ValueError(name+': prediction metadata/hash mismatch')
    mapped = {}
    for r in read_jsonl(path):
        if key(r) in mapped:
            raise ValueError(name+': duplicate prediction')
        mapped[key(r)] = r
    if set(mapped) != {key(r) for r in rows}:
        raise ValueError(name+': missing or extra prediction steps')
    ordered = [mapped[key(r)] for r in rows]
    p = np.asarray([r['p_error'] for r in ordered], dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(name+': invalid p_error')
    hashes = [r.get('input_sha256') for r in ordered]
    if any(not isinstance(h, str) or len(h) != 64 for h in hashes):
        raise ValueError(name+': missing input hashes')
    trunc = []
    for r in ordered:
        d = r.get('truncation', {})
        if any(type(d.get(k)) is not int or d[k] < 0 for k in TRUNC):
            raise ValueError(name+': missing/invalid truncation counters')
        trunc.append([d[k] for k in TRUNC])
    rp = None
    if 'role' in meta.get('active_heads', []):
        if (meta.get('schema') != 'mrben_three_model_prediction_v3_strict_role3'
                or not meta.get('loading_audit', {}).get('verified')):
            raise ValueError(name+': stale/unverified role cache; rerun corrected inference')
        if meta.get('role_order') != ROLES:
            raise ValueError(name+': unsupported role order')
        rp = np.asarray([r.get('role_probs') for r in ordered], dtype=float)
        if (rp.shape != (len(rows), 3) or not np.isfinite(rp).all() or
            ((rp < 0) | (rp > 1)).any() or not np.allclose(rp.sum(1), 1., atol=1e-5)):
            raise ValueError(name+': invalid role probabilities')
    elif any(r.get('role_probs') is not None for r in ordered):
        raise ValueError(name+': role probabilities present for inactive head')
    return dict(p=p, role=rp, hashes=hashes, trunc=np.array(trunc), meta=meta,
                path=str(Path(path).resolve()))


def trace_features(rows, predictions):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['trace_id']].append(i)
    traces = []
    # A common subset across all models: never give each model a different denominator.
    any_trunc = np.logical_or.reduce([p['trunc'].any(1) for p in predictions.values()])
    for tid, inds in sorted(groups.items()):
        inds.sort(key=lambda i: rows[i]['step_index'])
        ys = [rows[i]['y_error'] for i in inds]
        n, e = len(inds), sum(ys)
        first = next((j for j, y in enumerate(ys) if y), None)
        traces.append(dict(trace_id=tid, problem_id=rows[inds[0]]['problem_id'],
            inds=inds, length=n, errors=e, density=e/n, first=first,
            relative_first=(first+1)/n if first is not None else None,
            truncated=bool(any_trunc[inds].any())))
    return traces


def strata(traces, cuts):
    lo, hi = cuts
    specs = [('overall', 'all', lambda t: True),
        ('length', 'short', lambda t: t['length'] <= lo),
        ('length', 'medium', lambda t: lo < t['length'] <= hi),
        ('length', 'long', lambda t: t['length'] > hi),
        ('error_density', 'no_error', lambda t: t['errors'] == 0),
        ('error_density', 'sparse', lambda t: 0 < t['density'] <= .25),
        ('error_density', 'medium', lambda t: .25 < t['density'] <= .5),
        ('error_density', 'dense', lambda t: t['density'] > .5),
        ('first_error_position', 'no_error', lambda t: t['first'] is None),
        ('first_error_position', 'early', lambda t: t['first'] is not None and t['relative_first'] <= 1/3),
        ('first_error_position', 'middle', lambda t: t['first'] is not None and 1/3 < t['relative_first'] <= 2/3),
        ('first_error_position', 'late', lambda t: t['first'] is not None and t['relative_first'] > 2/3),
        ('context', 'untruncated', lambda t: not t['truncated']),
        ('context', 'truncated', lambda t: t['truncated'])]
    for name, cond in [('short', lambda t: t['length'] <= lo),
                       ('medium', lambda t: lo < t['length'] <= hi),
                       ('long', lambda t: t['length'] > hi)]:
        specs.append(('length_untruncated', name, lambda t, c=cond: not t['truncated'] and c(t)))
    return [(axis, name, [t for t in traces if predicate(t)]) for axis, name, predicate in specs]


class Metrics:
    def __init__(self, rows, traces, pred, threshold, problem_index):
        self.traces = traces
        inds = np.array([i for t in traces for i in t['inds']], dtype=int)
        self.y = np.array([rows[i]['y_error'] for i in inds], dtype=int)
        self.p = pred['p'][inds]
        self.hat = self.p >= threshold
        self.groups = np.array([problem_index[rows[i]['problem_id']] for i in inds], dtype=int)
        self.tgroups = np.array([problem_index[t['problem_id']] for t in traces], dtype=int)
        self.terr = np.array([t['errors'] > 0 for t in traces], dtype=bool)
        self.first_ok = np.array([next((j for j, i in enumerate(t['inds'])
            if pred['p'][i] >= threshold), None) == t['first'] for t in traces], dtype=bool)
        self.order = np.argsort(-self.p, kind='stable')
        self.ends = np.r_[np.flatnonzero(np.diff(self.p[self.order])), len(inds)-1]
        role_ids = np.array([ROLES.index(rows[i]['role']) if rows[i]['role'] in ROLES else -1 for i in inds])
        self.role_mask = role_ids >= 0
        self.roles = role_ids[self.role_mask]
        self.rp = pred['role'][inds][self.role_mask] if pred['role'] is not None else None
        if self.rp is not None and len(self.rp):
            self.rhat = self.rp.argmax(1)
            self.rorder = np.argsort(-self.rp[:, 1], kind='stable')
            self.rends = np.r_[np.flatnonzero(np.diff(self.rp[self.rorder, 1])), len(self.rp)-1]

    def compute(self, weights):
        if not self.traces:
            return {k: None for k in METRICS}
        w, tw = weights[self.groups], weights[self.tgroups]
        if w.sum() == 0:
            return {k: None for k in METRICS}
        cm = np.bincount(self.y*2+self.hat, weights=w, minlength=4).reshape(2, 2)
        tn, fp, fn, tp = cm.ravel()
        pos, neg = tp+fn, tn+fp
        rec = tp/pos if pos else None
        fpr = fp/neg if neg else None
        result = dict(error_ap=sorted_ap(self.y[self.order], w[self.order], self.ends),
            error_f1=float(2*tp/(2*tp+fp+fn)) if pos else None,
            balanced_accuracy=float((rec+1-fpr)/2) if pos and neg else None,
            error_recall=float(rec) if pos else None, false_positive_rate=float(fpr) if neg else None,
            first_error_accuracy=mean_or_none(self.first_ok, tw),
            first_error_accuracy_error_traces=mean_or_none(self.first_ok[self.terr], tw[self.terr]),
            no_error_trace_accuracy=mean_or_none(self.first_ok[~self.terr], tw[~self.terr]),
            new_error_ap=None, new_error_recall=None, propagated_error_recall=None, role_macro_f1=None)
        rw = w[self.role_mask]
        if self.rp is not None and rw.sum() > 0:
            rc = np.bincount(self.roles*3+self.rhat, weights=rw, minlength=9).reshape(3, 3)
            support, tp = rc.sum(1), np.diag(rc)
            den = support + rc.sum(0)
            rf1 = np.divide(2*tp, den, out=np.zeros(3), where=den > 0)
            result.update(new_error_ap=sorted_ap((self.roles[self.rorder] == 1).astype(int), rw[self.rorder], self.rends),
                new_error_recall=float(tp[1]/support[1]) if support[1] else None,
                propagated_error_recall=float(tp[2]/support[2]) if support[2] else None,
                role_macro_f1=float(rf1.mean()))
        return result


def ci(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    return dict(low=float(np.percentile(v, 2.5)) if len(v) else None,
                high=float(np.percentile(v, 97.5)) if len(v) else None, valid_replicates=len(v))


def difference(x, y):
    return x-y if x is not None and y is not None else None


def fmt(v):
    return 'N/A' if v is None else f'{100*v:.2f}'


def run(a):
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is nonempty. Choose a new OUTPUT_DIR; inputs will not be overwritten.')
    if not 0 <= a.threshold <= 1 or a.bootstrap_samples < 0:
        raise ValueError('Invalid threshold or bootstrap count')
    rows, ref = load_reference(a.reference)
    cuts, train_pids, trainmeta = train_boundaries(a.train_file)
    pids = sorted({r['problem_id'] for r in rows})
    overlap = set(pids) & train_pids
    if overlap and not getattr(a, 'skip_train_test_overlap_check', False):
        raise ValueError(f'Train/test overlap: {len(overlap)} problems; repair split first')
    if getattr(a, 'skip_train_test_overlap_check', False):
        print(
            'WARNING: train/test problem-overlap check is explicitly DISABLED '
            f'by --skip-train-test-overlap-check. Computed overlap count={len(overlap)} '
            'is not used to block evaluation.',
            flush=True,
        )
    pi = {p: i for i, p in enumerate(pids)}
    preds = {}
    for spec in a.prediction:
        name, sep, path = spec.partition('=')
        if not sep or not name or not path or name in preds:
            raise ValueError('--prediction must be unique NAME=PATH')
        preds[name] = load_prediction(path, rows, ref, name)
    if a.baseline not in preds or len(preds) < 2:
        raise ValueError('At least two models and an existing baseline required')
    base = preds[a.baseline]
    for n, p in preds.items():
        if (p['hashes'] != base['hashes'] or p['meta']['packing'] != base['meta']['packing'] or
            p['meta']['tokenizer_sha256'] != base['meta']['tokenizer_sha256']):
            raise ValueError(n+': input tokens/tokenizer/packing differ across models')
    traces = trace_features(rows, preds)
    groups = strata(traces, cuts)
    evaluators, estimates, samples, counts = {}, {}, {}, {}
    for axis, group, ts in groups:
        g = axis+'/'+group
        ii = [i for t in ts for i in t['inds']]
        counts[g] = dict(problems=len({t['problem_id'] for t in ts}), traces=len(ts), steps=len(ii),
            error_steps=sum(t['errors'] for t in ts), error_traces=sum(t['errors'] > 0 for t in ts),
            correct_traces=sum(t['errors'] == 0 for t in ts),
            new_error_steps=sum(rows[i]['role'] == 'new_error' for i in ii),
            propagated_error_steps=sum(rows[i]['role'] == 'propagated_error' for i in ii),
            truncated_traces=sum(t['truncated'] for t in ts),
            low_support=len({t['problem_id'] for t in ts}) < a.min_problems)
        for n, p in preds.items():
            k = (g, n)
            evaluators[k] = Metrics(rows, ts, p, a.threshold, pi)
            estimates[k] = evaluators[k].compute(np.ones(len(pids)))
            samples[k] = {m: [] for m in METRICS}
    rng = np.random.default_rng(a.seed)
    for b in range(a.bootstrap_samples):
        # ONE shared multiplicity vector per replicate for every model and subgroup.
        w = np.bincount(rng.integers(0, len(pids), size=len(pids)), minlength=len(pids))
        for k, ev in evaluators.items():
            metrics = ev.compute(w)
            for m in METRICS:
                samples[k][m].append(metrics[m])
        if (b+1) % 100 == 0 or b+1 == a.bootstrap_samples:
            print(f'bootstrap {b+1}/{a.bootstrap_samples}', flush=True)
    records = []
    for g, n in evaluators:
        bk = (g, a.baseline)
        metrics = {}
        for m in METRICS:
            value = estimates[(g, n)][m]
            delta_samples = [difference(x, y) for x, y in zip(samples[(g, n)][m], samples[bk][m])]
            metrics[m] = dict(value=value, ci=ci(samples[(g, n)][m]),
                delta_vs_baseline=difference(value, estimates[bk][m]), paired_delta_ci=ci(delta_samples))
        axis, group = g.split('/')
        records.append(dict(axis=axis, group=group, model=n, counts=counts[g], metrics=metrics))
    interactions = []
    for axis in ('length', 'length_untruncated'):
        for n in preds:
            if n == a.baseline:
                continue
            for m in METRICS:
                l, s, lb, sb = (axis+'/long', n), (axis+'/short', n), (axis+'/long', a.baseline), (axis+'/short', a.baseline)
                val = difference(difference(estimates[l][m], estimates[lb][m]), difference(estimates[s][m], estimates[sb][m]))
                draws = [difference(difference(x, y), difference(z, q)) for x, y, z, q in
                         zip(samples[l][m], samples[lb][m], samples[s][m], samples[sb][m])]
                interactions.append(dict(axis=axis, model=n, baseline=a.baseline, metric=m,
                    long_minus_short_gain=val, ci=ci(draws)))
    out.mkdir(parents=True, exist_ok=True)
    policy = dict(length_cuts=cuts, length_rule='short <= q1; q1 < medium <= q2; long > q2',
        error_density_cuts=[0, .25, .5, 1], first_position='(first_error_index+1)/trace_length; cuts 1/3,2/3',
        threshold=a.threshold, decision='ERROR iff p_error >= threshold',
        context_rule='trace truncated iff any step in ANY compared model drops problem/current tokens or prefix steps',
        scope='Retrospective subgroup diagnostics; no parameter or threshold search on test labels')
    result = dict(schema='prm_complexity_v1', baseline=a.baseline, policy=policy,
        train=trainmeta,
        train_test_shared_problems=len(overlap),
        train_test_overlap_check_skipped=bool(
            getattr(a, 'skip_train_test_overlap_check', False)
        ),
        reference=ref,
        models={n: dict(path=p['path'], metadata=p['meta'],
            truncation_step_counts={k:int((p['trunc'][:, j] > 0).sum()) for j,k in enumerate(TRUNC)}) for n,p in preds.items()},
        bootstrap=dict(samples=a.bootstrap_samples, seed=a.seed, unit='problem', interval='paired percentile 95%',
            warning='Test-sampling uncertainty only; no seed uncertainty, multiplicity adjustment, or causal supervision claim.'),
        results=records, interactions=interactions)
    write_json(out/'metrics.json', result)
    write_json(out/'strata_policy.json', policy)
    with (out/'trace_features.csv').open('w', newline='', encoding='utf-8') as f:
        cols = [k for k in traces[0] if k != 'inds']
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        wr.writerows({k:t[k] for k in cols} for t in traces)
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        cols = ['axis','group','model','problems','traces','steps','metric','value','ci_low','ci_high',
                'ci_valid_replicates','delta_vs_baseline','delta_ci_low','delta_ci_high','delta_ci_valid_replicates']
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for r in records:
            for m, v in r['metrics'].items():
                wr.writerow(dict(axis=r['axis'], group=r['group'], model=r['model'],
                    **{k:r['counts'][k] for k in ['problems','traces','steps']}, metric=m, value=v['value'],
                    ci_low=v['ci']['low'], ci_high=v['ci']['high'], ci_valid_replicates=v['ci']['valid_replicates'],
                    delta_vs_baseline=v['delta_vs_baseline'], delta_ci_low=v['paired_delta_ci']['low'],
                    delta_ci_high=v['paired_delta_ci']['high'], delta_ci_valid_replicates=v['paired_delta_ci']['valid_replicates']))
    main = ['error_ap','error_f1','balanced_accuracy','new_error_ap','role_macro_f1','first_error_accuracy','first_error_accuracy_error_traces','false_positive_rate']
    md = ['# PRM complexity diagnostics', '', f'Threshold: {a.threshold}; training-derived length boundaries: {cuts}. All metrics below are percentages.', '',
          'Complete trajectories; paired problem-cluster bootstrap. N/A denotes unavailable heads or undefined metrics. Error-density and first-position groups use gold labels for retrospective diagnosis only.', '']
    for axis in dict.fromkeys(r['axis'] for r in records):
        md += [f'## {axis}', '', '| Group | Model | Problems | Traces | Steps | '+' | '.join(main)+' |',
               '|---|---|---:|---:|---:|'+'---:|'*len(main)]
        for r in records:
            if r['axis'] == axis:
                md.append('| '+' | '.join([r['group'],r['model'],str(r['counts']['problems']),str(r['counts']['traces']),str(r['counts']['steps'])]+[fmt(r['metrics'][m]['value']) for m in main])+' |')
        md.append('')
    md += ['## Interpretation', '', 'See metrics.csv / metrics.json for per-model 95% intervals, paired differences and long-minus-short gain intervals. Subgroups with fewer than '+str(a.min_problems)+' problems are flagged low_support. Empty or single-class groups are retained; AP and balanced accuracy are N/A where undefined. Fixed three-class Role Macro-F1 assigns zero to absent classes.', '',
           'Longer traces are not evidence of longer dependency edges. Establishing non-local dependency benefits requires separately verified dependency annotations. These existing checkpoints are a model comparison, not a matched-training causal ablation. Historical initializer contamination is not ruled out by the train/test text overlap check.']
    (out/'report.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    tex = ['% Requires booktabs. Values in percent; -- means unavailable.', '\\begin{table}[t]', '\\centering\\small',
        '\\caption{Performance by training-defined trajectory length. Error detection uses a fixed threshold of 0.5.}',
        '\\label{tab:prm_length}', '\\begin{tabular}{llrrrr}', '\\toprule',
        'Length & Model & Error AP & Error F1 & Bal. Acc. & First-error Acc. \\\\', '\\midrule']
    tex[3] = tex[3].replace('0.5', str(a.threshold))
    for r in records:
        if r['axis'] == 'length':
            vals = [fmt(r['metrics'][m]['value']).replace('N/A', '--') for m in ('error_ap','error_f1','balanced_accuracy','first_error_accuracy')]
            tex.append(' & '.join([r['group'],r['model'].replace('_','\\_')]+vals)+' \\\\')
    tex += ['\\bottomrule','\\end{tabular}','\\end{table}']
    (out/'table_length.tex').write_text('\n'.join(tex)+'\n')
    for n in preds:
        print(n, {m:fmt(estimates[('overall/all',n)][m]) for m in ('error_ap','error_f1','balanced_accuracy','first_error_accuracy')})
    print('Saved:', out.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--train-file', required=True)
    p.add_argument('--prediction', action='append', required=True, help='NAME=merged_prediction.jsonl; repeatable')
    p.add_argument('--baseline', default='Three_class_v2')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20260922)
    p.add_argument('--min-problems', type=int, default=30)
    p.add_argument(
        '--skip-train-test-overlap-check',
        action='store_true',
        help=(
            'Explicitly skip the evaluator train/test exact-problem hash guard. '
            'Use only when split non-overlap has been established externally.'
        ),
    )
    run(p.parse_args())


if __name__ == '__main__':
    main()
