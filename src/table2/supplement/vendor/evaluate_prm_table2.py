#!/usr/bin/env python3
"""CPU evaluation for Table 2: frozen thresholds, complete trajectories, problem-cluster CIs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix
from prepare_prm_table2_data import (ROLES, TYPES, digest, file_hash, key, read_jsonl,
                                    validate_rows, write_json, write_jsonl)

MAIN = ['error_ap','error_f1','balanced_accuracy','new_error_ap','role_macro_f1','first_error_accuracy']
TITLES = ['Error AP','Error F1','Balanced Acc.','New-error AP','Role Macro-F1','First-error Acc.']


def reference(path):
    rows = list(read_jsonl(path))
    validate_rows(rows)
    meta = json.loads(Path(path+'.meta.json').read_text())
    if meta['reference_sha256'] != file_hash(path):
        raise ValueError('Reference was changed after preparation')
    return rows, meta


def signature(meta):
    # Validation and test must use the same checkpoint, tokenizer and input protocol.
    return digest({k: meta[k] for k in ('name','model','model_kind','active_heads','correctness_ids',
        'role_order','type_order','tokenizer_sha256','packing','model_config_sha256','checkpoint_files')})


def parse_models(specs):
    result = {}
    for item in specs:
        name, sep, path = item.partition('=')
        if not sep or not name or not path or name in result:
            raise ValueError('--prediction must be unique NAME=PATH')
        result[name] = path
    return result


def load_predictions(path, rows, ref_meta, name):
    meta = json.loads(Path(path+'.meta.json').read_text())
    if meta['name'] != name or meta['reference_sha256'] != ref_meta['reference_sha256'] or meta['split'] != ref_meta['split']:
        raise ValueError(f'{name}: prediction identity/reference mismatch')
    if meta.get('predictions_sha256') != file_hash(path):
        raise ValueError(f'{name}: prediction file hash mismatch')
    pred = {}
    for r in read_jsonl(path):
        if key(r) in pred:
            raise ValueError(f'{name}: duplicate prediction {key(r)}')
        pred[key(r)] = r
    if set(pred) != {key(r) for r in rows}:
        raise ValueError(f'{name}: prediction/reference step sets differ; all steps required')
    ordered = [pred[key(r)] for r in rows]
    errors = np.array([r['p_error'] for r in ordered], dtype=float)
    if not np.all(np.isfinite(errors)) or np.any((errors < 0) | (errors > 1)):
        raise ValueError(f'{name}: invalid p_error')
    probs = {}
    for head, field, n in [('role','role_probs',3),('type','type_probs',8)]:
        if head in meta['active_heads']:
            values = np.array([r.get(field) for r in ordered], dtype=float)
            if values.shape != (len(rows),n) or not np.isfinite(values).all() or (values<0).any() or (values>1).any() or not np.allclose(values.sum(1),1,atol=1e-5):
                raise ValueError(f'{name}: invalid {field}')
            probs[head] = values
        elif any(r.get(field) is not None for r in ordered):
            raise ValueError(f'{name}: inactive {head} has probabilities; declare trained heads explicitly')
    hashes = [r.get('input_sha256') for r in ordered]
    if any(not isinstance(x,str) or len(x)!=64 for x in hashes):
        raise ValueError(f'{name}: missing input-token hashes')
    return dict(meta=meta, p_error=errors, probs=probs, rows=ordered, input_hashes=hashes)


def matched_inputs(preds):
    first = next(iter(preds.values()))
    for name, pred in preds.items():
        if pred['input_hashes'] != first['input_hashes']:
            raise ValueError(f'{name}: packed input tokens differ across models. Re-run with shared tokenizer/packing; do not compare mismatched caches.')
        if pred['meta']['packing'] != first['meta']['packing'] or pred['meta']['tokenizer_sha256'] != first['meta']['tokenizer_sha256']:
            raise ValueError(f'{name}: tokenizer/packing metadata differs')


def choose_threshold(y, score):
    if set(y.tolist()) != {0,1}:
        raise ValueError('Validation must contain both correctness classes')
    order = np.argsort(-score, kind='stable')
    ys, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
    tp = np.cumsum(ys)[ends]
    fp = ends+1-tp
    ba = 0.5*(tp/y.sum() + 1-fp/(len(y)-y.sum()))
    thresholds = ss[ends]
    at_half = score >= .5
    half_ba = .5*(at_half[y==1].mean() + (~at_half[y==0]).mean())
    thresholds, ba = np.r_[thresholds,.5], np.r_[ba,half_ba]
    best = np.flatnonzero(np.isclose(ba, ba.max(), rtol=0, atol=1e-12))
    chosen = min(best, key=lambda i: (abs(thresholds[i]-.5), thresholds[i]))
    return float(thresholds[chosen]), float(ba[chosen])


def ap(y, score, weights):
    if weights[y==1].sum() <= 0 or weights[y==0].sum() <= 0:
        return None
    return float(average_precision_score(y, score, sample_weight=weights))


def class_stats(y, pred, n, w):
    cm = confusion_matrix(y, pred, labels=list(range(n)), sample_weight=w)
    support = cm.sum(1)
    tp = np.diag(cm)
    precision = np.divide(tp,cm.sum(0),out=np.zeros(n),where=cm.sum(0)>0)
    recall = np.divide(tp,support,out=np.zeros(n),where=support>0)
    f1 = np.divide(2*tp,support+cm.sum(0),out=np.zeros(n),where=(support+cm.sum(0))>0)
    return cm, support, precision, recall, f1


class Evaluation:
    def __init__(self, rows, pred, threshold):
        self.rows, self.pred, self.threshold = rows, pred, threshold
        self.y = np.array([r['y_error'] for r in rows])
        self.phat = pred['p_error'] >= threshold
        self.role = np.array([ROLES.index(r['role']) if r['role'] in ROLES else -1 for r in rows])
        self.typ = np.array([TYPES.index(r['error_type']) if r['error_type'] in TYPES else -1 for r in rows])
        self.type_supported = np.array([i in self.typ for i in range(8)])
        self.problems = sorted({r['problem_id'] for r in rows})
        pi = {v:i for i,v in enumerate(self.problems)}
        self.group = np.array([pi[r['problem_id']] for r in rows])
        traces = defaultdict(list)
        for i,r in enumerate(rows):
            traces[r['trace_id']].append(i)
        self.trace_groups, self.first_ok, self.has_error = [], [], []
        for tid, inds in sorted(traces.items()):
            inds.sort(key=lambda i: rows[i]['step_index'])
            gold = next((rows[i]['step_index'] for i in inds if self.y[i]), None)
            guess = next((rows[i]['step_index'] for i in inds if self.phat[i]), None)
            self.trace_groups.append(self.group[inds[0]])
            self.first_ok.append(gold == guess)
            self.has_error.append(gold is not None)
        self.trace_groups = np.array(self.trace_groups)
        self.first_ok, self.has_error = np.array(self.first_ok), np.array(self.has_error)

    def metrics(self, group_weights=None):
        if group_weights is None:
            group_weights = np.ones(len(self.problems))
        w = group_weights[self.group]
        cm,s,pr,re,f1 = class_stats(self.y, self.phat, 2, w)
        tw = group_weights[self.trace_groups]
        avg = lambda mask: float(np.average(self.first_ok[mask], weights=tw[mask])) if tw[mask].sum()>0 else None
        result = dict(error_ap=ap(self.y,self.pred['p_error'],w), error_f1=float(f1[1]) if s[1]>0 else None,
            balanced_accuracy=float(re.mean()) if np.all(s>0) else None,
            accuracy=float(np.trace(cm)/cm.sum()), error_precision=float(pr[1]), error_recall=float(re[1]),
            correct_recall=float(re[0]), false_positive_rate=float(1-re[0]) if s[0]>0 else None,
            correctness_macro_f1=float(f1.mean()),
            first_error_accuracy=avg(np.ones(len(tw),dtype=bool)),
            first_error_accuracy_error_traces=avg(self.has_error),
            no_error_trace_accuracy=avg(~self.has_error),
            new_error_ap=None, new_error_recall=None, role_macro_f1=None,
            type_macro_f1_supported=None, type_macro_f1_fixed8=None)
        m = self.role >= 0
        if 'role' in self.pred['probs'] and w[m].sum()>0:
            rp = self.pred['probs']['role']
            _,rs,_,rr,rf = class_stats(self.role[m],rp[m].argmax(1),3,w[m])
            result.update(new_error_ap=ap((self.role[m]==1).astype(int),rp[m,1],w[m]),
                new_error_recall=float(rr[1]) if rs[1]>0 else None,
                role_macro_f1=float(rf.mean()))  # fixed 3-class taxonomy
        m = self.typ >= 0
        if 'type' in self.pred['probs'] and w[m].sum()>0:
            _,ts,_,_,tf = class_stats(self.typ[m],self.pred['probs']['type'][m].argmax(1),8,w[m])
            # Condition on reference new_error + known type, NOT predicted role.
            result.update(type_macro_f1_supported=float(tf[self.type_supported].mean()), type_macro_f1_fixed8=float(tf.mean()))
        return result

    def diagnostics(self):
        result = dict(steps=len(self.rows), problems=len(self.problems), traces=len(self.first_ok),
            error_traces=int(self.has_error.sum()), no_error_traces=int((~self.has_error).sum()),
            role_labeled_steps=int((self.role>=0).sum()), valid_type_steps=int((self.typ>=0).sum()),
            type_supported_classes=[TYPES[i] for i in sorted(set(self.typ[self.typ>=0]))],
            threshold=self.threshold, flags='Metrics follow reference labels; review flags are not adjudicated by this script.')
        for field,n,labels,y,pred in [
            ('correctness',2,['NON_ERROR','ERROR'],self.y,self.phat),
            ('role',3,ROLES,self.role,self.pred['probs'].get('role')),
            ('type',8,TYPES,self.typ,self.pred['probs'].get('type'))]:
            if pred is None:
                continue
            m = y>=0
            guess = pred[m].argmax(1) if field != 'correctness' else pred[m]
            cm,s,pr,re,f1 = class_stats(y[m],guess,n,np.ones(m.sum()))
            result[field] = dict(labels=labels, confusion_matrix=cm.astype(int).tolist(),
                per_class={label:dict(support=int(s[i]),precision=float(pr[i]),recall=float(re[i]),f1=float(f1[i])) for i,label in enumerate(labels)})
        trunc = [r['truncation'] for r in self.pred['rows']]
        result['truncation'] = {f:sum(r[f]>0 for r in trunc) for f in ('problem_tokens_dropped','current_tokens_dropped','prefix_steps_dropped')}
        return result


def merge(a):
    rows, meta = reference(a.reference)
    parts, base, expected_ranks = [], None, set()
    for path in a.shards:
        m = json.loads(Path(path+'.meta.json').read_text())
        if m['reference_sha256'] != meta['reference_sha256']:
            raise ValueError('Shard belongs to other reference')
        if base is None:
            base = m
        if signature(m)!=signature(base) or m['world_size']!=base['world_size'] or m['rank'] in expected_ranks:
            raise ValueError('Mixed/duplicate inference shards')
        expected_ranks.add(m['rank'])
        parts.extend(read_jsonl(path))
    if expected_ranks != set(range(base['world_size'])):
        raise ValueError('Missing rank shards')
    if len(parts)!=len(rows) or len({key(r) for r in parts})!=len(parts) or {key(r) for r in parts}!={key(r) for r in rows}:
        raise ValueError('Incomplete/duplicate shard records')
    parts.sort(key=key)
    base = dict(base, rank='merged')
    write_jsonl(a.output,parts)
    base['predictions_sha256'] = file_hash(a.output)
    write_json(a.output+'.meta.json',base)
    load_predictions(a.output,rows,meta,base['name'])
    print('Merged',a.output)


def fit(a):
    rows, meta = reference(a.reference)
    if meta['split'] != 'validation':
        raise ValueError('Threshold fitting allowed only on declared validation reference')
    models = parse_models(a.prediction)
    preds = {n:load_predictions(f,rows,meta,n) for n,f in models.items()}
    matched_inputs(preds)
    y = np.array([r['y_error'] for r in rows])
    policy = dict(schema='prm_table2_policy_v1', objective='validation_correctness_balanced_accuracy',
        decision='ERROR iff p_error >= threshold', reference_sha256=meta['reference_sha256'],
        validation_problem_ids=sorted({r['problem_id'] for r in rows}), models={})
    for name,pred in preds.items():
        t,ba = choose_threshold(y,pred['p_error'])
        policy['models'][name] = dict(threshold=t,validation_balanced_accuracy=ba,signature=signature(pred['meta']))
        print(name,'threshold=',t,'validation BA=',ba)
    write_json(a.output,policy)


def table(a):
    rows, meta = reference(a.reference)
    if meta['split'] != 'test':
        raise ValueError('Main table requires reference declared test')
    models = parse_models(a.prediction)
    preds = {n:load_predictions(f,rows,meta,n) for n,f in models.items()}
    matched_inputs(preds)
    if a.baseline not in models:
        raise ValueError('Baseline must be one of the supplied model names')
    policy = json.loads(Path(a.policy).read_text()) if a.policy else None
    if policy and {r['problem_id'] for r in rows} & set(policy['validation_problem_ids']):
        raise ValueError('Test problems overlap validation used for threshold fitting')
    evaluations = {}
    for name,pred in preds.items():
        if policy:
            item = policy['models'][name]
            if item['signature'] != signature(pred['meta']):
                raise ValueError(f'{name}: checkpoint/input configuration differs from frozen validation policy')
            threshold = item['threshold']
        else:
            threshold = a.threshold
        evaluations[name] = Evaluation(rows,pred,threshold)
    points = {n:e.metrics() for n,e in evaluations.items()}
    boot = {n:defaultdict(list) for n in models}
    diffs = {n:defaultdict(list) for n in models if n != a.baseline}
    rng = np.random.default_rng(a.seed)
    g = len(next(iter(evaluations.values())).problems)
    if a.bootstrap_samples<0:
        raise ValueError('bootstrap-samples must be nonnegative')
    for b in range(a.bootstrap_samples):
        w = rng.multinomial(g,np.full(g,1/g))
        ms = {n:e.metrics(w) for n,e in evaluations.items()}
        for n,m in ms.items():
            for metric,v in m.items():
                if v is not None:
                    boot[n][metric].append(v)
                bv = ms[a.baseline][metric]
                if n!=a.baseline and v is not None and bv is not None:
                    diffs[n][metric].append(v-bv)
        if (b+1)%100==0:
            print('bootstrap',b+1,'/',a.bootstrap_samples,flush=True)
    interval = lambda xs: dict(low=float(np.quantile(xs,.025)),high=float(np.quantile(xs,.975)),valid_replicates=len(xs)) if xs else None
    results = dict(schema='prm_table2_results_v1',reference=meta, baseline=a.baseline,
        threshold_policy=policy or dict(fixed_error_threshold=a.threshold),bootstrap=dict(samples=a.bootstrap_samples,seed=a.seed,unit='problem',
        estimand='pooled-step metrics; per-trajectory first-error, clusters resampled by problem',
        note='Single-run test-sampling uncertainty, not training-seed variance'), models={})
    for n,e in evaluations.items():
        results['models'][n] = dict(metrics=points[n], ci95={k:interval(v) for k,v in boot[n].items()},
            paired_difference_vs_baseline={k:dict(delta=points[n][k]-points[a.baseline][k],ci95=interval(v)) for k,v in diffs.get(n,{}).items()},
            diagnostics=e.diagnostics(), inference_meta=preds[n]['meta'])
    out = Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'metrics.json',results)
    def fmt(v):
        return 'N/A' if v is None else f'{100*v:.2f}'
    text = ['| Method | '+' | '.join(TITLES)+' |', '|---|'+'---:|'*len(MAIN)]
    for n in models:
        text.append('| '+n+' | '+' | '.join(fmt(points[n][k]) for k in MAIN)+' |')
    text += ['', 'All values are percentages. First-error Acc. includes fully correct traces (predict no error).',
        'See metrics.json for error-only first-error accuracy, per-class metrics, truncation counts, CIs and paired differences.',
        'Role macro-F1 uses a fixed 3-class taxonomy. Type metrics condition on reference new_error with a known type.',
        'No role head => N/A. Bootstrap CIs describe test sampling only. Training matching and label adjudication require run records.']
    (out/'table2.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    ci_text = ['| Method | '+' | '.join(TITLES)+' |', '|---|'+'---:|'*len(MAIN)]
    for n in models:
        cells = []
        for k in MAIN:
            ci = results['models'][n]['ci95'].get(k)
            cell = fmt(points[n][k])
            if ci:
                cell += f" [{100*ci['low']:.2f}, {100*ci['high']:.2f}]"
            cells.append(cell)
        ci_text.append('| '+n+' | '+' | '.join(cells)+' |')
    (out/'table2_ci.md').write_text('\n'.join(ci_text)+'\n',encoding='utf-8')
    with (out/'table2.csv').open('w',encoding='utf-8',newline='') as f:
        w = csv.writer(f); w.writerow(['method']+MAIN)
        for n in models:
            w.writerow([n]+[points[n][k] for k in MAIN])
    with (out/'paired_differences.csv').open('w',encoding='utf-8',newline='') as f:
        w = csv.writer(f); w.writerow(['method','baseline','metric','delta_pp','ci95_low_pp','ci95_high_pp','valid_bootstraps'])
        for n in diffs:
            for k,v in diffs[n].items():
                ci = interval(v)
                w.writerow([n,a.baseline,k,100*(points[n][k]-points[a.baseline][k]),100*ci['low'],100*ci['high'],ci['valid_replicates']])
    print('\n'.join(text)); print('Saved',out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest='command',required=True)
    m = sp.add_parser('merge'); m.add_argument('--reference',required=True); m.add_argument('--shards',nargs='+',required=True); m.add_argument('--output',required=True)
    f = sp.add_parser('fit'); f.add_argument('--reference',required=True); f.add_argument('--prediction',action='append',required=True); f.add_argument('--output',required=True)
    t = sp.add_parser('table'); t.add_argument('--reference',required=True); t.add_argument('--prediction',action='append',required=True)
    t.add_argument('--baseline',required=True); t.add_argument('--policy'); t.add_argument('--threshold',type=float,default=.5)
    t.add_argument('--bootstrap-samples',type=int,default=2000); t.add_argument('--seed',type=int,default=42); t.add_argument('--output-dir',required=True)
    a = p.parse_args()
    if a.command=='table' and not 0<=a.threshold<=1:
        raise ValueError('threshold must be in [0,1]')
    {'merge':merge,'fit':fit,'table':table}[a.command](a)


if __name__ == '__main__':
    main()
