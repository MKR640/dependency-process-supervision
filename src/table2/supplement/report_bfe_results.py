"""CPU report from collected BFE predictions. Requires only NumPy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import file_hash, read_jsonl, write_json
from collect_bfe_results import ARMS


def key(r):
    return (r['trace_id'], r['step_index'])


def weighted_ap(y, scores, w):
    order = np.argsort(-scores, kind='stable')
    y, scores, w = y[order], scores[order], w[order]
    positives = np.sum(w*y)
    if positives == 0 or np.sum(w*(1-y)) == 0:
        return None
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores)-1]
    tp, total = np.cumsum(w*y)[ends], np.cumsum(w)[ends]
    precision = np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
    return float(np.sum(np.diff(np.r_[0.,tp])/positives * precision))


def threshold(y, scores):
    if set(y) != {0,1}:
        raise ValueError('Validation must include both classes')
    order = np.argsort(-scores,kind='stable')
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    tp = np.cumsum(ys)[ends]
    ba = .5*(tp/y.sum() + 1-(ends+1-tp)/(len(y)-y.sum()))
    ts = ss[ends]
    half = .5*((scores[y==1]>=.5).mean()+(scores[y==0]<.5).mean())
    ts, ba = np.r_[ts,.5], np.r_[ba,half]
    best = np.flatnonzero(np.isclose(ba,ba.max(),rtol=0,atol=1e-12))
    i = min(best, key=lambda j:(abs(ts[j]-.5),ts[j]))
    return float(ts[i])


def metrics(y, score, t, w):
    pred = score>=t
    tp, fp, fn, tn = [float(w[m].sum()) for m in
                      ((y==1)&pred,(y==0)&pred,(y==1)&~pred,(y==0)&~pred)]
    ratio = lambda x,d: x/d if d else None
    return dict(error_ap=weighted_ap(y,score,w), error_f1=ratio(2*tp,2*tp+fp+fn),
        balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if (tp+fn)*(tn+fp) else None,
        error_recall=ratio(tp,tp+fn), correct_recall=ratio(tn,tn+fp))


def load_predictions(path, refs, full):
    raw = list(read_jsonl(path))
    index = {key(r):r for r in raw}
    if len(index)!=len(raw) or set(index)!={key(r) for r in refs}:
        raise ValueError('Missing/duplicate/extra prediction steps: '+str(path))
    rows = [index[key(r)] for r in refs]
    for ref, row in zip(refs, rows):
        if row['reference_correctness'] != ref['correctness'] or row['problem_id'] != ref['problem_id']:
            raise ValueError('Prediction/reference label or problem mismatch')
    score = np.array([r['p_error'] for r in rows],dtype=float)
    if not np.isfinite(score).all() or np.any((score<0)|(score>1)):
        raise ValueError('Invalid probabilities')
    for field,n in [('role_probabilities',3),('type_probabilities',8)]:
        if full:
            arr=np.array([r[field] for r in rows],dtype=float)
            if arr.shape!=(len(rows),n) or not np.isfinite(arr).all() or (arr<0).any() or (arr>1).any() or not np.allclose(arr.sum(1),1,atol=1e-5):
                raise ValueError('Invalid head probabilities')
        elif any(field in r for r in rows):
            raise ValueError('Unexpected inactive head')
    return score, rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20260923)
    a=p.parse_args()
    if a.bootstrap<0: p.error('bootstrap must be nonnegative')
    root,out=Path(a.bundle),Path(a.output)
    if out.exists(): p.error('Output must be new')
    inventory=json.loads((root/'bundle_manifest.json').read_text())
    for name,h in inventory['files'].items():
        if file_hash(root/name)!=h: raise ValueError('Bundle changed: '+name)
    refs={s:sorted(read_jsonl(root/(s+'_reference.jsonl')),key=key) for s in ('validation','test')}
    for split,rs in refs.items():
        if not rs or len({key(r) for r in rs})!=len(rs): raise ValueError('Empty/duplicate references')
        groups={}
        for r in rs:
            groups.setdefault(r['trace_id'],[]).append(r)
            if r['correctness'] not in (0,1): raise ValueError('Invalid label')
        for group in groups.values():
            if [r['step_index'] for r in group]!=list(range(len(group))) or len({r['problem_sha256'] for r in group})!=1:
                raise ValueError('Incomplete or inconsistent trajectory')
    for field in ('problem_id','problem_sha256','trace_id'):
        if {r[field] for r in refs['validation']} & {r[field] for r in refs['test']}:
            raise ValueError('Validation/test overlap: '+field)
    identities={arm:json.loads((root/arm/'run_identity.json').read_text()) for arm in ARMS}
    protocols={arm:json.loads((root/arm/'protocol.json').read_text()) for arm in ARMS}
    controls=['epochs','learning_rate','warmup_ratio','weight_decay','classifier_dropout','max_length','prefix_policy','eval_steps','bf16','init_kind','batch_per_device']
    for field in controls:
        if len({json.dumps(r['config'][field],sort_keys=True) for r in identities.values()})!=1:
            raise ValueError('Training control mismatch: '+field)
    for field in ('initialization_sha256','data_manifest_sha256','seed'):
        if len({r[field] for r in identities.values()})!=1: raise ValueError('Mismatch: '+field)
    for field in ('global_batch','loss_reduction','selection','correctness_weights'):
        if len({json.dumps(r[field],sort_keys=True) for r in protocols.values()})!=1: raise ValueError('Protocol mismatch: '+field)
    y={s:np.array([1-r['correctness'] for r in rs]) for s,rs in refs.items()}
    test=refs['test']
    problems=sorted({r['problem_sha256'] for r in test})
    pi={v:i for i,v in enumerate(problems)}
    gi=np.array([pi[r['problem_sha256']] for r in test])
    traces={}
    for i,r in enumerate(test): traces.setdefault(r['trace_id'],[]).append(i)
    models={}
    for arm in ARMS:
        vs,_=load_predictions(root/arm/'validation_predictions.jsonl',refs['validation'],arm.startswith('M5'))
        scores,rows=load_predictions(root/arm/'test_predictions.jsonl',test,arm.startswith('M5'))
        t=threshold(y['validation'],vs)
        first_ok=[]; trace_group=[]; error_trace=[]
        pre=np.zeros(len(test),dtype=bool); after=pre.copy(); recovery=pre.copy()
        for inds in traces.values():
            gold=next((i for i in inds if y['test'][i]),None)
            guess=next((i for i in inds if scores[i]>=t),None)
            first_ok.append(gold==guess); trace_group.append(gi[inds[0]]); error_trace.append(gold is not None)
            for i in inds:
                pre[i]=gold is None or i<=gold
                after[i]=gold is not None and i>gold
                recovery[i]=after[i] and y['test'][i]==0
        tg=np.array(trace_group); ok=np.array(first_ok); has=np.array(error_trace)
        def evaluate(w,scores=scores,t=t,tg=tg,ok=ok,has=has):
            result=metrics(y['test'],scores,t,w[gi]); tw=w[tg]
            result['first_error_accuracy']=float(np.average(ok,weights=tw))
            result['first_error_accuracy_error_traces']=float(np.average(ok[has],weights=tw[has])) if tw[has].sum() else None
            return result
        ones=np.ones(len(problems))
        slices={}
        for name,mask in [('through_first_error',pre),('after_first_error',after),('correct_steps_after_error',recovery)]:
            slices[name]=dict(steps=int(mask.sum()),errors=int(y['test'][mask].sum()),
                metrics=metrics(y['test'][mask],scores[mask],t,np.ones(mask.sum())) if mask.any() else None)
        heads={}
        if arm.startswith('M5'):
            for field,prob,n in [('role','role_probabilities',3),('error_type','type_probabilities',8)]:
                target=np.array([r[field] for r in test]); eligible=target>=0
                pred=np.array([r[prob] for r in rows]).argmax(1)
                cm=np.zeros((n,n),dtype=int)
                np.add.at(cm,(target[eligible],pred[eligible]),1)
                den=cm.sum(0)+cm.sum(1)
                f1=np.divide(2*np.diag(cm),den,out=np.zeros(n,dtype=float),where=den>0)
                heads[field]=dict(confusion_matrix=cm.tolist(),support=cm.sum(1).tolist(),macro_f1_fixed_taxonomy=float(f1.mean()), eligible_steps=int(eligible.sum()))
        models[arm]=dict(threshold=t,metrics=evaluate(ones),fixed_05=metrics(y['test'],scores,.5,np.ones(len(test))),
            slices=slices,heads=heads,evaluate=evaluate,bootstrap={},differences={})
    rng=np.random.default_rng(a.seed)
    for b in range(a.bootstrap):
        w=rng.multinomial(len(problems),np.full(len(problems),1/len(problems)))
        ms={arm:m['evaluate'](w) for arm,m in models.items()}
        for arm,m in models.items():
            for k,v in ms[arm].items():
                if v is not None: m['bootstrap'].setdefault(k,[]).append(v)
        for base in (ARMS[0],ARMS[2]):
            for k,v in ms[ARMS[1]].items():
                bv=ms[base][k]
                if v is not None and bv is not None: models[ARMS[1]]['differences'].setdefault(base,{}).setdefault(k,[]).append(v-bv)
        if (b+1)%100==0: print(f'bootstrap {b+1}/{a.bootstrap}',flush=True)
    interval=lambda xs:dict(low=float(np.quantile(xs,.025)),high=float(np.quantile(xs,.975)),valid_replicates=len(xs)) if xs else None
    for arm,m in models.items():
        del m['evaluate']
        m['ci95']={k:interval(v) for k,v in m.pop('bootstrap').items()}
        m['paired_differences']={base:{k:dict(delta=m['metrics'][k]-models[base]['metrics'][k],ci95=interval(v)) for k,v in fields.items()} for base,fields in m.pop('differences').items()}
    warnings=['Single seed: bootstrap reflects test-problem sampling, not training-seed variance.',
              'Existing-label BFE adaptation; not an independent relabeling comparison.',
              'Original prediction export has no per-example token hash; matching relies on frozen data, initialization and packing settings.',
              'Correct steps after an error are a recovery-related slice, not an explicit cessation annotation.',
              'Initialization contamination and human label adjudication are not established by these files.']
    if len({r['config']['world_size'] for r in identities.values()})>1: warnings.append('Different GPU world sizes; effective batch matched, numerical equivalence not guaranteed.')
    if len({r['code_sha256'] for r in identities.values()})>1: warnings.append('Code fingerprints differ; manually review implementation parity before paper claims.')
    if len({json.dumps(r['software'],sort_keys=True) for r in protocols.values()})>1: warnings.append('Software environments differ; review before paper claims.')
    out.mkdir(parents=True)
    result=dict(models=models,bootstrap=dict(samples=a.bootstrap,seed=a.seed,unit='normalized-problem cluster',paired=True),
        threshold_policy='Per-model maximum validation balanced accuracy; test never tunes thresholds',
        test_steps=len(test),test_problems=len(problems),test_traces=len(traces),warnings=warnings,
        protocols=protocols,bundle_manifest_sha256=file_hash(root/'bundle_manifest.json'))
    write_json(out/'metrics.json',result)
    main_keys=['error_ap','error_f1','balanced_accuracy','first_error_accuracy','first_error_accuracy_error_traces']
    fmt=lambda v:'N/A' if v is None else f'{100*v:.2f}'
    table=['| Model | '+' | '.join(main_keys)+' |','|---|'+'---:|'*len(main_keys)]
    for arm,m in models.items():
        cells=[]
        for k in main_keys:
            cell=fmt(m['metrics'][k]); ci=m['ci95'].get(k)
            if ci: cell+=f" [{fmt(ci['low'])}, {fmt(ci['high'])}]"
            cells.append(cell)
        table.append('| '+arm+' | '+' | '.join(cells)+' |')
    (out/'table.md').write_text('\n'.join(table)+'\n\nValues are percentages; brackets are problem-cluster 95% CIs.\n',encoding='utf-8')
    with (out/'table.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['model']+main_keys)
        for arm,m in models.items(): writer.writerow([arm]+[m['metrics'][k] for k in main_keys])
    latex=['\\begin{tabular}{l'+'r'*len(main_keys)+'}', 'Model & Error AP & Error F1 & BA & First error & First error (error traces) \\\\', '\\hline']
    for arm,m in models.items(): latex.append(arm.replace('_','\\_')+' & '+' & '.join(fmt(m['metrics'][k]) for k in main_keys)+' \\\\')
    latex.append('\\end{tabular}')
    (out/'table.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    report=['# BFE matched-label evaluation','',*table,'','## Interpretation and limitations','']+['- '+w for w in warnings]
    report+=['','## M5 paired differences (percentage points)','']
    for base,fields in models[ARMS[1]]['paired_differences'].items():
        for k in main_keys:
            if k in fields:
                d=fields[k]; ci=d['ci95']
                report.append(f"- M5 minus {base}, {k}: {100*d['delta']:.2f}, 95% CI [{100*ci['low']:.2f}, {100*ci['high']:.2f}].")
    report+=['','Thresholds were selected on validation balanced accuracy. Fixed-0.5 metrics, slices, M5 head confusion matrices and run protocols are in metrics.json.',
             'No downstream Best-of-N or external-benchmark evaluation is inferred from these results.']
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Saved report:',out)

if __name__=='__main__': main()
