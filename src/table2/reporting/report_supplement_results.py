"""CPU report from collected BFE predictions. Requires only NumPy."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import file_hash, read_jsonl, write_json
from supplement_eval_protocol import ARMS, CONTRASTS, expected_heads


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


def load_predictions(path, refs, heads):
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
        if ('role' if field=='role_probabilities' else 'type') in heads:
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
    for field in ('global_batch','loss_reduction','selection'):
        if len({json.dumps(r[field],sort_keys=True) for r in protocols.values()})!=1: raise ValueError('Protocol mismatch: '+field)
    frozen=json.loads((root/'data_manifest.json').read_text(encoding='utf-8'))
    if frozen.get('label_source')!='existing': raise ValueError('Expected existing-label data')
    for arm,identity in identities.items():
        if identity['config']['world_size']!=8: raise ValueError('All main runs must use 8 GPUs')
        protocol=protocols[arm]
        if protocol['active_heads']!=expected_heads(arm): raise ValueError('Active-head mismatch: '+arm)
        expected=frozen['weights']['correctness'] if '-R2-' in arm else None
        if protocol['correctness_weights']!=expected: raise ValueError('Wrong correctness weighting: '+arm)
        for head,field in [('role','role_weights'),('type','type_weights')]:
            expected=frozen['weights']['role' if head=='role' else 'error_type'] if head in expected_heads(arm) else None
            if protocol[field]!=expected: raise ValueError('Wrong auxiliary weights: '+arm)
    y={s:np.array([1-r['correctness'] for r in rs]) for s,rs in refs.items()}
    test=refs['test']
    problems=sorted({r['problem_sha256'] for r in test})
    pi={v:i for i,v in enumerate(problems)}
    gi=np.array([pi[r['problem_sha256']] for r in test])
    traces={}
    for i,r in enumerate(test): traces.setdefault(r['trace_id'],[]).append(i)
    models={}
    for arm in ARMS:
        vs,_=load_predictions(root/arm/'validation_predictions.jsonl',refs['validation'],expected_heads(arm))
        scores,rows=load_predictions(root/arm/'test_predictions.jsonl',test,expected_heads(arm))
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
        if 'role' in expected_heads(arm) or 'type' in expected_heads(arm):
            for field,prob,n in [('role','role_probabilities',3),('error_type','type_probabilities',8)]:
                if ('role' if field=='role' else 'type') not in expected_heads(arm): continue
                target=np.array([r[field] for r in test]); eligible=target>=0
                pred=np.array([r[prob] for r in rows]).argmax(1)
                cm=np.zeros((n,n),dtype=int)
                np.add.at(cm,(target[eligible],pred[eligible]),1)
                den=cm.sum(0)+cm.sum(1)
                f1=np.divide(2*np.diag(cm),den,out=np.zeros(n,dtype=float),where=den>0)
                heads[field]=dict(confusion_matrix=cm.tolist(),support=cm.sum(1).tolist(),macro_f1_fixed_taxonomy=float(f1.mean()) if eligible.any() else None, eligible_steps=int(eligible.sum()))
        models[arm]=dict(threshold=t,metrics=evaluate(ones),fixed_05=metrics(y['test'],scores,.5,np.ones(len(test))),
            slices=slices,heads=heads,evaluate=evaluate,bootstrap={},differences={})
    contrast_boot={name:{} for name in CONTRASTS}
    rng=np.random.default_rng(a.seed)
    for b in range(a.bootstrap):
        w=rng.multinomial(len(problems),np.full(len(problems),1/len(problems)))
        ms={arm:m['evaluate'](w) for arm,m in models.items()}
        for arm,m in models.items():
            for k,v in ms[arm].items():
                if v is not None: m['bootstrap'].setdefault(k,[]).append(v)
        for name,terms in CONTRASTS.items():
            for k in next(iter(ms.values())):
                if all(ms[arm][k] is not None for arm in terms):
                    delta=sum(coef*ms[arm][k] for arm,coef in terms.items())
                    contrast_boot[name].setdefault(k,[]).append(delta)
        if (b+1)%100==0: print(f'bootstrap {b+1}/{a.bootstrap}',flush=True)
    interval=lambda xs:dict(low=float(np.quantile(xs,.025)),high=float(np.quantile(xs,.975)),valid_replicates=len(xs)) if xs else None
    for arm,m in models.items():
        del m['evaluate']
        m['ci95']={k:interval(v) for k,v in m.pop('bootstrap').items()}
        m['paired_differences']={base:{k:dict(delta=m['metrics'][k]-models[base]['metrics'][k],ci95=interval(v)) for k,v in fields.items()} for base,fields in m.pop('differences').items()}
    contrasts={name:{k:dict(delta=sum(c*models[arm]['metrics'][k] for arm,c in terms.items()),ci95=interval(values))
                         for k,values in contrast_boot[name].items()} for name,terms in CONTRASTS.items()}
    warnings=['Single seed: bootstrap reflects test-problem sampling, not training-seed variance.',
              'Existing-label BFE adaptation; not an independent relabeling comparison.',
              'Original prediction export has no per-example token hash; matching relies on frozen data, initialization and packing settings.',
              'Correct steps after an error are a recovery-related slice, not an explicit cessation annotation.',
              'Initialization contamination and human label adjudication are not established by these files.']
    if len({r['config']['world_size'] for r in identities.values()})>1: warnings.append('Different GPU world sizes; effective batch matched, numerical equivalence not guaranteed.')
    if len({r['code_sha256'] for r in identities.values()})>1: warnings.append('Code fingerprints differ; manually review implementation parity before paper claims.')
    if len({json.dumps(r['software'],sort_keys=True) for r in protocols.values()})>1: warnings.append('Software environments differ; review before paper claims.')
    out.mkdir(parents=True)
    result=dict(models=models,contrasts=contrasts,contrast_definitions=CONTRASTS,bootstrap=dict(samples=a.bootstrap,seed=a.seed,unit='normalized-problem cluster',paired=True),
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
    with (out/'contrasts.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['contrast','metric','delta_pp','low_pp','high_pp','valid_replicates'])
        for name,fields in contrasts.items():
            for k,d in fields.items():
                ci=d['ci95']; writer.writerow([name,k,100*d['delta'],100*ci['low'],100*ci['high'],ci['valid_replicates']])
    for name,subset in [('head_ablation',ARMS[:5]),('sampling_weighting',[ARMS[1],ARMS[4],*ARMS[5:]])]:
        lines=['| Model | '+' | '.join(main_keys)+' |','|---|'+'---:|'*len(main_keys)]
        for arm in subset:lines.append('| '+arm+' | '+' | '.join(fmt(models[arm]['metrics'][k]) for k in main_keys)+' |')
        (out/(name+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['model']+main_keys)
            for arm in subset:writer.writerow([arm]+[models[arm]['metrics'][k] for k in main_keys])
    with (out/'auxiliary_heads.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['model','head','eligible_steps','macro_f1_fixed_taxonomy','class_id','support','predicted','precision','recall','f1'])
        for arm,model in models.items():
            for head,h in model['heads'].items():
                cm=np.array(h['confusion_matrix']);support=cm.sum(1);predicted=cm.sum(0);tp=np.diag(cm)
                for i in range(len(cm)):
                    writer.writerow([arm,head,h['eligible_steps'],h['macro_f1_fixed_taxonomy'],i,int(support[i]),int(predicted[i]),float(tp[i]/predicted[i]) if predicted[i] else 0.,float(tp[i]/support[i]) if support[i] else None,float(2*tp[i]/(support[i]+predicted[i])) if support[i]+predicted[i] else 0.])
    with (out/'run_audit.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['model','seed','world_size','global_batch','train_input_tokens','code_sha256','best_checkpoint','train_hours'])
        for arm in ARMS:
            complete=json.loads((root/arm/'COMPLETED.json').read_text(encoding='utf-8'))
            train=json.loads((root/arm/'train_results.json').read_text(encoding='utf-8'))
            writer.writerow([arm,identities[arm]['seed'],identities[arm]['config']['world_size'],protocols[arm]['global_batch'],protocols[arm]['truncation']['train']['input_tokens'],identities[arm]['code_sha256'],complete['best_checkpoint'],train.get('train_runtime',0)/3600 if 'train_runtime' in train else None])
    report=['# Nine-condition seed42 comparison','',*table,'','## Paired contrasts (percentage points)','']
    for name,fields in contrasts.items():
        if 'error_ap' in fields:
            d=fields['error_ap'];ci=d['ci95']
            report.append(f"- {name}, Error AP: {100*d['delta']:+.2f} [{100*ci['low']:.2f}, {100*ci['high']:.2f}].")
    report+=['','## Limitations','']+['- '+w for w in warnings]
    report+=['- Contrasts and exploratory interaction are unadjusted for multiple comparisons; not preregistered confirmatory tests.',
             '- R1 matches row count, not token exposure. Inspect protocol truncation/input_tokens before causal claims.',
             '- Role/type metrics are only defined for trained heads; type is conditional on reference known new errors.',
             '- No Best-of-N, downstream SFT, semantic label adjudication, or extra training seeds are inferred.']
    report+=['','## Run metadata','']
    for arm in ARMS:
        ident=identities[arm];complete=json.loads((root/arm/'COMPLETED.json').read_text(encoding='utf-8'))
        report.append(f"- {arm}: best={complete['best_checkpoint']}; code={ident['code_sha256']}; train tokens per encoded dataset={protocols[arm]['truncation']['train']['input_tokens']}.")
    (out/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Saved expanded report:',out)

if __name__=='__main__': main()
