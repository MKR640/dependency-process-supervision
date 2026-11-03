"""Recompute experiment metrics from saved predictions and reference labels."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import string
import subprocess
import sys
import warnings
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RESOURCE_ROOT=ROOT
TABLE2_BUNDLE=None
RES=json.loads((ROOT/'configs/resources.json').read_text(encoding='utf-8'))
TASKS={'MATH':'math','GSM8K':'gsm8k','OmniMath':'OmniMath','UGPhysics MCQ':'UGPhysics-mcq-en-ppl',
       'UGPhysics True/False':'UGPhysics-true-false-en-ppl','UGPhysics Numeric':'UGPhysics-numeric-en-3shot-gen',
       'PROST':'prost-ppl','IFEval':'IFEval','LAMBADA':'lambada','ARC-Easy':'ARC-e','ARC-Challenge':'ARC-c'}

def resource(name):
    path=RESOURCE_ROOT/RES[name]
    if not path.is_file():
        raise FileNotFoundError('Required external asset missing: '+name)
    return path
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def one(x):
    if isinstance(x,list):
        if len(x)!=1:raise ValueError('Expected singleton record field')
        return x[0]
    return x
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def child(script,args,log):
    with log.open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,str(script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def annotation(out):
    dest=out/'annotation.json'
    child(ROOT/'scripts/replay_annotation.py',[
        '--archive',resource('prm800k_upstream_recovery.zip'),
        '--raw',resource('phase2_test.checked.jsonl'),
        '--audits',resource('prm800k_phase2_test.prm_materialized.cessation_refined.cessation_audits.jsonl'),
        '--materialized',resource('prm800k_phase2_test.prm_materialized.cessation_merged.jsonl'),
        '--trajectory',resource('prm800k_phase2_hierarchical_prm_trajectory_test.jsonl'),
        '--converter',ROOT/'src/annotation/prepare_prm800k_phase2_hierarchical.py',
        '--repo',ROOT/'src/annotation','--output',dest],out/'annotation.log')
    r=read(dest)
    assert all(r['comparisons'].values()) and not r['nonzero_issues']
    assert not r['final_record_field_mismatches'] and not r['trajectory_field_mismatches']
    return {'status':'passed','report':dest.name,'scope':'frozen Teacher outputs → materialization → trajectory test; no annotation inference'}

def existing(out):
    from cache_metrics import summarize
    expected=read(ROOT/'configs/expected_existing_metrics.json');checks={};results={}
    with zipfile.ZipFile(resource('prm_evaluation_predictions_recovery.zip')) as z:
        for domain,split in [('math_existing','test'),('physics_fixed','all')]:
            ref=[json.loads(x) for x in z.read(f'{domain}/data/{split}.jsonl').splitlines() if x.strip()]
            keys=[(x['trace_id'],x['step_index']) for x in ref]
            if len(set(keys))!=len(keys):raise ValueError('Duplicate reference keys')
            results[domain]={}
            for model in ['Binary_PRM','Three_class_v2','Full_hierarchy']:
                pred=[json.loads(x) for x in z.read(f'{domain}/predictions/{model}.{split}.jsonl').splitlines() if x.strip()]
                bykey={(x['trace_id'],x['step_index']):x for x in pred}
                assert len(bykey)==len(pred) and set(bykey)==set(keys)
                metrics,diag=summarize(ref,[bykey[k] for k in keys]);results[domain][model]={'metrics':metrics,'diagnostics':diag}
                for name,want in expected[domain][model]['metrics'].items():
                    got=metrics[name]
                    checks[f'{domain}:{model}:{name}']=got==want if want is None else abs(got-want)<1e-12
    save(out/'existing_metrics.json',{'checks':checks,'results':results})
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    return {'status':'passed','report':'existing_metrics.json','scope':'six existing-model cached evaluations; no new inference'}

def benchmarks(out):
    dest=out/'benchmark_sources.json'
    child(ROOT/'scripts/verify_benchmarks.py',[
        '--predictions',resource('prm_evaluation_predictions_recovery.zip'),
        '--records',resource('prm_evaluation_records_recovery.zip'),'--output',dest],out/'benchmarks.log')
    r=read(dest);assert not r['failed_checks'] and not r['physics_metric_or_interval_differences']
    return {'status':'passed','report':dest.name,'scope':'original physics statistics/CI and ProcessBench decoder; CPU only'}

def paired(a,b,repeats,seed):
    index={r['id']:r['prm_correct'] for r in b}
    assert len(index)==len(b)==len(a) and {r['id'] for r in a}==set(index)
    delta=[int(r['prm_correct'])-int(index[r['id']]) for r in a]
    rng=random.Random(seed)
    samples=sorted(100*sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(repeats))
    def quantile(q):
        pos=(len(samples)-1)*q;lo=int(pos);hi=min(lo+1,len(samples)-1)
        return samples[lo]*(hi-pos)+samples[hi]*(pos-lo) if hi!=lo else samples[lo]
    return dict(wins=delta.count(1),losses=delta.count(-1),ties=delta.count(0),delta_pp=100*sum(delta)/len(delta),
                delta_pp_ci95=[quantile(.025),quantile(.975)])

def voting(out):
    with zipfile.ZipFile(resource('prm_downstream_records_recovery.zip')) as z:
        expected=json.loads(z.read('voting_matched/comparison/comparison_2x2.json'))
    with zipfile.ZipFile(resource('prm_sft_benchmark_records_20261102T060645526206Z.zip')) as z:
        details=json.loads(z.read('voting/comparison_details.json'))
    results=[];checks={}
    for row in expected['results']:
        n=row['N'];cells=details[str(n)]
        accuracy={k:sum(r['prm_correct'] for r in v)/len(v) for k,v in cells.items()}
        checks[f'accuracy:{n}']=accuracy==row['accuracies']
        pairs={f'hierarchical_minus_threeclass_{m}':(cells['Hierarchical PRM / '+m],cells['Three-class PRM / '+m]) for m in ['prm','weighted_vote']}
        pairs.update({m+'_vote_minus_prm':(cells[m+' / weighted_vote'],cells[m+' / prm']) for m in ['Three-class PRM','Hierarchical PRM']})
        contrasts={}
        for label,(a,b) in pairs.items():
            got=paired(a,b,expected['bootstrap_samples'],expected['seed']);contrasts[label]=got
            checks[f'paired:{n}:{label}']=all(got[k]==row['paired_differences'][label][k] for k in got)
        results.append({'N':n,'accuracies':accuracy,'paired':contrasts})
    save(out/'voting.json',{'checks':checks,'results':results});assert all(checks.values())
    return {'status':'passed','report':'voting.json','scope':'stored per-question outcomes and historical paired intervals; not reranking raw generations'}

def evaluator_functions():
    src=ROOT/'src/downstream/opencompass/opencompass'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',SyntaxWarning)
        posttree=ast.parse((src/'utils/text_postprocessors.py').read_text(encoding='utf-8'))
    post=next(f for f in posttree.body if isinstance(f,ast.FunctionDef) and f.name=='general_postprocess');post.decorator_list=[]
    klass=next(c for c in ast.parse((src/'datasets/lambada.py').read_text(encoding='utf-8')).body if isinstance(c,ast.ClassDef) and c.name=='LambadaEvaluator')
    score=next(f for f in klass.body if isinstance(f,ast.FunctionDef) and f.name=='score')
    helpers=[f for f in ast.parse((src/'datasets/generic.py').read_text(encoding='utf-8')).body if isinstance(f,ast.FunctionDef) and f.name in ['_generic_llmjudge_postprocess','get_final_results']]
    assert len(helpers)==2
    env={'re':re,'string':string}
    exec(compile(ast.Module(body=[post,score]+helpers,type_ignores=[]),'<reviewed-offline-functions>','exec'),env)
    return env

def sft(out):
    env=evaluator_functions();rows=[];checks={}
    for archive in ['prm_sft_evaluator_sources_20261102T061439991002Z.zip','prm_sft_task_results_20261102T061101369402Z.zip']:
        with zipfile.ZipFile(resource(archive)) as z:
            for name in z.namelist():
                if not (name.startswith('paper_base_results/') or name.startswith('task_results/')) or not name.endswith('.json'):continue
                task=Path(name).stem
                if task not in TASKS.values():continue
                group='paper_base' if name.startswith('paper_base_results/') else next((g for g in ['selected','random','all'] if '-'+g+'-' in name),'recent_base')
                d=json.loads(z.read(name));v=d['details'];items=v if isinstance(v,list) else [r for k,r in v.items() if k!='type']
                metric='numeric_accuracy' if task=='UGPhysics-numeric-en-3shot-gen' else 'Prompt-level-strict-accuracy' if task=='IFEval' else 'accuracy'
                if task=='lambada':
                    value=env['score'](None,[r['predictions'] for r in items],[r['references'] for r in items])['accuracy']
                elif task in ['math','OmniMath']:
                    pred=[one(r['prediction']) for r in items]
                    value=env['get_final_results']([env['_generic_llmjudge_postprocess'](p) for p in pred],[one(r.get('gold','')) for r in items],pred)['accuracy']
                else:
                    field='is_strict_correct' if task=='IFEval' else 'is_correct' if task in ['prost-ppl','ARC-e','ARC-c'] else 'correct'
                    flags=[one(r[field]) for r in items];assert all(isinstance(f,bool) for f in flags)
                    value=100*sum(flags)/len(flags)
                checks[f'score:{group}:{task}']=abs(value-d[metric])<1e-9
                rows.append({'group':group,'task':task,'n':len(items),'value':value})
    table=read(ROOT/'configs/expected_sft_table.json')
    for row in table:
        for group in ['paper_base','random','selected','all']:
            got=next(r['value'] for r in rows if r['group']==group and r['task']==TASKS[row['task']])
            checks[f'paper:{group}:{row["task"]}']=round(got,2)==row[group]
    save(out/'sft.json',{'checks':checks,'recomputed':rows,'paper_table':table})
    with (out/'sft_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    assert all(checks.values())
    return {'status':'passed','report':'sft.json','scope':'44 paper cells plus separate recent Base; cached semantic judgments retained'}

def table2(out):
    bundle=TABLE2_BUNDLE or RESOURCE_ROOT/'inputs/table2/formal'
    if not (bundle/'bundle_manifest.json').is_file():
        return {'status':'blocked','reason':'Formal prediction bundle is required; pass --table2-bundle. See resources.json.'}
    identities=read(ROOT/'configs/table2/formal_identities.json')
    for arm in read(ROOT/'configs/table2/arms.json'):
        identity=read(bundle/arm['arm']/'run_identity.json')
        if identity['identity_hash']!=identities[arm['id']]['identity_hash']:
            raise ValueError('Not the formal run: '+arm['id'])
    if digest(bundle/'data_manifest.json')!=identities['M0']['data_manifest_sha256']:
        raise ValueError('Frozen data manifest mismatch')
    child(ROOT/'src/table2/reporting/report_supplement_results.py',[
        '--bundle',bundle,'--output',out/'table2','--bootstrap',2000,'--seed',20260923],out/'table2.log')
    return {'status':'passed','report':'table2/metrics.json','scope':'formal shared-label cached metrics; role sequential rule not covered'}

def verify_files():
    manifest=read(ROOT/'manifest.json');bad=[]
    for name,entry in manifest['files'].items():
        p=ROOT/name
        if not p.is_file() or p.stat().st_size!=entry['bytes'] or digest(p)!=entry['sha256']:bad.append(name)
    if bad:raise ValueError('Package files missing/changed: '+str(bad))
    return len(manifest['files'])

def roles(out):
    bundle=TABLE2_BUNDLE or RESOURCE_ROOT/'inputs/table2/formal'
    if not (bundle/'bundle_manifest.json').is_file():
        return {'status':'blocked','reason':'Formal prediction bundle required for role baseline replay; pass --table2-bundle.'}
    child(ROOT/'scripts/replay_role_baseline.py',['--bundle',bundle,'--output',out/'role_baseline.json'],out/'roles.log')
    return {'status':'passed','report':'role_baseline.json','scope':'Sequential role rule, validation calibration and paired problem bootstrap'}

def main():
    global RESOURCE_ROOT, TABLE2_BUNDLE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--module',choices=['annotation','existing','benchmarks','voting','sft','table2','roles','all'],default='all')
    p.add_argument('--output',type=Path)
    p.add_argument('--verify',action='store_true',help='Check bundled immutable file hashes before execution')
    p.add_argument('--strict',action='store_true',help='Compatibility flag; blocked modules always return nonzero')
    p.add_argument('--resource-root',type=Path,help='External asset root matching configs/resources.json')
    p.add_argument('--table2-bundle',type=Path,help='Exact formal prediction bundle directory')
    args=p.parse_args()
    RESOURCE_ROOT=args.resource_root.resolve() if args.resource_root else ROOT
    TABLE2_BUNDLE=args.table2_bundle.resolve() if args.table2_bundle else None
    if args.verify:print('Verified immutable files:',verify_files(),flush=True)
    out=args.output.resolve() if args.output else ROOT/'outputs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if out.exists():p.error('Output must be a new directory; existing results are never overwritten')
    # Keep all generated output within this local package.
    if not out.is_relative_to(ROOT):p.error('Output must be inside the package directory')
    out.mkdir(parents=True)
    modules=['annotation','existing','benchmarks','voting','sft','table2','roles'] if args.module=='all' else [args.module]
    summary={'modules':{},'output':str(out),'scope':'Offline cache reproduction; no GPU/LLM/training',
             'python':sys.version,'package_root':str(ROOT)}
    for name in modules:
        print('Running',name,flush=True)
        try:result=globals()[name](out)
        except FileNotFoundError as e:result={'status':'blocked','reason':str(e)}
        except Exception as e:result={'status':'failed','error':f'{type(e).__name__}: {e}'}
        summary['modules'][name]=result;print(name,result['status'],flush=True)
        save(out/'run_summary.json',summary)
    statuses=[r['status'] for r in summary['modules'].values()]
    summary['all_requested_complete']=all(s=='passed' for s in statuses)
    save(out/'run_summary.json',summary);print('Summary:',out/'run_summary.json')
    if 'failed' in statuses:sys.exit(1)
    if 'blocked' in statuses:sys.exit(2)

if __name__=='__main__':main()
