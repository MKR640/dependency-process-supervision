"""Evaluate the sequential role rule and the learned role head."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from role_baseline_functions import score, sequential_roles, confusion, macro_from_cm

ROOT=Path(__file__).resolve().parents[1]
BINARY='M1-BFE-R2-weighted'
HIERARCHY='M5-full-R2-weighted'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def replay(bundle):
    manifest=read(bundle/'bundle_manifest.json')
    ids=read(ROOT/'configs/table2/formal_identities.json')
    for arm,aid in [(BINARY,'M1'),(HIERARCHY,'M5')]:
        if read(bundle/arm/'run_identity.json')['identity_hash']!=ids[aid]['identity_hash']:
            raise ValueError('Unexpected run identity: '+aid)
    if digest(bundle/'data_manifest.json')!=ids['M1']['data_manifest_sha256']:
        raise ValueError('Unexpected frozen data manifest')
    def rows(name):
        p=bundle/name
        if digest(p)!=manifest['files'][name]:raise ValueError('Bundle file hash mismatch: '+name)
        result=sorted((json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()),key=lambda x:(x['trace_id'],x['step_index']))
        keys=[(x['trace_id'],x['step_index']) for x in result]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate step keys')
        return result,keys
    refs,rk=rows('test_reference.jsonl');binary,bk=rows(BINARY+'/test_predictions.jsonl');hier,hk=rows(HIERARCHY+'/test_predictions.jsonl')
    vr,vk=rows('validation_reference.jsonl');vp,vpk=rows(BINARY+'/validation_predictions.jsonl')
    if rk!=bk or rk!=hk or vk!=vpk:raise ValueError('Prediction/reference step alignment mismatch')
    expected=read(ROOT/'configs/expected_role_baseline.json')
    truth=[r['role'] for r in refs]
    result={}
    for name,threshold in [('fixed_0.5',.5),('existing_validation_balanced_accuracy',expected['existing_validation_balanced_accuracy']['threshold'])]:
        result[name]={'threshold':threshold,**score(truth,sequential_roles(binary,threshold).tolist())}
    hp=[max(range(3),key=lambda k:r['role_probabilities'][k]) for r in hier]
    result['hierarchical_role_argmax']=score(truth,hp)
    vy=np.array([r['role'] for r in vr]);grid=np.linspace(0,1,1001)
    grid_scores=np.array([macro_from_cm(confusion(vy,sequential_roles(vp,t))) for t in grid])
    selected=min(np.flatnonzero(grid_scores==grid_scores.max()),key=lambda i:(abs(grid[i]-.5),grid[i]))
    threshold=float(grid[selected]);bp=sequential_roles(binary,threshold)
    result['validation_role_macro_f1_calibrated']={'threshold':threshold,'validation_role_macro_f1':float(grid_scores[selected]),**score(truth,bp.tolist())}
    groups=sorted(set(r['problem_id'] for r in refs));index={g:i for i,g in enumerate(groups)}
    cmh=np.zeros((len(groups),3,3),dtype=int);cmb=cmh.copy()
    for r,a,b in zip(refs,hp,bp):
        cmh[index[r['problem_id']],r['role'],a]+=1
        cmb[index[r['problem_id']],r['role'],b]+=1
    rng=np.random.default_rng(20260925)
    weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    hm=np.einsum('rg,gij->rij',weights,cmh);bm=np.einsum('rg,gij->rij',weights,cmb)
    delta=100*(macro_from_cm(hm)-macro_from_cm(bm))
    result['hierarchy_minus_role_calibrated_heuristic']={'macro_f1_delta_pp':100*(float(macro_from_cm(cmh.sum(0)))-float(macro_from_cm(cmb.sum(0)))),
        'paired_ci95_pp':np.quantile(delta,[.025,.975]).tolist(),'bootstrap_replicates':2000,'bootstrap_unit':'normalized problem ID','groups':len(groups)}
    checks={}
    for name,value in result.items():
        for field,got in value.items():
            if field in {'macro_f1','threshold','validation_role_macro_f1','macro_f1_delta_pp','paired_ci95_pp','groups','classes'}:
                if field=='classes':checks[name+':'+field]=got==expected[name][field]
                else:checks[name+':'+field]=bool(np.allclose(got,expected[name][field],rtol=0,atol=1e-12))
    if not all(checks.values()):raise ValueError('Historical result mismatch: '+str([k for k,v in checks.items() if not v]))
    return {'checks':checks,'results':result,'scope':'Validation-selected sequential rule and paired problem-cluster bootstrap; fixed trained models.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a fresh output file')
    if not (a.bundle/'bundle_manifest.json').is_file():p.exit(2,'Formal bundle missing\n')
    result=replay(a.bundle)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Role comparison verified against historical results.')

if __name__=='__main__':main()
