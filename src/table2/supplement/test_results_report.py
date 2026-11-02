"""Synthetic end-to-end checks; no real experiment results are generated."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from common import file_hash, write_json, write_jsonl, error_ap
from collect_bfe_results import ARMS, FILES
from report_bfe_results import weighted_ap, threshold

assert weighted_ap(np.array([1,0]),np.array([.5,.5]),np.ones(2))==.5
assert weighted_ap(np.array([1,0]),np.array([1.,0.]),np.ones(2))==1
rng=np.random.default_rng(1)
for _ in range(20):
    y=np.r_[0,1,rng.integers(0,2,18)]; s=rng.integers(0,4,20)/4
    assert abs(weighted_ap(y,s,np.ones(20))-error_ap((1-y).tolist(),s.tolist()))<1e-12
root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    t=Path(tmp); data=t/'data';data.mkdir()
    refs={}
    for split in ('validation','test'):
        refs[split]=[dict(trace_id=f'{split}-{j}',step_index=i,problem_id=f'{split}-p{j}',problem_sha256=f'{split}-h{j}',correctness=[1,0,1][i],role=[0,1,0][i],error_type=-100) for j in range(4) for i in range(3)]
        write_jsonl(data/(split+'.jsonl'),refs[split])
    write_json(data/'manifest.json',dict(files={s+'.jsonl':file_hash(data/(s+'.jsonl')) for s in refs}))
    for arm in ARMS:
        run=t/'runs'/arm/'formal';run.mkdir(parents=True)
        cfg=dict(data_dir=str(data),world_size=8,batch_per_device=1,gradient_accumulation_steps=4)
        for k in ['epochs','learning_rate','warmup_ratio','weight_decay','classifier_dropout','max_length','prefix_policy','eval_steps','bf16','init_kind']: cfg[k]=1
        ident=dict(identity_hash=arm,config=cfg,seed=42,code_sha256='same',initialization_sha256='same',data_manifest_sha256=file_hash(data/'manifest.json'))
        write_json(run/'run_identity.json',ident)
        write_json(run/'COMPLETED.json',dict(identity_hash=arm,arm=arm,seed=42,smoke=False))
        write_json(run/'protocol.json',dict(global_batch=32,loss_reduction='same',selection='AP',correctness_weights=[1,1],software={}))
        for split,rs in refs.items():
            pred=[]
            for r in rs:
                d=dict(trace_id=r['trace_id'],step_index=r['step_index'],problem_id=r['problem_id'],reference_correctness=r['correctness'],p_error=.8 if r['correctness']==0 else .2)
                if arm.startswith('M5'): d.update(role_probabilities=[.8,.1,.1],type_probabilities=[.125]*8)
                pred.append(d)
            write_jsonl(run/(split+'_predictions.jsonl'),pred)
            write_json(run/(split+'_metrics.json'),{})
        write_json(run/'train_results.json',{})
    subprocess.run([sys.executable,str(root/'collect_bfe_results.py'),'--root',str(t/'runs'),'--output',str(t/'bundle')],check=True)
    subprocess.run([sys.executable,str(root/'report_bfe_results.py'),'--bundle',str(t/'bundle'),'--output',str(t/'report'),'--bootstrap','20'],check=True)
    result=json.loads((t/'report/metrics.json').read_text())
    for model in result['models'].values():
        assert model['metrics']['error_ap']==1 and model['metrics']['first_error_accuracy']==1
    assert result['models'][ARMS[1]]['paired_differences'][ARMS[0]]['error_ap']['delta']==0
    assert (t/'bundle.tar.gz').is_file() and (t/'report/table.tex').is_file()
    p=t/'bundle'/ARMS[0]/'test_predictions.jsonl';p.write_text('{}\n')
    bad=subprocess.run([sys.executable,str(root/'report_bfe_results.py'),'--bundle',str(t/'bundle'),'--output',str(t/'bad'),'--bootstrap','0'],capture_output=True)
    assert bad.returncode!=0 and b'Bundle changed' in bad.stderr
print('PASS: tie-aware AP parity, collection, matched report, paired bootstrap, exports, tamper rejection. Synthetic data only.')
