"""Offline matched-target and configuration tests. No GPU or torch dependency."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from expanded_protocol import ARMS, head_policy, targets
from common import ARMS as OLD_ARMS

row=dict(correctness=1,bfe=0,role=0,error_type=-100,first_error_mask=False)
for family,_ in OLD_ARMS.values():
    for is_train in (False,True):
        c=row['bfe'] if is_train and family=='bfe' else row['correctness']
        if is_train and family=='first_error' and not row['first_error_mask']: c=-100
        expected=(c,row['role'] if family=='full' else -100,row['error_type'] if family=='full' else -100)
        assert targets(row,family,is_train)==expected
error=dict(correctness=0,bfe=0,role=1,error_type=3,first_error_mask=True)
assert targets(error,'correctness_type',True)==(0,-100,3)
assert targets(error,'correctness_role',True)==(0,1,-100)
assert targets(row,'correctness_type',True)==(1,-100,-100)
assert head_policy('full')==(True,True)
assert head_policy('correctness_type')==(False,True)
assert head_policy('correctness_role')==(True,False)
root=Path(__file__).resolve().parent
# Execute the actual encoder function extracted from the trainer without loading torch.
import ast
node=next(n for n in ast.walk(ast.parse((root/'train_expanded.py').read_text(encoding='utf-8'))) if isinstance(n,ast.FunctionDef) and n.name=='encode')
for family in ('correctness_type','correctness_role','full','bfe','first_error'):
    env=dict(supervision_targets=targets,targets=['shadowing validation list'],family=family,
             pack=lambda *args:([7],{}),tok=None,cfg=dict(max_length=4096,prefix_policy='contiguous'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'encoder','exec'),env)
    actual=env['encode'](error,True)
    assert (actual['correctness_labels'],actual['role_labels'],actual['error_type_labels'])==targets(error,family,True)
with tempfile.TemporaryDirectory() as tmp:
    t=Path(tmp)
    for name in ('prepare_supplement.py','expanded_protocol.py','common.py'):
        (t/name).write_bytes((root/name).read_bytes())
    data=t/'frozen';data.mkdir();init=t/'model';init.mkdir()
    for name in ('train.jsonl','validation.jsonl','test.jsonl','enriched_indices.json'):(data/name).write_text('[]')
    (data/'manifest.json').write_text(json.dumps(dict(label_source='existing')))
    source=dict(world_size=8,batch_per_device=1,gradient_accumulation_steps=4,init_model=str(init),data_dir='STALE',output_root=str(t/'results'),learning_rate=5e-6)
    (t/'experiments_existing.json').write_text(json.dumps(source))
    (t/'pipeline_existing.json').write_text(json.dumps(dict(frozen_data_dir=str(data))))
    cmd=[sys.executable,str(t/'prepare_supplement.py'),'--arm','M3-correctness-type-R2-weighted','--seed','43']
    a=subprocess.run(cmd,check=True,capture_output=True,text=True)
    cfgpath=Path(a.stdout.strip());cfg=json.loads(cfgpath.read_text())
    assert cfg['world_size']==8 and cfg['gradient_accumulation_steps']==4 and cfg['data_dir']==str(data)
    assert cfg['learning_rate']==5e-6 and cfg['seeds']==[43] and cfg['arms']==['M3-correctness-type-R2-weighted']
    assert 'supplement_20260923_v1' in cfg['output_root']
    old=cfgpath.read_bytes();subprocess.run(cmd,check=True,capture_output=True);assert cfgpath.read_bytes()==old
    source['learning_rate']=1e-5;(t/'experiments_existing.json').write_text(json.dumps(source))
    bad=subprocess.run(cmd,capture_output=True);assert bad.returncode!=0 and b'Saved job config differs' in bad.stderr
print('PASS: legacy target parity, M3/M4 independent masks, fixed paths/batch, seeds, idempotent config and changed-config rejection. GPU training not tested.')
