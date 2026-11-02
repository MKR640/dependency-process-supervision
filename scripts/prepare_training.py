"""Validate external assets and prepare a new training command without starting it."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['M0','M1','M3','M4','M5'],required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--init-model',type=Path,required=True)
    p.add_argument('--plan-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args()
    expected=read(ROOT/'configs/table2/formal_identities.json')[a.arm]
    mf=a.data_dir/'manifest.json'
    if not mf.is_file() or digest(mf)!=expected['data_manifest_sha256']:
        p.error('Frozen data manifest missing or different from the formal experiment')
    for name,want in read(mf)['files'].items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
            p.error('Unsafe manifest file name')
        source=(a.data_dir/name).resolve()
        if not source.is_relative_to(a.data_dir.resolve()) or not source.is_file() or digest(source)!=want:
            p.error('Frozen data file missing or changed: '+name)
    if not (a.init_model/'config.json').is_file() or not list(a.init_model.glob('*.safetensors')):
        p.error('Initialization config and weights are required')
    if a.plan_dir.exists() or a.output_root.exists():p.error('Use new plan and output directories')
    arm=next(x for x in read(ROOT/'configs/table2/arms.json') if x['id']==a.arm)
    cfg=read(ROOT/arm['config'])
    cfg.update(data_dir=str(a.data_dir.resolve()),init_model=str(a.init_model.resolve()),output_root=str(a.output_root.resolve()))
    a.plan_dir.mkdir(parents=True)
    dest=a.plan_dir/'config.json'
    dest.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    cmd=['torchrun','--standalone','--nproc_per_node',str(cfg['world_size']),str(ROOT/'src/table2'/arm['source_view']/arm['entry']),
         '--config',str(dest.resolve()),'--arm',arm['arm'],'--seed',str(arm['seed'])]
    (a.plan_dir/'command.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join(cmd)+'\n',encoding='utf-8')
    print('Prepared configuration and Linux command. Training not started. Verify initialization payload identity before use.')

if __name__=='__main__':main()
