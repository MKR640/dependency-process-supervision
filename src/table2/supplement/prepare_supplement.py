"""Generate a stable per-job config for one 8-GPU node; no training or API calls."""
import argparse
import json
from pathlib import Path
import re
from expanded_protocol import ARMS


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',required=True,choices=list(ARMS))
    p.add_argument('--seed',required=True,type=int,choices=[42,43,44])
    p.add_argument('--source-config',default=str(root/'experiments_existing.json'))
    p.add_argument('--pipeline',default=str(root/'pipeline_existing.json'))
    p.add_argument('--campaign',default='supplement_20260923_v1')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.campaign):p.error('Use letters, digits, hyphens or underscores in campaign')
    if a.arm.startswith('M2-'):p.error('M2 duplicates M1 with existing labels; do not run twice')
    cfg=json.loads(Path(a.source_config).read_text(encoding='utf-8'))
    pipe=json.loads(Path(a.pipeline).read_text(encoding='utf-8'))
    data=Path(pipe['frozen_data_dir'])
    if not (data/'manifest.json').is_file():p.error('Frozen data not accessible: '+str(data))
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('label_source')!='existing':p.error('This campaign requires existing-label frozen data')
    for name in ('train.jsonl','validation.jsonl','test.jsonl','enriched_indices.json'):
        if not (data/name).is_file():p.error('Missing frozen asset: '+name)
    if not Path(cfg['init_model']).is_dir():p.error('Initialization not accessible: '+cfg['init_model'])
    # Preserve every model/optimizer/input setting; change only job selection/topology/output.
    old_batch=cfg['world_size']*cfg['batch_per_device']*cfg['gradient_accumulation_steps']
    denom=8*cfg['batch_per_device']
    if old_batch%denom or old_batch<denom:p.error('Cannot preserve effective batch on 8 GPUs')
    cfg.update(arms=[a.arm],seeds=[a.seed],data_dir=str(data),world_size=8,
               gradient_accumulation_steps=old_batch//denom,
               output_root=str(Path(cfg['output_root'])/a.campaign))
    path=root/'launch_configs'/a.campaign/f'{a.arm}_seed-{a.seed}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(cfg,ensure_ascii=False,indent=2)+'\n'
    try:
        with path.open('x',encoding='utf-8') as f:f.write(text)
    except FileExistsError:
        if json.loads(path.read_text(encoding='utf-8'))!=cfg:
            p.error('Saved job config differs; use a new CAMPAIGN, never overwrite for resume')
    print(path.resolve())

if __name__=='__main__':main()
