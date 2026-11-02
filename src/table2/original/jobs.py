"""Emit one independent 8-card job per arm/seed; never submits jobs itself."""
import argparse
import json
import shlex
from pathlib import Path
from common import ARMS, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', default=str(Path(__file__).with_name('experiments.json')))
p.add_argument('--output', default='jobs.json')
a = p.parse_args()
cfg = json.loads(Path(a.config).read_text())
config_path = Path(a.config).resolve()
command_config = config_path.name if config_path.parent == Path(__file__).resolve().parent else str(config_path)
if len(set(cfg['arms'])) != len(cfg['arms']) or len(set(cfg['seeds'])) != len(cfg['seeds']):
    raise ValueError('Duplicate arm or seed')
jobs = []
for arm in cfg['arms']:
    family, regime = ARMS[arm]
    for seed in cfg['seeds']:
        jobs.append(dict(job_id=f'{arm}-seed-{seed}', arm=arm, seed=seed,
                         nodes=1, gpus_per_node=8, family=family, regime=regime,
                         command=f'CONFIG={shlex.quote(command_config)} bash run_8gpu.sh {shlex.quote(arm)} {seed}'))
write_json(a.output, jobs)
print(f'{len(jobs)} jobs written to {a.output}; each job uses one 8-card node.')
