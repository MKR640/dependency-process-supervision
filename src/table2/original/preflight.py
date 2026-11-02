"""Run on the server; prints capability checks without starting training/API calls."""
import argparse
import json
from pathlib import Path
from common import ARMS, file_hash

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', default=str(Path(__file__).with_name('experiments.json')))
a = p.parse_args()
c = json.loads(Path(a.config).read_text())
assert all(x in ARMS for x in c['arms'])
assert len(c['arms']) == len(set(c['arms'])) and len(c['seeds']) == len(set(c['seeds']))
assert c['prefix_policy'] in ('contiguous', 'legacy_skip')
manifest_path = Path(c['data_dir']) / 'manifest.json'
m = json.loads(manifest_path.read_text())
for name, h in m['files'].items():
    assert file_hash(manifest_path.parent / name) == h, name
assert (Path(c['init_model']) / 'config.json').is_file()
import torch
import transformers
import datasets
import accelerate
assert torch.cuda.is_available(), 'No CUDA-compatible MACA device visible'
assert torch.cuda.device_count() == c['world_size'], 'Expected exactly 8 visible devices'
print(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__,
    datasets=datasets.__version__, accelerate=accelerate.__version__,
    devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    global_batch=c['world_size']*c['batch_per_device']*c['gradient_accumulation_steps']), indent=2))
print('Preflight passed. Run --smoke to verify model loading, DDP, backward, eval and checkpoint save.')
