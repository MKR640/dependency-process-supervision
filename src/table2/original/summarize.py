"""Gather completed formal runs, retain missing runs and report seed variation."""
import argparse
import csv
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', default=str(Path(__file__).with_name('experiments.json')))
p.add_argument('--output-dir', default='summary')
a = p.parse_args()
c = json.loads(Path(a.config).read_text())
rows, shared = [], set()
for arm in c['arms']:
    for seed in c['seeds']:
        path = Path(c['output_root']) / arm / f'seed-{seed}' / 'formal'
        row = dict(arm=arm, seed=seed, status='missing', test_error_ap='', test_error_f1='', test_accuracy='')
        if (path / 'COMPLETED.json').exists():
            completed = json.loads((path / 'COMPLETED.json').read_text())
            assert not completed['smoke'] and completed['arm'] == arm and completed['seed'] == seed
            identity = json.loads((path / 'run_identity.json').read_text())
            assert identity['identity_hash'] == completed['identity_hash']
            shared.add((identity['data_manifest_sha256'], identity['initialization_sha256'], identity['code_sha256'],
                        json.dumps(identity['config'], sort_keys=True)))
            metrics = json.loads((path / 'test_metrics.json').read_text())
            row.update(status='complete', **{k: metrics[k] for k in ('test_error_ap', 'test_error_f1', 'test_accuracy')})
        rows.append(row)
if len(shared) > 1:
    raise ValueError('Runs use different data/initialization/code/config; do not merge as matched comparisons')
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
with (out / 'runs.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
summary = []
for arm in c['arms']:
    done = [r for r in rows if r['arm'] == arm and r['status'] == 'complete']
    for metric in ('test_error_ap', 'test_error_f1', 'test_accuracy'):
        values = [r[metric] for r in done]
        summary.append(dict(arm=arm, metric=metric, completed_seeds=len(values), expected_seeds=len(c['seeds']),
            mean=statistics.mean(values) if values else None,
            sample_std=statistics.stdev(values) if len(values)>1 else None))
(out / 'seed_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f'{sum(r["status"]=="complete" for r in rows)}/{len(rows)} complete. Wrote {out}. Seed SD is not a test bootstrap CI.')
