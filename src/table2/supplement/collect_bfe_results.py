"""Collect seed-42 formal runs, frozen references and predictions; no model weights."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path
from common import file_hash, read_jsonl, write_json, write_jsonl

ARMS = ['M1-BFE-R2-weighted', 'M5-full-R2-weighted', 'M0-first-error-R2-weighted']
FILES = ['COMPLETED.json', 'run_identity.json', 'protocol.json', 'train_results.json',
         'validation_metrics.json', 'test_metrics.json', 'validation_predictions.jsonl', 'test_predictions.jsonl']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', action='append', default=[], help='Search root; repeat if needed')
    p.add_argument('--runs', help='Explicit JSON mapping ARM to formal directory, resolves duplicate runs')
    p.add_argument('--data-dir', help='Override moved frozen-data directory')
    p.add_argument('--output', required=True, help='New bundle directory')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() or Path(str(out)+'.tar.gz').exists():
        p.error('Choose a new output; existing bundles are never overwritten')
    if a.runs:
        selected = json.loads(Path(a.runs).read_text())
    else:
        if not a.root:
            cfg = json.loads(Path(__file__).with_name('experiments_existing.json').read_text())
            a.root = [str(Path(cfg['output_root']).parent)]
        print('Searching result roots:', a.root, flush=True)
        found = {arm: set() for arm in ARMS}
        for root in a.root:
            if not Path(root).is_dir():
                p.error('Missing search root: '+root)
            for f in Path(root).rglob('COMPLETED.json'):
                r = json.loads(f.read_text())
                if r.get('arm') in found and r.get('seed') == 42 and r.get('smoke') is False:
                    found[r['arm']].add(str(f.parent.resolve()))
        if any(len(v) != 1 for v in found.values()):
            print(json.dumps({k: sorted(v) for k,v in found.items()}, indent=2))
            p.error('Require one completed seed42 run per arm. Use --runs mapping if ambiguous; do not select by test score.')
        selected = {k: next(iter(v)) for k,v in found.items()}
    if set(selected) != set(ARMS):
        p.error('Runs mapping must contain exactly the three requested arms')
    loaded = {}
    for arm, folder in selected.items():
        run = Path(folder)
        for name in FILES:
            if not (run/name).is_file():
                raise ValueError(f'Missing {run/name}; evaluation may not have completed')
        identity = json.loads((run/'run_identity.json').read_text())
        complete = json.loads((run/'COMPLETED.json').read_text())
        if complete['arm'] != arm or complete['seed'] != 42 or complete['smoke'] or complete['identity_hash'] != identity['identity_hash']:
            raise ValueError('Run identity mismatch: '+arm)
        loaded[arm] = identity
    signatures = {r['data_manifest_sha256'] for r in loaded.values()}
    if len(signatures) != 1:
        raise ValueError('Runs use different frozen datasets')
    data = Path(a.data_dir or loaded[ARMS[0]]['config']['data_dir'])
    if file_hash(data/'manifest.json') != next(iter(signatures)):
        raise ValueError('Frozen manifest mismatch')
    manifest = json.loads((data/'manifest.json').read_text())
    for split in ('validation','test'):
        if file_hash(data/(split+'.jsonl')) != manifest['files'][split+'.jsonl']:
            raise ValueError('Frozen reference changed: '+split)
    out.mkdir(parents=True)
    write_json(out/'data_manifest.json', manifest)
    for split in ('validation','test'):
        keys = ['trace_id','step_index','problem_id','problem_sha256','correctness','role','error_type']
        write_jsonl(out/(split+'_reference.jsonl'), ({k:r[k] for k in keys} for r in read_jsonl(data/(split+'.jsonl'))))
    for arm, folder in selected.items():
        dest = out/arm
        dest.mkdir()
        for name in FILES:
            shutil.copyfile(Path(folder)/name, dest/name)
    inventory = {str(f.relative_to(out)).replace('\\','/'): file_hash(f) for f in out.rglob('*') if f.is_file()}
    write_json(out/'bundle_manifest.json', dict(files=inventory, runs=selected,
        note='Original frozen references verified; compact references omit question text. Prediction hashes recorded at collection, not at inference.'))
    with tarfile.open(str(out)+'.tar.gz','w:gz') as tf:
        tf.add(out, arcname=out.name)
    print('Send this archive:', str(out)+'.tar.gz')

if __name__ == '__main__':
    main()
