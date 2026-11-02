"""Explicit server stages; a command is executed only with --execute."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['split', 'requests', 'annotate', 'freeze'])
    p.add_argument('--config', default=str(root / 'pipeline.json'))
    p.add_argument('--label-source', choices=['teacher', 'existing'], help='freeze only: override label source')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--limit', type=int, help='Annotation pilot: maximum new traces this run')
    p.add_argument('--workers', type=int, help='Override annotation concurrency; does not change cache identity')
    a = p.parse_args()
    if (a.limit is not None or a.workers is not None) and a.stage != 'annotate':
        p.error('--limit/--workers apply only to annotate')
    if a.workers is not None and a.workers < 1:
        p.error('--workers must be positive')
    if a.label_source and a.stage != 'freeze':
        p.error('--label-source applies only to freeze')
    c = json.loads(Path(a.config).read_text())
    if a.stage == 'split':
        cmd = [sys.executable, str(root / 'split_by_problem.py')]
        for key in ('train_flat', 'validation_flat', 'test_flat'):
            cmd += ['--' + key.replace('_', '-'), c[key]]
        cmd += ['--output-dir', c['split_data_dir'], '--validation-fraction', str(c['validation_fraction']),
                '--seed', str(c['sampling_seed'])]
    elif a.stage == 'requests':
        cmd = [sys.executable, str(root / 'annotation.py'), '--materialized', c['materialized_train'],
               '--output', c['requests'], '--train-flat', str(Path(c['split_data_dir']) / 'train.jsonl')]
    elif a.stage == 'annotate':
        t = c['teacher']
        if any(not t.get(k) for k in ('endpoint', 'model', 'run_id')):
            raise SystemExit('Fill teacher.endpoint, teacher.model and teacher.run_id in pipeline.json first. No API call made.')
        cmd = [sys.executable, str(root / 'annotate_litellm.py'), '--requests', c['requests'], '--output', c['annotations']]
        for k in ('endpoint', 'model', 'run_id', 'api_key_env', 'workers', 'retries', 'timeout', 'max_tokens'):
            value = a.workers if k == 'workers' and a.workers is not None else t[k]
            cmd += ['--' + k.replace('_', '-'), str(value)]
        if t.get('json_mode'):
            cmd.append('--json-mode')
        if t.get('disable_thinking'):
            cmd.append('--disable-thinking')
        if t.get('response_format'):
            cmd += ['--response-format', t['response_format']]
        if t.get('max_retry_tokens'):
            cmd += ['--max-retry-tokens', str(t['max_retry_tokens'])]
        for source in c.get('annotation_import_successes', []):
            cmd += ['--import-successes', source]
        if a.limit is not None:
            cmd += ['--limit', str(a.limit)]
    else:
        cmd = [sys.executable, str(root / 'prepare.py')]
        source = a.label_source or c.get('label_source', 'teacher')
        cmd += ['--label-source', source]
        keys = ['train_flat', 'validation_flat', 'test_flat']
        if source == 'teacher':
            keys += ['requests', 'annotations']
        for key in keys:
            value = str(Path(c['split_data_dir']) / (key.split('_')[0] + '.jsonl')) if key.endswith('_flat') else c[key]
            cmd += ['--' + key.replace('_', '-'), value]
        cmd += ['--output-dir', c['frozen_data_dir'], '--enriched-error-fraction', str(c['enriched_error_fraction']),
                '--sampling-seed', str(c['sampling_seed'])]
    print(shlex.join(cmd), flush=True)
    if a.execute:
        if a.stage == 'requests':
            Path(c['requests']).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
    else:
        print('Preview only. Add --execute on the server to run this stage.')


if __name__ == '__main__':
    main()
