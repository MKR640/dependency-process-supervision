"""Rebuild train/validation by connected problem identity; keep test unchanged.

Training-pool traces sharing normalized question text, provided problem_id or
trace ID with test are excluded and recorded. Originals are never overwritten.
"""
import argparse
from collections import defaultdict
from pathlib import Path
from common import digest, file_hash, problem_hash, read_jsonl, write_json, write_jsonl


def rebuild(train_path, validation_path, test_path, output_dir, fraction=.05, seed=20260921):
    out = Path(output_dir)
    if out.exists():
        raise ValueError('Split output directory must not exist')
    pool = list(read_jsonl(train_path)) + list(read_jsonl(validation_path))
    test = list(read_jsonl(test_path))
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root
    def keys(row):
        tid = str(row.get('trace_id', row.get('id', '')))
        if not tid or not row.get('problem'):
            raise ValueError('Missing ID/problem')
        keys = ['trace:' + tid, 'text:' + problem_hash(row['problem'])]
        if row.get('problem_id'):
            keys.append('problem:' + str(row['problem_id']))
        return keys
    for row in pool + test:
        group = keys(row)
        for key in group[1:]:
            a, b = find(group[0]), find(key)
            parent[max(a, b)] = min(a, b)
    forbidden = {find(keys(r)[0]) for r in test}
    groups = defaultdict(list)
    excluded, unique = [], set()
    for r in pool:
        key = (str(r.get('trace_id', r.get('id'))), r['step_index'])
        if key in unique:
            raise ValueError('Duplicate row in original train/validation pool')
        unique.add(key)
        group = find(keys(r)[0])
        if group in forbidden:
            excluded.append(dict(trace_id=key[0], step_index=key[1], reason='problem/trace overlaps fixed test'))
        else:
            groups[group].append(r)
    partitions = {'train': [], 'validation': [], 'test': test}
    for group, rows in sorted(groups.items()):
        ratio = int(digest([seed, group])[:16], 16) / 2**64
        split = 'validation' if ratio < fraction else 'train'
        # This optional bookkeeping field is changed only in the new copy.
        partitions[split].extend(dict(r, split=split) for r in rows)
    if not all(partitions.values()):
        raise ValueError('Empty partition; inspect input size/fraction before freezing')
    out.mkdir(parents=True)
    for split, rows in partitions.items():
        write_jsonl(out / (split + '.jsonl'), rows)
    write_jsonl(out / 'excluded_training_rows.jsonl', excluded)
    write_json(out / 'split_manifest.json', dict(seed=seed, fraction=fraction, rows={s: len(v) for s, v in partitions.items()},
        excluded_rows=len(excluded), algorithm='connected components of normalized problem text, provided problem ID, trace ID',
        sources={str(p): file_hash(p) for p in (train_path, validation_path, test_path)},
        limitations='Does not detect semantic near-duplicates or audit source checkpoint pretraining history.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train-flat', 'validation-flat', 'test-flat', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--validation-fraction', type=float, default=.05)
    p.add_argument('--seed', type=int, default=20260921)
    a = p.parse_args()
    if not 0 < a.validation_fraction < 1:
        p.error('fraction must be between 0 and 1')
    rebuild(a.train_flat, a.validation_flat, a.test_flat, a.output_dir, a.validation_fraction, a.seed)
    print('Wrote new problem-disjoint splits:', a.output_dir)
