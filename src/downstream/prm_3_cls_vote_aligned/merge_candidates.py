#!/usr/bin/env python3
"""Merge only the explicit completed ranking shards from one manifest (no glob)."""
# Merge only the shards listed in the manifest.
# Validate shard completion and alignment with the candidate pool.
# Preserve candidate order for both single- and multi-device runs.
# This merger consumes PRM ranking shards.
import argparse
from pathlib import Path
from evaluate import (atomic_write, complete, file_hash, read_json, read_jsonl,
                        validate_ranked, verified, key)


def merge(manifest_path, output):
    manifest = read_json(manifest_path)
    protected = {Path(manifest_path).resolve(), Path(manifest['candidates']).resolve()}
    protected.update((Path(manifest['shard_dir']) / f'rank{r}.jsonl').resolve()
                     for r in range(manifest['world_size']))
    if Path(output).resolve() in protected:
        raise ValueError('Merged output must not overwrite candidates, manifest or shards')
    if file_hash(manifest['candidates']) != manifest['candidates_sha256']:
        raise ValueError('Candidate content changed')
    candidates = read_jsonl(manifest['candidates'])
    merged = []
    for rank in range(manifest['world_size']):
        path = Path(manifest['shard_dir']) / f'rank{rank}.jsonl'
        identity = dict(run_id=manifest['run_id'], rank=rank, world_size=manifest['world_size'])
        if not verified(path, identity):
            raise ValueError(f'Missing, incomplete or stale shard: {path}')
        rows = read_jsonl(path)
        validate_ranked(rows, candidates[rank::manifest['world_size']], require_steps=True, allow_empty=True)
        merged.extend(rows)
    validate_ranked(merged, candidates, require_steps=True)
    by_id = {key(row): row for row in merged}
    atomic_write(output, (by_id[key(row)] for row in candidates), jsonl=True)
    atomic_write(str(output) + '.meta.json', manifest)
    complete(output, dict(run_id=manifest['run_id']), len(merged))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    merge(a.manifest, a.output)
