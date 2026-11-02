"""Freeze matched data/weights/sampling before dispatching any training arm."""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from common import digest, file_hash, problem_hash, read_jsonl, weights, write_json, write_jsonl
from annotation import validate_result


def normalize(row):
    if len(row.get('completions', [])) != 1 or len(row.get('labels', [])) != 1 or type(row['labels'][0]) is not bool:
        raise ValueError('Expected one completion and one BOOL correctness per flat row')
    tid = row.get('trace_id', row.get('id'))
    idx = row.get('step_index')
    if tid is None or type(idx) is not int or idx < 0:
        raise ValueError('Missing trace_id/id or step_index')
    problem, prefix, text = row.get('problem'), row.get('prefix'), row['completions'][0]
    if not isinstance(problem, str) or not problem.strip() or not isinstance(text, str) or not text.strip():
        raise ValueError('Empty problem/current step')
    if not isinstance(prefix, list) or len(prefix) != idx or any(not isinstance(x, str) for x in prefix):
        raise ValueError('Incomplete prefix')
    role = row.get('role_id', -100)
    typ = row.get('error_type_id', -100) if row.get('error_type_mask') else -100
    if role not in (0, 1, 2) or typ not in (-100, *range(8)) or (typ != -100 and role != 1):
        raise ValueError('Invalid hierarchy targets')
    label = int(row['labels'][0])
    if (role == 0) != (label == 1):
        raise ValueError('Reference role/correctness conflict')
    return dict(trace_id=str(tid), step_index=idx, problem_id=str(row.get('problem_id') or problem_hash(problem)),
                problem_sha256=problem_hash(problem), problem=problem, prefix=prefix,
                current_step=text, correctness=label, role=role, error_type=typ)


def load_flat(path):
    rows = [normalize(r) for r in read_jsonl(path)]
    rows.sort(key=lambda r: (r['trace_id'], r['step_index']))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['trace_id']].append(r)
    for tid, group in grouped.items():
        if [r['step_index'] for r in group] != list(range(len(group))):
            raise ValueError('Missing/duplicate steps: ' + tid)
        texts = [r['current_step'] for r in group]
        if any(r['prefix'] != texts[:r['step_index']] or r['problem'] != group[0]['problem'] for r in group):
            raise ValueError('Prefix/problem inconsistency: ' + tid)
    if not rows:
        raise ValueError('Empty flat input: ' + str(path))
    return rows


def audit_splits(splits):
    seen = {name: set() for name in ('trace_id', 'problem_id', 'problem_sha256')}
    for split, rows in splits.items():
        for field in seen:
            values = {r[field] for r in rows}
            if seen[field] & values:
                raise ValueError(f'Cross-split {field} overlap at {split}. Rebuild problem-disjoint splits before training.')
            seen[field].update(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-flat', required=True)
    p.add_argument('--validation-flat', required=True)
    p.add_argument('--test-flat', required=True)
    p.add_argument('--requests')
    p.add_argument('--label-source', choices=['teacher', 'existing'], default='teacher',
                   help='existing reuses validated flat correctness labels; no teacher files required')
    p.add_argument('--annotations')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--enriched-error-fraction', type=float, default=0.7)
    p.add_argument('--sampling-seed', type=int, default=20260921)
    a = p.parse_args()
    if not 0 < a.enriched_error_fraction < 1:
        p.error('enriched fraction must be between 0 and 1')
    if a.label_source == 'teacher' and (not a.requests or not a.annotations):
        p.error('teacher mode requires --requests and --annotations')
    out = Path(a.output_dir)
    if out.exists():
        p.error('Output directory must be new to keep frozen data immutable')
    splits = {s: load_flat(getattr(a, s + '_flat')) for s in ('train', 'validation', 'test')}
    audit_splits(splits)
    if len({r['correctness'] for r in splits['validation']}) != 2:
        raise ValueError('Validation needs both classes for Error AP selection')
    teacher_conditions = set()
    first_error = {}
    if a.label_source == 'existing':
        for row in splits['train']:
            row['bfe'] = row['correctness']
            if row['correctness'] == 0:
                first_error.setdefault(row['trace_id'], row['step_index'])
    else:
        reqs, annotations = {}, {}
        for r in read_jsonl(a.requests):
            if r['trace_id'] in reqs:
                raise ValueError('Duplicate request')
            reqs[r['trace_id']] = r
        for r in read_jsonl(a.annotations):
            tid = r.get('trace_id')
            if tid in annotations or tid not in reqs:
                raise ValueError('Duplicate/unknown annotation')
            validate_result(reqs[tid], r)
            annotations[tid] = r
        trace_lengths = Counter(r['trace_id'] for r in splits['train'])
        for row in splits['train']:
            tid, idx = row['trace_id'], row['step_index']
            if tid not in annotations:
                raise ValueError('Missing independent BFE annotation: ' + tid)
            req, ann = reqs[tid], annotations[tid]
            if len(req['payload']['steps']) != trace_lengths[tid]:
                raise ValueError('Request contains a different number of steps')
            texts = [f"Step{i+1}: {s['text'].strip()}" for i, s in enumerate(req['payload']['steps'])]
            if row['problem'] != req['payload']['problem'] or row['current_step'] != texts[idx] or row['prefix'] != texts[:idx]:
                raise ValueError('Teacher input differs from frozen flat text: ' + tid)
            teacher_conditions.add((ann['teacher_model'], ann['annotation_run_id']))
            row['bfe'] = ann['labels'][idx]
            if row['correctness'] == 0:
                first_error.setdefault(tid, idx)
        if len(teacher_conditions) != 1:
            raise ValueError('Mixed teacher models/runs')
    for row in splits['train']:
        row['first_error_mask'] = row['step_index'] <= first_error.get(row['trace_id'], float('inf'))
    train = splits['train']
    ids = {v: [i for i, r in enumerate(train) if r['correctness'] == v] for v in (0, 1)}
    if not all(ids.values()):
        raise ValueError('Training reference needs both classes')
    rng = random.Random(a.sampling_seed)
    n_error = round(len(train) * a.enriched_error_fraction)
    enriched = rng.choices(ids[0], k=n_error) + rng.choices(ids[1], k=len(train) - n_error)
    rng.shuffle(enriched)
    counts = {field: [sum(r[field] == k for r in train) for k in range(n)]
              for field, n in [('correctness', 2), ('role', 3), ('error_type', 8)]}
    out.mkdir(parents=True)
    for split, rows in splits.items():
        write_jsonl(out / (split + '.jsonl'), rows)
    write_json(out / 'enriched_indices.json', enriched)
    def distribution(rows):
        return {k: dict(Counter(str(r[k]) for r in rows)) for k in ('correctness', 'bfe', 'role', 'error_type')}
    files = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    source_keys = ['train_flat', 'validation_flat', 'test_flat']
    if a.label_source == 'teacher':
        source_keys += ['requests', 'annotations']
    manifest = dict(label_source=a.label_source,
        label_mapping='existing flat BOOL correctness -> binary 0/1' if a.label_source == 'existing' else 'independent teacher',
        equivalent_arms=[['M1-BFE-R2-weighted', 'M2-correctness-R2-weighted']] if a.label_source == 'existing' else [],
        schema='bfe_matched_data_v1', files=files, counts=counts,
        weights={k: weights(v) for k, v in counts.items()},
        teacher_conditions=sorted(teacher_conditions),
        sources={k: dict(path=str(getattr(a, k)), sha256=file_hash(getattr(a, k))) for k in
                 source_keys},
        split_audit='trace_id + provided/fallback problem_id + normalized problem text; no fuzzy/initialization-history audit',
        rows={s: len(rows) for s, rows in splits.items()},
        natural_distribution=distribution(train), enriched_distribution=distribution([train[i] for i in enriched]),
        enrichment=dict(seed=a.sampling_seed, unit='flat step with full prefix', replacement=True,
                        target_reference_error_fraction=a.enriched_error_fraction,
                        unique_rows=len(set(enriched)), duplicate_fraction=1-len(set(enriched))/len(enriched)),
        label_disagreement=sum(r['bfe'] != r['correctness'] for r in train),
        first_error_supervised=sum(r['first_error_mask'] for r in train))
    write_json(out / 'manifest.json', manifest)
    print(f'Frozen {len(train)} training rows at {out}; label_source={a.label_source}; teacher={teacher_conditions}')


if __name__ == '__main__':
    main()
