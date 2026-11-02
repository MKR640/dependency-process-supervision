#!/usr/bin/env python3
"""Freeze complete step-level reference sets. No label-based test filtering."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ['correct', 'new_error', 'propagated_error']
TYPES = ['calculation', 'derivation_or_logic', 'method_or_formula',
         'problem_understanding', 'assumption_or_condition', 'answer_mismatch',
         'missing_evidence', 'unit']


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as e:
                    raise ValueError(f'{path}:{n}: {e}') from e


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def problem_text(r):
    if 'problem' in r:
        p = r['problem']
    else:
        c = r.get('context', {})
        p = str(c.get('problem', '')).strip()
        opts = c.get('options') or []
        if not isinstance(opts, list):
            raise ValueError('context.options must be a list')
        opts = [str(x).strip() for x in opts if str(x).strip()]
        if opts:
            p += '\n\n' + '\n'.join(opts)
    if not isinstance(p, str) or not p.strip():
        raise ValueError('Missing nonempty problem text')
    return p


def problem_key(text):
    # Same text across different trace IDs must share one bootstrap cluster.
    # Exact whitespace-normalized matching is a check, NOT semantic decontamination.
    return hashlib.sha256(re.sub(r'\s+', ' ', text).strip().encode()).hexdigest()


def key(row):
    return (row['trace_id'], row['step_index'])


def validate_rows(rows):
    if not rows:
        raise ValueError('Empty reference set')
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        k = key(r)
        if k in seen:
            raise ValueError(f'Duplicate step: {k}')
        seen.add(k)
        if type(r['step_index']) is not int or r['step_index'] < 0:
            raise ValueError(f'Invalid 0-based step_index: {k}')
        if type(r['y_error']) is not int or r['y_error'] not in (0, 1):
            raise ValueError(f'Invalid correctness: {k}')
        if r.get('role') is not None and r['role'] not in ROLES:
            raise ValueError(f'Unknown role: {k}')
        if r.get('role') is not None and (r['role'] != 'correct') != bool(r['y_error']):
            raise ValueError(f'Correctness/role contradiction: {k}')
        if r.get('error_type') is not None and (r['error_type'] not in TYPES or r['role'] != 'new_error'):
            raise ValueError(f'Invalid type eligibility: {k}')
        if not isinstance(r['prefix'], list) or not all(isinstance(s, str) for s in r['prefix']):
            raise ValueError(f'Invalid prefix: {k}')
        if not isinstance(r['current_step'], str) or not r['current_step'].strip():
            raise ValueError(f'Empty step: {k}')
        if r['problem_id'] != problem_key(r['problem']):
            raise ValueError(f'Wrong problem hash: {k}')
        groups[r['trace_id']].append(r)
    for tid, rs in groups.items():
        rs.sort(key=lambda x: x['step_index'])
        if [r['step_index'] for r in rs] != list(range(len(rs))):
            raise ValueError(f'{tid}: incomplete trajectory; require contiguous 0-based steps')
        if len({r['problem_id'] for r in rs}) != 1:
            raise ValueError(f'{tid}: multiple problems')
        for i, r in enumerate(rs):
            if r['prefix'] != [p['current_step'] for p in rs[:i]]:
                raise ValueError(f'{tid}:{i}: prefix differs from complete preceding steps')
    return groups


def normalize(path, style, audit_source=None):
    audits = {}
    if audit_source:
        for r in read_jsonl(audit_source):
            tid = str(r['id'])
            if tid in audits:
                raise ValueError(f'Duplicate audit trace: {tid}')
            audits[tid] = r.get('audit', {})
    rows = []
    for r in read_jsonl(path):
        if 'y_error' in r and 'current_step' in r:
            rows.append(r)
            continue
        problem = problem_text(r)
        trace_id = str(r.get('trace_id') or r.get('id') or '')
        if not trace_id:
            raise ValueError('Missing id/trace_id')
        audit = r.get('audit') or audits.get(trace_id, {})
        if audit_source and trace_id not in audits:
            raise ValueError(f'{trace_id}: missing audit-source trace')
        flags = [str(x) for x in audit.get('soft_flags', [])]
        if audit.get('status') == 'review':
            flags.append('audit_status_review')
        if 'steps' in r:
            if style is None:
                raise ValueError('Materialized input requires --materialized-step-style numbered|raw; match training format explicitly')
            steps, prefix = r['steps'], []
            for i, s in enumerate(steps):
                text = str(s.get('text') or '').strip()
                if not text:
                    raise ValueError(f'{trace_id}: empty step; adjudicate, do not silently remove')
                current = f'Step{i+1}: {text}' if style == 'numbered' else text
                role = s.get('role')
                if role == 'cessation':
                    role = 'correct'
                label = s.get('prm_label')
                if label not in ('C', 'I'):
                    raise ValueError(f'{trace_id}: unsupported prm_label={label}')
                et = s.get('error_type')
                rows.append(dict(trace_id=trace_id, problem_id=problem_key(problem),
                    source_problem_id=r.get('problem_id'), problem=problem,
                    step_index=i, source_step_id=s.get('id'), prefix=list(prefix), current_step=current,
                    y_error=int(label == 'I'), role=role,
                    error_type=et if role == 'new_error' and et in TYPES else None,
                    raw_error_type=et, audit_flags=flags))
                prefix.append(current)
        else:
            labels, completions = r.get('labels'), r.get('completions')
            if not isinstance(labels, list) or len(labels) != 1 or type(labels[0]) is not bool:
                raise ValueError(f'{trace_id}: flat labels must contain one bool')
            if not isinstance(completions, list) or len(completions) != 1:
                raise ValueError(f'{trace_id}: flat completions must contain one step')
            role = r.get('role')
            if role == 'cessation':
                role = 'correct'
            et = r.get('error_type')
            valid_type = role == 'new_error' and et in TYPES and r.get('error_type_mask', True)
            rows.append(dict(trace_id=trace_id, problem_id=problem_key(problem),
                source_problem_id=r.get('problem_id'), problem=problem,
                step_index=r['step_index'], source_step_id=r.get('step_id'),
                prefix=r['prefix'], current_step=completions[0], y_error=int(not labels[0]),
                role=role, error_type=et if valid_type else None,
                raw_error_type=r.get('raw_error_type', et), audit_flags=flags))
    groups = validate_rows(rows)
    rows.sort(key=key)
    stats = dict(traces=len(groups), problems=len({r['problem_id'] for r in rows}), steps=len(rows),
        correctness=dict(Counter('ERROR' if r['y_error'] else 'NON_ERROR' for r in rows)),
        roles=dict(Counter(r['role'] or 'unavailable' for r in rows)),
        valid_types=dict(Counter(r['error_type'] for r in rows if r['error_type'])),
        flagged_traces=len({r['trace_id'] for r in rows if r['audit_flags']}),
        audit_flags_available=bool(audit_source) or any('audit' in r for r in read_jsonl(path)))
    return rows, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', required=True, choices=['validation', 'test'])
    p.add_argument('--materialized-step-style', choices=['numbered', 'raw'])
    p.add_argument('--audit-source')
    p.add_argument('--exclude', action='append', default=[], help='Training/history/validation files to check for problem overlap; repeatable')
    a = p.parse_args()
    if Path(a.input).resolve() == Path(a.output).resolve():
        raise ValueError('Output must not overwrite source data')
    rows, stats = normalize(a.input, a.materialized_step_style, a.audit_source)
    ids = {r['problem_id'] for r in rows}
    checks = []
    for path in a.exclude:
        other = {problem_key(problem_text(r)) for r in read_jsonl(path)}
        overlap = sorted(ids & other)
        checks.append(dict(path=str(Path(path).resolve()), sha256=file_hash(path), overlap_problems=len(overlap), examples=overlap[:10]))
    manifest = dict(schema='prm_table2_reference_v1', split=a.split, source=str(Path(a.input).resolve()),
        source_sha256=file_hash(a.input), statistics=stats, overlap_checks=checks,
        audit_source_sha256=file_hash(a.audit_source) if a.audit_source else None,
        note='Teacher-derived unless separately independently adjudicated. Exact text overlap check is not full decontamination.')
    if any(c['overlap_problems'] for c in checks):
        write_json(a.output + '.rejected.json', manifest)
        raise ValueError('Problem overlap detected; no reference output written. See .rejected.json; repair split, do not silently drop rows.')
    write_jsonl(a.output, rows)
    manifest['reference_sha256'] = file_hash(a.output)
    write_json(a.output + '.meta.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
