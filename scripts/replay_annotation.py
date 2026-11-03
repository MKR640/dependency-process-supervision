"""Reconstruct step labels and training targets from saved annotation outputs.

Inputs are read-only. Recorded timestamps are retained for exact output comparison.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.request
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rows(data):
    return [json.loads(line) for line in data.decode('utf-8-sig').splitlines() if line.strip()]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def forbid_network(*args, **kwargs):
    raise RuntimeError('Network calls are disabled for this offline replay')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--audits', required=True, type=Path)
    p.add_argument('--materialized', required=True, type=Path)
    p.add_argument('--trajectory', required=True, type=Path)
    p.add_argument('--converter', required=True, type=Path)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    urllib.request.urlopen = forbid_network
    mpath = args.repo / 'pipeline_code/prm/260826/prm_step_materializer.py'
    cpath = args.repo / 'pipeline_code/prm/260827/cessation_candidate_audit_client.py'
    mat = load_module('offline_materializer', mpath)
    cessation = load_module('offline_cessation', cpath)
    converter = load_module('offline_trajectory_converter', args.converter)
    source = {}
    with zipfile.ZipFile(args.archive) as z:
        for tag, suffix in [('sat', 'sat'), ('a', 'teacher_a_v2_1'), ('b', 'teacher_b_v2')]:
            source[tag] = rows(z.read(f'260827-2/prm800k_phase2_test.{suffix}.jsonl'))
        stats = json.loads(z.read('260827-2/prm800k_phase2_test.prm_materialized.stats.json'))
        merge = json.loads(z.read('260827-2/prm800k_phase2_test.prm_materialized.cessation_merge_report.json'))
        refinement_stats = json.loads(z.read('260827-2/prm800k_phase2_test.prm_materialized.cessation_refined.cessation_stats.json'))
    source['raw'] = rows(args.raw.read_bytes())
    target = rows(args.materialized.read_bytes())
    target_by_id = {r['id']: r for r in target}
    target_trajectories = {r['id']: r for r in rows(args.trajectory.read_bytes())}
    issues = Counter()
    issues['raw_file_hash_mismatch'] += sha(args.raw.read_bytes()) != stats['inputs']['raw_sha256']
    lookup = {}
    for tag, rr in source.items():
        issues[f'{tag}_duplicate_ids'] += len(rr) != len({r['id'] for r in rr})
        lookup[tag] = {
            r['id']: (r, mat.IndexEntry(0, i, mat.record_sha256(r)))
            for i, r in enumerate(rr, 1)}
    settings = stats['settings'].copy()
    settings['no_think_required'] = not settings.pop('require_think')
    settings = SimpleNamespace(**settings)
    counts, hard_codes, soft_codes = Counter(), Counter(), Counter()
    role_counts, label_counts = Counter(), Counter()
    base = []
    for line, b in enumerate(source['b'], 1):
        counts['teacher_b_records'] += 1
        hard = mat.teacher_b_prefilter(b)
        soft = []
        if not hard:
            rr, aa, ss = [lookup[tag][b['id']] for tag in ['raw', 'a', 'sat']]
            original = target_by_id.get(b['id'])
            mat.utc_now = lambda: original['source_lineage']['created_at'] if original else 'REPLAY_TIMESTAMP'
            result, hard, soft = mat.materialize_and_audit(
                record_id=b['id'], sat=ss[0], sat_entry=ss[1],
                teacher_a=aa[0], teacher_a_entry=aa[1], teacher_b=b,
                teacher_b_line=line, teacher_b_hash=mat.record_sha256(b),
                raw=rr[0], raw_entry=rr[1], args=settings)
        if hard:
            counts['rejected'] += 1
            hard_codes.update(x['code'] for x in hard)
            soft_codes.update(x['code'] for x in soft)
        else:
            role_counts.update(str(s['role']) for s in result['steps'])
            label_counts.update(str(s['prm_label']) for s in result['steps'])
            if soft:
                counts['review'] += 1
                soft_codes.update(x['code'] for x in soft)
            else:
                counts['core'] += 1
                base.append(result)
    comparisons = {
        'counts': dict(counts) == stats['counts'],
        'hard_issue_counts': dict(hard_codes) == stats['hard_issue_counts'],
        'soft_issue_counts': dict(soft_codes) == stats['soft_issue_counts'],
        'role_counts': dict(role_counts) == stats['role_counts'],
        'prm_label_counts': dict(label_counts) == stats['prm_label_counts'],
        'retained_id_order': [r['id'] for r in base] == [r['id'] for r in target],
    }
    base_bytes = ''.join(mat.json_line(r) for r in base).encode('utf-8')
    comparisons['base_file_hash'] = sha(base_bytes) == merge['base_sha256']
    sources = []
    for i, record in enumerate(base, 1):
        actual = mat.record_sha256(record)
        expected = target_by_id[record['id']]['source_lineage']['cessation_refinement']['input_record_sha256']
        issues['base_record_hash_mismatch'] += actual != expected
        sources.append(cessation.SourceRecord(i, record, actual))
    candidates = cessation.build_candidates(sources)
    audit_rows = rows(args.audits.read_bytes())
    audits = {a['key']: a for a in audit_rows}
    issues['duplicate_audit_keys'] += len(audit_rows) != len(audits)
    run = {'run_id': refinement_stats['run_id'], 'prompt_sha256': target[0]['source_lineage']['cessation_refinement']['prompt_sha256']}
    usable, stale = cessation.reusable_audits(audits, candidates, run['prompt_sha256'])
    issues['stale_audit_records'] += stale
    issues['unmatched_candidate_count'] += len(candidates) - len(usable)
    grouped = defaultdict(list)
    decisions = Counter()
    for c in candidates:
        grouped[c.record_id].append(c)
        audit = usable.get(c.key)
        if audit is None:
            continue
        cessation.validate_decision(json.dumps(audit['decision'], ensure_ascii=False), c)
        issues['audit_base_record_hash_mismatch'] += audit['lineage']['input_record_sha256'] != c.source.sha256
        issues['audit_base_line_mismatch'] += audit['lineage']['input_line_number'] != c.source.line_number
        d = audit['decision']
        decisions[f"{str(d['is_cessation']).lower()}_{d['confidence']}"] += 1
    rebuilt = []
    field_diffs, trajectory_diffs = Counter(), Counter()
    refinement_counts = Counter()
    for source_record in sources:
        expected = target_by_id[source_record.raw['id']]
        cessation.utc_now = lambda: expected['source_lineage']['cessation_refinement']['created_at']
        record, needs_review, info = cessation.apply_audits(
            source_record, grouped[source_record.raw['id']], usable, 'high', run)
        refinement_counts['review' if needs_review else 'core'] += 1
        refinement_counts.update(info)
        for field in set(record) | set(expected):
            field_diffs[field] += record.get(field) != expected.get(field)
        converted, flat = converter.convert_record(record, 'test', 'test', args.materialized)
        expected_traj = target_trajectories[record['id']]
        for field in set(converted) | set(expected_traj):
            trajectory_diffs[field] += converted.get(field) != expected_traj.get(field)
        rebuilt.append(record)
    final_bytes = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rebuilt).encode('utf-8')
    comparisons.update({
        'final_file_hash': sha(final_bytes) == sha(args.materialized.read_bytes()) == merge['output_sha256'],
        'candidate_count': len(candidates) == refinement_stats['candidate_count'],
        'decision_counts': dict(decisions) == refinement_stats['decision_counts_this_run'],
        'promoted_count': refinement_counts['promoted'] == refinement_stats['promoted_cessation_steps'],
        'validated_existing_count': refinement_counts['validated_existing'] == refinement_stats['validated_existing_cessation_steps'],
        'review_count': refinement_counts['review'] == refinement_stats['review_records'],
    })
    report = {
        'scope': 'Offline replay of materialization, cached cessation decisions, and trajectory conversion; no LLM or training calls.',
        'timestamp_handling': 'Original created_at metadata restored from retained provenance; no scientific fields copied from targets.',
        'inputs': {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in
                   [args.archive, args.raw, args.audits, args.materialized, args.trajectory, args.converter, mpath, cpath]},
        'source_records': {k: len(v) for k, v in source.items()},
        'materialization_counts': dict(counts), 'materialization_hard_issues': dict(hard_codes),
        'materialization_soft_issues': dict(soft_codes),
        'base_file_sha256': sha(base_bytes), 'final_file_sha256': sha(final_bytes),
        'comparisons': comparisons, 'nonzero_issues': {k: v for k, v in issues.items() if v},
        'candidate_count': len(candidates), 'cached_audit_records': len(audit_rows),
        'cached_decisions': dict(decisions), 'refinement_counts': dict(refinement_counts),
        'final_record_field_mismatches': {k: v for k, v in field_diffs.items() if v},
        'trajectory_field_mismatches': {k: v for k, v in trajectory_diffs.items() if v},
        'remaining_scope': 'Does not regenerate SaT spans or LLM outputs, measure semantic label accuracy, or reproduce model training.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in [
        'source_records', 'materialization_counts', 'comparisons', 'nonzero_issues',
        'candidate_count', 'cached_audit_records', 'refinement_counts',
        'final_record_field_mismatches', 'trajectory_field_mismatches']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
