#!/usr/bin/env python3
"""Read-only problem-level split audit. Never edits, filters or re-splits input data."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from prepare_prm_table2_data import file_hash, problem_key, problem_text, read_jsonl, write_json, write_jsonl


def index_file(path):
    problems = {}
    total = 0
    for row in read_jsonl(path):
        total += 1
        text = problem_text(row)
        pid = problem_key(text)
        tid = str(row.get('trace_id') or row.get('id') or '')
        item = problems.setdefault(pid, dict(problem=text, rows=0, traces=set(), first_record=total))
        item['rows'] += 1
        if tid:
            item['traces'].add(tid)
    if not total:
        raise ValueError(f'Empty file: {path}')
    return dict(path=str(Path(path).resolve()), sha256=file_hash(path), rows=total,
                unique_problems=len(problems), traces=len(set().union(*(v['traces'] for v in problems.values())))), problems


def audit(train, validation, test, out, extra=None):
    out = Path(out)
    paths = dict(train=train, validation=validation, test=test)
    for i,path in enumerate(extra or []):
        paths[f'history_{i+1}'] = path
    stats, indices = {}, {}
    for name,path in paths.items():
        print(f'AUDIT reading {name}: {path}', flush=True)
        stats[name], indices[name] = index_file(path)
    pairs = [('train','validation'),('train','test'),('validation','test')]
    for name in paths:
        if name.startswith('history_'):
            pairs += [(name,'validation'),(name,'test')]
    summaries, details = [], []
    for left,right in pairs:
        shared = sorted(indices[left].keys() & indices[right].keys())
        lt,rt=set(),set()
        for pid in shared:
            a,b=indices[left][pid],indices[right][pid]
            lt.update(a['traces']); rt.update(b['traces'])
            details.append(dict(left=left,right=right,problem_sha256=pid,problem=a['problem'],
                left_rows=a['rows'],right_rows=b['rows'],left_trace_count=len(a['traces']),right_trace_count=len(b['traces']),
                left_trace_examples=sorted(a['traces'])[:5],right_trace_examples=sorted(b['traces'])[:5],
                left_first_nonempty_record=a['first_record'],right_first_nonempty_record=b['first_record']))
        summaries.append(dict(left=left,right=right,shared_problems=len(shared),
            left_affected_traces=len(lt),right_affected_traces=len(rt),
            left_affected_rows=sum(indices[left][k]['rows'] for k in shared),
            right_affected_rows=sum(indices[right][k]['rows'] for k in shared)))
    report=dict(schema='prm_split_audit_v1', matching='SHA256 of whitespace-normalized problem text; exact match only',
        datasets=stats,pairs=summaries,
        note='No source data changed. No semantic near-duplicate guarantee. Historical checkpoint training is checked only when explicitly supplied.')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'overlap_report.json',report)
    write_jsonl(out/'overlap_examples.jsonl',details)
    columns=['left','right','shared_problems','left_affected_traces','right_affected_traces','left_affected_rows','right_affected_rows']
    with (out/'overlap_summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(summaries)
    print('\n| Pair | Shared problems | Left affected traces | Right affected traces |',flush=True)
    print('|---|---:|---:|---:|',flush=True)
    for r in summaries:
        print(f"| {r['left']} / {r['right']} | {r['shared_problems']} | {r['left_affected_traces']} | {r['right_affected_traces']} |",flush=True)
    print('Audit saved:',out,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data/prm800k/hierarchical_prm')
    p.add_argument('--train-input'); p.add_argument('--validation-input'); p.add_argument('--test-input')
    p.add_argument('--exclude',action='append',default=[],help='Extra historical training file; repeatable')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    root=Path(a.data_root)
    audit(a.train_input or root/'prm800k_phase2_hierarchical_prm_flat_train.jsonl',
          a.validation_input or root/'prm800k_phase2_hierarchical_prm_flat_validation.jsonl',
          a.test_input or root/'prm800k_phase2_hierarchical_prm_flat_test.jsonl',a.output_dir,a.exclude)


if __name__=='__main__': main()
