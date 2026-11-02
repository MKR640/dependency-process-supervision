"""Resumable LiteLLM annotation with Pydantic validation and corrective retries."""
import argparse
import concurrent.futures
import json
import os
import time
import urllib.request
import urllib.error
import itertools
import re
import threading
from collections import Counter
from urllib.parse import urlsplit
from pathlib import Path
import pydantic
from annotation_schema import validate_output
from annotation_worker import AnnotationWorker
from annotation import validate_result
from common import read_jsonl, file_hash, write_json, digest


def parse_content(content):
    if not isinstance(content, str):
        raise ValueError('Expected string message.content, not reasoning_content')
    text = content.strip()
    # Some local reasoning deployments put analysis into content and emit only
    # a closing tag (the opening tag was inserted in the generation template).
    # Parse strictly AFTER that boundary; never extract example JSON from analysis.
    boundaries = list(re.finditer(r'</think\s*>', text, flags=re.IGNORECASE))
    if boundaries:
        text = text[boundaries[-1].end():].strip()
    if re.search(r'<think(?:\s[^>]*)?>', text, flags=re.IGNORECASE):
        raise ValueError('Unclosed reasoning block; refusing to parse analysis as labels')
    if text.startswith('```'):
        fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n?```', text, flags=re.IGNORECASE | re.DOTALL)
        if not fenced:
            raise ValueError('Incomplete or unsupported JSON code fence')
        text = fenced.group(1).strip()
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('Final answer must be a JSON object')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--api-key-env', default='LITELLM_API_KEY')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--retries', type=int, default=5, help='Maximum attempts per trace, including first attempt')
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--max-tokens', type=int, default=16384)
    p.add_argument('--run-id', required=True, help='Immutable teacher deployment/version identifier')
    p.add_argument('--json-mode', action='store_true', help='Enable response_format=json_object if backend supports it')
    p.add_argument('--limit', type=int, help='Pilot: process at most this many uncached traces; full run reuses them')
    p.add_argument('--disable-thinking', action='store_true', help='Send chat_template_kwargs.enable_thinking=false; backend must support it')
    p.add_argument('--response-format', choices=['none', 'json_object', 'json_schema'])
    p.add_argument('--max-retry-tokens', type=int, help='Legacy option; retries no longer increase the initial output budget')
    p.add_argument('--import-successes', action='append', default=[], help='Revalidate and reuse previous successes; original files untouched')
    a = p.parse_args()
    if min(a.workers, a.retries, a.max_tokens) < 1 or a.timeout <= 0:
        p.error('Positive limits required')
    if a.limit is not None and a.limit < 1:
        p.error('--limit must be positive')
    if a.max_retry_tokens is None:
        a.max_retry_tokens = a.max_tokens
    if a.max_retry_tokens < a.max_tokens:
        p.error('--max-retry-tokens must be >= --max-tokens')
    url = urlsplit(a.endpoint)
    if url.scheme not in ('http', 'https') or not url.hostname or any(x in a.endpoint for x in ('[http', '](', ' ', '\n')):
        p.error('endpoint must be a plain HTTP(S) URL, not a Markdown link')
    mode = a.response_format or ('json_object' if a.json_mode else 'none')
    settings = dict(endpoint=a.endpoint, model=a.model, max_tokens=a.max_tokens,
                    temperature=0, json_mode=a.json_mode, run_id=a.run_id,
                    disable_thinking=a.disable_thinking, response_format=mode,
                    max_retry_tokens=a.max_retry_tokens, max_attempts=a.retries,
                    retry_protocol='bfe_pydantic_context_v2', validator_version=pydantic.__version__)
    out = Path(a.output)
    meta_path = Path(a.output + '.meta.json')
    imports = {str(Path(f).resolve()): file_hash(f) for f in a.import_successes}
    if str(out.resolve()) in imports:
        p.error('Import source must differ from output')
    meta = dict(settings=settings, requests_sha256=file_hash(a.requests),
                annotation_code_sha256=file_hash(__file__), imports=imports,
                worker_code_sha256=file_hash(Path(__file__).with_name('annotation_worker.py')),
                schema_code_sha256=file_hash(Path(__file__).with_name('annotation_schema.py')))
    if out.exists():
        if not meta_path.exists() or json.loads(meta_path.read_text()) != meta:
            raise ValueError('Resume identity differs; choose a new output')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(meta_path, meta)
    completed = {}
    if out.exists():
        for r in read_jsonl(out):
            if r['trace_id'] in completed:
                raise ValueError('Duplicate cached result')
            completed[r['trace_id']] = r
    reqs = {}
    for r in read_jsonl(a.requests):
        if r['trace_id'] in reqs:
            raise ValueError('Duplicate request ID')
        if r['request_sha256'] != digest(dict(system=r['system'], payload=r['payload'])):
            raise ValueError('Invalid request hash')
        reqs[r['trace_id']] = r
    if set(completed) - set(reqs):
        raise ValueError('Cached result has unknown ID')
    for tid, r in completed.items():
        validate_result(reqs[tid], r)
        validate_output(dict(labels=r['labels'], reasons=r['reasons']), len(reqs[tid]['payload']['steps']))
    imported = []
    for source, source_hash in imports.items():
        old_meta_path = Path(source + '.meta.json')
        if not old_meta_path.exists():
            raise ValueError('Import requires original metadata: ' + str(old_meta_path))
        old_meta = json.loads(old_meta_path.read_text())
        if old_meta['requests_sha256'] != meta['requests_sha256']:
            raise ValueError('Import uses different requests')
        for key in ('model', 'run_id', 'disable_thinking', 'temperature'):
            if old_meta['settings'].get(key) != settings.get(key):
                raise ValueError('Import teacher condition differs: ' + key)
        for row in read_jsonl(source):
            tid = row.get('trace_id')
            if tid not in reqs:
                raise ValueError('Unknown imported trace')
            validate_result(reqs[tid], row)
            validate_output(dict(labels=row['labels'], reasons=row['reasons']), len(reqs[tid]['payload']['steps']))
            if row['teacher_model'] != a.model or row['annotation_run_id'] != a.run_id:
                raise ValueError('Imported result has different teacher provenance')
            if tid in completed:
                if any(completed[tid][k] != row[k] for k in ('request_sha256', 'labels', 'reasons')):
                    raise ValueError('Conflicting cached/imported results: ' + tid)
                continue
            row = dict(row, imported_from=dict(path=source, sha256=source_hash),
                       revalidated_with='pydantic_v2_strict')
            completed[tid] = row; imported.append(row)
    if imported:
        with out.open('a', encoding='utf-8') as f:
            for row in imported:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush(); os.fsync(f.fileno())
        print(f'Imported {len(imported)} validated successes; original labels/provenance preserved.', flush=True)
    stop_event = threading.Event()
    call = AnnotationWorker(a, settings, parse_content, stop_event)
    # Fixed hashed order avoids taking only the first contiguous source block.
    pending = (reqs[tid] for tid in sorted(reqs, key=digest) if tid not in completed)
    if a.limit is not None:
        pending = itertools.islice(pending, a.limit)
    success = failed = 0
    cancelled = 0
    error_counts, backend_counts = Counter(), Counter()
    batch_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor, \
            out.open('a', encoding='utf-8') as dest, \
            Path(a.output + '.failures.jsonl').open('a', encoding='utf-8') as errors:
        active = set()
        for _ in range(a.workers):
            r = next(pending, None)
            if r is not None:
                active.add(executor.submit(call, r))
        while active:
            done, active = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                result = future.result()
                if result.get('cancelled'):
                    cancelled += 1
                    continue
                for event in result.get('attempt_details', []):
                    if event.get('error_type'):
                        error_counts[event['stage'] + ':' + event['error_type']] += 1
                    if event.get('backend_id'):
                        backend_counts[event['backend_id']] += 1
                target = errors if result.get('failed') else dest
                target.write(json.dumps(result, ensure_ascii=False) + '\n'); target.flush(); os.fsync(target.fileno())
                failed += bool(result.get('failed')); success += not result.get('failed')
                print(f'cached={len(completed)} new_ok={success} new_failed={failed}', flush=True)
                if result.get('failed'):
                    last = result['attempt_details'][-1]
                    print('Failure:', result['trace_id'], last['stage'], last.get('detail', '')[:300], flush=True)
                r = None if stop_event.is_set() else next(pending, None)
                if r is not None:
                    active.add(executor.submit(call, r))
    elapsed = time.monotonic() - batch_started
    remaining = len(reqs) - len(completed) - success
    report = dict(total_requests=len(reqs), cached_before=len(completed), new_success=success,
                  new_failed=failed, remaining=remaining, wall_seconds=elapsed, workers=a.workers,
                  success_per_hour=success / elapsed * 3600 if elapsed else None,
                  estimated_remaining_hours=remaining * elapsed / success / 3600 if success else None,
                  imported_successes=len(imported), cancelled=cancelled, aborted=stop_event.is_set(),
                  attempt_error_counts=dict(error_counts), backend_request_counts=dict(backend_counts),
                  estimate_assumption='Same service capacity, concurrency, input/output lengths and failure rate; pilot estimate only.',
                  limit=a.limit)
    write_json(str(out) + f'.runs/{time.time_ns()}.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if failed or stop_event.is_set():
        raise SystemExit('Some annotations failed. Re-run the same command to retry missing IDs.')


if __name__ == '__main__':
    main()
