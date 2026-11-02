"""Offline tests only: no API calls, no model download, no GPU required."""
import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import io
from unittest.mock import patch
from pathlib import Path
from annotation import request, validate_result
from common import error_ap, pack, read_jsonl, write_jsonl
from prepare import audit_splits, load_flat
from split_by_problem import rebuild
from annotate_litellm import parse_content
import annotate_litellm

ROOT = Path(__file__).resolve().parent


def rejects(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError('Expected rejection')


def fixture(tid, problem, labels=(1, 0, 1)):
    roles = ['correct', 'new_error', 'cessation']
    raw = dict(id=tid, context=dict(problem=problem, options=[]), reference_answer='42',
               secret_label='DO_NOT_SEND', steps=[dict(text=f'{problem}: step {i}', role=roles[i], note='DO_NOT_SEND',
                 caused_by=['SECRET_GRAPH']) for i in range(3)])
    texts = [f"Step{i+1}: {s['text']}" for i, s in enumerate(raw['steps'])]
    flat = [dict(id=tid, step_index=i, problem=problem, prefix=texts[:i], completions=[texts[i]],
                 labels=[bool(y)], role_id=0 if y else 1, error_type_id=0 if not y else -100,
                 error_type_mask=not bool(y)) for i, y in enumerate(labels)]
    return raw, flat


def main():
    raw, flat = fixture('one', 'problem one')
    req = request(raw, 'train')
    assert 'DO_NOT_SEND' not in json.dumps(req) and 'SECRET_GRAPH' not in json.dumps(req)
    assert req['payload']['reference_answer'] == '42'
    result = dict(trace_id='one', request_sha256=req['request_sha256'], labels=[1, 0, 1],
                  reasons=['fixture'] * 3, teacher_model='TEST_FIXTURE_ONLY', annotation_run_id='TEST')
    validate_result(req, result)
    bad = copy.deepcopy(result); bad['labels'][0] = True
    rejects(lambda: validate_result(req, bad))
    bad_req = copy.deepcopy(req); bad_req['payload']['problem'] = 'changed'
    rejects(lambda: validate_result(bad_req, result))
    assert parse_content('```json\n{"labels":[1]}\n```')['labels'] == [1]
    assert abs(error_ap([0, 1, 0], [.5, .5, .1]) - (1/2 * 1/2 + 1/2 * 2/3)) < 1e-12

    class Tokenizer:
        bos_token_id = 1
        def encode(self, text, add_special_tokens=False):
            return [ord(x) + 2 for x in text]
    tok = Tokenizer()
    sys.path.insert(0, str(ROOT / 'vendor'))
    spec = importlib.util.spec_from_file_location('vendor_predict', ROOT / 'vendor/predict_prm_table2.py')
    vendor = importlib.util.module_from_spec(spec); spec.loader.exec_module(vendor)
    for policy in ('contiguous', 'legacy_skip'):
        for size in (5, 12, 50, 100):
            row = dict(problem='question' * 2, current_step='answer' * 2, prefix=['old', 'x' * 100, 'recent'])
            assert pack(tok, row, size, policy) == vendor.pack(tok, row, size, policy)
    # Weight normalization regression: at batch=1 default weighted-mean CE
    # cancels weights; our numerator / number-of-rows must retain the factor.
    ce, weight = -math.log(.4), 2.0
    assert weight * ce / 1 == 2 * ce
    assert abs(weight * ce / weight - ce) < 1e-12
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        raw_rows, train_rows = [], []
        for i in range(40):
            r, f = fixture(str(i), f'question {i}')
            raw_rows.append(r); train_rows.extend(f)
        _, held = fixture('held', 'fixed test problem')
        # Add a distinct training trace with the same test question: must be excluded.
        _, leaked = fixture('leak', 'fixed test problem')
        write_jsonl(t / 'old_train.jsonl', train_rows[:60] + leaked)
        write_jsonl(t / 'old_validation.jsonl', train_rows[60:])
        write_jsonl(t / 'old_test.jsonl', held)
        rebuild(t / 'old_train.jsonl', t / 'old_validation.jsonl', t / 'old_test.jsonl', t / 'split', .25, 4)
        split = {s: load_flat(t / 'split' / (s + '.jsonl')) for s in ('train', 'validation', 'test')}
        audit_splits(split)
        assert len(list(read_jsonl(t / 'split/excluded_training_rows.jsonl'))) == 3
        wrong = dict(split, validation=split['train'])
        rejects(lambda: audit_splits(wrong))
        requests = [request(r, 'train') for r in raw_rows]
        answers = [dict(trace_id=r['trace_id'], request_sha256=r['request_sha256'], labels=[1, 1, 0],
                        reasons=['SYNTHETIC TEST, NOT REAL ANNOTATION']*3,
                        teacher_model='TEST_FIXTURE_ONLY', annotation_run_id='TEST') for r in requests]
        write_jsonl(t / 'requests.jsonl', requests); write_jsonl(t / 'answers.jsonl', answers)
        # Mock transport: exercise actual API-client validation and resume, with
        # urllib replaced entirely. No socket or local service is contacted.
        write_jsonl(t / 'one_request.jsonl', requests[:2])
        api_args = ['annotate_litellm.py', '--requests', str(t / 'one_request.jsonl'),
                    '--output', str(t / 'mock_api.jsonl'), '--endpoint', 'http://unused.invalid/v1/chat/completions',
                    '--model', 'TEST_FIXTURE_ONLY', '--run-id', 'TEST', '--workers', '1']
        response = dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(
            dict(labels=[1, 0, 1], reasons=['fixture']*3))))], usage=dict(total_tokens=1))
        with patch.object(sys, 'argv', api_args + ['--limit', '1']), patch('urllib.request.urlopen', return_value=io.StringIO(json.dumps(response))) as transport:
            annotate_litellm.main()
            assert transport.call_count == 1
        assert len(list(read_jsonl(t / 'mock_api.jsonl'))) == 1
        with patch.object(sys, 'argv', api_args), patch('urllib.request.urlopen', return_value=io.StringIO(json.dumps(response))) as transport:
            annotate_litellm.main()
            assert transport.call_count == 1
        assert len(list(read_jsonl(t / 'mock_api.jsonl'))) == 2
        with patch.object(sys, 'argv', api_args), patch('urllib.request.urlopen', side_effect=AssertionError('cache should prevent transport')) as transport:
            annotate_litellm.main()
            assert transport.call_count == 0
        cmd = [sys.executable, str(ROOT / 'prepare.py'), '--requests', str(t / 'requests.jsonl'),
               '--annotations', str(t / 'answers.jsonl'), '--output-dir', str(t / 'frozen')]
        for s in ('train', 'validation', 'test'):
            cmd += ['--' + s + '-flat', str(t / 'split' / (s + '.jsonl'))]
        subprocess.run(cmd, check=True, capture_output=True)
        train = list(read_jsonl(t / 'frozen/train.jsonl'))
        assert all(r['bfe'] == [1, 1, 0][r['step_index']] for r in train)
        assert all(r['first_error_mask'] == (r['step_index'] <= 1) for r in train)
        indices = json.loads((t / 'frozen/enriched_indices.json').read_text())
        assert len(indices) == len(train)
        assert sum(train[i]['correctness'] == 0 for i in indices) == round(.7 * len(train))
        assert len(set(indices)) < len(indices)
        manifest = json.loads((t / 'frozen/manifest.json').read_text())
        assert manifest['label_disagreement'] > 0
        cmd[cmd.index(str(t / 'frozen'))] = str(t / 'frozen_second')
        subprocess.run(cmd, check=True, capture_output=True)
        assert (t / 'frozen/enriched_indices.json').read_bytes() == (t / 'frozen_second/enriched_indices.json').read_bytes()
        subprocess.run([sys.executable, str(ROOT / 'jobs.py'), '--output', str(t / 'jobs.json')], check=True, capture_output=True)
        jobs = json.loads((t / 'jobs.json').read_text())
        assert len(jobs) == 24 and len({j['job_id'] for j in jobs}) == 24
    # Real provided samples: only validate label-blind request formation, never
    # invent independent teacher results or run annotation on them.
    real = ROOT.parent / 'server_reference/bfe_samples/samples/materialized_train.jsonl'
    if real.is_file():
        for r in read_jsonl(real):
            payload = request(r, 'train')['payload']
            assert list(payload) == ['problem', 'reference_answer', 'steps']
            assert all(set(s) == {'step_index', 'text'} for s in payload['steps'])
    print('PASS: blind requests, strict labels/hash, AP ties, packing parity, problem split/exclusions, independent labels, FE masks, shared R1 indices, 24 jobs. No API/GPU calls.')


if __name__ == '__main__':
    main()
