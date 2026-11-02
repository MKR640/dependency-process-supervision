"""Offline tests: strict schema, repair feedback, retries, imports and concurrency config."""
import contextlib
import io
import json
import os
import sys
import tempfile
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from pydantic import ValidationError
import annotate_litellm
from annotation import request
from annotation_schema import validate_output, schema_for_steps
from annotation_worker import AnnotationWorker
from common import file_hash, read_jsonl, write_json, write_jsonl
from activate_pydantic import main as activate
from activate_context_budget import main as activate_context


GOOD = {'labels': [1, 0], 'reasons': ['correct step', 'incorrect step']}
REQ = request(dict(id='a', context={'problem': 'p'}, reference_answer='42',
                   steps=[{'text': 'one'}, {'text': 'two'}]), 'train')


def args():
    return SimpleNamespace(api_key_env='BFE_TEST_KEY', model='TEST', run_id='TEST',
        max_tokens=100, max_retry_tokens=200, retries=3, timeout=5, disable_thinking=True,
        endpoint='http://unused.invalid/v1/chat/completions')


class Response(io.BytesIO):
    status = 200
    headers = {'x-litellm-model-id': 'test-deployment-1'}


def response(body=GOOD, finish='stop', raw=False):
    content = body if raw else json.dumps(body)
    return Response(json.dumps(dict(model='TEST', choices=[dict(finish_reason=finish,
        message=dict(content=content))], usage={'total_tokens': 12})).encode())


def run_worker(sequence, mode='json_object'):
    worker = AnnotationWorker(args(), {'response_format': mode}, annotate_litellm.parse_content)
    with patch('urllib.request.urlopen', side_effect=sequence) as transport, patch('annotation_worker.time.sleep'):
        result = worker(REQ)
        bodies = [json.loads(c.args[0].data) for c in transport.call_args_list]
    return result, bodies, worker


def main():
    validate_output(GOOD, 2)
    bad_cases = [dict(labels=[True, 0], reasons=['a', 'b']), dict(labels=['1', 0], reasons=['a', 'b']),
                 dict(labels=[1.0, 0], reasons=['a', 'b']), dict(labels=[2, 0], reasons=['a', 'b']),
                 dict(labels=[1], reasons=['a']), dict(labels=[1, 0], reasons=[' ', 'b']),
                 dict(GOOD, extra='forbidden')]
    for bad in bad_cases:
        try:
            validate_output(bad, 2)
        except ValidationError:
            pass
        else:
            raise AssertionError('Invalid output was coerced or accepted')
    schema = schema_for_steps(2)
    assert schema['additionalProperties'] is False
    assert schema['properties']['labels']['minItems'] == schema['properties']['labels']['maxItems'] == 2

    result, bodies, _ = run_worker([response(bad_cases[0]), response()])
    assert result['attempts'] == 2 and result['labels'] == GOOD['labels']
    assert result['attempt_details'][0]['stage'] == 'schema'
    assert 'validation_errors' in result['attempt_details'][0]
    assert bodies[1]['messages'][-1]['role'] == 'user'
    assert 'labels' in bodies[1]['messages'][-1]['content']
    assert bodies[1]['messages'][:2] == bodies[0]['messages'][:2]
    assert bodies[0]['response_format'] == {'type': 'json_object'}
    assert bodies[0]['chat_template_kwargs'] == {'enable_thinking': False}

    result, _, _ = run_worker([response('not JSON', raw=True), response()])
    assert result['attempt_details'][0]['stage'] == 'model_json'
    result, bodies, _ = run_worker([response('{', 'length', raw=True), response()])
    assert [b['max_tokens'] for b in bodies] == [100, 100]
    assert all(m['role'] != 'assistant' for m in bodies[1]['messages'])
    def overflow():
        return urllib.error.HTTPError('unused', 400, 'Bad Request', {}, io.BytesIO(
            b"model's context length is only 32768 tokens"))
    result, bodies, worker = run_worker([overflow(), response()])
    assert [b['max_tokens'] for b in bodies] == [100, 50]
    assert bodies[0]['messages'] == bodies[1]['messages']
    assert result['attempt_details'][0]['stage'] == 'context_budget' and not worker.stop.is_set()
    result, bodies, worker = run_worker([overflow(), overflow(), overflow()])
    assert result['failed'] and not result['fatal'] and not worker.stop.is_set()
    assert [b['max_tokens'] for b in bodies] == [100, 50, 25]
    result, bodies, worker = run_worker([response(bad_cases[0]), overflow(), response()])
    assert bodies[2]['messages'] == bodies[0]['messages'] and not worker.stop.is_set()
    result, bodies, _ = run_worker([response()], 'json_schema')
    assert bodies[0]['response_format']['json_schema']['schema']['properties']['reasons']['maxItems'] == 2

    limited = urllib.error.HTTPError('unused', 429, 'Rate limited', {'Retry-After': '1'}, io.BytesIO(b'busy'))
    result, bodies, _ = run_worker([limited, response()])
    assert result['attempt_details'][0]['http_status'] == 429 and len(bodies) == 2
    unauthorized = urllib.error.HTTPError('unused', 401, 'Unauthorized', {}, io.BytesIO(b'key=EXAMPLE_SECRET'))
    with patch.dict(os.environ, {'BFE_TEST_KEY': 'EXAMPLE_SECRET'}):
        result, bodies, worker = run_worker([unauthorized])
    assert result['fatal'] and len(bodies) == 1 and worker.stop.is_set()
    assert 'EXAMPLE_SECRET' not in json.dumps(result)
    assert worker(REQ)['cancelled']

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        req2 = dict(REQ, trace_id='b')
        write_jsonl(t / 'requests.jsonl', [REQ, req2])
        old = dict(trace_id='a', request_sha256=REQ['request_sha256'], **GOOD,
                   teacher_model='TEST', annotation_run_id='TEST', old_provenance='KEEP')
        write_jsonl(t / 'old.jsonl', [old])
        write_json(t / 'old.jsonl.meta.json', dict(requests_sha256=file_hash(t / 'requests.jsonl'),
                  settings=dict(model='TEST', run_id='TEST', disable_thinking=True, temperature=0)))
        write_json(t / 'pipeline.json', dict(annotations=str(t / 'old.jsonl'),
                  teacher=dict(model='TEST', run_id='TEST', endpoint='KEEP', max_tokens=100, disable_thinking=True)))
        with patch.object(sys, 'argv', ['activate', '--config', str(t / 'pipeline.json')]), contextlib.redirect_stdout(io.StringIO()):
            activate()
        cfg = json.loads((t / 'pipeline.json').read_text())
        assert cfg['teacher']['workers'] == 32 and cfg['teacher']['model'] == 'TEST'
        assert cfg['teacher']['run_id'] == 'TEST' and cfg['annotation_import_successes'] == [str(t / 'old.jsonl')]
        assert (t / 'old.jsonl').is_file()
        with patch.object(sys, 'argv', ['activate_context', '--config', str(t / 'pipeline.json')]), contextlib.redirect_stdout(io.StringIO()):
            activate_context()
            snapshot = (t / 'pipeline.json').read_bytes()
            activate_context()
            assert snapshot == (t / 'pipeline.json').read_bytes()
        cfg = json.loads((t / 'pipeline.json').read_text())
        assert cfg['teacher']['max_tokens'] == cfg['teacher']['max_retry_tokens'] == 100
        assert cfg['teacher']['run_id'] == 'TEST' and cfg['teacher']['workers'] == 32
        assert cfg['annotation_import_successes'] == [str(t / 'old.jsonl')]
        argv = ['annotate', '--requests', str(t / 'requests.jsonl'), '--output', cfg['annotations'],
                '--endpoint', 'http://unused.invalid/v1/chat/completions', '--model', 'TEST', '--run-id', 'TEST',
                '--workers', '1', '--disable-thinking', '--response-format', 'json_object',
                '--import-successes', str(t / 'old.jsonl')]
        with patch.object(sys, 'argv', argv), patch('urllib.request.urlopen', return_value=response()) as transport, contextlib.redirect_stdout(io.StringIO()):
            annotate_litellm.main()
            assert transport.call_count == 1
        saved = list(read_jsonl(cfg['annotations']))
        assert len(saved) == 2 and saved[0]['old_provenance'] == 'KEEP'
        assert saved[0]['labels'] == old['labels'] and saved[0]['imported_from']['sha256'] == file_hash(t / 'old.jsonl')
        argv[argv.index('--workers') + 1] = '32'
        with patch.object(sys, 'argv', argv), patch('urllib.request.urlopen', side_effect=AssertionError('must use cache')) as transport, contextlib.redirect_stdout(io.StringIO()):
            annotate_litellm.main()
            assert transport.call_count == 0
    print('PASS: strict Pydantic schema, corrective retry, JSON modes, bounded output budget, context-overflow recovery/isolation, 429 backoff, fatal-stop, redaction, old-success import, worker-change resume. No network/GPU calls.')


if __name__ == '__main__':
    main()
