#!/usr/bin/env python3
"""Repair local endpoint/config; optionally probe ONE real request on the server.

Default invocation makes no network call. --probe explicitly sends one request.
Never deletes existing annotation results. No API key is printed.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).resolve().parent / 'pipeline.json'))
    p.add_argument('--probe', action='store_true')
    p.add_argument('--timeout', type=float, default=180)
    a = p.parse_args()
    path = Path(a.config)
    cfg = json.loads(path.read_text(encoding='utf-8'))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    teacher = cfg['teacher']
    old = str(teacher.get('endpoint', ''))
    # Construct without a literal hyperlink to avoid copy/paste link formatting.
    endpoint = 'http' + '://' + '127.0.0.1' + ':4000' + '/v1/chat/completions'
    parts = urlsplit(endpoint)
    print('Old endpoint contains square brackets:', '[' in old or ']' in old)
    print('Old endpoint equals expected local URL:', old == endpoint)
    changed = old != endpoint
    if changed:
        backup = path.with_name(path.name + '.' + stamp + '.bak')
        backup.write_bytes(path.read_bytes())
        teacher['endpoint'] = endpoint
        # Endpoint belongs to annotation cache identity: keep old data untouched.
        output = Path(cfg['annotations'])
        cfg['annotations'] = str(output.with_name('annotations_urlfix_' + stamp + '.jsonl'))
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('Configuration repaired; backup:', backup)
        print('New output file:', cfg['annotations'])
    else:
        print('URL already correct; configuration/output unchanged.')
    print('Verified scheme:', parts.scheme)
    print('Verified host:', parts.hostname)
    print('Verified port:', parts.port)
    print('Verified path:', parts.path)
    if not a.probe:
        print('No network call made. Add --probe to diagnose ONE request.')
        return
    req_path = Path(cfg['requests'])
    with req_path.open(encoding='utf-8') as f:
        req = next((json.loads(line) for line in f if line.strip()), None)
    if req is None:
        raise ValueError('Empty requests file')
    body = dict(model=teacher['model'], temperature=0, max_tokens=teacher['max_tokens'],
                messages=[dict(role='system', content=req['system']),
                          dict(role='user', content=json.dumps(req['payload'], ensure_ascii=False))])
    if teacher.get('json_mode'):
        body['response_format'] = dict(type='json_object')
    if teacher.get('response_format') == 'json_object':
        body['response_format'] = dict(type='json_object')
    elif teacher.get('response_format') == 'json_schema':
        from annotation_schema import schema_for_steps
        body['response_format'] = dict(type='json_schema', json_schema=dict(name='bfe_step_labels', strict=True,
            schema=schema_for_steps(len(req['payload']['steps']))))
    if teacher.get('disable_thinking'):
        body['chat_template_kwargs'] = {'enable_thinking': False}
    key = os.environ.get(teacher.get('api_key_env', 'LITELLM_API_KEY'))
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    print('API key environment variable is set:', bool(key))
    print('Sending ONE request. Timeout seconds:', a.timeout, flush=True)
    diagnostic = dict(trace_id=req.get('trace_id'), expected_steps=len(req['payload']['steps']),
                      requested_disable_thinking=bool(teacher.get('disable_thinking')))
    def preview(text):
        value = str(text)
        if key:
            value = value.replace(key, '[REDACTED]')
        return value[:1200]
    try:
        call = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(call, timeout=a.timeout) as response:
            diagnostic['http_status'] = response.status
            diagnostic['content_type'] = response.headers.get('Content-Type')
            raw = response.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as e:
        diagnostic.update(stage='http_error', http_status=e.code,
                          response_preview=preview(e.read().decode('utf-8', errors='replace')))
    except Exception as e:
        diagnostic.update(stage='connection_error', error_type=type(e).__name__, error=preview(e))
    else:
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as e:
            diagnostic.update(stage='http_body_not_json', error=str(e), response_preview=preview(raw))
        else:
            try:
                choice = envelope['choices'][0]
                msg = choice['message']
                content = msg.get('content')
                diagnostic.update(finish_reason=choice.get('finish_reason'), usage=envelope.get('usage'),
                    content_type_in_message=type(content).__name__,
                    content_chars=len(content) if isinstance(content, str) else None,
                    has_reasoning_content=bool(msg.get('reasoning_content')),
                    content_preview=preview(content))
                if choice.get('finish_reason') != 'stop':
                    diagnostic['stage'] = 'incomplete_generation'
                elif not isinstance(content, str):
                    diagnostic['stage'] = 'missing_string_content'
                else:
                    from annotate_litellm import parse_content
                    try:
                        result = parse_content(content)
                    except (ValueError, TypeError) as e:
                        diagnostic.update(stage='model_content_not_json', error=str(e))
                    else:
                        from annotation import validate_result
                        from annotation_schema import validate_output
                        try:
                            validate_output(result, len(req['payload']['steps']))
                            validate_result(req, dict(trace_id=req['trace_id'], request_sha256=req['request_sha256'],
                                labels=result['labels'], reasons=result['reasons'], teacher_model=teacher['model'],
                                annotation_run_id=teacher['run_id']))
                        except Exception as e:
                            diagnostic.update(stage='annotation_schema_invalid', error_type=type(e).__name__, error=preview(e))
                        else:
                            diagnostic['stage'] = 'success'
            except (KeyError, IndexError, TypeError) as e:
                diagnostic.update(stage='chat_envelope_invalid', error=preview(e), response_preview=preview(raw))
    target = path.parent / ('annotation_diagnostic_' + stamp + '.json')
    target.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    print('Diagnostic saved:', target)
    print('Probe result is diagnostic only; annotation output was not appended.')


if __name__ == '__main__':
    main()
