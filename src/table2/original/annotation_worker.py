"""Bounded worker with typed validation, corrective feedback and transport backoff."""
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pydantic import ValidationError
from annotation import validate_result
from annotation_schema import validate_output, schema_for_steps
from common import digest


class CompletionError(ValueError):
    pass


def is_context_overflow(status, detail):
    text = detail.lower()
    return status in (400, 413, 422) and any(marker in text for marker in (
        'context_length_exceeded', 'maximum context length', "model's context length",
        'exceeds the context', 'exceed the context', 'max_model_len',
        'maximum input length', 'too many tokens'))


class AnnotationWorker:
    def __init__(self, args, settings, parse_content, stop_event=None):
        self.args, self.settings, self.parse_content = args, settings, parse_content
        self.stop = stop_event or threading.Event()

    def __call__(self, req):
        a = self.args
        key = os.environ.get(a.api_key_env)
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        n = len(req['payload']['steps'])
        original_messages = [dict(role='system', content=req['system']),
                             dict(role='user', content=json.dumps(req['payload'], ensure_ascii=False))]
        correction = None
        prior_content = None
        budget = a.max_tokens
        events, usage = [], []
        started = time.monotonic()

        def safe(text):
            text = str(text)
            if key:
                text = text.replace(key, '[REDACTED]')
            return text[:1500]

        for attempt in range(a.retries):
            if self.stop.is_set():
                return dict(trace_id=req['trace_id'], cancelled=True)
            payload = dict(model=a.model, temperature=0, max_tokens=budget, messages=list(original_messages))
            if correction:
                if isinstance(prior_content, str) and prior_content:
                    payload['messages'].append(dict(role='assistant', content=prior_content[:6000]))
                payload['messages'].append(dict(role='user', content=(
                    f'The previous response failed output validation: {correction}\n'
                    f'Return a COMPLETE replacement JSON object with exactly {n} integer 0/1 labels and '
                    f'exactly {n} nonempty string reasons, one per original step in order. '
                    'Use only keys labels and reasons. No Markdown or analysis outside JSON. '
                    'Re-evaluate the supplied steps if needed; do not omit steps or invent labels to fill gaps.')))
            mode = self.settings['response_format']
            if mode == 'json_object':
                payload['response_format'] = {'type': 'json_object'}
            elif mode == 'json_schema':
                payload['response_format'] = dict(type='json_schema', json_schema=dict(
                    name='bfe_step_labels', strict=True, schema=schema_for_steps(n)))
            if a.disable_thinking:
                payload['chat_template_kwargs'] = {'enable_thinking': False}
            stage = 'transport'
            event = dict(attempt=attempt + 1, max_tokens=budget,
                         feedback_included=bool(correction), request_body_sha256=digest(payload))
            delay = min(2 ** attempt, 30) + random.uniform(0, .5)
            fatal = False
            try:
                request = urllib.request.Request(a.endpoint, data=json.dumps(payload).encode(), headers=headers)
                with urllib.request.urlopen(request, timeout=a.timeout) as response:
                    event['http_status'] = getattr(response, 'status', 200)
                    hdr = getattr(response, 'headers', {})
                    event['backend_id'] = hdr.get('x-litellm-model-id') or hdr.get('model_id')
                    stage = 'http_body_json'
                    data = json.loads(response.read())
                stage = 'chat_envelope'
                choice = data['choices'][0]
                event.update(finish_reason=choice.get('finish_reason'), usage=data.get('usage'),
                             response_id=data.get('id'))
                usage.append(data.get('usage'))
                prior_content = choice['message'].get('content')
                event['content_chars'] = len(prior_content) if isinstance(prior_content, str) else None
                stage = 'generation'
                if choice.get('finish_reason') != 'stop':
                    raise CompletionError('finish_reason=' + str(choice.get('finish_reason')))
                stage = 'model_json'
                body = self.parse_content(prior_content)
                stage = 'schema'
                parsed = validate_output(body, n)
                result = dict(trace_id=req['trace_id'], request_sha256=req['request_sha256'],
                    **parsed.model_dump(), teacher_model=a.model, served_model=data.get('model'),
                    annotation_run_id=a.run_id, attempts=attempt + 1, attempt_usage=usage,
                    failed_attempts=[e['error_type'] for e in events if 'error_type' in e],
                    latency_seconds=time.monotonic() - started, response_id=data.get('id'),
                    settings_sha256=digest(self.settings), validator='pydantic_v2_strict',
                    annotation_protocol='bfe_pydantic_context_v2')
                validate_result(req, result)
                event['stage'] = 'success'; events.append(event)
                result['attempt_details'] = events
                return result
            except urllib.error.HTTPError as exc:
                stage = 'http_error'
                raw_detail = exc.read().decode('utf-8', errors='replace')
                event.update(http_status=exc.code, error_type=type(exc).__name__,
                             detail=safe(raw_detail))
                # Invalid auth/model/unsupported schema should not hit 40k tasks.
                fatal = exc.code in (400, 401, 403, 404, 422)
                if is_context_overflow(exc.code, raw_detail):
                    stage = 'context_budget'
                    fatal = False
                    # The server tokenizer is authoritative. Retry with less output
                    # and no repair history, preserving ALL original input steps.
                    budget = max(1, budget // 2)
                    correction = prior_content = None
                    event['next_max_tokens'] = budget
                if exc.code == 429:
                    try:
                        delay = min(60., max(delay, float(exc.headers.get('Retry-After', 0))))
                    except (ValueError, TypeError):
                        pass
            except Exception as exc:
                event['error_type'] = type(exc).__name__
                if isinstance(exc, ValidationError):
                    details = exc.errors(include_url=False, include_input=False, include_context=False)
                    event['validation_errors'] = details[:10]
                    detail = '; '.join(f'{e["loc"]}: {e["msg"]}' for e in details[:10])
                else:
                    detail = str(exc)
                event['detail'] = safe(detail)
                if stage in ('model_json', 'schema', 'generation'):
                    correction = safe(detail)
                if stage == 'generation' and event.get('finish_reason') == 'length':
                    # A completion cap is not the total context window. Never
                    # double it blindly, even with a legacy max_retry_tokens value.
                    prior_content = None
                    correction = 'Output was truncated. Keep each reason concise while preserving every step.'
                # HTTP-envelope JSON errors are transport/proxy faults, not label
                # validation feedback. Retry with the last valid request intent.
            event['stage'] = stage
            event['fatal'] = fatal
            events.append(event)
            if fatal:
                self.stop.set()
            if fatal or attempt + 1 == a.retries:
                return dict(trace_id=req['trace_id'], request_sha256=req['request_sha256'], failed=True,
                    fatal=fatal, errors=[e.get('error_type') for e in events], attempt_details=events,
                    attempt_usage=usage, latency_seconds=time.monotonic() - started)
            time.sleep(delay)
