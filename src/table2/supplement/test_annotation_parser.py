"""Offline regression for local reasoning-model final JSON parsing; no API calls."""
import json
from annotate_litellm import parse_content

gold = {'labels': [1, 0, 1], 'reasons': ['a', 'b', 'c']}
text = json.dumps(gold)
for content in (
    text,
    '```json\n' + text + '\n```',
    '<think>Some analysis</think>\n' + text,
    'Analysis without opening tag\n</think>\n' + text,
    'Analysis with example {"labels":[0]}\n</think>\n```json\n' + text + '\n```',
    '<THINK>analysis</THINK>\n' + text,
):
    assert parse_content(content) == gold
for bad in (
    '<think>' + text,
    'Analysis ' + text,
    '</think>{"labels": [1',
    '</think>```json\n' + text,
    '</think>' + text + '\n' + text,
    '</think>[]',
    None,
):
    try:
        parse_content(bad)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError('Accepted malformed/ambiguous output')
print('PASS: plain/fenced JSON, full/closing-only think tags, no analysis-label extraction, truncated/ambiguous output rejected. No API calls.')
