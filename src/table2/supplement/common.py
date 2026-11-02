"""Pure-Python shared protocol and deterministic utilities."""
import hashlib
import json
import math
import re
from pathlib import Path

ARMS = {
    'M1-BFE-R0-natural': ('bfe', 'R0'),
    'M1-BFE-R1-enriched': ('bfe', 'R1'),
    'M1-BFE-R2-weighted': ('bfe', 'R2'),
    'M5-full-R0-natural': ('full', 'R0'),
    'M5-full-R1-enriched': ('full', 'R1'),
    'M5-full-R2-weighted': ('full', 'R2'),
    'M2-correctness-R2-weighted': ('correctness', 'R2'),
    'M0-first-error-R2-weighted': ('first_error', 'R2'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('expected object')
                    yield row
                except Exception as e:
                    raise ValueError(f'{path}:{n}: {e}') from e


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def problem_hash(text):
    return digest(re.sub(r'\s+', ' ', text).strip())


def weights(counts, cap=4.0):
    # Shared TRAIN reference only. Absent classes have zero weight and no targets.
    raw = [1 / math.sqrt(c) if c else 0.0 for c in counts]
    mean = sum(raw) / max(1, sum(c > 0 for c in counts))
    return [min(cap, w / mean) if mean else 0.0 for w in raw]


def error_ap(labels, scores):
    """Tie-aware, non-interpolated AP; labels are 0=error, 1=correct."""
    pairs = sorted(zip(scores, labels), reverse=True)
    positives = sum(y == 0 for _, y in pairs)
    if not positives or positives == len(pairs):
        raise ValueError('Validation must contain both correctness classes')
    tp = seen = 0
    result = 0.0
    i = 0
    while i < len(pairs):
        j = i
        hits = 0
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            hits += pairs[j][1] == 0
            j += 1
        tp += hits
        seen += j - i
        result += hits / positives * tp / seen
        i = j
    return result


def pack(tokenizer, row, max_length, policy='contiguous'):
    # Same input protocol as supplied Table 2 inference. No labels or future steps.
    if policy not in ('contiguous', 'legacy_skip'):
        raise ValueError('Invalid prefix policy')
    sep = tokenizer.encode('\n\n', add_special_tokens=False)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    if not sep or max_length < len(bos) + len(sep) + 1:
        raise ValueError('Invalid separator/length')
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    raw_p, raw_c = enc(row['problem']), enc(row['current_step'])
    p, c = raw_p + sep, raw_c + sep
    p_drop = c_drop = 0
    if len(bos) + len(p) + len(c) > max_length:
        avail = max_length - len(bos) - len(c)
        if avail <= len(sep):
            keep = max_length - len(bos) - len(sep)
            c = raw_c[-keep:] + sep
            c_drop, p_drop, p = max(0, len(raw_c) - keep), len(raw_p), []
        else:
            keep = avail - len(sep)
            p = raw_p[:keep] + sep
            p_drop = max(0, len(raw_p) - keep)
    remaining = max_length - len(bos) - len(p) - len(c)
    pieces = []
    for text in reversed(row['prefix']):
        piece = enc(text) + sep
        if len(piece) > remaining:
            if policy == 'contiguous':
                break
            continue
        pieces.append(piece)
        remaining -= len(piece)
    pieces.reverse()
    ids = bos + p + [t for piece in pieces for t in piece] + c
    return ids, {'problem_tokens_dropped': p_drop, 'current_tokens_dropped': c_drop,
                 'prefix_steps_dropped': len(row['prefix']) - len(pieces)}
