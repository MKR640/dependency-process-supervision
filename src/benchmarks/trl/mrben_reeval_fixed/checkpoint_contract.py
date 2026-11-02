"""Checkpoint contract for the supplied 2/3/8-head hierarchical PRM.

Only reads safetensors headers during preflight (no GPU / tensor materialization).
The separate Transformers loading-info check is still mandatory at inference.
"""
import argparse
import json
import struct
from pathlib import Path

EXPECTED_MAPS = {
    'correctness_id2label': ['ERROR', 'NON_ERROR'],
    'role_id2label': ['correct', 'new_error', 'propagated_error'],
    'error_type_id2label': ['calculation', 'derivation_or_logic', 'method_or_formula',
                          'problem_understanding', 'assumption_or_condition',
                          'answer_mismatch', 'missing_evidence', 'unit'],
}


def check_config(config):
    evidence = {}
    for field, expected in EXPECTED_MAPS.items():
        value = config.get(field)
        if value:
            actual = [value.get(str(i), value.get(i)) for i in range(len(expected))]
            if actual != expected:
                raise ValueError(f'{field}: expected {expected}, got {actual}; do not guess label order')
            evidence[field] = 'verified in config.json'
        else:
            evidence[field] = 'supplied training-script convention (map absent in config)'
    for key, n in [('correctness_num_labels', 2), ('role_num_labels', 3), ('error_type_num_labels', 8)]:
        if key in config and config[key] != n:
            raise ValueError(f'{key} must be {n}, got {config[key]}')
    return evidence


def check_loading_info(info):
    bad = {k: info.get(k, []) for k in
           ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs') if info.get(k)}
    if bad:
        raise RuntimeError('Checkpoint not fully loaded; inference aborted: ' + json.dumps(bad, default=str))
    return {'verified': True, 'missing_keys': [], 'unexpected_keys': [],
            'mismatched_keys': [], 'error_msgs': []}


def inspect_checkpoint(model_path):
    root = Path(model_path)
    cfg = json.loads((root / 'config.json').read_text())
    maps = check_config(cfg)
    index_path = root / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map'] if index_path.exists() else None
    names = sorted(set(index.values())) if index else ['model.safetensors']
    tensors = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'{path}: this preflight requires HF safetensors checkpoints')
        with path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            if n <= 0 or n > 100_000_000:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(f.read(n))
        for key, val in header.items():
            if key == '__metadata__':
                continue
            if key in tensors:
                raise ValueError('Duplicate tensor: ' + key)
            if index and index.get(key) != name:
                raise ValueError('Index/header mismatch: ' + key)
            tensors[key] = val['shape']
    if index and set(index) != set(tensors):
        raise ValueError('Index keys do not match shard keys')
    hidden = int(cfg['hidden_size'])
    expected = {}
    for head, n in [('correctness_head', 2), ('role_head', 3), ('error_type_head', 8)]:
        expected[head + '.weight'] = [n, hidden]
        expected[head + '.bias'] = [n]
    for key, shape in expected.items():
        if tensors.get(key) != shape:
            raise ValueError(f'{key}: expected {shape}, got {tensors.get(key)}')
    if any(k.startswith('error_role_head.') for k in tensors):
        raise ValueError('This bundle targets role_head (3 classes), not error_role_head (2 classes)')
    if not any(k.startswith('backbone.') for k in tensors):
        raise ValueError('No backbone.* tensors found')
    return {'model_path': str(root.resolve()), 'head_shapes': expected, 'label_maps': maps,
            'tensor_count': len(tensors), 'header_check_passed': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    a = p.parse_args()
    print(json.dumps(inspect_checkpoint(a.model_path), ensure_ascii=False, indent=2))
