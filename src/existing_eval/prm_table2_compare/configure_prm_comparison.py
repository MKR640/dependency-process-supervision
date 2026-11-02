#!/usr/bin/env python3
"""Validate local checkpoint configs before launching Table 2 comparisons (stdlib only)."""
import argparse
import json
from pathlib import Path
from prepare_prm_table2_data import ROLES, TYPES, write_json
from predict_prm_table2 import resolve_ids


def label_map(config, field):
    raw = config.get(field) or {}
    return {int(k): str(v) for k, v in raw.items()}


def inspect_model(item, root, overrides):
    path = (root / item['directory']).resolve()
    cfg_path = path / 'config.json'
    if not cfg_path.is_file():
        raise ValueError(f'Missing config.json: {cfg_path}. Point to the actual saved checkpoint directory.')
    cfg = json.loads(cfg_path.read_text())
    kind = item['kind']
    arch = cfg.get('architectures') or []
    if kind == 'token_classifier' and not any('ForTokenClassification' in x for x in arch):
        raise ValueError(f'{path}: expected a token-classification PRM; architectures={arch}')
    field = 'correctness_id2label' if kind == 'hierarchical' else 'id2label'
    mapping = label_map(cfg, field)
    expected = item['expected_labels']
    if kind == 'token_classifier':
        n = cfg.get('num_labels', len(mapping))
        if n != expected or (mapping and set(mapping) != set(range(expected))):
            raise ValueError(f'{path}: expected {expected} classification outputs; num_labels={n}, map={mapping}')
    elif mapping and set(mapping) != {0, 1}:
        raise ValueError(f'{path}: expected 2 correctness outputs; map={mapping}')
    override = overrides.get(item['name'])
    inferred = None
    try:
        inferred = resolve_ids(mapping, [{'error'}, {'correct', 'non_error'}])
    except ValueError:
        pass
    if override is not None:
        ids = [override.get('error_id'), override.get('correct_id')]
        if any(type(x) is not int or not 0 <= x < expected for x in ids) or ids[0] == ids[1]:
            raise ValueError(f"{item['name']}: invalid explicit error_id/correct_id")
        if inferred is not None and ids != inferred:
            raise ValueError(f"{item['name']}: explicit IDs contradict semantic checkpoint labels")
    elif inferred is not None:
        ids = inferred
    else:
        raise ValueError(f"{item['name']}: cannot determine ERROR/CORRECT from {mapping}. "
                         'Set CLASS_IDS_JSON to a JSON file of {model_name: {error_id: integer, correct_id: integer}} '
                         'using the actual training label definitions. No generic label order is assumed.')
    heads = item['active_heads'].split(',')
    if kind == 'token_classifier' and heads != ['correctness']:
        raise ValueError('Token classifiers cannot supply role/type metrics')
    if kind == 'hierarchical':
        if 'role' in heads:
            resolve_ids(label_map(cfg, 'role_id2label'), [{r} for r in ROLES])
        if 'type' in heads:
            resolve_ids(label_map(cfg, 'error_type_id2label'), [{t} for t in TYPES])
    weights = [p for p in path.iterdir() if p.is_file() and p.suffix in ('.safetensors', '.bin')]
    if not weights:
        raise ValueError(f'{path}: no .safetensors/.bin weights at checkpoint root')
    result = dict(name=item['name'], enabled=True, kind=kind, path=str(path),
                  active_heads=item['active_heads'], error_id=ids[0], correct_id=ids[1],
                  comparison_scope='existing_checkpoint_comparison', training_matching_verified=False,
                  note=item.get('note', ''))
    inspection = dict(name=item['name'], path=str(path), architectures=arch, correctness_map=mapping,
                      error_id=ids[0], correct_id=ids[1], active_heads=heads,
                      neutral_excluded=kind == 'token_classifier' and expected == 3,
                      note='Head training status follows the catalog; confirm against training records. '
                           'Directory names and config fields do not establish matched supervision.')
    return result, inspection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', required=True)
    p.add_argument('--model-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--models', help='Comma-separated catalog names; default is all enabled entries')
    p.add_argument('--class-ids-json')
    a = p.parse_args()
    catalog = json.loads(Path(a.catalog).read_text())
    names = [x['name'] for x in catalog]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate model names in catalog')
    if a.models:
        selected = a.models.split(',')
        if len(selected) != len(set(selected)) or set(selected) - set(names):
            raise ValueError(f'Invalid/duplicate selected models: {selected}; available: {names}')
        chosen = [next(x for x in catalog if x['name'] == n) for n in selected]
    else:
        chosen = [x for x in catalog if x.get('enabled', True)]
    if not chosen:
        raise ValueError('No selected models')
    overrides = json.loads(Path(a.class_ids_json).read_text()) if a.class_ids_json else {}
    if not isinstance(overrides, dict) or set(overrides) - set(names):
        raise ValueError('CLASS_IDS_JSON must map known catalog names to ID definitions')
    results, checks, errors = [], [], []
    for item in chosen:
        try:
            result, check = inspect_model(item, Path(a.model_root), overrides)
            results.append(result)
            checks.append(check)
            print(f"OK {item['name']}: ERROR={result['error_id']} CORRECT={result['correct_id']} "
                  f"heads={result['active_heads']}", flush=True)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            errors.append(dict(name=item['name'], error=str(exc)))
    write_json(str(a.output)+'.inspection.json', dict(models=checks, errors=errors,
        comparison_scope='existing_checkpoint_comparison', training_matching_verified=False))
    if errors:
        for error in errors:
            print(f"BLOCKED {error['name']}: {error['error']}", flush=True)
        raise SystemExit('Checkpoint preflight failed; no GPU inference started; model config not rewritten.')
    write_json(a.output, results)
    print('Saved', a.output)


if __name__ == '__main__':
    main()
