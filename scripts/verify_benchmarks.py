"""Recompute physics statistics and ProcessBench decoding metrics on the CPU.

Uses saved predictions and scoring functions; models are not loaded.
"""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import zipfile


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def definition(path, name, class_name=None):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    func = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.Module(body=[func], type_ignores=[])
    namespace = {'List': list, 'Dict': dict}
    exec(compile(ast.fix_missing_locations(code), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True, type=Path)
    p.add_argument('--records', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1] / 'src/benchmarks'
    pb = root / 'prm/ProcessBench/processbench_3model_hier_v2'
    phys = root / 'trl/mrben_reeval_fixed'
    source_files = {f.relative_to(root).as_posix(): {
        'bytes': f.stat().st_size, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
        for f in root.rglob('*') if f.is_file()}
    for f in root.rglob('*.py'):
        ast.parse(f.read_text(encoding='utf-8-sig'))
    metric_function = definition(pb/'main.py', 'calculate_metrics')
    resolve_mapping = definition(pb/'prm_eval/qwen3_math_prm.py', '_resolve_score_mapping', 'Qwen3MathPRM')
    base = load_module('recovered_pb_base', pb/'prm_eval/base_prm.py')
    checks, pb_metrics, physics_differences = {}, {}, {}
    mapping = {}
    for label, count, labels in [('binary', 2, {'0': 'ERROR', '1': 'CORRECT'}),
                                  ('three_class', 3, {'0': 'ERROR', '1': 'NEUTRAL', '2': 'CORRECT'})]:
        instance = SimpleNamespace(error_label_ids=None, non_error_label_ids=None, NON_ERROR_LABEL=1,
                                   model=SimpleNamespace(config=SimpleNamespace(id2label=labels)))
        resolve_mapping(instance, count)
        mapping[label] = {'mode': instance.score_mode, 'label_ids': instance.score_label_ids}
    checks['pb_binary_mapping'] = mapping['binary'] == {'mode': 'sum_non_error', 'label_ids': [1]}
    checks['pb_three_class_mapping'] = mapping['three_class'] == {'mode': 'one_minus_error', 'label_ids': [0]}
    with zipfile.ZipFile(args.predictions) as z, zipfile.ZipFile(args.records) as records:
        def rows(name):
            return [json.loads(line) for line in z.read(name).splitlines() if line.strip()]
        for name in z.namelist():
            if name.startswith('processbench/') and name.endswith('/all_results.jsonl'):
                rr = rows(name)
                key = name[len('processbench/'):].removesuffix('/all_results.jsonl')
                guesses = [base.BasePRM.scores_to_prediction(SimpleNamespace(threshold=.5), r['step_scores']) for r in rr]
                checks['pb_source_predictions:' + key] = all(
                    steps == r['step_predictions'] and prediction == r['prediction']
                    for r, (steps, prediction) in zip(rr, guesses))
                actual = metric_function(rr)
                expected = json.loads(records.read(name.removesuffix('all_results.jsonl') + 'metrics.json'))
                checks['pb_source_metrics:' + key] = actual == expected
                pb_metrics[key] = actual
        # Restore exact cached data/metadata in a disposable, task-local directory.
        # The source evaluator requires an empty output and validates cache hashes.
        with tempfile.TemporaryDirectory(prefix='benchmark-cpu-', dir=args.output.parent) as td:
            work = Path(td)
            paths = ['data/all.jsonl'] + [f'predictions/{n}.all.jsonl' for n in
                    ['Binary_PRM', 'Three_class_v2', 'Full_hierarchy']]
            for name in paths:
                target = work/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read('physics_fixed/' + name))
                Path(str(target)+'.meta.json').write_bytes(records.read('physics_fixed/' + name + '.meta.json'))
            sys.path.insert(0, str(phys))
            evaluator = load_module('recovered_physics_evaluator', phys/'evaluate_all.py')
            previous_argv = sys.argv
            sys.argv = [str(phys/'evaluate_all.py'), '--reference', str(work/'data/all.jsonl'),
                        '--output-dir', str(work/'results'), '--baseline', 'Three_class_v2',
                        '--threshold', '.5', '--bootstrap-samples', '2000', '--seed', '20260924']
            for n in ['Binary_PRM', 'Three_class_v2', 'Full_hierarchy']:
                sys.argv.extend(['--prediction', f'{n}={work}/predictions/{n}.all.jsonl'])
            log = io.StringIO()
            try:
                with redirect_stdout(log):
                    evaluator.main()
            finally:
                sys.argv = previous_argv
            actual = json.loads((work/'results/metrics.json').read_text(encoding='utf-8'))
            expected = json.loads(records.read('physics_fixed/overall_evaluation/metrics.json'))
            checks['physics_reference_exact'] = actual['reference'] == expected['reference']
            checks['physics_bootstrap_settings'] = actual['bootstrap'] == expected['bootstrap']
            expected_rows = {r['model']: r for r in expected['results']}
            for row in actual['results']:
                old = expected_rows[row['model']]
                checks['physics_counts:' + row['model']] = row['counts'] == old['counts']
                for metric, item in row['metrics'].items():
                    if item != old['metrics'][metric]:
                        physics_differences[row['model']+':'+metric] = {'recorded': old['metrics'][metric], 'recomputed': item}
            physics_log = log.getvalue().splitlines()
    candidates = {}
    # Use the implementation associated with these benchmark outputs.
    for variant in ['processbench_3model_hier_v2']:
        path = root / 'prm/ProcessBench' / variant / 'prm_eval/qwen3_math_prm.py'
        content = path.read_text(encoding='utf-8-sig')
        candidates[variant] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'has_cached_conditional_role_field': 'p_new_given_error_role' in content,
                                'has_cached_joint_raw_field': 'p_joint_first_error_raw' in content}
    report = {
        'checked_on': '2026-11-02', 'source_files': source_files,
        'all_python_syntax_valid': True, 'checks_total': len(checks),
        'failed_checks': {k: v for k, v in checks.items() if not v},
        'checks': checks, 'processbench_source_metrics': pb_metrics,
        'processbench_flat_mapping': mapping, 'processbench_candidate_signatures': candidates,
        'processbench_selected_reproduction_source': 'prm/ProcessBench/processbench_3model_hier_v2',
        'physics_source': 'trl/mrben_reeval_fixed',
        'physics_metric_or_interval_differences': physics_differences,
        'physics_replay_log': physics_log,
        'selection_evidence': 'hier_v2 uniquely matches cached auxiliary score fields among supplied candidates; source BasePRM decoder/metrics reproduce all twelve cached model/subset records. Exact historical source bytes were not independently logged.',
        'source_shell_note': 'Original ProcessBench launch shells delete their chosen output directory and include threshold sweeps. They were not executed. A release entry must use fresh outputs and separate fixed-threshold reproduction from optional searches.',
        'dependencies': {'processbench_inference': ['torch', 'transformers', 'datasets', 'tqdm'],
                         'physics_inference': ['torch', 'transformers', 'safetensors', 'numpy'],
                         'physics_cpu_statistics': ['numpy'], 'cache_reanalysis': ['numpy']},
        'execution_scope': 'Only CPU metric/decoder functions and physics cached-data statistics entry executed. No tensor payloads, model constructors, training or inference shells executed.',
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['checks_total', 'failed_checks', 'physics_metric_or_interval_differences',
                                           'processbench_flat_mapping', 'processbench_candidate_signatures']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
