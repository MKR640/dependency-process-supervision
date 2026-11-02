"""Activate bounded-context retries, preserving prior successes and configuration."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).resolve().parent / 'pipeline.json'))
    a = p.parse_args()
    path = Path(a.config)
    cfg = json.loads(path.read_text(encoding='utf-8'))
    teacher = cfg['teacher']
    if teacher.get('client_protocol') == 'pydantic_context_v2':
        print('Already activated; resume with run_annotation.sh.')
        return
    old = Path(cfg['annotations'])
    sources = list(cfg.get('annotation_import_successes', []))
    if old.is_file() and old.stat().st_size:
        if not Path(str(old) + '.meta.json').is_file():
            raise ValueError('Previous output has no metadata; cannot safely import it')
        sources = [str(old)]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path.with_name(path.name + '.' + stamp + '.bak').write_bytes(path.read_bytes())
    teacher['max_tokens'] = min(16384, int(teacher.get('max_tokens', 16384)))
    teacher['max_retry_tokens'] = teacher['max_tokens']
    teacher['client_protocol'] = 'pydantic_context_v2'
    cfg['annotations'] = str(old.with_name('annotations_contextfix_' + stamp + '.jsonl'))
    cfg['annotation_import_successes'] = sources
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('New output:', cfg['annotations'])
    print('Prior successes to validate/import:', sources)
    print('Output token cap:', teacher['max_tokens'], '| workers:', teacher.get('workers'))
    print('Original outputs untouched. No API call made. Stop old writers before importing.')


if __name__ == '__main__':
    main()
