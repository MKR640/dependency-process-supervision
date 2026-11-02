"""Upgrade annotation settings; preserve old files and import validated successes."""
import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).resolve().parent / 'pipeline.json'))
    p.add_argument('--workers', type=int, default=32)
    a = p.parse_args()
    if a.workers < 1:
        p.error('--workers must be positive')
    path = Path(a.config)
    cfg = json.loads(path.read_text(encoding='utf-8'))
    teacher = cfg['teacher']
    if teacher.get('client_protocol') == 'pydantic_feedback_v1':
        print('Already activated; no new output created. Override concurrency with run_annotation.sh --workers N.')
        return
    old = Path(cfg['annotations'])
    sources = []
    if old.is_file() and old.stat().st_size:
        if not Path(str(old) + '.meta.json').is_file():
            raise ValueError('Previous output has no metadata; cannot safely import it')
        sources.append(str(old))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path.with_name(path.name + '.' + stamp + '.bak').write_bytes(path.read_bytes())
    teacher.update(workers=a.workers, retries=5, response_format='json_object', json_mode=False,
                   max_retry_tokens=int(teacher['max_tokens']), client_protocol='pydantic_feedback_v1')
    # Keep model, run_id, thinking mode and original task prompt unchanged.
    cfg['annotations'] = str(old.with_name('annotations_pydantic_' + stamp + '.jsonl'))
    cfg['annotation_import_successes'] = sources
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('New output:', cfg['annotations'])
    print('Prior successes to validate/import:', sources)
    print('Workers:', a.workers, '| attempts:', teacher['retries'], '| response format:', teacher['response_format'])
    print('Original files untouched. No API call made. Do not keep an old annotation process writing the import source.')


if __name__ == '__main__':
    main()
