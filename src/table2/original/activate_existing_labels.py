"""Write separate no-teacher pipeline/experiment configs, preserving originals."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
pipeline_path = root / 'pipeline_existing.json'
experiment_path = root / 'experiments_existing.json'
if pipeline_path.exists() or experiment_path.exists():
    raise SystemExit('Existing-mode configs already exist; reuse them or inspect before replacing.')
pipeline = json.loads((root / 'pipeline.json').read_text(encoding='utf-8'))
experiments = json.loads((root / 'experiments.json').read_text(encoding='utf-8'))
pipeline['label_source'] = 'existing'
pipeline['frozen_data_dir'] = pipeline['frozen_data_dir'].rstrip('/') + '_existing_labels_v1'
experiments['data_dir'] = pipeline['frozen_data_dir']
experiments['output_root'] = experiments['output_root'].rstrip('/') + '_existing_labels_v1'
experiments['seeds'] = [42]
experiments['arms'] = ['M1-BFE-R2-weighted', 'M5-full-R2-weighted', 'M0-first-error-R2-weighted']
for path, data in [(pipeline_path, pipeline), (experiment_path, experiments)]:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Wrote:', path)
print('Check init_model and output_root are accessible on your server:', experiments['init_model'], experiments['output_root'])
print('No API calls. Existing annotation files and original configs untouched.')
