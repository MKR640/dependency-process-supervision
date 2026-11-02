"""Enable the optional backend thinking switch; do not contact the service."""
import json
from datetime import datetime
from pathlib import Path

p = Path(__file__).resolve().parent / 'pipeline.json'
c = json.loads(p.read_text(encoding='utf-8'))
teacher = c['teacher']
old = Path(c['annotations'])
if teacher.get('disable_thinking') is True and old.name.startswith('annotations_nothink_'):
    print('Non-thinking condition already configured:', old)
else:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = p.with_name(p.name + '.' + stamp + '.bak')
    backup.write_bytes(p.read_bytes())
    teacher['disable_thinking'] = True
    teacher['run_id'] = (teacher.get('run_id') or 'bfe-annotation') + '-nothink-' + stamp
    c['annotations'] = str(old.with_name('annotations_nothink_' + stamp + '.jsonl'))
    p.write_text(json.dumps(c, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Configuration backed up:', backup)
    print('New annotation output:', c['annotations'])
print('Request will include chat_template_kwargs.enable_thinking=false.')
print('Backend support is required; run one probe first. No API call made.')
