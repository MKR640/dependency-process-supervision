"""Offline tests for topology adaptation and launch command construction."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from launch_multinode import adapt_config, main

cfg = dict(world_size=8, batch_per_device=1, gradient_accumulation_steps=4,
           output_root='/shared/results', arms=['M1-BFE-R2-weighted'], seeds=[42])
adapted = adapt_config(cfg, 2, 8)
assert adapted['world_size'] == 16 and adapted['gradient_accumulation_steps'] == 2
assert cfg['world_size'] == 8 and cfg['gradient_accumulation_steps'] == 4
assert adapt_config(cfg, 4, 8)['gradient_accumulation_steps'] == 1
try:
    adapt_config(cfg, 3, 8)
except ValueError:
    pass
else:
    raise AssertionError('Nonintegral global batch accepted')
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'config.json'
    path.write_text(json.dumps(cfg), encoding='utf-8')
    for rank in (0, 1):
        argv = ['launch', 'M1-BFE-R2-weighted', '42', '--config', str(path),
                '--node-rank', str(rank), '--master-addr', '10.0.0.10', '--interface', 'eth0', '--dry-run', '--smoke']
        output = io.StringIO()
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output), patch('subprocess.call', side_effect=AssertionError('dry-run launched process')):
            main()
        text = output.getvalue()
        assert '--node_rank ' + str(rank) in text and '--nnodes 2' in text
        assert '--nproc_per_node 8' in text and '--smoke' in text
        assert '"global_batch": 32' in text and '--standalone' not in text
    for flag, value in [('--node-rank', '2'), ('--master-addr', '127.0.0.1'), ('--interface', 'lo')]:
        bad = list(argv)
        bad[bad.index(flag)+1] = value
        with patch.object(sys, 'argv', bad), contextlib.redirect_stderr(io.StringIO()):
            try:
                main()
            except SystemExit as e:
                assert e.code != 0
            else:
                raise AssertionError('Invalid launch accepted')
print('PASS: 2x8/4x8 batch preservation, invalid topology/loopback rejection, both node commands, dry-run without launch. No GPU/network calls.')
