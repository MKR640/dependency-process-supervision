"""Offline platform environment and torchrun handoff test; no GPUs needed."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import launch_multinode

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    cfg = dict(world_size=8, batch_per_device=1, gradient_accumulation_steps=4,
               output_root='/shared/results', arms=['M1-BFE-R2-weighted'], seeds=[42],
               init_model='/models/init', data_dir='/data/frozen')
    config_path = root / 'experiments_existing.json'
    config_path.write_text(json.dumps(cfg), encoding='utf-8')
    for rank in ('0', '1'):
        environment = dict(TQ_GPU_NUM='8', WORLD_SIZE='2', RANK=rank,
                           MASTER_ADDR='10.0.0.10', MASTER_PORT='29600',
                           NCCL_SOCKET_IFNAME='ib0', MCCL_IB_DISABLE='0')
        with patch.dict(os.environ, environment, clear=True), patch.object(sys, 'argv', ['launch', 'M1-BFE-R2-weighted', '42']), patch.object(sys, 'platform', 'linux'), patch.object(launch_multinode, '__file__', str(root / 'launch_multinode.py')), patch.object(Path, 'is_dir', return_value=True), patch('subprocess.call', return_value=0) as call, contextlib.redirect_stdout(io.StringIO()):
            try:
                launch_multinode.main()
            except SystemExit as e:
                assert e.code == 0
            command = call.call_args.args[0]
            assert command[0] == 'torchrun'
            for flag, value in [('--nnodes', '2'), ('--nproc_per_node', '8'), ('--node_rank', rank), ('--master_addr', '10.0.0.10'), ('--master_port', '29600')]:
                assert command[command.index(flag)+1] == value
            passed_env = call.call_args.kwargs['env']
            assert passed_env['NCCL_SOCKET_IFNAME'] == 'ib0' and passed_env['MCCL_IB_DISABLE'] == '0'
            generated = json.loads(Path(command[command.index('--config')+1]).read_text())
            assert generated['world_size'] == 16 and generated['gradient_accumulation_steps'] == 2
            assert passed_env['WORLD_SIZE'] == '2'  # torchrun assigns worker WORLD_SIZE=16
            if rank == '0':
                first = generated
            else:
                assert generated == first
print('PASS: platform env -> torchrun for both nodes, identical configs, batch32, network settings preserved. Mocked launch only.')
