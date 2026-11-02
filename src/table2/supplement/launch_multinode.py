"""Launch one node of a fixed-size multi-node DDP experiment."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def adapt_config(cfg, nodes, gpus):
    cfg = dict(cfg)
    old_batch = cfg['world_size'] * cfg['batch_per_device'] * cfg['gradient_accumulation_steps']
    world = nodes * gpus
    denominator = world * cfg['batch_per_device']
    if old_batch % denominator or old_batch < denominator:
        raise ValueError(f'Cannot preserve effective batch {old_batch} with {world} devices; choose fewer devices')
    cfg['world_size'] = world
    cfg['gradient_accumulation_steps'] = old_batch // denominator
    cfg['output_root'] = cfg['output_root'].rstrip('/') + f'_ddp_{nodes}x{gpus}'
    return cfg


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('arm')
    p.add_argument('seed', type=int)
    p.add_argument('--config', default=os.environ.get('CONFIG', str(root / 'experiments_existing.json')))
    p.add_argument('--nodes', type=int, default=os.environ.get('WORLD_SIZE', 2))
    p.add_argument('--gpus-per-node', type=int, default=os.environ.get('TQ_GPU_NUM', 8))
    p.add_argument('--node-rank', type=int, default=os.environ.get('RANK'))
    p.add_argument('--master-addr', default=os.environ.get('MASTER_ADDR'), help='Node 0 address reachable from all nodes')
    p.add_argument('--master-port', type=int, default=os.environ.get('MASTER_PORT', 29500))
    p.add_argument('--interface', help='Local inter-node network interface, e.g. eth0; not lo')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.node_rank is None or not a.master_addr:
        p.error('Set RANK/MASTER_ADDR or pass --node-rank/--master-addr')
    if a.nodes < 2 or a.gpus_per_node < 1 or not 0 <= a.node_rank < a.nodes:
        p.error('Require nodes >= 2, gpus >= 1 and 0 <= node-rank < nodes')
    if not 1 <= a.master_port <= 65535:
        p.error('Invalid port')
    if a.interface is not None and (a.interface == 'lo' or not a.interface or any(c in a.interface for c in ',=^ /')):
        p.error('Supply one actual inter-node interface name, not lo or a selector')
    if a.master_addr.lower() == 'localhost':
        p.error('master-addr cannot be localhost')
    try:
        addr = ipaddress.ip_address(a.master_addr)
    except ValueError:
        addr = None
    if addr is not None and (addr.is_loopback or addr.is_unspecified):
        p.error('master-addr must be reachable across nodes')
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8'))
    if a.arm not in cfg['arms'] or a.seed not in cfg['seeds']:
        p.error('Arm/seed not enabled in source config')
    cfg = adapt_config(cfg, a.nodes, a.gpus_per_node)
    # Node-specific files avoid two launchers racing on a shared directory.
    generated = root / 'launch_configs' / f'{a.arm}_seed-{a.seed}_{a.nodes}x{a.gpus_per_node}_node-{a.node_rank}.json'
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    if a.interface:
        env.update(NCCL_SOCKET_IFNAME=a.interface, MCCL_SOCKET_IFNAME=a.interface, GLOO_SOCKET_IFNAME=a.interface)
    for name in ('NCCL_SOCKET_IFNAME', 'MCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME'):
        if env.get(name) in ('lo', '=lo'):
            p.error(f'{name}=lo prevents multi-node communication; unset it or pass --interface')
    env.setdefault('OMP_NUM_THREADS', '4')
    # Preserve cluster-provided RDMA settings; do not copy single-node IB-disable defaults.
    cmd = ['torchrun', '--nnodes', str(a.nodes),
           '--nproc_per_node', str(a.gpus_per_node), '--node_rank', str(a.node_rank),
           '--master_addr', a.master_addr, '--master_port', str(a.master_port),
           '--max-restarts', '0', str(root / 'train.py'), '--config', str(generated),
           '--arm', a.arm, '--seed', str(a.seed)]
    if a.smoke:
        cmd.append('--smoke')
    if a.resume:
        cmd.append('--resume')
    print(json.dumps(dict(world_size=cfg['world_size'], gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        global_batch=cfg['world_size'] * cfg['batch_per_device'] * cfg['gradient_accumulation_steps'],
        output_root=cfg['output_root'], node_rank=a.node_rank, interface=a.interface,
        NCCL_IB_DISABLE=env.get('NCCL_IB_DISABLE', '(runtime default)'),
        MCCL_IB_DISABLE=env.get('MCCL_IB_DISABLE', '(runtime default)')), indent=2), flush=True)
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        print('Preview only; no files written or distributed processes started.')
        return
    if sys.platform != 'linux':
        p.error('Execute on the Linux training nodes')
    if a.interface and not (Path('/sys/class/net') / a.interface).exists():
        p.error('Network interface does not exist on this node')
    for key in ('init_model', 'data_dir'):
        if not Path(cfg[key]).is_dir():
            p.error(f'Missing {key}: {cfg[key]}')
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    raise SystemExit(subprocess.call(cmd, env=env, cwd=root))


if __name__ == '__main__':
    main()
