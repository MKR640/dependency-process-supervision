"""One arm/seed per 8-card node. Uses the uploaded model architecture, new matched protocol."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from common import digest, error_ap, file_hash, pack, write_json

from expanded_protocol import ARMS, head_policy, targets as supervision_targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--arm', choices=list(ARMS), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--smoke', action='store_true', help='Two updates in separate output; never formal results')
    p.add_argument('--resume', action='store_true', help='Resume latest checkpoint with identical identity')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.arm not in cfg['arms'] or a.seed not in cfg['seeds']:
        p.error('Arm/seed not enabled in configuration')
    family, regime = ARMS[a.arm]
    use_role, use_type = head_policy(family)
    data_root = Path(cfg['data_dir'])
    manifest = json.loads((data_root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if file_hash(data_root / name) != expected:
            raise ValueError('Frozen data changed: ' + name)
    init = Path(cfg['init_model'])
    if not init.is_dir():
        raise ValueError('Missing initialization: ' + str(init))
    if cfg['init_kind'] not in ('prm', 'base'):
        raise ValueError('init_kind must be prm or base')
    source_config = json.loads((init / 'config.json').read_text())
    if cfg['init_kind'] == 'prm':
        mapping = {str(v).upper() for v in source_config.get('id2label', {}).values()}
        if 'ERROR' not in mapping or not ({'CORRECT', 'NON_ERROR'} & mapping):
            raise ValueError('Initialization needs explicit ERROR/CORRECT class semantics; no guessed row IDs')
    init_files = [f for f in sorted(init.iterdir()) if f.is_file() and f.suffix in ('.json', '.safetensors', '.bin')]
    if not any(f.suffix in ('.safetensors', '.bin') for f in init_files):
        raise ValueError('No initialization model weights')
    # Each rank hashes one shard; gather below avoids all 8 reading all shards.
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != cfg['world_size'] and not (a.smoke and world == 1):
        raise ValueError(f'Expected {cfg["world_size"]} processes; got {world}')
    run_id = f'{a.arm}/seed-{a.seed}' + ('/smoke' if a.smoke else '/formal')
    output = Path(cfg['output_root']) / run_id
    import numpy as np
    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from losses import weighted_ce_per_row

    source = Path(__file__).parent / 'vendor/original_model.py'
    spec = importlib.util.spec_from_file_location('bfe_original_model', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    settings = dict(
        output_dir=str(output), per_device_train_batch_size=cfg['batch_per_device'],
        per_device_eval_batch_size=cfg['eval_batch_per_device'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        num_train_epochs=cfg['epochs'], max_steps=2 if a.smoke else -1,
        learning_rate=cfg['learning_rate'], warmup_ratio=cfg['warmup_ratio'],
        lr_scheduler_type='cosine', weight_decay=cfg['weight_decay'], max_grad_norm=1.0,
        eval_strategy='steps', eval_steps=1 if a.smoke else cfg['eval_steps'],
        save_strategy='steps', save_steps=1 if a.smoke else cfg['eval_steps'],
        save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model='error_ap', greater_is_better=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=cfg['bf16'], fp16=False, optim='adamw_torch',
        logging_steps=1 if a.smoke else 20, report_to='none', seed=a.seed, data_seed=a.seed,
        remove_unused_columns=False, label_names=['correctness_labels', 'role_labels', 'error_type_labels'],
        ddp_find_unused_parameters=False, ddp_broadcast_buffers=False, dataloader_num_workers=0,
        dataloader_drop_last=False, eval_accumulation_steps=16, save_safetensors=True,
    )
    args = TrainingArguments(**settings)
    rank = args.process_index
    own_hashes = {str(f.name): file_hash(f) for i, f in enumerate(init_files) if i % world == rank}
    if dist.is_initialized():
        hashes = [None] * world
        dist.all_gather_object(hashes, own_hashes)
        init_hashes = {k: v for h in hashes for k, v in h.items()}
    else:
        init_hashes = own_hashes
    code_hashes = {str(f.relative_to(Path(__file__).parent)): file_hash(f)
                   for f in sorted(Path(__file__).parent.rglob('*.py'))}
    identity = dict(arm=a.arm, seed=a.seed, smoke=a.smoke, config=cfg,
                    data_manifest_sha256=file_hash(data_root / 'manifest.json'),
                    initialization_sha256=digest(init_hashes), initialization_files=init_hashes,
                    code_sha256=digest(code_hashes), training_arguments=settings)
    identity_hash = digest(identity)
    resume = None
    if a.resume:
        if not (output / 'run_identity.json').is_file():
            raise ValueError('No run identity to resume')
        old = json.loads((output / 'run_identity.json').read_text())
        if old['identity_hash'] != identity_hash:
            raise ValueError('Cannot resume changed config/data/initialization/code')
        resume = get_last_checkpoint(str(output))
        if not resume:
            raise ValueError('No resumable checkpoint')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Output already exists; use --resume or a new output_root')
    if dist.is_initialized():
        dist.barrier()  # All ranks finish the existence check before rank 0 writes.
    if args.should_save:
        write_json(output / 'run_identity.json', dict(identity_hash=identity_hash, **identity))
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(str(init), use_fast=True, local_files_only=True,
                                        trust_remote_code=cfg['trust_remote_code'])
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError('Tokenizer needs PAD or EOS')
        tok.pad_token = tok.eos_token
    with args.main_process_first(desc='load/tokenize shared data'):
        ds = load_dataset('json', data_files={s: str(data_root / (s + '.jsonl')) for s in ('train', 'validation', 'test')})
        train = ds['train']
        if regime == 'R1':
            indices = json.loads((data_root / 'enriched_indices.json').read_text())
            train = train.select(indices)
        if a.smoke:
            train = train.select(range(min(128, len(train))))
            # Include both classes for AP, independent of model predictions.
            for split in ('validation', 'test'):
                targets = ds[split]['correctness']
                indices = sorted(set(list(range(min(32, len(targets)))) +
                                     [targets.index(k) for k in (0, 1) if k in targets]))
                ds[split] = ds[split].select(indices)
        def encode(row, is_train=False):
            ids, trunc = pack(tok, row, cfg['max_length'], cfg['prefix_policy'])
            corr, role, typ = supervision_targets(row, family, is_train)
            return dict(input_ids=ids, correctness_labels=corr,
                        role_labels=role, error_type_labels=typ, **trunc)
        encoded = {'train': train.map(lambda r: encode(r, True), remove_columns=train.column_names,
                                     desc='Encode training arm')}
        for split in ('validation', 'test'):
            encoded[split] = ds[split].map(encode, remove_columns=ds[split].column_names, desc='Encode ' + split)
    truncation = {}
    for split, dataset in list(encoded.items()):
        names = ['problem_tokens_dropped', 'current_tokens_dropped', 'prefix_steps_dropped']
        truncation[split] = {name: dict(sum=int(sum(dataset[name])), affected_rows=int(sum(x > 0 for x in dataset[name]))) for name in names}
        truncation[split]['input_tokens'] = sum(len(x) for x in dataset['input_ids'])
        truncation[split]['supervised_correctness_rows'] = sum(x != -100 for x in dataset['correctness_labels'])
        truncation[split]['supervised_role_rows'] = sum(x != -100 for x in dataset['role_labels'])
        truncation[split]['supervised_type_rows'] = sum(x != -100 for x in dataset['error_type_labels'])
        encoded[split] = dataset.remove_columns(names)

    class StepModel(module.HierarchicalPRMModel):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            pos = attention_mask.sum(-1) - 1
            bi = torch.arange(input_ids.shape[0], device=input_ids.device)
            return module.HierarchicalPRMOutput(correctness_logits=out.correctness_logits[bi, pos],
                role_logits=out.role_logits[bi, pos], error_type_logits=out.error_type_logits[bi, pos])

    model = StepModel.from_source_checkpoint(str(init), cfg['init_kind'] == 'prm',
                classifier_dropout=cfg['classifier_dropout'], trust_remote_code=cfg['trust_remote_code'],
                attn_implementation='eager', torch_dtype=torch.bfloat16 if cfg['bf16'] else torch.float32)
    model.config.use_cache = False
    model.config.pad_token_id = tok.pad_token_id
    active_heads = ['correctness'] + (['role'] if use_role else []) + (['type'] if use_type else [])
    model.role_head.requires_grad_(use_role)
    model.error_type_head.requires_grad_(use_type)
    model.config.bfe_active_heads = active_heads
    model.config.bfe_run_identity = identity_hash
    def collate(features):
        batch = tok.pad([dict(input_ids=f['input_ids']) for f in features], return_tensors='pt')
        for name in args.label_names:
            batch[name] = torch.tensor([f[name] for f in features], dtype=torch.long)
        return batch

    corr_weights = manifest['weights']['correctness'] if regime == 'R2' else None
    role_weights = manifest['weights']['role']
    type_weights = manifest['weights']['error_type']

    class MatchedTrainer(Trainer):
        def __init__(self, **kw):
            super().__init__(**kw)
            # compute_loss owns the microbatch mean; Trainer handles grad accumulation.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = [inputs[n] for n in args.label_names]
            out = model(**{k: v for k, v in inputs.items() if k not in args.label_names})
            loss = out.correctness_logits.sum() * 0
            for logits, target, w, coefficient in zip(
                [out.correctness_logits, out.role_logits, out.error_type_logits], labels,
                [corr_weights, role_weights, type_weights], [1., .5 if use_role else 0., .5 if use_type else 0.]):
                # Divide by number of input rows, NOT sum of weights/eligible targets.
                # This preserves weighting at microbatch=1 and fixes a global objective
                # across DDP ranks, masks, heads and accumulation microbatches.
                loss = loss + coefficient * weighted_ce_per_row(logits, target, w)
            out.loss = loss
            return (loss, out) if return_outputs else loss

    def metrics(pred):
        logits = np.asarray(pred.predictions[0], dtype=np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(axis=-1, keepdims=True)
        labels = np.asarray(pred.label_ids[0])
        positive, predicted = labels == 0, probs[:, 0] >= 0.5
        tp = int((positive & predicted).sum()); fp = int((~positive & predicted).sum()); fn = int((positive & ~predicted).sum())
        return dict(error_ap=error_ap(labels.tolist(), probs[:, 0].tolist()),
                    error_f1=2 * tp / max(1, 2 * tp + fp + fn), accuracy=float((positive == predicted).mean()))
    trainer = MatchedTrainer(model=model, args=args, processing_class=tok, data_collator=collate,
        train_dataset=encoded['train'], eval_dataset=encoded['validation'], compute_metrics=metrics)
    if args.should_save:
        write_json(output / 'protocol.json', dict(active_heads=active_heads, loss_reduction='weighted sum / input rows',
            correctness_weights=corr_weights, role_weights=role_weights if use_role else None,
            type_weights=type_weights if use_type else None, prefix_policy=cfg['prefix_policy'],
            max_length=cfg['max_length'], selection='shared validation Error AP', truncation=truncation,
            global_batch=world*cfg['batch_per_device']*cfg['gradient_accumulation_steps'],
            software=dict(torch=torch.__version__, transformers=__import__('transformers').__version__,
                          datasets=__import__('datasets').__version__),
            initialization_history=cfg.get('initialization_history', 'not independently audited')))
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / 'final'))  # load_best_model_at_end => best validation checkpoint
    trainer.save_state()
    if args.should_save:
        tok.save_pretrained(str(output / 'final'))
        write_json(output / 'train_results.json', result.metrics)
    for split in ('validation', 'test'):
        prediction = trainer.predict(encoded[split], metric_key_prefix=split)
        if args.should_save:
            write_json(output / (split + '_metrics.json'), prediction.metrics)
            with (output / (split + '_predictions.jsonl')).open('w', encoding='utf-8') as f:
                for i, row in enumerate(ds[split]):
                    logits = np.asarray(prediction.predictions[0][i], dtype=np.float64)
                    prob = np.exp(logits - logits.max()); prob /= prob.sum()
                    record = dict(trace_id=row['trace_id'], step_index=row['step_index'], problem_id=row['problem_id'],
                                  p_error=float(prob[0]), reference_correctness=row['correctness'])
                    for enabled, name, j in [(use_role, 'role_probabilities', 1), (use_type, 'type_probabilities', 2)]:
                        if enabled:
                            z = np.asarray(prediction.predictions[j][i], dtype=np.float64)
                            z = np.exp(z - z.max()); z /= z.sum(); record[name] = z.tolist()
                    f.write(json.dumps(record) + '\n')
    if args.should_save:
        write_json(output / 'COMPLETED.json', dict(arm=a.arm, seed=a.seed, smoke=a.smoke,
            identity_hash=identity_hash, best_checkpoint=trainer.state.best_model_checkpoint,
            best_error_ap=trainer.state.best_metric, global_step=trainer.state.global_step,
            artifact='final', active_heads=active_heads))


if __name__ == '__main__':
    main()
