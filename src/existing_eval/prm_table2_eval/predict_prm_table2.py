#!/usr/bin/env python3
"""Shared prefix-only inference for hierarchical and HF token-classification PRMs.
No gold labels, graph edges, references, notes or future steps enter model inputs.
"""
import argparse
import json
import os
from pathlib import Path
from prepare_prm_table2_data import (ROLES, TYPES, digest, file_hash, key, read_jsonl,
                                    validate_rows, write_json, write_jsonl)


def pack(tokenizer, row, max_length, policy):
    sep = tokenizer.encode('\n\n', add_special_tokens=False)
    if not sep:
        raise ValueError('Empty separator tokens')
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    if max_length < len(bos) + len(sep) + 1:
        raise ValueError('max-length too small')
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    raw_p, raw_c = enc(row['problem']), enc(row['current_step'])
    p, c = raw_p + sep, raw_c + sep
    p_drop = c_drop = 0
    if len(bos) + len(p) + len(c) > max_length:
        avail_p = max_length - len(bos) - len(c)
        if avail_p <= len(sep):
            keep = max_length - len(bos) - len(sep)
            c = raw_c[-keep:] + sep
            c_drop, p_drop, p = max(0, len(raw_c)-keep), len(raw_p), []
        else:
            keep = avail_p - len(sep)
            p = raw_p[:keep] + sep
            p_drop = max(0, len(raw_p)-keep)
    remaining = max_length - len(bos) - len(p) - len(c)
    pieces = []
    for text in reversed(row['prefix']):
        ids = enc(text) + sep
        if len(ids) > remaining:
            if policy == 'contiguous':
                break
            continue
        pieces.append(ids)
        remaining -= len(ids)
    pieces.reverse()
    ids = bos + p + [t for piece in pieces for t in piece] + c
    if not ids or len(ids) > max_length:
        raise ValueError('Invalid packed input')
    return ids, dict(problem_tokens_dropped=p_drop, current_tokens_dropped=c_drop,
        prefix_steps_dropped=len(row['prefix'])-len(pieces))


def resolve_ids(mapping, names):
    mapping = {int(k): str(v).lower() for k, v in mapping.items()}
    result = []
    for aliases in names:
        found = [i for i, label in mapping.items() if label in aliases]
        if len(found) != 1:
            raise ValueError(f'Cannot resolve {aliases} in {mapping}; use explicit class IDs for generic correctness labels')
        result.append(found[0])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--model-kind', choices=['hierarchical', 'token_classifier'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--active-heads', default='correctness', help='Explicit trained heads only: correctness[,role,type]. Zero-loss/random heads must be omitted.')
    p.add_argument('--error-id', type=int)
    p.add_argument('--correct-id', type=int)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--prefix-policy', choices=['contiguous', 'legacy_skip'], default='contiguous')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--device', default='cuda')
    p.add_argument('--trust-remote-code', action='store_true')
    p.add_argument('--run-info', help='Optional JSON describing TRAINING seed, data hashes, initialization, updates, losses, etc.')
    a = p.parse_args()
    heads = a.active_heads.split(',')
    if len(set(heads)) != len(heads) or 'correctness' not in heads or set(heads)-{'correctness','role','type'}:
        raise ValueError('Invalid --active-heads')
    if a.model_kind == 'token_classifier' and heads != ['correctness']:
        raise ValueError('Flat token classifier cannot expose role/type heads')
    if (a.error_id is None) != (a.correct_id is None):
        raise ValueError('Supply both --error-id and --correct-id, or neither')
    if a.batch_size < 1:
        raise ValueError('batch-size must be positive')
    rows = list(read_jsonl(a.reference))
    validate_rows(rows)
    ref_meta = json.loads(Path(a.reference + '.meta.json').read_text())
    if file_hash(a.reference) != ref_meta['reference_sha256']:
        raise ValueError('Reference hash mismatch')
    # Shard without changing order or inputs. Launch with torchrun if desired.
    rank, world = int(os.getenv('RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    shard = rows[rank::world]
    import torch
    import torch.nn as nn
    from types import SimpleNamespace
    from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer, PreTrainedModel

    class HierarchicalModel(PreTrainedModel):
        _supports_sdpa = True
        def __init__(self, config):
            super().__init__(config)
            self.backbone = AutoModel.from_config(config, trust_remote_code=a.trust_remote_code)
            self.dropout = nn.Dropout(float(getattr(config, 'hierarchical_prm_classifier_dropout', 0.1)))
            self.correctness_head = nn.Linear(config.hidden_size, 2)
            self.role_head = nn.Linear(config.hidden_size, 3)
            self.error_type_head = nn.Linear(config.hidden_size, 8)
        def forward(self, input_ids=None, attention_mask=None):
            h = self.dropout(self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                          use_cache=False, return_dict=True).last_hidden_state)
            return SimpleNamespace(correctness_logits=self.correctness_head(h),
                role_logits=self.role_head(h), error_type_logits=self.error_type_head(h))

    kw = dict(local_files_only=True, trust_remote_code=a.trust_remote_code)
    config = AutoConfig.from_pretrained(a.model, **kw)
    cls = HierarchicalModel if a.model_kind == 'hierarchical' else AutoModelForTokenClassification
    model, info = cls.from_pretrained(a.model, config=config, torch_dtype=getattr(torch, a.dtype),
                                     output_loading_info=True, **kw)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(f'Checkpoint does not match architecture: {info}')
    device = f'cuda:{local_rank}' if a.device == 'cuda' else a.device
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, **kw)
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError('No PAD/EOS token')
        tok.pad_token = tok.eos_token
    corr_map = getattr(config, 'correctness_id2label', {}) if a.model_kind == 'hierarchical' else config.id2label
    if a.error_id is None:
        ei, ci = resolve_ids(corr_map, [{'error'}, {'correct','non_error'}])
    else:
        ei, ci = a.error_id, a.correct_id
    nlabels = 2 if a.model_kind == 'hierarchical' else config.num_labels
    if ei == ci or not all(0 <= i < nlabels for i in (ei,ci)):
        raise ValueError('Invalid correctness IDs')
    role_ids = resolve_ids(getattr(config, 'role_id2label', {}), [{r} for r in ROLES]) if 'role' in heads else None
    type_ids = resolve_ids(getattr(config, 'error_type_id2label', {}), [{t} for t in TYPES]) if 'type' in heads else None
    common = dict(schema='prm_table2_predictions_v1', name=a.name, reference_sha256=file_hash(a.reference),
        split=ref_meta['split'], model=str(Path(a.model).resolve()), model_kind=a.model_kind,
        active_heads=heads, correctness_ids=[ei, ci], role_order=ROLES, type_order=TYPES,
        correctness_definition='softmax(ERROR,CORRECT); neutral excluded when present',
        tokenizer_sha256=digest(dict(vocab=tok.get_vocab(), special_tokens=tok.special_tokens_map)),
        packing=dict(max_length=a.max_length, prefix_policy=a.prefix_policy, separator='\n\n', target='last_separator_token'),
        model_config_sha256=digest(config.to_dict()),
        training_run_info=json.loads(Path(a.run_info).read_text()) if a.run_info else None,
        checkpoint_files=[dict(name=str(x.relative_to(a.model)), size=x.stat().st_size, mtime_ns=x.stat().st_mtime_ns)
                          for x in sorted(Path(a.model).rglob('*')) if x.is_file() and x.suffix in ('.json','.bin','.safetensors')],
        software=dict(torch=torch.__version__, transformers=__import__('transformers').__version__),
        world_size=world, rank=rank)
    def predict():
        for start in range(0, len(shard), a.batch_size):
            batch = shard[start:start+a.batch_size]
            packed = [pack(tok, r, a.max_length, a.prefix_policy) for r in batch]
            tokens = tok.pad([dict(input_ids=x[0]) for x in packed], padding=True, return_tensors='pt')
            tokens = {k: v.to(device) for k,v in tokens.items() if k in ('input_ids','attention_mask')}
            with torch.inference_mode():
                out = model(**tokens)
                pos = tokens['attention_mask'].sum(1) - 1
                bi = torch.arange(len(batch), device=device)
                logits = out.correctness_logits if a.model_kind == 'hierarchical' else out.logits
                err = logits[bi,pos][:,[ei,ci]].float().softmax(-1)[:,0].cpu().tolist()
                rp = out.role_logits[bi,pos][:,role_ids].float().softmax(-1).cpu().tolist() if role_ids else None
                tp = out.error_type_logits[bi,pos][:,type_ids].float().softmax(-1).cpu().tolist() if type_ids else None
            for i,r in enumerate(batch):
                yield dict(trace_id=r['trace_id'], step_index=r['step_index'], p_error=err[i],
                    role_probs=rp[i] if rp else None, type_probs=tp[i] if tp else None,
                    input_sha256=digest(packed[i][0]), truncation=packed[i][1])
            print(f'[{a.name} rank {rank}] {min(start+a.batch_size,len(shard))}/{len(shard)}', flush=True)
    dest = a.output if world == 1 else a.output + f'.rank{rank}.jsonl'
    write_jsonl(dest, predict())
    common['predictions_sha256'] = file_hash(dest)
    write_json(dest+'.meta.json', common)
    print('Saved', dest)


if __name__ == '__main__':
    main()
