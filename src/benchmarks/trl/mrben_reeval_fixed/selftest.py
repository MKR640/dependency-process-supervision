#!/usr/bin/env python3
"""CPU tests: metadata gates, role-head contract, complete offline evaluation.
--with-model additionally checks a tiny real Transformers Qwen3 save/reload.
"""
import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from checkpoint_contract import EXPECTED_MAPS, check_config, check_loading_info, inspect_checkpoint
from reference_io import write_json, write_jsonl, file_hash, problem_key
from evaluate_complexity import average_precision, load_prediction
ROOT = Path(__file__).resolve().parent


def must_fail(fn):
    try:
        fn()
    except (ValueError, RuntimeError, FileNotFoundError):
        return
    raise AssertionError('Expected rejection did not occur')


def run(with_model=False):
    assert np.isclose(average_precision([1, 0], [.5, .5], [1, 1]), .5)
    check_loading_info({})
    for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']:
        must_fail(lambda k=key: check_loading_info({k: ['bad_head']}))
    must_fail(lambda: check_config({'role_num_labels': 2}))
    must_fail(lambda: check_config({'role_id2label': {'0':'new_error', '1':'correct', '2':'propagated_error'}}))
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td)
        cfg = {'hidden_size': 4, **{k: {str(i):v for i,v in enumerate(vals)} for k,vals in EXPECTED_MAPS.items()}}
        ck = temp/'checkpoint'; ck.mkdir()
        write_json(ck/'config.json', cfg)
        header = {'backbone.embed_tokens.weight': {'shape':[8,4]}}
        for name, n in [('correctness_head', 2), ('role_head', 3), ('error_type_head', 8)]:
            header[name+'.weight']={'shape':[n,4]}
            header[name+'.bias']={'shape':[n]}
        def save_header():
            raw=json.dumps(header).encode()
            (ck/'model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw)
        save_header(); inspect_checkpoint(ck)
        header['role_head.weight']['shape']=[2,4]; save_header()
        must_fail(lambda: inspect_checkpoint(ck))
        header['role_head.weight']['shape']=[3,4];save_header()
        del header['role_head.bias'];save_header()
        must_fail(lambda: inspect_checkpoint(ck))

        # Four complete trajectories, two distinct problems; all reference roles covered.
        role_sequences=[['correct','correct'],['correct','new_error'],
                        ['new_error','propagated_error'],['correct','correct']]
        flat=[]
        for t, roles in enumerate(role_sequences):
            for i, role in enumerate(roles):
                flat.append({'id':f't{t}', 'problem':f'Question {t//2}',
                             'prefix':['step 0'][:i], 'completions':[f'step {i}'],
                             'step_index':i, 'labels':[role=='correct'], 'prm_label':'C' if role=='correct' else 'I',
                             'role':role, 'error_type':'calculation' if role=='new_error' else None})
        src=temp/'input.jsonl';ref=temp/'all.jsonl'
        write_jsonl(src,flat)
        subprocess.run([sys.executable,str(ROOT/'reference_io.py'),'--input',str(src),'--output',str(ref),'--split','all'],check=True,stdout=subprocess.DEVNULL)
        rows=[json.loads(x) for x in ref.read_text().splitlines()]
        rmeta=json.loads(Path(str(ref)+'.meta.json').read_text())
        cli=[sys.executable,str(ROOT/'evaluate_all.py'),'--reference',str(ref),
             '--output-dir',str(temp/'results'),'--baseline','Three_class_v2','--bootstrap-samples','40']
        for name in ['Binary_PRM','Three_class_v2','Full_hierarchy']:
            pred=[]
            for r in rows:
                role=['correct','new_error','propagated_error'].index(r['role'])
                rp=[.01,.01,.01];rp[role]=.98
                pred.append({'trace_id':r['trace_id'],'step_index':r['step_index'],
                             'p_error':.9 if r['y_error'] else .1,
                             'role_probs':rp if name=='Full_hierarchy' else None,
                             'input_sha256':'0'*64,
                             'truncation':{'problem_tokens_dropped':0,'current_tokens_dropped':0,'prefix_steps_dropped':0}})
            path=temp/(name+'.jsonl');write_jsonl(path,pred)
            meta={'schema':'mrben_three_model_prediction_v3_strict_role3','loading_audit':check_loading_info({}),
                  'name':name,'split':'all','reference_sha256':file_hash(ref),'predictions_sha256':file_hash(path),
                  'active_heads':['correctness','role'] if name=='Full_hierarchy' else ['correctness'],
                  'role_order':['correct','new_error','propagated_error'],'packing':{'fixture':True},'tokenizer_sha256':'0'*64}
            write_json(str(path)+'.meta.json',meta)
            if name=='Full_hierarchy':
                wrong=dict(meta,schema='mrben_three_model_prediction_v2_8gpu')
                write_json(str(path)+'.meta.json',wrong)
                must_fail(lambda:load_prediction(path,rows,rmeta,name))
                write_json(str(path)+'.meta.json',meta)
            cli+=['--prediction',name+'='+str(path)]
        subprocess.run(cli,check=True,stdout=subprocess.DEVNULL)
        result=json.loads((temp/'results/metrics.json').read_text())
        for r in result['results']:
            for k in ['error_ap','error_f1','balanced_accuracy','first_error_accuracy',
                      'first_error_accuracy_error_traces','no_error_trace_accuracy']:
                assert np.isclose(r['metrics'][k]['value'],1.),(r['model'],k)
            if r['model']=='Full_hierarchy':
                assert np.isclose(r['metrics']['role_macro_f1']['value'],1.)
                assert np.isclose(r['metrics']['new_error_ap']['value'],1.)
        assert (temp/'results/diagnostics.md').is_file()
        assert (temp/'results/table_overall.tex').is_file()
        print('PASS: head/config checks, strict loading gate, stale-cache rejection, AP ties, full offline pipeline')

    if with_model:
        import torch
        from transformers import Qwen3Config
        from predict_mrben_model_8gpu import HierarchicalPRMModel
        cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=2,head_dim=8)
        cfg._attn_implementation='eager'
        for key,vals in EXPECTED_MAPS.items():setattr(cfg,key,dict(enumerate(vals)))
        torch.manual_seed(7)
        model=HierarchicalPRMModel(cfg).eval()
        ids=torch.tensor([[1,2,3]])
        with tempfile.TemporaryDirectory() as td:
            model.save_pretrained(td,safe_serialization=True)
            inspect_checkpoint(td)
            restored,info=HierarchicalPRMModel.from_pretrained(td,output_loading_info=True)
            check_loading_info(info);restored.eval()
            with torch.no_grad():
                a=model(input_ids=ids);b=restored(input_ids=ids)
            for field in ['correctness_logits','role_logits','error_type_logits']:
                torch.testing.assert_close(getattr(a,field),getattr(b,field))
            # Corrupt the saved role checkpoint: must reject on missing weight.
            from safetensors.torch import load_file,save_file
            path=Path(td)/'model.safetensors';state=load_file(str(path))
            del state['role_head.bias'];save_file(state,str(path),metadata={'format':'pt'})
            must_fail(lambda:inspect_checkpoint(td))
            _,bad=HierarchicalPRMModel.from_pretrained(td,output_loading_info=True)
            must_fail(lambda:check_loading_info(bad))
        print('PASS: real tiny Qwen3 save/reload predictions and corrupt-checkpoint rejection')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--with-model',action='store_true')
    run(p.parse_args().with_model)
