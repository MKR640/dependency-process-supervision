"""CPU regression tests; no checkpoint required. Run: python3 -m unittest -v test_prm_table2.py"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.metrics import balanced_accuracy_score, average_precision_score
import prepare_prm_table2_data as data
import evaluate_prm_table2 as ev
from predict_prm_table2 import pack


def make_rows(prefix='p'):
    result=[]
    # Two traces of SAME problem: they must be one bootstrap group.
    for tid,question,roles in [('t1',prefix+'Q',['correct','new_error','propagated_error']),
                               ('t2',prefix+'Q',['correct','new_error']),
                               ('t3',prefix+'R',['correct','correct'])]:
        past=[]
        for i,role in enumerate(roles):
            cur=f'Step{i+1}: {tid} text'
            result.append(dict(trace_id=tid,step_index=i,problem=question,problem_id=data.problem_key(question),
                prefix=list(past),current_step=cur,y_error=int(role!='correct'),role=role,
                error_type='calculation' if role=='new_error' else None,audit_flags=[]))
            past.append(cur)
    return result


def fake_pred(rows,name,path,refmeta,role=True,shift=0.):
    preds=[]
    for r in rows:
        probs=[.01,.01,.01]; probs[data.ROLES.index(r['role'])]=.98
        preds.append(dict(trace_id=r['trace_id'],step_index=r['step_index'],
            p_error=float(np.clip((.8 if r['y_error'] else .2)+shift,0,1)),
            role_probs=probs if role else None,type_probs=None,
            input_sha256=data.digest([r['problem'],r['prefix'],r['current_step']]),
            truncation=dict(problem_tokens_dropped=0,current_tokens_dropped=0,prefix_steps_dropped=0)))
    meta=dict(name=name, reference_sha256=refmeta['reference_sha256'],split=refmeta['split'],
        model='/fake/'+name,model_kind='hierarchical',active_heads=['correctness','role'] if role else ['correctness'],
        correctness_ids=[0,1],role_order=data.ROLES,type_order=data.TYPES,
        tokenizer_sha256='fake',packing={'max_length':4096},model_config_sha256='fake',checkpoint_files=[])
    data.write_jsonl(path,preds)
    meta['predictions_sha256']=data.file_hash(path)
    data.write_json(str(path)+'.meta.json',meta)
    return ev.load_predictions(str(path),rows,refmeta,name)


class Tests(unittest.TestCase):
    def test_incomplete_and_wrong_prefix_fail(self):
        rows=make_rows(); data.validate_rows(rows)
        with self.assertRaises(ValueError): data.validate_rows(rows[1:])
        rows[1]['prefix']=[]
        with self.assertRaises(ValueError): data.validate_rows(rows)

    def test_contradictory_role_fails(self):
        rows=make_rows(); rows[0]['y_error']=1
        with self.assertRaises(ValueError): data.validate_rows(rows)

    def test_threshold_matches_bruteforce_with_ties(self):
        rng=np.random.default_rng(24)
        for _ in range(20):
            y=rng.integers(0,2,100); p=np.round(rng.uniform(size=100),2)
            t,ba=ev.choose_threshold(y,p)
            want=max(balanced_accuracy_score(y,p>=t) for t in list(np.unique(p))+[.5])
            self.assertAlmostEqual(ba,want)
            self.assertAlmostEqual(ba,balanced_accuracy_score(y,p>=t))

    def test_unknown_type_kept(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.jsonl'
            r=dict(id='t',context={'problem':'Q'},audit={'status':'review'},steps=[
                dict(id='s1',text='Wrong',prm_label='I',role='new_error',error_type='other')])
            data.write_jsonl(p,[r]); rows,stats=data.normalize(p,'numbered')
            self.assertEqual(rows[0]['y_error'],1)
            self.assertEqual(rows[0]['role'],'new_error')
            self.assertIsNone(rows[0]['error_type'])
            self.assertEqual(stats['flagged_traces'],1)

    def test_perfect_and_absent_heads(self):
        rows=make_rows()
        with tempfile.TemporaryDirectory() as d:
            meta=dict(reference_sha256='ref',split='test')
            pred=fake_pred(rows,'m',Path(d)/'p',meta)
            e=ev.Evaluation(rows,pred,.5)
            m=e.metrics()
            self.assertEqual(len(e.problems),2)
            for k in ev.MAIN: self.assertAlmostEqual(m[k],1.)
            flat=fake_pred(rows,'f',Path(d)/'f',meta,role=False)
            self.assertIsNone(ev.Evaluation(rows,flat,.5).metrics()['new_error_ap'])
            self.assertIsNone(ev.Evaluation(rows,flat,.5).metrics()['role_macro_f1'])

    def test_first_error_is_not_new_error_recall(self):
        rows=make_rows()
        with tempfile.TemporaryDirectory() as d:
            pred=fake_pred(rows,'m',Path(d)/'p',dict(reference_sha256='r',split='test'))
            pred['p_error'][0]=.9  # falsely flag earlier correct step on t1
            m=ev.Evaluation(rows,pred,.5).metrics()
            self.assertAlmostEqual(m['first_error_accuracy'],2/3)
            self.assertAlmostEqual(m['first_error_accuracy_error_traces'],.5)
            self.assertAlmostEqual(m['new_error_recall'],1.)

    def test_weighted_ap_matches_duplicated_clusters(self):
        y=np.array([0,1,0,1]); p=np.array([.2,.3,.8,.7]); w=np.array([2,2,1,1])
        self.assertAlmostEqual(ev.ap(y,p,w),average_precision_score(np.repeat(y,w),np.repeat(p,w)))

    def test_contiguous_does_not_skip_near_step(self):
        class Tokenizer:
            bos_token_id=None
            def encode(self,text,**kwargs): return [ord(x) for x in text]
        row=dict(problem='Q',current_step='C',prefix=['a','longlonglonglong'])
        ids,c=pack(Tokenizer(),row,10,'contiguous')
        legacy,l=pack(Tokenizer(),row,10,'legacy_skip')
        self.assertEqual(c['prefix_steps_dropped'],2)
        self.assertEqual(l['prefix_steps_dropped'],1)
        self.assertNotEqual(ids,legacy)

    def test_end_to_end_fit_table_and_leakage_rejection(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            d=Path(d); specs={}
            for split in ['validation','test']:
                rows=make_rows(split)
                path=d/(split+'.jsonl'); data.write_jsonl(path,rows)
                meta=dict(reference_sha256=data.file_hash(path),split=split)
                data.write_json(str(path)+'.meta.json',meta)
                specs[split]=[]
                for name,shift in [('base',0),('full',.05)]:
                    pp=d/(name+'.'+split+'.jsonl'); fake_pred(rows,name,pp,meta,shift=shift)
                    specs[split].append(name+'='+str(pp))
            policy=d/'policy.json'
            ev.fit(SimpleNamespace(reference=str(d/'validation.jsonl'),prediction=specs['validation'],output=str(policy)))
            args=SimpleNamespace(reference=str(d/'test.jsonl'),prediction=specs['test'],baseline='base',policy=str(policy),
                                 threshold=.5,bootstrap_samples=20,seed=42,output_dir=str(d/'out'))
            ev.table(args)
            result=json.loads((d/'out'/'metrics.json').read_text())
            self.assertEqual(result['models']['full']['paired_difference_vs_baseline']['error_ap']['delta'],0)
            self.assertTrue((d/'out'/'table2.csv').exists())
            bad=json.loads(policy.read_text()); bad['validation_problem_ids']=[data.problem_key('testQ')]
            data.write_json(policy,bad)
            with self.assertRaisesRegex(ValueError,'overlap'): ev.table(args)

    def test_hash_tampering_and_inputs_rejected(self):
        rows=make_rows()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p'; meta=dict(reference_sha256='r',split='test')
            pred=fake_pred(rows,'m',path,meta)
            other=dict(pred,input_hashes=['different']*len(rows))
            with self.assertRaises(ValueError): ev.matched_inputs({'a':pred,'b':other})
            with path.open('a') as f: f.write('\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): ev.load_predictions(str(path),rows,meta,'m')


if __name__=='__main__': unittest.main()
