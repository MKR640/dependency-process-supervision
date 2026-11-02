"""Regression tests for overlap audit and fixed-threshold launch; GPU inference is stubbed."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from audit_problem_overlap import audit
from prepare_prm_table2_data import file_hash, write_json, write_jsonl
from test_prm_table2 import make_rows


class OverlapTests(unittest.TestCase):
    def fixture(self,root,test_seen=False):
        paths=[]
        for split,prefix in [('train','seen'),('validation','seen'),('test','seen' if test_seen else 'unseen')]:
            path=root/(split+'.jsonl'); write_jsonl(path,make_rows(prefix)); paths.append(path)
        return paths

    def test_audit_counts_problems_not_steps_and_keeps_sources(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); paths=self.fixture(root); hashes=[file_hash(p) for p in paths]
            report=audit(*paths,root/'audit')
            self.assertEqual(report['pairs'][0]['shared_problems'],2)
            self.assertEqual(report['pairs'][0]['left_affected_rows'],7)
            self.assertEqual(report['pairs'][0]['left_affected_traces'],3)
            self.assertEqual(report['pairs'][1]['shared_problems'],0)
            self.assertEqual(hashes,[file_hash(p) for p in paths])

    def launch(self,root,mode,test_seen=False):
        source=Path(__file__).parent
        bundle=root/'bundle'; bundle.mkdir()
        for name in ['run_prm_table2.sh','audit_problem_overlap.py','prepare_prm_table2_data.py',
                     'evaluate_prm_table2.py','test_prm_table2.py']:
            shutil.copy2(source/name,bundle/name)
        # Pure CPU protocol stub. Does NOT validate actual GPU model loading.
        (bundle/'predict_prm_table2.py').write_text('''
import argparse
from pathlib import Path
from prepare_prm_table2_data import read_jsonl
from evaluate_prm_table2 import reference
# test_prm_table2 imports pack at module load; provide a dummy since it is not used here.
def pack(*args): raise NotImplementedError
if __name__=='__main__':
    from test_prm_table2 import fake_pred
    p=argparse.ArgumentParser(); p.add_argument('--reference'); p.add_argument('--output'); p.add_argument('--name')
    a,_=p.parse_known_args(); rows,meta=reference(a.reference)
    fake_pred(rows,a.name,Path(a.output),meta)
''')
        paths=self.fixture(root,test_seen)
        model=root/'model'; model.mkdir()
        config=root/'models.json'
        write_json(config,[dict(name='Full',enabled=True,path=str(model),kind='hierarchical',active_heads='correctness,role')])
        out=root/'out'
        env=dict(os.environ,MODELS_JSON=str(config),TRAIN_INPUT=str(paths[0]),VAL_INPUT=str(paths[1]),
            TEST_INPUT=str(paths[2]),OUTPUT_DIR=str(out),NUM_GPUS='1',BOOTSTRAP_SAMPLES='5',
            EVAL_MODE=mode,ERROR_THRESHOLD='0.5',PYTHON=sys.executable,BASELINE='Full')
        for k in ['EXTRA_EXCLUDE_JSON','VAL_AUDIT_SOURCE','TEST_AUDIT_SOURCE']:
            env.pop(k,None)
        result=subprocess.run(['bash',str(bundle/'run_prm_table2.sh')],env=env,text=True,capture_output=True)
        return result,out

    def test_default_blocks_before_inference(self):
        with tempfile.TemporaryDirectory() as d:
            result,out=self.launch(Path(d),'validation_tuned')
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('train / validation',result.stdout)
            self.assertFalse((out/'predictions').exists())
            self.assertEqual(json.loads((out/'run_context.json').read_text())['status'],'blocked_overlap')

    def test_fixed_rejects_test_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            result,out=self.launch(Path(d),'fixed_threshold',test_seen=True)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('train / test',result.stdout)
            self.assertFalse((out/'predictions').exists())

    def test_fixed_clean_test_does_not_score_or_fit_validation(self):
        with tempfile.TemporaryDirectory() as d:
            result,out=self.launch(Path(d),'fixed_threshold')
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertTrue((out/'predictions'/'Full.test.jsonl').exists())
            self.assertFalse((out/'predictions'/'Full.validation.jsonl').exists())
            self.assertFalse((out/'validation_policy.json').exists())
            metrics=json.loads((out/'results'/'metrics.json').read_text())
            self.assertEqual(metrics['threshold_policy']['fixed_error_threshold'],.5)
            self.assertIn('Training/validation overlap',(out/'results'/'table2.md').read_text())
            context=json.loads((out/'results'/'evaluation_context.json').read_text())
            self.assertFalse(context['uses_validation_for_threshold'])
            self.assertEqual(context['status'],'completed')

    def test_history_test_overlap_counted(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); paths=self.fixture(root)
            hist=root/'history.jsonl'; shutil.copy2(paths[2],hist)
            report=audit(*paths,root/'audit',[hist])
            pair=next(p for p in report['pairs'] if p['left']=='history_1' and p['right']=='test')
            self.assertEqual(pair['shared_problems'],2)


if __name__=='__main__': unittest.main()
