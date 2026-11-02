"""CPU checks for release boundaries and recovered role semantics; no real-data claims."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from role_baseline_functions import sequential_roles, confusion, macro_from_cm

class ReleaseTests(unittest.TestCase):
    def test_sequential_rule_keeps_error_history_across_correct_steps(self):
        rows=[{'trace_id':t,'p_error':p} for t,p in [('a',.1),('a',.9),('a',.1),('a',.8),('b',.1),('b',.9)]]
        np.testing.assert_array_equal(sequential_roles(rows,.5),[0,1,0,2,0,1])

    def test_macro_f1_with_absent_class(self):
        # Perfect classes 0 and 1; missing class 2 contributes zero to the fixed three-class average.
        cm=confusion([0,0,1],[0,0,1])
        self.assertAlmostEqual(float(macro_from_cm(cm)),2/3)

    def test_all_missing_resources_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--module','all','--resource-root',td],capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,2,result.stderr)
        path=Path(result.stdout.split('Summary: ',1)[1].strip())
        report=json.loads(path.read_text(encoding='utf-8'))
        self.assertFalse(report['all_requested_complete'])
        self.assertEqual(len(report['modules']),7)
        self.assertTrue(all(r['status']=='blocked' for r in report['modules'].values()))

    def test_training_plan_requires_frozen_data(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'plan'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_training.py'),'--arm','M5','--data-dir',td,
                '--init-model',td,'--plan-dir',str(target),'--output-root',str(Path(td)/'train')],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(result.returncode,2)
            self.assertFalse(target.exists())

if __name__=='__main__':unittest.main()
