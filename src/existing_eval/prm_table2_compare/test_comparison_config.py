"""Checkpoint configuration checks on synthetic local files; no GPU required."""
import json
import tempfile
import unittest
from pathlib import Path
from configure_prm_comparison import inspect_model
from prepare_prm_table2_data import ROLES, TYPES

class ConfigTests(unittest.TestCase):
    def fixture(self, root, labels, kind='token_classifier'):
        p=root/'checkpoint'; p.mkdir()
        cfg={'architectures':['Qwen3ForTokenClassification'], 'id2label':labels}
        if kind=='hierarchical':
            cfg.update(correctness_id2label=labels,
                       role_id2label=dict(enumerate(ROLES)),error_type_id2label=dict(enumerate(TYPES)))
        (p/'config.json').write_text(json.dumps(cfg))
        (p/'model.safetensors').write_bytes(b'synthetic fixture')
        return dict(name='M',directory='checkpoint',kind=kind,expected_labels=len(labels),
                    active_heads='correctness,role,type' if kind=='hierarchical' else 'correctness')

    def test_three_class_ids_resolved_not_assumed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); item=self.fixture(root,{0:'CORRECT',1:'NEUTRAL',2:'ERROR'})
            model,check=inspect_model(item,root,{})
            self.assertEqual((model['error_id'],model['correct_id']),(2,0))
            self.assertTrue(check['neutral_excluded'])

    def test_generic_labels_need_explicit_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); item=self.fixture(root,{0:'LABEL_0',1:'LABEL_1'})
            with self.assertRaisesRegex(ValueError,'cannot determine'):
                inspect_model(item,root,{})
            model,_=inspect_model(item,root,{'M':{'error_id':1,'correct_id':0}})
            self.assertEqual(model['error_id'],1)

    def test_contradictory_override_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); item=self.fixture(root,{0:'ERROR',1:'NON_ERROR'})
            with self.assertRaisesRegex(ValueError,'contradict'):
                inspect_model(item,root,{'M':{'error_id':1,'correct_id':0}})

    def test_hierarchy_requires_complete_role_type_maps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); item=self.fixture(root,{0:'ERROR',1:'NON_ERROR'},'hierarchical')
            model,_=inspect_model(item,root,{})
            self.assertEqual(model['active_heads'],'correctness,role,type')
            p=root/'checkpoint'/'config.json'; cfg=json.loads(p.read_text()); cfg.pop('role_id2label')
            p.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError): inspect_model(item,root,{})

    def test_generator_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); item=self.fixture(root,{0:'ERROR',1:'CORRECT'})
            p=root/'checkpoint'/'config.json'; cfg=json.loads(p.read_text())
            cfg['architectures']=['Qwen3ForCausalLM']; p.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'token-classification'):
                inspect_model(item,root,{})

if __name__=='__main__': unittest.main()
