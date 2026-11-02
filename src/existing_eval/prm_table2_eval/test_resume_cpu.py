"""CPU-only resume regression tests with synthetic predictions; no real checkpoint."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
import test_overlap_modes
from prepare_prm_table2_data import write_json, write_jsonl, read_jsonl, file_hash

class ResumeTests(unittest.TestCase):
    def setup_run(self, root):
        result, out = test_overlap_modes.OverlapTests().launch(root, 'fixed_threshold')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        dest = out/'predictions'/'Full.test.jsonl'
        meta = json.loads(Path(str(dest)+'.meta.json').read_text())
        meta.update(model=str((root/'model').resolve()),
                    packing=dict(max_length=4096, prefix_policy='contiguous'))
        rows = list(read_jsonl(dest))
        for rank in range(2):
            shard = Path(str(dest)+f'.rank{rank}.jsonl')
            write_jsonl(shard, rows[rank::2])
            write_json(str(shard)+'.meta.json',dict(meta, rank=rank, world_size=2,
                       predictions_sha256=file_hash(shard)))
        # A resumed run must never call the predictor, even on incomplete caches.
        (root/'bundle'/'predict_prm_table2.py').write_text('raise RuntimeError("GPU MUST NOT RUN")')
        dest.unlink()
        Path(str(dest)+'.meta.json').unlink()
        return out

    def resume(self, root, out, **overrides):
        env=dict(os.environ,MODELS_JSON=str(root/'models.json'),
            TRAIN_INPUT=str(root/'train.jsonl'),VAL_INPUT=str(root/'validation.jsonl'),
            TEST_INPUT=str(root/'test.jsonl'),OUTPUT_DIR=str(out),NUM_GPUS='2',
            BOOTSTRAP_SAMPLES='5',EVAL_MODE='fixed_threshold',ERROR_THRESHOLD='0.5',
            PYTHON=__import__('sys').executable,BASELINE='Full',SKIP_INFERENCE='1',
            AUTO_INSTALL_DEPS='0',MAX_LENGTH='4096',PREFIX_POLICY='contiguous')
        for k in ['EXTRA_EXCLUDE_JSON','VAL_AUDIT_SOURCE','TEST_AUDIT_SOURCE']:
            env.pop(k,None)
        env.update(overrides)
        return subprocess.run(['bash',str(root/'bundle'/'run_prm_table2.sh')],
                              env=env,text=True,capture_output=True)

    def test_resume_shards_completes_without_predictor(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=self.setup_run(root)
            result=self.resume(root,out)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('CPU RESUME',result.stdout)
            self.assertTrue((out/'predictions'/'Full.test.jsonl').exists())
            ctx=json.loads((out/'results'/'evaluation_context.json').read_text())
            self.assertTrue(ctx['skip_inference'])
            self.assertEqual(ctx['status'],'completed')

    def test_missing_shard_does_not_start_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=self.setup_run(root)
            (out/'predictions'/'Full.test.jsonl.rank1.jsonl').unlink()
            result=self.resume(root,out)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Missing prediction or metadata',result.stderr)
            self.assertNotIn('GPU MUST NOT RUN',result.stderr)

    def test_changed_packing_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=self.setup_run(root)
            result=self.resume(root,out,MAX_LENGTH='2048')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('cached packing differs',result.stderr)

    def test_changed_shard_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=self.setup_run(root)
            shard=out/'predictions'/'Full.test.jsonl.rank0.jsonl'
            shard.write_text(shard.read_text()+'\n')
            result=self.resume(root,out)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('checksum mismatch',result.stderr)

if __name__=='__main__': unittest.main()
