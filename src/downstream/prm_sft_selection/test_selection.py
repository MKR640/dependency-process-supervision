import argparse
import json
from pathlib import Path
import tempfile
import unittest

import select_prm_sft as s


class CharTokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class SelectionTests(unittest.TestCase):
    def test_field_contract(self):
        answer = ' Step1: <span>First</span>\nStep2: Second\n Answer：First again.  '
        obj = {"problem": "Q?", "options": ["one", "B. two"], "solution": answer,
               "answer": "DO NOT USE", "raw_str": {"question": "LEAK", "answer": "LEAK"}}
        prompt, completion = s.make_pair(obj)
        self.assertEqual(prompt, "Q?\n\nOptions:\nA. one\nB. two")
        self.assertEqual(completion, answer)
        self.assertNotIn("LEAK", prompt + completion)
        self.assertEqual(s.make_pair({**obj, "options": []})[0], "Q?")
        with self.assertRaises(ValueError):
            s.make_pair({**obj, "solution": " "})
        with self.assertRaises(ValueError):
            s.options_text([{"unknown": "x"}])

    def test_splitting(self):
        parts, mode = s.split_steps("Intro\n Step1: first\nline two\n Step2: last\n Answer：last again")
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[-1], "Answer：last again")
        self.assertIn("line two", parts[1])
        self.assertEqual(mode, "step_and_answer_markers")
        self.assertEqual(s.split_steps("A\n\nB")[0], ["A", "B"])
        self.assertEqual(s.split_steps("A\nB")[0], ["A\nB"])

    def test_context_window(self):
        tok = CharTokenizer()
        steps = ["a", "much longer previous", "z"]
        encoded, info = s.encode_inputs(tok, "Q", steps, 100)
        expected = [1] + tok.encode("Q\n\na\n\nmuch longer previous\n\nz\n\n")
        self.assertEqual(encoded[-1], expected)
        # Q + each current step fits, but no room for immediate predecessor at step3.
        encoded, info = s.encode_inputs(tok, "Q", steps, 26)
        self.assertEqual(info["dropped_prefix_steps"][-1], 2)
        self.assertEqual(encoded[-1], [1] + tok.encode("Q\n\nz\n\n"))
        encoded, info = s.encode_inputs(tok, "long question", ["x" * 30], 10)
        self.assertIsNone(encoded)
        self.assertEqual(info["step_index"], 0)

    def test_aggregation(self):
        steps = [{"q_correct_vs_error": q, "p_error": 1-q, "p_neutral": 0} for q in [.25, 1]]
        self.assertAlmostEqual(s.aggregate(steps, 0)["mixed_q"], .5)
        self.assertEqual(s.aggregate(steps, 1)["mixed_q"], .25)
        self.assertAlmostEqual(s.aggregate(steps, .5)["mixed_q"], (.25 * .5)**.5)
        steps[0]["q_correct_vs_error"] = 0
        self.assertEqual(s.aggregate(steps, 0)["mixed_q"], 0)

    def test_cpu_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "work"
            source = Path(temp) / "source.jsonl"
            originals = [{"id": "duplicate-id", "problem": f"Q{i}", "options": [],
                          "solution": f"Step1: reasoning {i}\nAnswer：{i}"} for i in range(4)]
            with open(source, "w") as f:
                for row in originals + [{"problem": "", "solution": "invalid"}]:
                    s.dump(f, row)
            s.prepare(argparse.Namespace(data=str(source), output_dir=str(root)))
            manifest = s.read_json(root / "prepare_manifest.json")
            self.assertEqual(manifest["counts"]["usable_records"], 4)
            config = {"version": s.VERSION, "world_size": 2, "all_sha256": manifest["all_sha256"]}
            for rank in range(2):
                path = root / f"scores.rank{rank:04d}.jsonl"
                with s.atomic(path) as f:
                    for i in range(rank, 4, 2):
                        q = [.95, .4, .9, .1][i]
                        row = {"index": i, "source_line": i + 1, "id": "duplicate-id", "num_steps": 2,
                               "status": "ok", "dropped_prefix_steps": [0, 0],
                               "step_scores": [{"q_correct_vs_error": q, "p_error": 1-q,
                                                "p_neutral": 0, "p_correct": q}] * 2}
                        if i == 3:
                            row.update(status="unscorable", reason="problem_plus_step_exceeds_max_length")
                            row.pop("step_scores")
                        s.dump(f, row)
                s.write_json(root / f"scores.rank{rank:04d}.done.json",
                             {"config": {**config, "rank": rank}, "count": 2, "sha256": s.digest(path)})
            dest = root / "chosen"
            args = argparse.Namespace(output_dir=str(root), selection_dir=str(dest), mix_weight=.5,
                                      min_step_q=.5, min_score=.8, top_fraction=None,
                                      require_full_prefix=False, seed=42, review_size=2)
            s.select(args)
            chosen = [row for _, row in s.rows(dest / "selected_sft.jsonl")]
            self.assertEqual([row["source_line"] for row in chosen], [1, 3])
            self.assertEqual(chosen[0]["completion"], originals[0]["solution"])
            self.assertEqual(chosen[1]["completion"], originals[2]["solution"])
            self.assertEqual(len(list(s.rows(dest / "random_matched_sft.jsonl"))), 2)
            report = s.read_json(dest / "selection_report.json")
            self.assertEqual(report["selection_rate"], .5)
            self.assertEqual(report["unscorable_records"], 1)
            # Missing shards cannot silently make a partial dataset look complete.
            (root / "scores.rank0001.done.json").unlink()
            with self.assertRaises(FileNotFoundError):
                s.select(args)


if __name__ == "__main__":
    unittest.main()
