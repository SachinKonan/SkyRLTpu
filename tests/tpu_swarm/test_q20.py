import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tpu.science.routing_suite import manifest, rescore_verified, select_suite
from tpu.science.rewards import qubit
from tpu.science.training_setup import check_references
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.commands import client_environment


class Q20Test(unittest.TestCase):
    def row(self):
        specs = manifest()['cases']
        counts = [c['original_cnot_added'] for c in specs]
        self.assertTrue(all(c % 3 == 0 for c in counts))
        reward, metrics = qubit(counts, counts, [c['weight'] for c in specs])
        metrics['cases'] = [dict(case=c['id'], added_cnots=n, swaps=n // 3)
                            for c, n in zip(specs, counts)]
        metrics['source_sha256'] = hashlib.sha256(b'code').hexdigest()
        return dict(correctness=1, code='code', reward=reward, metrics=metrics)

    def test_rescore_exact_subset(self):
        row = self.row()
        result = rescore_verified(row)
        self.assertEqual(result['reward'], .5)
        self.assertEqual(result['metrics']['case_count'], 24)
        self.assertEqual(result['metrics']['swaps'], 22714)
        # Changes on Willow/Heron must have no influence on Q20.
        specs = manifest()['cases']
        for case in row['metrics']['cases']:
            if not case['case'].endswith('_q20'):
                case['swaps'] *= 2
                case['added_cnots'] *= 2
        row['reward'] = qubit([c['original_cnot_added'] for c in specs],
                             [c['added_cnots'] for c in row['metrics']['cases']],
                             [c['weight'] for c in specs])[0]
        self.assertEqual(rescore_verified(row)['reward'], .5)

    def test_refuse_unverified_rescoring(self):
        for mutation in ('duplicate', 'hash', 'reward', 'negative', 'invalid'):
            row = self.row()
            if mutation == 'duplicate':
                row['metrics']['cases'][-1] = row['metrics']['cases'][0]
            elif mutation == 'hash':
                row['code'] = 'different'
            elif mutation == 'reward':
                row['reward'] = .9
            elif mutation == 'negative':
                row['metrics']['cases'][0]['swaps'] = -1
            else:
                row['correctness'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                rescore_verified(row)

    def test_selection_preserves_native_relative_paths_and_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'python/benchmarks/sabre_suite.json'
            original.parent.mkdir(parents=True)
            value = dict(cases=copy.deepcopy(manifest()['cases']))
            original.write_text(json.dumps(value))
            pinned = manifest()
            pinned['suite_sha256'] = hashlib.sha256(original.read_bytes()).hexdigest()
            self.assertEqual(select_suite(root, root, 'full'), original)
            with patch('tpu.science.routing_suite.manifest', return_value=pinned):
                selected = json.loads(select_suite(root, root, 'q20').read_text())['cases']
                self.assertEqual(len(selected), 24)
                self.assertEqual(selected[0]['seed'], 42)
                self.assertEqual(selected[0]['qasm3_path'], str(original.parent / value['cases'][0]['qasm3_path']))
                original.write_text('{}')
                with self.assertRaises(ValueError):
                    select_suite(root, root, 'q20')

    def test_prompt_and_client_propagation(self):
        from tpu.science.training_env import candidate_prompt
        cfg = Config.load('tpu/swarm/ray_train/profiles/science-qubit-v6e-qwen-runtime-001.json')
        cfg.client_env['SCIENCE_ROUTING_SUITE'] = 'q20'
        env = client_environment(cfg, Path('/tmp/test'), '127.0.0.1')
        self.assertEqual(env['SCIENCE_ROUTING_SUITE'], 'q20')
        with patch.dict(os.environ, env):
            for code, repair in [('', False), ('code', False), ('code', True)]:
                prompt = candidate_prompt('routing', code, 'feedback', repair=repair)
                self.assertEqual(prompt.count('Q20 ONLY'), 1)
                self.assertIn('22714/(22714+S)', prompt)
        cfg.client_env['SCIENCE_ROUTING_SUITE'] = 'typo'
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_reference_gate_rejects_wrong_suite(self):
        rows = [dict(correctness=1, reward=.5, metrics=dict(case_count=24, routing_suite='q20'))] * 8
        check_references('routing', rows, routing_suite='q20')
        with self.assertRaises(RuntimeError):
            check_references('routing', rows)
        rows[0]['metrics']['case_count'] = 72
        with self.assertRaises(RuntimeError):
            check_references('routing', rows, routing_suite='q20')


if __name__ == '__main__':
    unittest.main()
