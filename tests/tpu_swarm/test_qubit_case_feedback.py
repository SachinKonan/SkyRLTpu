import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tpu.science.feedback import feedback_limit, observation
from tpu.science.rewards import invalid
from tpu.science.routing_suite import manifest, rescore_verified


ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / 'tpu/science/results/full-topology-replay-20260919'


def replay(model='qwen'):
    return json.loads((REPLAY / f'{model}-replay.json').read_text())


class QubitFeedbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_cases_survive_grade_state_and_next_prompt(self):
        from tpu.science import training_env as env
        specs = {row['id']: row for row in manifest()['cases']}
        for model in ('qwen', 'gemma', 'muse'):
            result = replay(model)
            for suite in ('full', 'q20'):
                with self.subTest(model=model, suite=suite):
                    graded = result
                    if suite == 'q20':
                        graded = rescore_verified(dict(result, code=(REPLAY / f'{model}.py').read_text()))
                    e = object.__new__(env.RoutingTrainingEnv)
                    e.problem_type = 'routing'
                    e.log_path = '/tmp'
                    e.eval_timeout = 1800
                    e.num_cpus_per_task = 4
                    e.eval_backend = 'local'

                    async def evaluate(*args):
                        return graded

                    with patch.object(env, 'evaluate', evaluate):
                        verdict = await e._safe_grade('RUST_CODE = "policy"', 0)
                    state = e._create_next_state(0, 'RUST_CODE = "policy"', verdict)
                    e.initial_state = state
                    with patch.dict('os.environ', SCIENCE_ROUTING_SUITE=suite):
                        prompt = e.get_question()
                    self.assertIn(state.observation, prompt)
                    self.assertEqual(verdict.reward, graded['reward'])
                    self.assertEqual(verdict.raw_score, graded['raw_score'])
                    self.assertEqual(verdict.metrics, graded['metrics'])
                    data = json.loads(state.observation)['metrics']
                    self.assertEqual(len(data['cases']), 72 if suite == 'full' else 24)
                    self.assertEqual(data['case_columns'],
                                     ['case', 'swaps', 'baseline_swaps', 'delta_swaps'])
                    for row, recorded in zip(data['cases'], graded['metrics']['cases'], strict=True):
                        baseline = specs[recorded['case']]['original_cnot_added'] / 3
                        self.assertEqual(row, [recorded['case'], recorded['swaps'], baseline,
                                               recorded['swaps'] - baseline])
                    self.assertEqual(sum(t['swaps'] for t in data['topologies'].values()),
                                     graded['metrics']['swaps'])
                    self.assertEqual(sum(t['case_count'] for t in data['topologies'].values()),
                                     len(data['cases']))
                    self.assertLessEqual(len(state.observation), feedback_limit('routing'))

    async def test_invalid_error_reaches_repair_without_fabricated_case_counts(self):
        from tpu.science.training_env import candidate_prompt
        for error in ('ValueError: 4gt13_92_q20: illegal SWAP',
                      'compile failed: error[E0308]: mismatched types',
                      'TimeoutError: independent verification exhausted budget'):
            result = invalid(error)
            feedback = observation('routing', result)
            parsed = json.loads(feedback)
            self.assertEqual(parsed['reward'], 0)
            self.assertEqual(parsed['message'], error)
            self.assertNotIn('cases', parsed['metrics'])
            self.assertIn(feedback, candidate_prompt('routing', 'broken code', feedback, repair=True))

    async def test_explicit_baselines_and_unknown_cases_do_not_invent_references(self):
        result = replay()
        result['metrics']['cases'] = [dict(case='custom', swaps=2, added_cnots=6,
                                          baseline_added_cnots=9, topology='custom_graph'),
                                     dict(case='unknown', swaps=1, added_cnots=3)]
        data = json.loads(observation('routing', result))['metrics']
        self.assertEqual(data['cases'], [['custom', 2, 3, -1], ['unknown', 1, None, None]])
        self.assertIsNone(data['topologies']['unknown']['baseline_swaps'])

    async def test_old_aggregate_only_results_remain_usable(self):
        result = replay()
        del result['metrics']['cases']
        data = json.loads(observation('routing', result))
        self.assertEqual(data['reward'], result['reward'])
        self.assertNotIn('cases', data['metrics'])


if __name__ == '__main__':
    unittest.main()
