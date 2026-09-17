import asyncio
import json
import os
import shutil
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.bootstrap import workload_resources
from tpu.science.training_setup import split_roles, check_references
from tpu.science.placement_slots import device_paths, tpu_environment


class ScienceTopologyTest(unittest.TestCase):
    def test_packaged_cpu_placement_prompts_for_all_models(self):
        from tpu.science import training_env
        from tpu.swarm.ray_train.overlay import manifest, install
        root = Path(__file__).resolve().parents[2]
        for model in ('qwen', 'muse', 'gemma'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temp:
                config = Config.load(root / (
                    f'tpu/swarm/ray_train/profiles/science-placement-v6e-{model}-cpu-seeded-train-001.json'))
                records = manifest(root, config)
                overlay, installed = Path(temp) / 'overlay', Path(temp) / 'installed'
                overlay.mkdir()
                for name in records:
                    target = overlay / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root / name, target)
                (overlay / 'manifest.json').write_text(json.dumps(records))
                install(overlay, installed)
                with patch.object(training_env, '__file__', str(installed / 'tpu/science/training_env.py')), \
                        patch.dict(os.environ, {'SCIENCE_PLACEMENT_BACKEND': 'cpu'}):
                    for code, repair in [('', False), ('pass', False), ('pass', True)]:
                        self.assertTrue(training_env.candidate_prompt(
                            'placement', code, 'Feedback', repair=repair).strip())
                    (installed / 'tpu/science/prompts/placement-jax-cpu.txt').unlink()
                    with self.assertRaises(FileNotFoundError):
                        training_env.candidate_prompt('placement', 'pass', 'Feedback')

    def test_gemma_science_retains_validated_replay_backend(self):
        from tpu.swarm.ray_train.overlay import manifest
        root = Path(__file__).resolve().parents[2]
        profiles = root / 'tpu/swarm/ray_train/profiles'
        replay = Config.load(profiles / 'science-gemma-v4-tp4-fsdp4-replay-001.json')
        validated = manifest(root, replay)
        for task in ('routing', 'placement'):
            with self.subTest(task=task):
                config = Config.load(profiles / f'science-{task}-v4-gemma-16x32-e2e-001.json')
                shipped = manifest(root, config)
                self.assertFalse(config.attention_replay)
                for name in ('skyrl/backends/tunix_backend.py', 'skyrl/backends/lora_init.py'):
                    self.assertEqual(shipped[name], validated[name])
                self.assertEqual(config.trainer_env['TUNIX_SHARD_INPUTS'], '1')

    def test_full_qwen_batch_reaches_client_without_request_splitting(self):
        from tpu.swarm.ray_train.commands import client_environment
        from tpu.swarm.ray_train.overlay import manifest, REPEATED_KV_FILES
        root = Path(__file__).resolve().parents[2]
        for task, engines in [('routing', 4), ('placement', 3)]:
            c = Config.load(root / f'tpu/swarm/ray_train/profiles/science-{task}-v4-qwen-16x32-e2e-001.json')
            env = client_environment(c, Path('/runtime'), 'head')
            self.assertEqual((env['GROUPS_PER_BATCH'], env['GROUP_SIZE']), ('16', '32'))
            self.assertEqual(env['TTD_QWEN_SAMPLE_GROUP_CHUNK_SIZE'], '0')
            self.assertEqual(env['TTD_MIN_VALID_PER_GROUP'], '0')
            self.assertEqual(env['TTD_ONE_STEP_SMOKE'], '1')
            self.assertEqual(env['NUM_EPOCHS'], '1')
            self.assertEqual((c.trainer.tp, c.trainer.fsdp, c.trainer.logical_kv_heads), (8, 2, 8))
            self.assertEqual(c.trainer.token_budget, 45056)
            self.assertEqual(c.inference_hosts, engines)
            self.assertTrue(c.inference.prefix_caching)
            self.assertEqual(c.inference.max_sequences, 16)
            self.assertTrue(REPEATED_KV_FILES <= manifest(root, c).keys())
            # All 512 candidates can wait for their hard-limited worker slots.
            worst = 512 * 300 if task == 'placement' else 512 / 16 * 1800
            self.assertGreater(float(env['EVAL_TIMEOUT']), worst)

    def test_shuffled_physical_training_block_is_preserved(self):
        train, infer, grader = split_roles('placement', [6, 0, 7, 2], [1, 3, 4, 5])
        self.assertEqual(train, [6, 0, 7, 2])
        self.assertEqual(infer, [1, 3, 4])
        self.assertEqual(grader, 5)
        self.assertNotIn(grader, train)
        self.assertEqual(split_roles('routing', train, [1, 3, 4, 5])[1], [1, 3, 4, 5])

    def test_profiles_keep_training_and_native_sampling(self):
        for task, count in [('routing', 4), ('placement', 3)]:
            c = Config.load(f'tpu/swarm/ray_train/profiles/science-{task}-v6e-muse-grpo-001.json')
            self.assertFalse(c.inference_only)
            self.assertEqual((c.trainer.hosts, c.trainer.tp, c.trainer.fsdp), (4, 2, 8))
            self.assertEqual(c.inference_hosts, count)
            self.assertTrue(c.inference.native_thinking_budget)
            self.assertEqual(c.client_env['TTD_LOSS_FN'], 'importance_sampling')
            self.assertEqual(c.client_env['GROUP_SIZE'], '8')
            if task == 'placement':
                self.assertEqual(workload_resources(c, 0), {'TPU': 4, 'placement_tpu_host': 4})
                with self.assertRaises(ValueError):
                    replace(c, client_env={**c.client_env, 'PLACEMENT_TPU_RANKS': '7'}).validate()

    def test_muse_training_rejects_indivisible_kv_heads(self):
        from tpu.swarm.ray_train.commands import maxtext_kwargs
        for task in ('routing', 'placement'):
            c = Config.load(f'tpu/swarm/ray_train/profiles/science-{task}-v6e-muse-grpo-001.json')
            with self.assertRaisesRegex(ValueError, 'two KV heads'):
                replace(c, trainer=replace(c.trainer, tp=4, fsdp=4)).validate()
            kwargs = maxtext_kwargs(c, Path('/runtime'))
            self.assertEqual(kwargs['ici_tensor_parallelism'], 2)
            self.assertEqual(kwargs['ici_fsdp_parallelism'], 8)
            self.assertNotIn('base_num_kv_heads', kwargs)
            self.assertNotIn('override_model_config', kwargs)

    def test_v6e_exposes_only_assigned_chip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'vfio').mkdir()
            for name in ('0', '1', '2', '3', 'vfio'):
                (root / 'vfio' / name).symlink_to('/dev/null')
            self.assertEqual(device_paths(3, 'tpu-v6e-32', root), [root/'vfio/3', root/'vfio/vfio'])
        self.assertEqual(tpu_environment(3, 'tpu-v6e-32')['TPU_VISIBLE_CHIPS'], '3')

    def test_reference_failure_blocks_training(self):
        good = [{'correctness': 1, 'reward': .5, 'metrics': {'case_count': 72}} for _ in range(8)]
        check_references('routing', good)
        good[3]['correctness'] = 0
        with self.assertRaises(RuntimeError):
            check_references('routing', good)


class ScienceRewardsTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_placement_feedback_preserves_actual_errors_for_every_case(self):
        from tpu.science import training_env as env
        from tpu.science.challenge_contract import CASES, aggregate
        from tpu.science.rewards import invalid
        rows = []
        for case in CASES:
            row = invalid('CalledProcessError: Command ' + '/long/path/' * 500)
            row['metrics'] = {'case': case, 'grader.log': (
                'NOISY_INIT\n' * 300 + "ValueError: illegal placement: ['Macros 0 and 166 overlap']\n")}
            rows.append(row)
        result = aggregate(rows)
        e = object.__new__(env.PlacementTrainingEnv)
        e.problem_type = 'placement'; e.log_path = '/tmp'; e.eval_timeout = 1200
        e.num_cpus_per_task = 4; e.eval_backend = 'local'
        async def evaluate(*args):
            return result
        with patch.object(env, 'evaluate', evaluate):
            verdict = await e._safe_grade('candidate', 0)
        self.assertEqual(verdict.reward, 0)
        self.assertEqual(verdict.metrics, result['metrics'])
        decoded = json.loads(verdict.stdout)
        self.assertEqual([row['case'] for row in decoded['metrics']['cases']], list(CASES))
        for row in decoded['metrics']['cases']:
            self.assertFalse(row['valid'])
            self.assertIn('Macros 0 and 166 overlap', row['message'])
        self.assertNotIn('NOISY_INIT', verdict.stdout)
        self.assertNotIn('/long/path/', verdict.stdout)
        self.assertLessEqual(len(verdict.stdout), 3000)
        with patch.dict('os.environ', SCIENCE_ACCELERATOR='tpu-v4-64'):
            prompt = env.candidate_prompt('placement', 'candidate', verdict.stdout, repair=True)
        self.assertIn('Macros 0 and 166 overlap', prompt)

    async def test_candidate_exception_and_long_errors_leave_complete_feedback(self):
        from tpu.science.feedback import observation
        from tpu.science.challenge_contract import CASES, aggregate
        from tpu.science.rewards import invalid
        rows = []
        for case in CASES:
            row = invalid('candidate exited 1')
            row['metrics'] = {'case': case, 'candidate.log': (
                'Traceback (most recent call last):\n  File "/runner.py"\n'
                'jax.errors.ConcretizationTypeError: invalid dynamic shape ' + 'x' * 8000)}
            rows.append(row)
        output = observation('placement', aggregate(rows))
        self.assertLessEqual(len(output), 3000)
        self.assertEqual(len(json.loads(output)['metrics']['cases']), 4)
        self.assertIn('ConcretizationTypeError', output)

    async def test_placement_components_reach_next_prompt_without_worker_logs(self):
        from tpu.science import training_env as env
        from tpu.science.challenge_contract import CASES, aggregate
        from tpu.science.rewards import valid
        from ttt_discover import State
        rows = [valid(.5, dict(case=case, proxy_cost=1.2 + i,
                               wirelength_cost=.2 + i, density_cost=.8, congestion_cost=1.2,
                               candidate_wall_seconds=170.123456789, grading_seconds=6.123456789,
                               **{'candidate.log': 'DO_NOT_FORWARD' * 4000}))
                for i, case in enumerate(CASES)]
        result = aggregate(rows)
        e = object.__new__(env.PlacementTrainingEnv)
        e.problem_type = 'placement'; e.log_path = '/tmp'; e.eval_timeout = 1200
        e.num_cpus_per_task = 4; e.eval_backend = 'local'
        e.state = State(timestep=-1, construction=None, code='', value=0.)
        async def evaluate(*args):
            return result
        with patch.object(env, 'evaluate', evaluate):
            verdict = await e._safe_grade('def place(): pass', 0)
        self.assertEqual(verdict.reward, result['reward'])
        self.assertEqual(verdict.raw_score, result['raw_score'])
        self.assertEqual(verdict.metrics, result['metrics'])
        state = e._create_next_state(0, 'def place(): pass', verdict)
        feedback = json.loads(state.observation)
        self.assertEqual([r['case'] for r in feedback['metrics']['cases']], list(CASES))
        for i, row in enumerate(feedback['metrics']['cases']):
            self.assertAlmostEqual(row['proxy_cost'], 1.2 + i)
            self.assertAlmostEqual(row['wirelength_cost'], .2 + i)
            self.assertEqual(row['density_cost'], .8)
            self.assertEqual(row['congestion_cost'], 1.2)
            self.assertAlmostEqual(row['candidate_wall_seconds'], 170.123, places=3)
            self.assertAlmostEqual(row['grading_seconds'], 6.12346, places=5)
        self.assertLess(len(state.observation), 3000)
        self.assertNotIn('DO_NOT_FORWARD', state.observation)
        e.initial_state = state
        with patch.dict('os.environ', SCIENCE_ACCELERATOR='tpu-v4-64'):
            prompt = e.get_question()
        self.assertIn(state.observation, prompt)
        self.assertEqual(prompt.count(state.code), 1)
        self.assertNotIn('Starting implementation (replace with your improved algorithm):', prompt)

    async def test_starter_only_on_initial_round_for_both_tasks(self):
        from tpu.science import training_env as env
        from ttt_discover import State
        for task, marker in [('routing', 'Valid initial policy block:'),
                             ('placement', 'Starting implementation (replace with your improved algorithm):')]:
            e = object.__new__(env.ScienceTrainingEnv)
            e.problem_type = task
            e.initial_state = State(timestep=-1, construction=None, code='', value=0.)
            with patch.dict('os.environ', SCIENCE_ACCELERATOR='tpu-v4-64'):
                initial = e.get_question()
                self.assertEqual(initial, env.task_prompt(task))
                self.assertIn(marker, initial)
                e.initial_state = State(timestep=0, construction=None, code='SELECTED_PROGRAM',
                                        value=.5, observation='SAVED_FEEDBACK')
                improved = e.get_question()
            self.assertTrue(improved.startswith(initial.partition(marker)[0].rstrip()))
            self.assertNotIn(marker, improved)
            self.assertEqual(improved.count('SELECTED_PROGRAM'), 1)
            self.assertIn('SAVED_FEEDBACK', improved)
            if task == 'routing':
                self.assertIn('Fixed Rust scaffold (read-only):', improved)

    async def test_placement_dispatch_and_prompt_follow_accelerator(self):
        from tpu.science import training_env as env
        from tpu.science import placement_ray, challenge_contract
        from tpu.science.rewards import valid
        result = valid(.5, {})
        class Ref:
            def future(self):
                future = Future(); future.set_result(result); return future
        for accelerator, chip in [('tpu-v4-64', 'TPU v4 chip'), ('tpu-v6e-32', 'TPU v6e chip')]:
            with patch.object(env, 'connect'), patch.dict('os.environ',
                    SCIENCE_ACCELERATOR=accelerator, SCIENCE_WORKER_ROOT='/payload', RAY_NAMESPACE='run'), \
                    patch('ray.util.placement_group.get_placement_group', return_value=object()), \
                    patch.object(placement_ray.grade_case, 'options') as opts, \
                    patch.object(challenge_contract, 'aggregate', return_value=result), \
                    patch.object(env.ray, 'cancel'):
                opts.return_value.remote.side_effect = lambda *a, **k: Ref()
                self.assertEqual(await env.evaluate('placement', 'code', 10), result)
                self.assertEqual(opts.return_value.remote.call_count, 4)
                for call in opts.return_value.remote.call_args_list:
                    self.assertEqual(call.kwargs['accelerator'], accelerator)
                prompt = env.task_prompt('placement')
                self.assertIn(chip, prompt)
                self.assertNotIn('TPU v6e chip' if 'v4' in accelerator else 'TPU v4 chip', prompt)

    async def test_reward_and_state_direction_from_real_environment(self):
        from tpu.science import training_env as env
        from tpu.science.rewards import valid
        from ttt_discover import State
        e = object.__new__(env.RoutingTrainingEnv)
        e.problem_type = 'routing'; e.log_path = '/tmp'; e.eval_timeout = 7200
        e.num_cpus_per_task = 4; e.eval_backend = 'local'; e.state = State(timestep=-1, construction=None, code='', value=0.)
        result = valid(.53, {'swaps': 123, 'case_count': 72})
        async def evaluate(*args):
            return result
        with patch.object(env, 'evaluate', evaluate):
            verdict = await e._safe_grade('RUST_CODE="policy"', 0)
        state = e._create_next_state(0, 'RUST_CODE="policy"', verdict)
        self.assertEqual(state.value, .53)
        self.assertIn('123', state.observation)
        self.assertEqual(verdict.correctness, 1.)

    async def test_infrastructure_failure_is_fatal_and_cancels_ray(self):
        from tpu.science import training_env as env
        from tpu.science import ray_cpu
        future = Future(); future.set_exception(RuntimeError('host lost'))
        class Ref:
            def future(self): return future
        ref = Ref()
        with patch.object(env, 'connect'), patch.dict('os.environ', SCIENCE_WORKER_ROOT='/payload'), \
                patch.object(ray_cpu.grade, 'options') as opts, patch.object(env.ray, 'cancel') as cancel:
            opts.return_value.remote.return_value = ref
            with self.assertRaises(env.ScienceInfrastructureError) as error:
                await env.evaluate('routing', 'code', 1)
            self.assertEqual(opts.return_value.remote.call_args.kwargs['admission_timeout_s'], 1)
            self.assertTrue(error.exception.abort_training_step)
            cancel.assert_called_once_with(ref, force=True)

    async def test_cancellation_stops_underlying_ray_task(self):
        from tpu.science import training_env as env
        from tpu.science import ray_cpu
        future = Future()
        class Ref:
            def future(self): return future
        ref = Ref()
        with patch.object(env, 'connect'), patch.dict('os.environ', SCIENCE_WORKER_ROOT='/payload'), \
                patch.object(ray_cpu.grade, 'options') as opts, patch.object(env.ray, 'cancel') as cancel:
            opts.return_value.remote.return_value = ref
            task = asyncio.create_task(env.evaluate('routing', 'code', 60))
            await asyncio.sleep(.01); task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            cancel.assert_called_once_with(ref, force=True)


if __name__ == '__main__':
    unittest.main()
