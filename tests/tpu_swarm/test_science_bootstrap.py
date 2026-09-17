import asyncio
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tpu.science import bootstrap
from tpu.swarm.ray_train.config import Config

PROFILE = 'tpu/swarm/ray_train/profiles/science-routing-v4-qwen-16x32-e2e-001.json'


class Remote:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args):
        result = self.fn(*args)
        return await result if inspect.isawaitable(result) else result


class CompatibilityTest(unittest.TestCase):
    def test_controller_waits_on_real_serve_response_without_ray_wait(self):
        from concurrent.futures import Future
        from ray.serve.handle import DeploymentResponse
        from tpu.swarm.ray_train.controller import Controller
        controller = Controller(Config.load(PROFILE), ['127.0.0.1'])
        attempts = []
        def get(timeout_s):
            attempts.append(timeout_s)
            if len(attempts) == 1:
                raise TimeoutError('not yet complete')
            return ['retired-host']
        future = Future()
        future.set_result(SimpleNamespace(get=get))
        response = DeploymentResponse(future, SimpleNamespace(request_id='handoff-test'))
        with patch('ray.wait', side_effect=AssertionError('Serve responses are not ObjectRefs')):
            self.assertEqual(controller.checked_serve(response, 5), ['retired-host'])
        self.assertEqual(len(attempts), 2)
        controller.stopping.set()
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            controller.checked_serve(response, 5)
        controller.stopping.clear()
        with self.assertRaisesRegex(TimeoutError, 'deadline'):
            controller.checked_serve(response, 0)

    def test_packaged_bootstrap_ignores_stale_source_config(self):
        from tpu.science.package_training import package
        from tpu.swarm.ray_train import host as host_module
        from tpu.swarm.ray_train.host import Host
        repo = Path(__file__).resolve().parents[2]
        profile = repo / 'tpu/swarm/ray_train/profiles/science-routing-v4-qwen-bootstrap-l2-001.json'
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            package(profile, temp / 'build')
            bundle = temp / 'bundle'
            bundle.mkdir()
            with tarfile.open(temp / 'build/science-training.tar.gz') as archive:
                archive.extractall(bundle, filter='data')
            source = temp / 'frozen-source'
            stale = source / 'tpu/swarm/ray_train'
            stale.mkdir(parents=True)
            (stale / '__init__.py').write_text('')
            (stale / 'config.py').write_text("raise RuntimeError('STALE_SOURCE_CONFIG')\n")
            (source / 'third_party').mkdir()
            (source / 'third_party/discover').symlink_to(repo / 'third_party/discover')
            (temp / 'envs').mkdir()
            (temp / 'envs/client').symlink_to(sys.prefix, target_is_directory=True)
            fake = SimpleNamespace(rank=0, config=Config.load(profile), root=temp,
                                   source=source, snapshot=temp/'snapshot',
                                   ips=['127.0.0.1'], trainer_leader=0)
            with patch.object(host_module, '__file__', str(bundle/'tpu/swarm/ray_train/host.py')):
                command, env, cwd = Host.bootstrap_launch(fake)
            self.assertEqual(cwd, bundle)
            # This is the exact production module invocation and environment.
            good = subprocess.run(command + ['--check-only'], cwd=cwd, env=env,
                                  capture_output=True, text=True, timeout=90)
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
            result = json.loads(good.stdout.splitlines()[-1])
            self.assertEqual(Path(result['config']), bundle/'tpu/swarm/ray_train/config.py')
            self.assertEqual(Path(result['bootstrap']), bundle/'tpu/science/bootstrap.py')
            bad = subprocess.run(command + ['--check-only'], cwd=source, env=env,
                                 capture_output=True, text=True, timeout=90)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('STALE_SOURCE_CONFIG', bad.stderr)

    def test_opt_in_validation_and_old_profiles(self):
        old = Config.load(PROFILE)
        self.assertEqual(old.bootstrap_layers, 0)
        self.assertFalse(old.bootstrap_all_hosts)
        self.assertEqual(Config.from_dict(old.to_dict()), old)
        replace(old, bootstrap_layers=2, bootstrap_all_hosts=True).validate()
        for changes in ({'bootstrap_layers': 3}, {'bootstrap_all_hosts': True},
                        {'bootstrap_layers': 2, 'inference_only': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(old, **changes).validate()

    def test_seed_only_config_and_training_pool_gate(self):
        from tpu.science.training_setup import check_references
        from tpu.science.seed_pool import verify_pool
        old = Config.load(PROFILE)
        seed = replace(old, bootstrap_layers=2, bootstrap_only=True, inference_only=True, training_smoke=False,
                       accelerator='tpu-v4-32', hosts=4, trainer=replace(old.trainer, hosts=0))
        seed.validate()
        self.assertEqual(seed.inference_hosts, 4)
        for change in ({'bootstrap_layers': 0}, {'inference_only': False},
                       {'accelerator': 'tpu-v4-64', 'hosts': 8},
                       {'trainer': old.trainer}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(seed, **change).validate()
        good = [dict(correctness=1, reward=.5, metrics={'case_count': 72}) for _ in range(4)]
        check_references('routing', good, expected_hosts=4)
        with self.assertRaises(RuntimeError):
            check_references('routing', good[:3], expected_hosts=4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'pool.json'
            pool = dict(step=0, states=[dict(code='good', value=.5)])
            bootstrap.save(path, pool)
            verify_pool(path, bootstrap.identity(pool))
            with self.assertRaises(ValueError):
                verify_pool(path, '0'*64)
            pool['states'][0]['value'] = 0
            bootstrap.save(path, pool)
            with self.assertRaises(ValueError):
                verify_pool(path, bootstrap.identity(pool))

    def test_seed_only_controller_finishes_without_client_or_trainer(self):
        from tpu.swarm.ray_train.controller import Controller
        from unittest.mock import Mock
        old = Config.load(PROFILE)
        controller = Controller(replace(old, bootstrap_only=True), ['head'])
        controller.setup = Mock()
        controller.report = Mock()
        host = Mock()
        controller.hosts = [host]
        with patch('ray.get', return_value={'retained': 1, 'optimizer_steps': 0}):
            self.assertEqual(controller.run(), 0)
        host.start_client.remote.assert_not_called()
        host.start_trainer.remote.assert_not_called()
        self.assertEqual(controller.trainers, [])

    def test_repairs_exclude_valid_duplicate_and_oversize_programs(self):
        rows = [dict(code='valid', correctness=1, message='OK'),
                dict(code='invalid', correctness=0, message='NameError'),
                dict(code='invalid', correctness=0, message='NameError'),
                dict(code='too large', correctness=0, message='SyntaxError'),
                dict(code='', correctness=0, message='bad fence')]
        self.assertEqual(bootstrap.repair_parents(rows, 16, lambda r: r['code'] != 'too large'), rows[1:2])


class BootstrapIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_two_layers_resume_and_valid_only_puct_without_training_api(self):
        from tpu.science import training_env
        from ttt_discover.tinker_utils import renderers
        from ttt_discover.tinker_utils.dataset_builder import VerifyResult
        from ttt_discover.tinker_utils.sampler import get_or_create_sampler_with_default
        import tinker
        import transformers
        import httpx

        calls, grades, questions = [], [], []
        class Renderer:
            def build_generation_prompt(self, messages):
                question = messages[0]['content']
                questions.append(question)
                # Short encoding used solely to replace the tokenizer/device.
                return SimpleNamespace(to_ints=lambda: [int('Repair this invalid' in question), 7])
            def get_stop_sequences(self):
                return [2]
            def parse_response(self, ids):
                return {'content': ''.join(map(chr, ids))}, True

        class HTTP:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, json):
                calls.append(json)
                repair = json['prompt'][0] == 1
                choices = []
                for i in range(json['n']):
                    code = f'valid_{len(calls)}_{i}' if repair or i == 0 else f'broken_{len(calls)}'
                    text = f'</think>\n```python\n{code}\n```'
                    ids = list(map(ord, text))
                    choices.append(dict(token_ids=ids, loss_mask=[1]*len(ids), finish_reason='stop',
                        thinking_budget=dict(enforced=True, budget_basis='phase1_generated_tokens',
                                             counted_phase1_tokens=0)))
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(choices=choices))

        async def grade(env, code, step):
            grades.append(code)
            valid = code.startswith('valid')
            return VerifyResult(reward=.5 if valid else 0, correctness=int(valid),
                raw_score=.5 if valid else 0, msg='OK' if valid else 'NameError: missing_helper',
                stdout='score: 0.5' if valid else 'NameError: missing_helper',
                result_construction=None, metrics={})

        with tempfile.TemporaryDirectory() as temp:
            old = Config.load(PROFILE)
            cfg = replace(old, bootstrap_layers=2, bootstrap_all_hosts=True,
                client_env={**old.client_env, 'GROUPS_PER_BATCH': '2', 'GROUP_SIZE': '2'})
            with patch.dict('os.environ', TTD_RUN_DIR=temp), \
                    patch.object(training_env, 'connect'), \
                    patch.object(training_env.ScienceTrainingEnv, '_safe_grade', grade), \
                    patch.object(renderers, 'get_renderer', return_value=Renderer()), \
                    patch.object(transformers.AutoTokenizer, 'from_pretrained',
                                 return_value=SimpleNamespace(get_vocab=lambda: {'x': 1})), \
                    patch.object(httpx, 'AsyncClient', return_value=HTTP()), \
                    patch.object(tinker, 'ServiceClient', side_effect=AssertionError('No training API allowed')):
                summary = await bootstrap.run(cfg, 'fake-snapshot', 'head')
                self.assertEqual((summary['total'], summary['valid'], summary['retained']), (8, 6, 4))
                self.assertEqual(summary['optimizer_steps'], 0)
                plan = bootstrap.read(Path(temp)/'bootstrap/layer-1-plan.json')
                self.assertEqual(len(plan), 2)
                for entry in plan:
                    self.assertIn('broken_', entry['question'])
                    self.assertIn('NameError: missing_helper', entry['question'])
                    self.assertNotIn('Valid initial policy block:', entry['question'])
                    self.assertIn('Fixed Rust scaffold (read-only):', entry['question'])
                log = Path(temp)/'tinker_log'/cfg.run_id
                pool = bootstrap.read(log/'puct_sampler_step_000000.json')
                self.assertEqual(pool['puct_T'], 8)
                self.assertFalse(any(s['code'].startswith('broken') for s in pool['states']))
                root_ids = {s['id'] for s in pool['initial_states']}
                self.assertTrue(all(s['parents'][0]['id'] in root_ids for s in pool['states'] if s['code']))
                # The unchanged normal-training sampler must load the promoted
                # seed pool, with optimizer timestamp still at step zero.
                normal = get_or_create_sampler_with_default(str(log), training_env.RoutingTrainingEnv,
                                                            'routing', 2)
                self.assertEqual(len(normal._states), 6)
                self.assertEqual(normal._current_step, 0)
                previous = (len(calls), len(grades))
                self.assertEqual(await bootstrap.run(cfg, 'fake-snapshot', 'head'), summary)
                # Simulate a crash immediately before writing the commit marker.
                (Path(temp)/'bootstrap/complete.json').unlink()
                self.assertEqual(await bootstrap.run(cfg, 'fake-snapshot', 'head'), summary)
                self.assertEqual((len(calls), len(grades)), previous)
                self.assertEqual(bootstrap.read(log/'puct_sampler_step_000000.json'), pool)
                with self.assertRaisesRegex(ValueError, 'contract changed'):
                    await bootstrap.run(replace(cfg, bootstrap_layers=1), 'fake-snapshot', 'head')

                # Three independent inference-only slices share the original
                # total group budget; the normal sampler merges all journals.
                from tpu.science.seed_pool import merge, verify_pool
                clients = []
                for shard in range(3):
                    client = Path(temp)/f'shard-{shard}'
                    seed = replace(cfg, run_id=f'seed-{shard}', root=str(client),
                        accelerator='tpu-v4-32', hosts=4, inference_only=True, bootstrap_only=True, training_smoke=False,
                        trainer=replace(cfg.trainer, hosts=0),
                        client_env={**cfg.client_env, 'GROUPS_PER_BATCH': '1'})
                    with patch.dict('os.environ', TTD_RUN_DIR=str(client)):
                        result = await bootstrap.run(seed, 'fake-snapshot', 'head')
                    self.assertEqual(result['optimizer_steps'], 0)
                    clients.append(client)
                target = replace(cfg, bootstrap_layers=0, bootstrap_all_hosts=False,
                                 client_env={**cfg.client_env, 'GROUPS_PER_BATCH': '3'})
                report = merge(clients, target, Path(temp)/'merged')
                self.assertEqual((report['roots'], report['total'], report['valid']), (3, 12, 9))
                merged = verify_pool(Path(temp)/'merged/puct_sampler_step_000000.json', report['pool_sha256'])
                self.assertEqual(merged['puct_T'], 12)
                self.assertFalse(any(s['code'].startswith('broken') for s in merged['states']))
                with self.assertRaisesRegex(ValueError, 'duplicate seed shard'):
                    merge([clients[0], clients[0], clients[2]], target, Path(temp)/'duplicate')
                plan_path = clients[0]/'bootstrap/layer-1-plan.json'
                plan = bootstrap.read(plan_path)
                plan[0]['repair_parent_id'] = 'not-an-invalid-draft'
                bootstrap.save(plan_path, plan)
                with self.assertRaisesRegex(ValueError, 'invalid draft'):
                    merge(clients, target, Path(temp)/'tampered')


class ServingTransitionTest(unittest.IsolatedAsyncioTestCase):
    async def test_drain_fences_retired_engines_and_removes_all_routes(self):
        from tpu.swarm.ray_train import serving
        from fastapi import HTTPException
        ips = ['10.0.0.1', '10.0.0.2', '10.0.0.3']
        catalog = serving.Catalog.__ray_metadata__.modified_class(ips, 0)
        for ip in ips:
            catalog.register(ip, ip)
        retired = []
        routed = []
        handles = [SimpleNamespace(retire=Remote(lambda ip=ip: retired.append(ip)),
                                   generate=Remote(lambda payload, ip=ip: routed.append(ip)),
                                   tokenize=Remote(lambda payload, ip=ip: routed.append(ip))) for ip in ips]
        with tempfile.TemporaryDirectory() as temp:
            cfg = replace(Config.load(PROFILE), root=temp, bootstrap_layers=2, bootstrap_all_hosts=True)
            original = serving.Ingress.func_or_class.__mro__[1]
            gateway = original(cfg.to_dict(), handles, SimpleNamespace(
                restrict=Remote(catalog.restrict), snapshot=Remote(catalog.snapshot)), ips)
            gateway.active = 1
            try:
                transition = asyncio.create_task(gateway.restrict_engines(ips[1:]))
                await asyncio.sleep(0)
                self.assertTrue(gateway.updating)
                self.assertEqual(retired, [])
                request = SimpleNamespace(json=lambda: None)
                async def data():
                    return {'model': cfg.model}
                request.json = data
                with self.assertRaises(HTTPException):
                    await gateway.tokenize(request)
                async with gateway.condition:
                    gateway.active = 0
                    gateway.condition.notify_all()
                await transition
                self.assertEqual(retired, ips[:1])
                self.assertEqual(gateway.engine_urls, [f'http://{ip}:{cfg.ports.engine}' for ip in ips[1:]])
                self.assertEqual(catalog.snapshot()['expected'], ips[1:])
                with self.assertRaises(RuntimeError):
                    catalog.register(ips[0], 'late-restart')
                with self.assertRaises(RuntimeError):
                    catalog.claim(ips[0])
                # Once the controller has installed the reduced graph, every
                # generated request must route only to surviving handles.
                gateway.updating = False
                for _ in range(6):
                    await gateway.generate(request)
                self.assertEqual(set(routed), set(ips[1:]))
                # Default profiles retain tokenization behavior during adapter
                # updates; expanded bootstrap is the only opt-in change.
                gateway.config = replace(cfg, bootstrap_layers=0, bootstrap_all_hosts=False)
                gateway.updating = True
                await gateway.tokenize(request)
            finally:
                await gateway.http.aclose()


if __name__ == '__main__':
    unittest.main()
