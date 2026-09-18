"""Admission/resource regression checks; no TPU or cloud allocations."""
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tpu.science.cpu_slots import acquire_slot, slot_cpus
from tpu.swarm.ray_train.config import Config

PROFILE = 'tpu/swarm/ray_train/profiles/science-routing-v4-qwen-bootstrap-l2-001.json'


def competing_worker(root, queue):
    try:
        slot, handle = acquire_slot(root, slots=16, deadline_seconds=.1)
        with handle:
            queue.put(slot)
    except TimeoutError:
        queue.put('full')


class CpuSlotsTest(unittest.TestCase):
    def test_sixteen_process_safe_slots_and_reuse_after_release(self):
        ctx = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as root:
            handles = []
            try:
                assigned = []
                for _ in range(16):
                    slot, handle = acquire_slot(root, slots=16, deadline_seconds=1)
                    assigned.append(slot); handles.append(handle)
                self.assertEqual(len(set(assigned)), 16)
                cpus = [cpu for slot in assigned for cpu in slot_cpus(slot)]
                self.assertEqual(len(set(cpus)), 64)
                self.assertTrue(set(cpus).isdisjoint(range(16)))
                queue = ctx.Queue()
                worker = ctx.Process(target=competing_worker, args=(root, queue))
                worker.start(); self.assertEqual(queue.get(timeout=10), 'full')
                worker.join(10); self.assertEqual(worker.exitcode, 0)
                handles[7].close()
                worker = ctx.Process(target=competing_worker, args=(root, queue))
                worker.start(); self.assertEqual(queue.get(timeout=10), 7)
                worker.join(10); self.assertEqual(worker.exitcode, 0)
                queue.close(); queue.join_thread()
            finally:
                for handle in handles: handle.close()

    def test_legacy_default_still_admits_only_two(self):
        with tempfile.TemporaryDirectory() as root:
            _, first = acquire_slot(root)
            _, second = acquire_slot(root)
            try:
                with self.assertRaises(TimeoutError): acquire_slot(root, deadline_seconds=.01)
            finally:
                first.close(); second.close()

    def test_rejects_unsafe_or_overbudget_admission(self):
        with tempfile.TemporaryDirectory() as root:
            for slots in (0, 17, True, 2.5):
                with self.subTest(slots=slots), self.assertRaises(ValueError):
                    acquire_slot(root, slots=slots)
            path = Path(root)/'cpu-slot-0.lock'
            path.symlink_to(Path(root)/'outside')
            with self.assertRaises(OSError): acquire_slot(root, deadline_seconds=.01)
            path.unlink(); os.chmod(root, 0o777)
            with self.assertRaises(RuntimeError): acquire_slot(root, deadline_seconds=.01)

    def test_profile_capacity_propagates_without_changing_candidate_budget(self):
        from tpu.swarm.ray_train.commands import client_environment
        from tpu.swarm.ray_train.overlay import manifest
        old = Config.load(PROFILE)
        self.assertEqual(old.science_routing_slots_per_host, 2)
        self.assertEqual(old.ray_cpus_per_host, 32)
        new = replace(old, science_routing_slots_per_host=16)
        new.validate()
        self.assertEqual(Config.from_dict(new.to_dict()), new)
        self.assertGreaterEqual(new.ray_cpus_per_host, 16 * 4 + 1)
        env = client_environment(new, Path('/runtime'), 'head')
        self.assertEqual(env['SCIENCE_ROUTING_SLOTS_PER_HOST'], '16')
        self.assertEqual(env['NUM_CPUS_PER_TASK'], '4')
        self.assertEqual(env['EVAL_TIMEOUT'], old.client_env['EVAL_TIMEOUT'])
        self.assertIn('tpu/science/cpu_slots.py', manifest(Path(__file__).resolve().parents[2], new))
        for slots in (0, 17, True):
            with self.assertRaises(ValueError): replace(new, science_routing_slots_per_host=slots).validate()
        with self.assertRaises(ValueError):
            replace(new, cache=replace(new.cache, inference_gib=192)).validate()
        placement = Config.load('tpu/swarm/ray_train/profiles/science-placement-v4-qwen-xplace-bootstrap-l2-001.json')
        self.assertEqual(placement.ray_cpus_per_host, 32)
        with self.assertRaises(ValueError): replace(placement, science_routing_slots_per_host=16).validate()

    def test_only_known_legacy_seed_implementation_is_compatible(self):
        from tpu.science.seed_pool import compatible_bootstrap_implementation, PRE_SLOT_UPDATE_BOOTSTRAP_SHA256
        import hashlib
        path = Path(__file__).resolve().parents[2]/'tpu/science/bootstrap.py'
        self.assertTrue(compatible_bootstrap_implementation(hashlib.sha256(path.read_bytes()).hexdigest()))
        self.assertTrue(compatible_bootstrap_implementation(PRE_SLOT_UPDATE_BOOTSTRAP_SHA256))
        self.assertFalse(compatible_bootstrap_implementation('0' * 64))

    def test_task_failure_releases_admission(self):
        from tpu.science import ray_cpu
        with tempfile.TemporaryDirectory() as root:
            payload = Path(root)/'payload'
            (payload/'.science').mkdir(parents=True)
            (payload/'.science/ready.json').write_text('{}')
            locks = Path(root)/'locks'
            def acquire(**kwargs):
                return acquire_slot(locks, **kwargs)
            with patch.object(ray_cpu, 'acquire_slot', side_effect=acquire), \
                    patch.object(ray_cpu, '_grade_admitted', side_effect=RuntimeError('worker failed')):
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    ray_cpu.grade._function('routing', 'code', payload, slots_per_host=16)
            slot, lock = acquire_slot(locks, slots=16, deadline_seconds=.1)
            with lock: self.assertEqual(slot, 0)

    def test_completed_build_cleanup_preserves_evidence_and_symlink_targets(self):
        from tpu.science.ray_cpu import cleanup_builds
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'job'; work=folder/'evaluation'; work.mkdir(parents=True)
            for name in ['target','rust','out']:
                (work/name).mkdir(); (work/name/'artifact').write_text(name)
            (folder/'candidate.py').write_text('candidate')
            (folder/'verdict.json').write_text('{}')
            (work/'build.log').write_text('log')
            self.assertEqual(cleanup_builds(folder), ['target','rust'])
            for p in [folder/'candidate.py',folder/'verdict.json',work/'build.log',work/'out/artifact']:
                self.assertTrue(p.exists())
            self.assertEqual(cleanup_builds(folder), [])
            (work/'target').symlink_to(work/'out', target_is_directory=True)
            cleanup_builds(folder)
            self.assertEqual((work/'out/artifact').read_text(), 'out')


class DispatchTest(unittest.IsolatedAsyncioTestCase):
    async def test_client_passes_expanded_limit_to_ray(self):
        from concurrent.futures import Future
        from tpu.science import training_env as env, ray_cpu
        future = Future(); future.set_result({'reward': .5, 'raw_score': .5})
        class Ref:
            def future(self): return future
        for suite in ('full', 'q20'):
            with patch.object(env, 'connect'), patch.dict(os.environ, SCIENCE_WORKER_ROOT='/payload',
                    SCIENCE_ROUTING_SLOTS_PER_HOST='16', SCIENCE_ROUTING_SUITE=suite), \
                    patch.object(ray_cpu.grade, 'options') as opts, patch.object(env.ray, 'cancel'):
                opts.return_value.remote.return_value = Ref()
                await env.evaluate('routing', 'code', 100)
                self.assertEqual(opts.return_value.remote.call_args.kwargs,
                                 {'admission_timeout_s': 100, 'slots_per_host': 16, 'routing_suite': suite})



if __name__ == '__main__':
    unittest.main()
