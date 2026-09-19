import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import subprocess

from tpu.science.gpu_ray_queue import Queue, atomic_json, has_time
from tpu.science.gpu_ray_queue import job_alive, _JOB_STATUS


def try_claim(path, output):
    q = Queue(path, alive=lambda job: False)
    claim = q.claim({'job': 'test'})
    output.put(None if claim is None else claim[0])
    if claim:
        claim[2].close()


class QueueTests(unittest.TestCase):
    def test_scheduler_timeout_preserves_owner_and_is_cached(self):
        _JOB_STATUS.clear()
        self.addCleanup(_JOB_STATUS.clear)
        with mock.patch('tpu.science.gpu_ray_queue.subprocess.run',
                        side_effect=subprocess.TimeoutExpired('squeue', 15)) as run:
            self.assertTrue(job_alive('owner'))
            self.assertTrue(job_alive('owner'))
            self.assertEqual(run.call_count, 1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy = self.root / 'legacy'
        self.legacy.mkdir()
        self.task = {'method': 'abuplace', 'case': 'ibm01'}
        atomic_json(self.root / 'plan.json', dict(jobs=[self.task], legacy=str(self.legacy)))

    def test_cross_process_exclusion(self):
        claim = Queue(self.root, alive=lambda _: False).claim({'job': 'first'})
        ctx = mp.get_context('spawn')
        output = ctx.Queue()
        process = ctx.Process(target=try_claim, args=(str(self.root), output))
        process.start()
        self.assertIsNone(output.get(timeout=10))
        process.join(10)
        self.assertEqual(process.exitcode, 0)
        claim[2].close()

    def test_owner_must_end_before_retry(self):
        q = Queue(self.root, alive=lambda _: False)
        first = q.claim({'job': 'old'})
        first[2].close()
        self.assertIsNone(Queue(self.root, alive=lambda _: True).claim({'job': 'new'}))
        retry = q.claim({'job': 'new'})
        self.assertNotEqual(first[1], retry[1])
        retry[2].close()

    def test_invalid_results_are_terminal(self):
        folder = self.legacy / 'abuplace-ibm01'
        folder.mkdir()
        atomic_json(folder / 'report.json', {'valid': False})
        self.assertIsNone(Queue(self.root).claim({'job': 'new'}))

    def test_published_results_are_terminal(self):
        q = Queue(self.root, alive=lambda _: False)
        claim = q.claim({'job': 'old'})
        atomic_json(claim[1].parent / 'done.json', {'valid': True})
        claim[2].close()
        self.assertIsNone(q.claim({'job': 'new'}))

    def test_legacy_active_is_not_duplicated(self):
        plan = json.loads((self.root / 'plan.json').read_text())
        plan['legacy_active'] = {'abuplace-ibm01': 'old'}
        atomic_json(self.root / 'plan.json', plan)
        self.assertIsNone(Queue(self.root, alive=lambda _: True).claim({'job': 'new'}))

    def test_deadline_reserves_full_task_and_cleanup(self):
        self.assertTrue(has_time(3600, 270))
        self.assertFalse(has_time(3600, 271))

    def test_incomplete_legacy_method_is_recomputed(self):
        folder = self.legacy / 'abuplace-ibm01'
        folder.mkdir()
        atomic_json(folder / 'report.json', {'valid': True})
        plan = json.loads((self.root / 'plan.json').read_text())
        plan['ignore_legacy_methods'] = ['abuplace']
        atomic_json(self.root / 'plan.json', plan)
        claim = Queue(self.root, alive=lambda _: False).claim({'job': 'new'})
        self.assertIsNotNone(claim)
        claim[2].close()


if __name__ == '__main__':
    unittest.main()
