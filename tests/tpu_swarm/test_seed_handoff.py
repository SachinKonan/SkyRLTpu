import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tpu.science.seed_handoff import seed_job_status


class SeedHandoffTest(unittest.TestCase):
    def test_timeout_is_unknown_then_successful_poll_recovers(self):
        shards = [{'job_id': 1}, {'job_id': 2}]
        with patch('subprocess.run', side_effect=[subprocess.TimeoutExpired('sky', 60),
                   SimpleNamespace(returncode=0, stdout='1 x SUCCEEDED\n2 x SUCCEEDED\n', stderr='')]):
            pending, error = seed_job_status('sky', shards, {})
            self.assertIsNone(pending)
            self.assertIn('retry', error)
            self.assertEqual(seed_job_status('sky', shards, {}), ([], None))

    def test_missing_rows_remain_pending_and_failed_jobs_block_training(self):
        with patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')):
            self.assertEqual(seed_job_status('sky', [{'job_id': 1}], {}), (['job-1'], None))
        for status in ('FAILED', 'FAILED_SETUP', 'CANCELLED'):
            with self.subTest(status=status), patch('subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout='1 x '+status, stderr='')):
                with self.assertRaisesRegex(RuntimeError, 'no training submitted'):
                    seed_job_status('sky', [{'job_id': 1}], {})

    def test_control_plane_error_is_retriable(self):
        with patch('subprocess.run', return_value=SimpleNamespace(returncode=1, stdout='', stderr='Unavailable')):
            pending, error = seed_job_status('sky', [{'job_id': 1}], {})
            self.assertIsNone(pending)
            self.assertIn('Unavailable', error)


if __name__ == '__main__':
    unittest.main()
