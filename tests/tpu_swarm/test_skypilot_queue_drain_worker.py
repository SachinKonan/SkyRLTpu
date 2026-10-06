"""The operational queue bridge only selects this user's pending pool launches."""
import importlib.util
from pathlib import Path
import sqlite3


def test_pending_launch_scope(tmp_path):
    script = Path(__file__).resolve().parents[2] / 'tpu/swarm/skypilot_queue_drain_worker.py'
    spec = importlib.util.spec_from_file_location('queue_drain', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    database = tmp_path / '.sky/api_server/requests.db'
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE requests (request_id, name, status, user_id, cluster_name, created_at)')
        rows = [
            ('pending', 'sky.launch', 'PENDING', '7bfcb694', 'tpuswarm-test-1', 1),
            ('waiting', 'sky.launch', 'WAITING', '7bfcb694', 'tpuswarm-test-2', 2),
            ('running', 'sky.launch', 'RUNNING', '7bfcb694', 'tpuswarm-test-3', 3),
            ('finished', 'sky.launch', 'SUCCEEDED', '7bfcb694', 'tpuswarm-test-4', 4),
            ('other-user', 'sky.launch', 'PENDING', 'other', 'tpuswarm-test-5', 5),
            ('other-cluster', 'sky.launch', 'PENDING', '7bfcb694', 'experiment', 6),
            ('other-command', 'sky.down', 'PENDING', '7bfcb694', 'tpuswarm-test-6', 7),
        ]
        connection.executemany('INSERT INTO requests VALUES (?,?,?,?,?,?)', rows)
    assert module.pending_launches(tmp_path) == [('pending',), ('waiting',)]
