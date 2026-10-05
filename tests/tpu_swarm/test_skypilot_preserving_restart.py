import importlib.util
from pathlib import Path
import sqlite3

import pytest


SPEC = importlib.util.spec_from_file_location(
    'preserving_restart', Path(__file__).resolve().parents[2] /
    'tpu/swarm/skypilot_preserving_restart.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(name='sky.launch', status='RUNNING', request_id='launch'):
    return dict(request_id=request_id, name=name, status=status, pid=123,
                cluster_name='tpuswarm-test-1', user_id='7bfcb694', created_at=1)


def test_exec_replayed_before_launch():
    rows = [row(), row('sky.exec', 'PENDING', 'exec')]
    assert MODULE.validate_requests(rows, {'requests': rows}, set())[0][
        'request_id'] == 'exec'


@pytest.mark.parametrize('change', [dict(name='sky.down'),
                                   dict(name='sky.exec'),
                                   dict(user_id='someone-else'),
                                   dict(cluster_name='unrelated')])
def test_unsafe_operations_refused(change):
    request = row() | change
    with pytest.raises(RuntimeError):
        MODULE.validate_requests([request], {'requests': [request]}, set())


def test_changed_or_missing_request_refused():
    with pytest.raises(RuntimeError):
        MODULE.validate_requests([row() | dict(pid=456)],
                                 {'requests': [row()]}, set())
    with pytest.raises(RuntimeError):
        MODULE.validate_requests([], {'requests': [row()]}, set())


def test_database_preserves_terminal_records_and_request_ids(tmp_path):
    database = tmp_path / 'requests.db'
    rows = [row(), row('sky.exec', 'PENDING', 'exec'),
            row('sky.pool-status-refresh', 'RUNNING', 'daemon'),
            row('sky.exec', 'SUCCEEDED', 'done')]
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE requests (request_id TEXT PRIMARY KEY, '
                           'name TEXT,status TEXT,pid INTEGER,cluster_name TEXT,'
                           'user_id TEXT,created_at REAL)')
        connection.executemany('INSERT INTO requests VALUES (?,?,?,?,?,?,?)',
                               [tuple(r.values()) for r in rows])
    assert MODULE.preserve_database(database, {'requests': rows}, {'daemon'}) == [
        'exec', 'launch']
    with sqlite3.connect(database) as connection:
        assert connection.execute('SELECT request_id,status,pid FROM requests '
                                  'ORDER BY request_id').fetchall() == [
            ('done', 'SUCCEEDED', 123), ('exec', 'PENDING', None),
            ('launch', 'PENDING', None)]
