"""One-shot API startup using an audited, quiesced request database.

Not a shutdown command. The operator must stop the old API without cancellation,
pause its local controllers, back up state, and provide the exact request
manifest. Normal starts should continue to use sky.server.server directly.
"""
import json
import os
from pathlib import Path
import runpy
import sqlite3


ACTIVE = ('PENDING', 'WAITING', 'RUNNING')


def validate_requests(rows, manifest, daemon_ids):
    expected = {r['request_id']: r for r in manifest['requests']}
    replay = []
    for row in rows:
        if row['status'] not in ACTIVE or row['request_id'] in daemon_ids:
            continue
        original = expected.get(row['request_id'])
        if original is None or any(row[k] != original[k] for k in (
                'name', 'status', 'pid', 'cluster_name', 'user_id')):
            raise RuntimeError('Active request changed after snapshot')
        if row['name'] not in ('sky.launch', 'sky.exec'):
            raise RuntimeError('Unsupported active operation: ' + row['name'])
        if row['user_id'] != '7bfcb694' or not row['cluster_name'].startswith(
                'tpuswarm-'):
            raise RuntimeError('Request outside the approved pools')
        if row['name'] == 'sky.exec' and row['status'] != 'PENDING':
            raise RuntimeError('Cannot replay an ambiguously executed workload')
        replay.append(row)
    if {r['request_id'] for r in replay} != {
            r['request_id'] for r in manifest['requests']
            if r['request_id'] not in daemon_ids and r['status'] in ACTIVE}:
        raise RuntimeError('An audited active request is missing')
    return sorted(replay, key=lambda r: (r['name'] != 'sky.exec', r['created_at']))


def preserve_database(database, manifest, daemon_ids):
    with sqlite3.connect(database, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('BEGIN IMMEDIATE')
        rows = [dict(r) for r in connection.execute('SELECT * FROM requests')]
        replay = validate_requests(rows, manifest, daemon_ids)
        # Internal daemons must be freshly registered by the new API workers.
        connection.executemany('DELETE FROM requests WHERE request_id=?',
                               [(name,) for name in daemon_ids])
        for row in replay:
            connection.execute(
                "UPDATE requests SET status='PENDING',pid=NULL "
                'WHERE request_id=?', (row['request_id'],))
    return [r['request_id'] for r in replay]


def main():
    import psutil
    from sky.server import daemons
    from sky.server.requests import executor, requests, storage

    manifest_path = Path(os.environ['SKYPILOT_RESTART_MANIFEST'])
    manifest = json.loads(manifest_path.read_text())
    if os.environ['HOME'] != manifest['home']:
        raise RuntimeError('Durable HOME mismatch')
    for record in manifest['api_processes']:
        try:
            process = psutil.Process(record['pid'])
            if (process.create_time() == record['created_at'] and
                    process.status() != psutil.STATUS_ZOMBIE):
                raise RuntimeError('Old API process is still alive')
        except psutil.NoSuchProcess:
            pass
    marker = manifest_path.parent / 'preserving-start-claimed'
    with marker.open('x') as stream:
        stream.write(str(os.getpid()))
    replay_ids = []

    def preserve_instead_of_reset():
        nonlocal replay_ids
        database = Path(manifest['home']) / '.sky/api_server/requests.db'
        replay_ids = preserve_database(
            database, manifest, {d.id for d in daemons.INTERNAL_REQUEST_DAEMONS})
        storage.get_request_backend().reset_on_startup()
        print(f'Preserved request database and client files; '
              f'{len(replay_ids)} requests to resume', flush=True)

    original_start = executor.start

    def start_and_requeue(config):
        result = original_start(config)
        for request_id in replay_ids:
            request = requests.get_request(request_id)
            if request is None:
                raise RuntimeError('Preserved request disappeared')
            executor._get_queue(request.schedule_type).put(
                (request_id, False, request.name == 'sky.launch'))
        (manifest_path.parent / 'requests-requeued.json').write_text(
            json.dumps(replay_ids))
        print(f'Requeued {len(replay_ids)} original request IDs', flush=True)
        return result

    requests.reset_db_and_logs = preserve_instead_of_reset
    executor.start = start_and_requeue
    runpy.run_module('sky.server.server', run_name='__main__')


if __name__ == '__main__':
    main()
