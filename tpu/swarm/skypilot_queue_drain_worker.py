"""Run one supplemental executor for stalled SkyPilot pool launch requests.

Operational bridge for deploying the TPU queued-resource pause fix without
restarting an API whose original executors are inside blocking capacity waits.
This starts no API, controller, or queue manager and never invents requests.
The normal execution wrapper atomically claims existing PENDING/WAITING IDs;
the original executor skips an ID already RUNNING or finished. Processing one
ID at a time avoids a second prefetched backlog. Use the same HOME/config and
credentials as the existing bounded API. Stop only when this worker is idle.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sqlite3
import time
import urllib.request


def pending_launches(home):
    database = Path(home) / '.sky' / 'api_server' / 'requests.db'
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
        return connection.execute(
            "SELECT request_id FROM requests WHERE name='sky.launch' "
            "AND status IN ('PENDING', 'WAITING') AND user_id='7bfcb694' "
            "AND cluster_name LIKE 'tpuswarm-%' ORDER BY "
            "CASE WHEN cluster_name LIKE 'tpuswarm-v6e32-east5b-%' "
            "THEN 1 ELSE 0 END, created_at").fetchall()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--api-pid', type=int, required=True)
    args = parser.parse_args()

    import psutil
    from sky import exceptions
    from sky.provision.gcp import instance_utils
    from sky.server import clean_env
    from sky.server.requests import executor
    from sky.server.requests import requests

    api = psutil.Process(args.api_pid)
    assert api.uids().real == os.getuid(), 'API belongs to another user'
    assert 'sky.server.server' in api.cmdline(), 'Not a SkyPilot API process'
    assert '--deploy' in api.cmdline(), 'Requires a shared multiprocessing queue'
    assert '--port=46580' in api.cmdline(), 'Unexpected API port'
    assert os.environ['HOME'] == api.environ()['HOME'], 'Durable HOME mismatch'
    with urllib.request.urlopen('http://127.0.0.1:46580/api/health',
                                timeout=5) as response:
        assert response.status == 200
    # Check the installed fix, not just the file on disk.
    assert any(getattr(value, 'co_name', None) == 'pause_if_scheduled'
               for value in instance_utils.GCPTPUVMInstance.
               wait_for_queued_resource.__code__.co_consts)
    pending = pending_launches(os.environ['HOME'])
    print(f'Found {len(pending)} existing pending/waiting pool launch IDs; '
          'supplemental parallelism=1, burst=0', flush=True)
    if not args.run:
        return

    lock_path = Path.home() / '.sky' / 'queue-drain-worker.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.sched_setaffinity(0, sorted(api.cpu_affinity())[:2])
        clean_env.capture_clean_server_env()
        executor.executor_initializer('pool-drain', dict(os.environ))
        api_started = api.create_time()
        not_before = {}
        while api.is_running() and api.create_time() == api_started:
            candidates = pending_launches(os.environ['HOME'])
            eligible = [request_id for (request_id,) in candidates
                        if not_before.get(request_id, 0) <= time.monotonic()]
            if not eligible:
                time.sleep(5)
                continue
            # Visit every new request before cycling back to earlier waits.
            request_id = min(eligible, key=lambda item: not_before.get(item, 0))
            retry_wait = 60
            try:
                executor._request_execution_wrapper(request_id, False, 2)
            except exceptions.ExecutionRetryableError as error:
                retry_wait = max(1, error.retry_wait_seconds)
                with requests.update_request(request_id) as request:
                    if (request is not None and request.status ==
                            requests.RequestStatus.RUNNING and request.pid is None):
                        request.status = requests.RequestStatus.WAITING
                        request.status_msg = executor._waiting_status_msg(
                            str(error), f'retrying in {retry_wait}s')
                print(f'Parked existing request {request_id}: {error}', flush=True)
            not_before[request_id] = time.monotonic() + retry_wait
            time.sleep(1)


if __name__ == '__main__':
    main()
