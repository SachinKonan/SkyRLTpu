"""Release a campaign's stale pool claim only after a terminal worker audit.

The caller holds SkyPilot's pool lock across inspection and release. This never
stops processes, changes job status, or submits a second attempt; the existing
managed-job controller performs recovery and its normal clean-host checks.
"""


REMOTE_ATTEMPT_AUDIT = r'''
import json, sqlite3
from pathlib import Path
path = Path.home() / '.sky/jobs.db'
db = sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True)
db.row_factory = sqlite3.Row
latest = db.execute('SELECT job_id,job_name,status,pid,end_at FROM jobs '
                    'ORDER BY job_id DESC LIMIT 1').fetchone()
active = db.execute("SELECT job_id FROM jobs WHERE status NOT IN "
                    "('SUCCEEDED','FAILED','FAILED_SETUP','CANCELLED')").fetchall()
row = dict(latest) if latest else None
print(json.dumps(dict(latest=row, active=[x['job_id'] for x in active],
                      driver_alive=bool(row and row['pid'] and
                                        Path('/proc', str(row['pid'])).exists()))))
'''


def release_stale_claim(job_id, allowed_runs, *, read_job, claimants,
                        inspect_worker, release, pool_lock, now):
    """Return evidence after a verified release, or None when unsafe/unneeded.

    Observation errors propagate: a missing SSH handle is never terminal proof.
    Re-read identity after remote inspection to reject controller transitions.
    """
    candidate = read_job(job_id)
    if not candidate or candidate['name'] not in allowed_runs:
        return None
    if candidate['status'] not in {'PENDING', 'RECOVERING'} or not candidate['cluster']:
        return None
    with pool_lock(candidate['pool']):
        current = read_job(job_id)
        if current != candidate or current.get('recovering_from_failure') != 1:
            return None
        if set(claimants(current['pool'], current['cluster'])) != {job_id}:
            return None
        observed = inspect_worker(current['cluster'])
        last = observed.get('latest') or {}
        if (observed.get('active') != [] or observed.get('driver_alive') is not False
                or last.get('job_id') != current.get('worker_job_id')
                or last.get('job_name') != current['name']
                or last.get('status') not in {'FAILED', 'FAILED_SETUP', 'CANCELLED'}
                or not last.get('end_at') or now() - last['end_at'] < 120):
            return None
        if read_job(job_id) != current:
            return None
        release(job_id)
        return {'job_id': job_id, 'cluster': current['cluster'],
                'worker_job_id': last['job_id'], 'observed': observed}
