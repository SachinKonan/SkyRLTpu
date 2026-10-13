from contextlib import contextmanager
from copy import deepcopy

import pytest

from tpu.swarm.ray_train.pool_recovery import release_stale_claim


def attempt():
    return dict(name='ours', status='PENDING', cluster='worker', pool='pool',
                worker_job_id=4, recovering_from_failure=1)


def evidence():
    return dict(latest=dict(job_id=4, job_name='ours', status='FAILED', end_at=10),
                active=[], driver_alive=False)


def run(job=None, observed=None, owners=(7,), transition=False):
    job = attempt() if job is None else job
    observed = evidence() if observed is None else observed
    released, locked = [], []

    @contextmanager
    def lock(pool):
        locked.append(pool)
        yield
        locked.pop()

    reads = 0

    def read(_):
        nonlocal reads
        reads += 1
        result = deepcopy(job)
        if transition and reads == 3:
            result['status'] = 'RUNNING'
        return result

    def inspect(_):
        assert locked == ['pool']
        if isinstance(observed, Exception):
            raise observed
        return observed

    def release(j):
        assert locked == ['pool']
        released.append(j)

    proof = release_stale_claim(7, {'ours'}, read_job=read, claimants=lambda *_: owners,
                               inspect_worker=inspect, release=release,
                               pool_lock=lock, now=lambda: 200)
    return proof, released


def test_verified_terminal_attempt_releases_under_lock():
    proof, released = run()
    assert released == [7]
    assert proof['worker_job_id'] == 4


@pytest.mark.parametrize('key,value', [('name', 'other'), ('status', 'RUNNING'),
                                     ('status', 'STARTING'), ('cluster', None),
                                     ('recovering_from_failure', 0), ('worker_job_id', 3)])
def test_nonrecovering_or_changed_identity_preserved(key, value):
    j = attempt(); j[key] = value
    assert run(job=j)[1] == []


@pytest.mark.parametrize('key,value', [('job_name', 'other'), ('status', 'RUNNING'),
                                     ('end_at', None), ('end_at', 150), ('job_id', 5)])
def test_unproven_or_new_attempt_preserved(key, value):
    e = evidence(); e['latest'][key] = value
    assert run(observed=e)[1] == []


def test_active_worker_job_or_driver_preserved():
    for key, value in [('active', [5]), ('driver_alive', True)]:
        e = evidence(); e[key] = value
        assert run(observed=e)[1] == []


def test_shared_claim_and_controller_transition_preserved():
    assert run(owners=(7, 8))[1] == []
    assert run(transition=True)[1] == []


def test_timeout_is_not_terminal_proof():
    with pytest.raises(TimeoutError):
        run(observed=TimeoutError())
