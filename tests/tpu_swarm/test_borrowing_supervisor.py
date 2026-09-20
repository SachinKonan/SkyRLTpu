import copy

from tpu.swarm.ray_train.borrowing_supervisor import tick


def fixture():
    rows = [
        dict(job_id=1, status='RUNNING', cluster='farm-1', run_id='qwen-farm'),
        dict(job_id=2, status='RUNNING', cluster='farm-2', run_id='gemma-farm'),
        dict(job_id=3, status='STARTING', cluster='farm-3', run_id='not-ready'),
        dict(job_id=4, status='RUNNING', cluster='other-4', run_id='unrelated'),
        dict(job_id=10, status='RUNNING', cluster='train-10', run_id='train-qwen'),
        dict(job_id=11, status='RUNNING', cluster='train-11', run_id='train-gemma'),
    ]
    farms = {'farm-1': dict(models=['qwen'], url='http://10.0.0.1:24800'),
             'farm-2': dict(models=['gemma'], url='http://10.0.0.2:24800')}
    targets = {'train-10': dict(enabled=True, model='qwen', run_id='train-qwen', instance='a', urls=[]),
               'train-11': dict(enabled=True, model='gemma', run_id='train-gemma', instance='b', urls=[])}
    calls = []
    def call(cluster, action, body=None):
        calls.append((cluster, action, copy.deepcopy(body)))
        source = farms if action == 'farm' else targets
        if cluster not in source:
            return dict(ok=False, error='unreachable')
        if body is not None:
            assert body['model'] == source[cluster]['model']
            assert body['run_id'] == source[cluster]['run_id']
            assert body['instance'] == source[cluster]['instance']
            source[cluster].update(body)
        return dict(ok=True, result=copy.deepcopy(source[cluster]))
    return rows, farms, targets, calls, call


def test_model_matched_push_and_replacement_ip():
    rows, farms, targets, calls, call = fixture()
    result = tick(rows, 'farm', [10, 11], call)
    assert [r['state'] for r in result['targets']] == ['updated', 'updated']
    assert targets['train-10']['urls'] == ['http://10.0.0.1:24800']
    assert targets['train-11']['urls'] == ['http://10.0.0.2:24800']
    assert not any(cluster in ('farm-3', 'other-4') for cluster, _, _ in calls)
    farms['farm-1']['url'] = 'http://10.0.1.9:24800'
    result = tick(rows, 'farm', [10, 11], call)
    assert [r['state'] for r in result['targets']] == ['updated', 'unchanged']
    assert targets['train-10']['urls'] == ['http://10.0.1.9:24800']


def test_dry_run_never_writes_or_touches_unlisted_trainers():
    rows, farms, targets, calls, call = fixture()
    result = tick(rows, 'farm', [10], call, dry_run=True)
    assert result['targets'][0]['state'] == 'would_update'
    assert all(body is None for _, _, body in calls)
    assert not any(cluster == 'train-11' for cluster, _, _ in calls)


def test_missing_farm_clears_candidate_list():
    rows, farms, targets, calls, call = fixture()
    tick(rows, 'farm', [10], call)
    del farms['farm-1']
    result = tick(rows, 'farm', [10], call)
    assert result['targets'][0]['state'] == 'updated'
    assert targets['train-10']['urls'] == []


def test_reused_worker_or_disabled_target_is_never_updated():
    rows, farms, targets, calls, call = fixture()
    targets['train-10']['run_id'] = 'some-other-job'
    targets['train-11']['enabled'] = False
    result = tick(rows, 'farm', [10, 11, 99], call)
    assert [r['state'] for r in result['targets']] == ['target_identity_mismatch', 'not_opted_in', 'not_running']
    assert all(body is None for _, _, body in calls)
