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


def test_named_farms_across_pools_keep_existing_reservations():
    rows, farms, targets, calls, call = fixture()
    rows.append(dict(job_id=20, status='RUNNING', cluster='v5p-20',
                     run_id='Inference-Farm-v5p32-qwen', pool='v5p'))
    farms['v5p-20'] = dict(models=['qwen'], url='http://10.0.5.1:24800',
                           state='unleased', capabilities={'compatibility_sha256': 'match'})
    target = targets['train-10']
    target.update(lease_scope='run', compatibility_sha256='match')
    farms['farm-1'].update(state='ready', owner_run='train-qwen:instance', active=4)
    tick(rows, 'farm', [10], call, run_scoped_only=True)
    assert target['urls'] == ['http://10.0.0.1:24800']
    # When the old farm disappears, the named v5p capacity becomes eligible.
    rows[0]['status'] = 'PENDING'
    tick(rows, 'farm', [10], call, run_scoped_only=True)
    assert target['urls'] == ['http://10.0.5.1:24800']


def test_name_only_discovery_skips_unready_unrelated_and_incompatible_farms():
    rows, farms, targets, calls, call = fixture()
    target = targets['train-10']
    target.update(lease_scope='run', compatibility_sha256='match')
    for job, state, digest in [(20, 'RUNNING', 'wrong'), (21, 'STARTING', 'match'),
                                (22, 'RUNNING', 'match')]:
        cluster = f'v5p-{job}'
        rows.append(dict(job_id=job, status=state, cluster=cluster,
                         run_id=f'inference-farm-qwen-{job}', pool='v5p'))
        farms[cluster] = dict(models=['qwen'], url=f'http://10.0.5.{job}:24800',
                               state='unleased', capabilities={'compatibility_sha256': digest})
    # A name match alone cannot make an unreachable/non-farm service eligible.
    rows.append(dict(job_id=23, status='RUNNING', cluster='v5p-23',
                     run_id='inference-farm-broken', pool='v5p'))
    result = tick(rows, None, [10], call, run_scoped_only=True)
    assert target['urls'] == ['http://10.0.5.22:24800']
    assert result['unavailable'][0]['job_id'] == 23
    assert {c for c, action, _ in calls if action == 'farm'} == {'v5p-20', 'v5p-22', 'v5p-23'}


def test_named_farm_dry_run_and_trainer_scope():
    rows, farms, targets, calls, call = fixture()
    rows[0]['run_id'] = 'qwen-inference-farm'
    result = tick(rows, None, [10], call, dry_run=True)
    assert result['targets'][0]['state'] == 'would_update'
    assert targets['train-10']['urls'] == []
    assert all(body is None for _, _, body in calls)
    assert not any(cluster == 'train-11' for cluster, _, _ in calls)


def test_legacy_pool_job_ceiling_preserves_general_name_discovery():
    rows, farms, targets, calls, call = fixture()
    for job, pool, name in [(1372, 'farm', 'old'), (1373, 'farm', 'old-boundary'),
                            (1374, 'farm', 'new-unrelated'), (1370, 'other', 'old-other-pool'),
                            (2000, 'other', 'inference-farm-new')]:
        cluster = f'{pool}-{job}'
        rows.append(dict(job_id=job, pool=pool, cluster=cluster, run_id=name, status='RUNNING'))
        farms[cluster] = dict(models=['qwen'], url=f'http://farm-{job}:24800')
    tick(rows, 'farm', [10], call, farm_pool_max_job_id=1373)
    assert {c for c, action, _ in calls if action == 'farm'} == {
        'farm-1372', 'farm-1373', 'other-2000'}


def test_render_bounded_legacy_pool_exception():
    from tpu.swarm.ray_train.borrowing_service import unit
    text = unit('/code', '/python', '/env', '/ssh', 'farm', ['train'], '/lock',
                farm_pool_max_job_id=1373)
    assert '"--farm-pool" "farm"' in text
    assert '"--farm-pool-max-job-id" "1373"' in text
    assert '"--farm-name-contains" "inference-farm"' in text
