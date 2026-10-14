import json
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train import discovery


def farm_result(instance, model='Qwen/Qwen3.5-27B', url='http://10.0.0.5:24800'):
    return dict(models=[model], url=url, owner_run=None, state='unleased', active=0,
                capabilities=dict(compatibility_sha256='c' * 64), instance=instance,
                grading=dict(ready=True, families=['routing'], queue_depth=0))


def args(tmp_path, **kw):
    base = dict(registry=tmp_path / 'farms.json', ssh_config_dir=tmp_path, name_contains=['inference-farm'],
                trainer_pool=[], trainer_job_id=[], discovery_group='', run_scoped_only=False,
                farm_port=24800, tunnels=False, port_base=29000, push=False)
    base.update(kw)
    return SimpleNamespace(**base)


ROW = dict(job_id=2457, status='RUNNING', cluster='tpuswarm-v6e8-east5b-3861',
           run_id='inference-farm-rq4-qwen35-east5-v6e8-001', pool='tpuswarm-v6e8-east5b', priority=0)


def test_directory_health_and_incarnation():
    d = discovery.Directory()
    assert d.observe('a', dict(ok=True, result=dict(instance='i1')), 0) is None
    assert d.entries['a']['health'] == 'healthy'
    for t in (1, 2):
        d.observe('a', dict(ok=False), t)
    assert d.entries['a']['health'] == 'suspect' and d.entries['a']['misses'] == 2
    d.observe('a', dict(ok=False), 3)
    assert d.entries['a']['health'] == 'unhealthy'
    # Preempted and relaunched: same job, new ingress process.
    assert d.observe('a', dict(ok=True, result=dict(instance='i2')), 4) == 'returned'
    entry = d.entries['a']
    assert entry['health'] == 'healthy' and entry['first_seen'] == 4 and entry['returns'] == 1
    assert d.retire({'b'}) == ['a'] and not d.entries


def test_tick_publishes_local_farms_schema_and_detects_return_and_gone(tmp_path, monkeypatch, capsys):
    instances = iter(['i1', 'i1', 'i2'])
    monkeypatch.setattr(discovery, 'rpc', lambda d, cluster, action, port, body=None, group='':
                        dict(ok=True, result=farm_result(next(instances))))
    clock = iter(range(100))
    service = discovery.Service(args(tmp_path), dict(farms=[], trainers=[]), now=lambda: next(clock))
    for _ in range(3):
        doc = service.tick([ROW, dict(ROW, job_id=1, status='RUNNING', run_id='lab-gradeab-v6e8')])
    farm, = doc['farms']
    for key in ('models', 'url', 'internal_url', 'owner_run', 'state', 'ok', 'tunnel_ok', 'job_id', 'name',
                'cluster', 'capabilities', 'error'):
        assert key in farm
    assert farm['internal_url'] == 'http://10.0.0.5:24800' and farm['job_id'] == '2457'
    assert farm['incarnation'] == 'i2' and farm['returns'] == 1 and farm['health'] == 'healthy'
    assert json.loads((tmp_path / 'farms.json').read_text())['shadow'] is True
    assert 'farm_returned' in capsys.readouterr().out
    doc = service.tick([dict(ROW, status='RECOVERING')])
    assert doc['farms'] == [] and 'farm_gone' in capsys.readouterr().out


def test_private_farm_and_shadow_direct_trainer_do_not_post(tmp_path, monkeypatch):
    posted = []
    def private_rpc(entry, action, port, body=None, group=''):
        if body is not None:
            posted.append(entry)
        if action == 'farm':
            return dict(ok=True, result=farm_result('p1', url='http://10.128.0.9:24800'))
        return dict(ok=True, result=dict(run_id='od-run', model='Qwen/Qwen3.8', instance='t1', enabled=True,
                                         lease_scope='run', urls=[], candidate_limit=4))
    monkeypatch.setattr(discovery, 'private_rpc', private_rpc)
    monkeypatch.setattr(discovery, 'rpc', lambda *a, **k: dict(ok=False, error='ssh_failed'))
    private = dict(farms=[dict(name='v5p64-farm-0', external_ip='34.0.0.1', ssh_key='~/.ssh/k')],
                   trainers=[dict(run_id='od-run', model='Qwen/Qwen3.8', group='g', external_ip='34.0.0.2',
                                  ssh_key='~/.ssh/k', stable_urls=['http://10.128.0.9:24800'])])
    service = discovery.Service(args(tmp_path), private, now=lambda: 0)
    doc = service.tick([ROW])
    by_source = {f['source']: f for f in doc['farms']}
    assert by_source['private']['ok'] and by_source['private']['job_id'] == str(discovery.PRIVATE_ID_BASE)
    # Ready-only contract: the unreachable pool farm is listed under `unready`, never under `farms`.
    assert 'pool' not in by_source
    unready, = doc['unready']
    assert unready['job_id'] == str(ROW['job_id']) and unready['health'] == 'suspect'
    assert doc['trainers']['direct'] == [dict(run_id='od-run', state='would_update',
                                              urls=['http://10.128.0.9:24800'])]
    assert posted == []


def test_private_hosts_file_is_validated(tmp_path):
    path = tmp_path / 'private.json'
    path.write_text(json.dumps(dict(farms=[dict(name='x', external_ip='1.2.3.4')])))
    with pytest.raises(ValueError, match='ssh_key'):
        discovery.load_private(path)
    assert discovery.load_private(None) == dict(farms=[], trainers=[])
