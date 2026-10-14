"""Discovery output contract: only ready, attested, healthy farms; a timestamp; staleness; watchdog unit."""
import json
import os
import socket

from tpu.swarm.ray_train import discovery
from test_discovery import ROW, args, farm_result


def test_registry_lists_only_ready_attested_healthy_farms(tmp_path, monkeypatch):
    probes = {
        2457: dict(ok=True, result=farm_result('i1')),
        2458: dict(ok=False, error='ValueError'),                     # engines loading: probe not ok
        2459: dict(ok=True, result=dict(farm_result('i3'), capabilities={})),  # not attested
    }
    clusters = {ROW['cluster']: 2457, 'c-2458': 2458, 'c-2459': 2459}
    monkeypatch.setattr(discovery, 'rpc', lambda d, cluster, action, port, body=None, group='': probes[clusters[cluster]])
    service = discovery.Service(args(tmp_path), dict(farms=[], trainers=[]), now=lambda: 1000.0)
    rows = [ROW, dict(ROW, job_id=2458, cluster='c-2458', run_id='inference-farm-rq4-gemma-east5-v6e8-002'),
            dict(ROW, job_id=2459, cluster='c-2459', run_id='inference-farm-rq4-muse-east5-v6e8-001')]
    doc = service.tick(rows)
    assert [f['job_id'] for f in doc['farms']] == ['2457']
    assert {u['job_id'] for u in doc['unready']} == {'2458', '2459'}
    assert doc['updated'] == 1000.0 and doc['contract'] == discovery.READY_ONLY
    on_disk = json.loads((tmp_path / 'farms.json').read_text())
    assert [f['job_id'] for f in on_disk['farms']] == ['2457']


def test_a_suspect_farm_leaves_the_list_until_it_answers_again(tmp_path, monkeypatch):
    answers = iter([dict(ok=True, result=farm_result('i1')), dict(ok=False, error='Timeout'),
                    dict(ok=True, result=farm_result('i1'))])
    monkeypatch.setattr(discovery, 'rpc', lambda *a, **kw: next(answers))
    service = discovery.Service(args(tmp_path), dict(farms=[], trainers=[]), now=lambda: 0)
    assert len(service.tick([ROW])['farms']) == 1
    assert service.tick([ROW])['farms'] == []
    assert len(service.tick([ROW])['farms']) == 1


def test_registry_freshness(tmp_path):
    path = tmp_path / 'farms.json'
    assert discovery.registry_is_fresh(path, 60) == (False, None)
    path.write_text(json.dumps(dict(updated=1000.0, farms=[])))
    assert discovery.registry_is_fresh(path, 60, now=lambda: 1030.0) == (True, 30.0)
    assert discovery.registry_is_fresh(path, 60, now=lambda: 1100.0) == (False, 100.0)
    path.write_text(json.dumps(dict(time=1000.0, farms=[])))  # older registries carry only `time`
    assert discovery.registry_is_fresh(path, 60, now=lambda: 1010.0)[0]


def test_sd_notify_feeds_the_unit_watchdog(tmp_path, monkeypatch):
    address = str(tmp_path / 'notify.sock')
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(address)
    monkeypatch.setenv('NOTIFY_SOCKET', address)
    assert discovery.sd_notify('WATCHDOG=1')
    assert server.recv(64) == b'WATCHDOG=1'
    server.close()
    monkeypatch.delenv('NOTIFY_SOCKET')
    assert discovery.sd_notify('WATCHDOG=1') is False


def test_discovery_unit_restarts_on_exit_and_on_a_missed_watchdog():
    argv = discovery.unit_argv('rq4-discovery', ['python3', '-m', 'tpu.swarm.ray_train.discovery'], '/w', '/w/d.log', 30)
    assert '--property=Restart=always' in argv and '--property=Type=notify' in argv
    assert '--property=WatchdogSec=90' in argv and '--property=Type=exec' not in argv
