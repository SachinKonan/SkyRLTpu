import json
import os
import pwd
import subprocess
import sys

import pytest

from tpu.swarm.ray_train import host_claim


def fake_run(probes, writes):
    def run(host, script, sudo):
        if 'record' in script.split('\n', 1)[0]:
            writes.append(host)
            return json.dumps(dict(ok=True))
        value = probes[host]
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)
    return run


FREE = dict(holders=[], uninspectable=[], claim=None)


@pytest.mark.parametrize('probe,reason', [
    (FREE, None),
    (dict(FREE, holders=[dict(pid=7, user='alice')]), 'other users hold the TPU: alice'),
    (dict(FREE, uninspectable=[dict(pid=7, user='alice')]), 'cannot be inspected'),
    (dict(FREE, claim=dict(run_id='other', unit='skyrl-job-x.scope', live=True)), 'claimed by other'),
    (dict(FREE, claim=dict(run_id='other', unit='skyrl-job-x.scope', live=False)), None),
    (dict(FREE, claim=dict(run_id='mine', unit='skyrl-job-y.scope', live=True)), None),
    (dict(FREE, claim=dict(invalid=True, live=True)), 'unreadable claim'),
])
def test_verdict(probe, reason):
    got = host_claim.verdict(probe, 'mine')
    assert (got is None) if reason is None else (reason in got)


def test_claim_is_all_or_none():
    writes = []
    probes = {'h0': FREE, 'h1': dict(FREE, holders=[dict(pid=1, user='bob')]), 'h2': RuntimeError('ssh')}
    result = host_claim.claim_hosts(['h0', 'h1', 'h2'], fake_run(probes, writes), 'mine', 'u', 'me', 0)
    assert not result['claimed'] and writes == []
    assert set(result['reasons']) == {'h1', 'h2'} and 'probe failed' in result['reasons']['h2']
    probes['h1'] = probes['h2'] = FREE
    assert host_claim.claim_hosts(['h0', 'h1', 'h2'], fake_run(probes, writes), 'mine', 'u', 'me', 0)['claimed']
    assert writes == ['h0', 'h1', 'h2']


def test_probe_script_runs_and_reports_shape(tmp_path):
    user = pwd.getpwuid(os.getuid()).pw_name
    claim = tmp_path / 'claim.json'
    claim.write_text(json.dumps(dict(run_id='r', unit='skyrl-test-not-a-unit.scope')))
    out = subprocess.run([sys.executable, '-'], input='PARAMS = %r\n' % dict(user=user, claim=str(claim))
                         + host_claim.PROBE, text=True, capture_output=True, check=True).stdout
    probe = json.loads(out)
    assert probe['holders'] == [] and probe['claim']['live'] is False
    assert isinstance(probe['uninspectable'], list)
