"""Farms as capabilities: a zero-engine (grading-only) farm, no-sudo units, host sizing, draining, discovery."""
import asyncio
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request
import uuid

import httpx
import pytest

from tpu.science import cgroup_limits, core_pool, unit
from tpu.swarm.ray_train import discovery
from tpu.swarm.ray_train.borrowing_supervisor import REMOTE
from tpu.swarm.ray_train.config import Config, Grading
from tpu.swarm.ray_train.grading_service import GradingService

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'tpu/swarm/ray_train/profiles/inference-farm-grading-della-cpu-001.json'
GIB = 1024 ** 3


def supervised(code, *, memory_gib=1, seconds=30, tasks_max=64, cpus=None, private_tmp=False, timeout=60):
    """Run ``python -c code`` under the unit supervisor; returns (exit code, stderr)."""
    cpus = cpus or sorted(os.sched_getaffinity(0))[:1]
    argv = unit.command([sys.executable, '-c', code], cpus=cpus, memory_gib=memory_gib, seconds=seconds,
                        tasks_max=tasks_max, private_tmp=private_tmp)
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)))
    return proc.returncode, proc.stdout, proc.stderr


def finished(stderr):
    return json.loads([line for line in stderr.splitlines() if 'unit_finished' in line][-1])


def alive(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except OSError:
        return False
    return state != 'Z'


# -- the no-sudo unit ---------------------------------------------------------------------------

def test_unit_passes_exit_code_and_limit_and_pins_cpus():
    cpus = sorted(os.sched_getaffinity(0))[:1]
    code, out, err = supervised(f'import os,sys; print(os.environ["{unit.LIMIT}"], sorted(os.sched_getaffinity(0)));'
                                ' sys.exit(3)', cpus=cpus)
    assert code == 3 and out.split()[0] == str(1024 ** 3) and str(cpus) in out
    assert finished(err)['reason'] == 'exited'


def test_unit_kills_the_whole_tree_over_its_memory_limit():
    code, _, err = supervised('import subprocess,sys,time\n'
                              'subprocess.Popen([sys.executable,"-c","b=bytearray(700*2**20);import time;time.sleep(60)"])\n'
                              'b=bytearray(700*2**20); time.sleep(60)', memory_gib=1)
    record = finished(err)
    assert code == 137 and record['reason'].startswith('memory limit')
    assert record['peak_memory_bytes'] > 1024 ** 3


def test_unit_stops_detached_descendants_at_its_deadline(tmp_path):
    pidfile = tmp_path / 'pid'
    started = time.monotonic()
    code, _, err = supervised('import os,time\n'
                              'if os.fork()==0:\n'
                              '    os.setsid()\n'
                              '    if os.fork()==0:\n'
                              f'        open("{pidfile}","w").write(str(os.getpid())); time.sleep(120)\n'
                              '    os._exit(0)\n'
                              'time.sleep(120)', seconds=2)
    assert code == 143 and finished(err)['reason'].startswith('runtime limit')
    assert time.monotonic() - started < 15
    assert not alive(int(pidfile.read_text()))


def test_unit_stops_leftovers_when_the_command_exits(tmp_path):
    pidfile = tmp_path / 'pid'
    code, _, err = supervised('import os,subprocess,sys\n'
                              'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"],start_new_session=True)\n'
                              f'open("{pidfile}","w").write(str(p.pid))')
    assert code == 0 and finished(err)['reason'] == 'exited'
    assert not alive(int(pidfile.read_text()))


def test_unit_enforces_tasks_and_pins_widened_affinity_back():
    code, _, err = supervised('import threading,time\n'
                              '[threading.Thread(target=time.sleep,args=(30,),daemon=True).start() for _ in range(40)]\n'
                              'time.sleep(30)', tasks_max=16)
    assert code == 137 and finished(err)['reason'].startswith('tasks limit')
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip('needs two CPUs')
    code, out, _ = supervised(f'import os,time; os.sched_setaffinity(0,{allowed}); time.sleep(1);'
                              ' print(sorted(os.sched_getaffinity(0)))', cpus=allowed[:1])
    assert code == 0 and out.strip() == str(allowed[:1])


def test_unit_private_tmp_and_owner_death(tmp_path):
    marker = f'/tmp/unit-test-{uuid.uuid4().hex}'
    code, out, _ = supervised(f'open("{marker}","w").write("x"); import os; print(os.listdir("/tmp"))',
                              private_tmp=True)
    assert code == 0 and not Path(marker).exists()
    # The owner (a Ray worker) dies: the unit and its tree stop.
    pidfile = tmp_path / 'pid'
    argv = unit.command([sys.executable, '-c', f'import os,time; open("{pidfile}","w").write(str(os.getpid())); '
                         'time.sleep(120)'], cpus=sorted(os.sched_getaffinity(0))[:1], memory_gib=1, seconds=120,
                        tasks_max=64)
    owner = subprocess.Popen([sys.executable, '-c', 'import subprocess,sys,time; subprocess.Popen(sys.argv[1:]); '
                              'time.sleep(120)', *argv], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(.05)
    owner.kill()
    owner.wait()
    for _ in range(100):
        if not alive(int(pidfile.read_text())):
            break
        time.sleep(.05)
    assert not alive(int(pidfile.read_text()))


def test_envelope_trusts_the_unit_limit_only_inside_a_unit(monkeypatch):
    monkeypatch.setenv(unit.LIMIT, str(2 * 1024 ** 3))
    assert cgroup_limits.envelope(2) == (None, 2 * 1024 ** 3)
    assert cgroup_limits.metrics(None) == {}
    with pytest.raises(RuntimeError):
        cgroup_limits.envelope(1)
    monkeypatch.delenv(unit.LIMIT)
    monkeypatch.delenv(unit.MODE, raising=False)
    assert unit.mode() == 'systemd' and not unit.inside()
    env = unit.environment()
    assert env[unit.MODE] == 'bwrap' and 'PYTHONPATH' not in env and unit.LIMIT not in env


def test_launch_sites_run_the_systemd_program_under_the_supervisor(monkeypatch, tmp_path):
    """bwrap mode keeps the exact program argv systemd-run would start, under the same numbers."""
    from tpu.science import math_grade
    calls = []

    class Proc:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            folder = Path(command[command.index('--result') + 1]).parent
            (folder / math_grade.STARTED_MARKER).write_text('{}')
            (folder / 'result.json').write_text(json.dumps(dict(result=[1.0], error=None, stdout='', metrics={})))

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0
    monkeypatch.setattr(math_grade.subprocess, 'Popen', Proc)
    monkeypatch.setattr(math_grade.subprocess, 'run', lambda *a, **k: pytest.fail('no systemctl without sudo'))
    monkeypatch.setattr(math_grade, 'admit', lambda families, deadline: (None, [3, 4], __import__('contextlib').nullcontext(), 'core-pool-v1'))
    monkeypatch.setenv(unit.MODE, 'bwrap')
    families = {'math': dict(slots_per_host=4, cpus=2, memory_gib=4, memory_max_gib=8)}
    spec = dict(program_code='x', function_name='f', eval_timeout_seconds=60, systemd=True)
    result = math_grade.grade_math_admitted(spec, families, tmp_path)
    (command, kwargs), = calls
    assert result['result'] == [1.0]
    assert command[:3] == [sys.executable, '-m', 'tpu.science.unit']
    assert command[command.index('--cpus') + 1] == '3,4' and command[command.index('--memory-gib') + 1] == '8'
    assert command[command.index('--seconds') + 1] == '75' and '--private-tmp' in command
    program = command[command.index('--') + 1:]
    assert program[:3] == [sys.executable, '-m', 'tpu.science.math_runner']
    assert kwargs['env'][unit.MODE] == 'bwrap' and kwargs['cwd'] == str(tmp_path)


# -- a zero-engine farm profile -------------------------------------------------------------------

def test_cpu_farm_has_no_engines_and_sizes_grading_from_the_host(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURMD_NODENAME', 'della-h14n13')
    monkeypatch.setenv('SLURM_JOB_ID', '77')
    config = Config.load(PROFILE)
    assert config.root.endswith('/farms/della-h14n13-77') and config.controller_env.endswith('/envs/controller')
    assert config.engines_per_host == 0 and config.engine_slots(['10.0.0.1']) == []
    assert config.served_models == [] and not config.writes_back and config.grading_host_sized
    host = config.for_host(16, 32 * GIB)
    assert {n: f['slots_per_host'] for n, f in host.grading_families.items()} == {'placement': 7, 'routing': 1, 'math': 6}
    assert host.grading_memory_gib == 26 and host.grading_service_cpus == 2 and not host.grading_host_sized
    assert host.grading_families['math']['memory_max_gib'] == 8 and host.grading.isolation == 'bwrap'
    small = config.for_host(8, 32 * GIB)
    assert 'routing' not in small.grading_families and small.grading_families['placement']['slots_per_host'] == 3
    with pytest.raises(RuntimeError):
        config.for_host(3, 8 * GIB)
    raw = json.loads(PROFILE.read_text())
    raw['grading']['isolation'] = 'systemd'  # systemd units need the systemd runtime
    with pytest.raises(ValueError, match='systemd'):
        Config.from_dict(raw)


def test_tpu_farms_keep_their_shape():
    farm = Config.load(ROOT / 'tpu/swarm/ray_train/profiles/inference-farm-rq4-gemma-east5-v6e8-001.json')
    assert farm.engines_per_host == 1 and farm.writes_back and farm.served_models
    assert farm.grading.isolation == 'systemd' and farm.grading.drain_seconds == 0 and not farm.grading_host_sized
    assert farm.for_host(16, 32 * GIB) is farm and farm.grading_service_cpus == 36
    assert farm.grading_memory_gib == farm.cache.reserve_gib - 64


def test_small_service_block_pool(tmp_path):
    document = core_pool.install(2, 26, root=tmp_path / 'locks', affinity=set(range(16)),
                                 topology_root=str(tmp_path / 'none'), minimum=1)
    assert len(document['grading']) == 14 and len(document['service']) == 2
    with pytest.raises(ValueError):
        core_pool.layout(2, set(range(16)), str(tmp_path / 'none'))  # TPU farms keep their 24-CPU minimum


def test_host_end_time_and_slurm_scoped_reaping(monkeypatch, tmp_path):
    from tpu.swarm.ray_train import controller, lifecycle
    monkeypatch.setenv('SKYRL_FARM_STOP_AT', '1234.5')
    assert controller.host_end_time() == 1234.5
    monkeypatch.delenv('SKYRL_FARM_STOP_AT')
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    assert controller.host_end_time() is None
    procs = [(11, '/system.slice/slurmstepd.scope/job_7/step_0/user/task_0', 'ray'),
             (12, '/system.slice/slurmstepd.scope/job_8/step_0/user/task_0', 'ray')]
    monkeypatch.setattr(lifecycle, 'workload_processes', lambda proc: procs)
    monkeypatch.setattr(lifecycle, 'active_units', lambda run: [])
    killed = []
    report = lifecycle.reap_leftovers('', kill=lambda pid, sig: killed.append(pid), scope='7')
    assert killed == [11] and len(report['processes']) == 1  # another allocation's processes are not ours


# -- draining -------------------------------------------------------------------------------------

def service(slots=1):
    config = SimpleNamespace(grading=Grading(long_poll_seconds=1), accelerator='cpu', run_id='g',
                             grading_families={'math': dict(slots_per_host=slots, cpus=2, memory_gib=4, memory_max_gib=8)})
    grading = GradingService(config, ['10.0.0.1'], warmup=False)
    grading.ready = True
    futures = []

    def dispatch(task, spec):
        future = asyncio.get_running_loop().create_future()
        futures.append(future)
        return future
    grading.dispatch = dispatch
    grading.cancel_ref = lambda ref: ref.cancel()
    return grading, futures


def math_body():
    return dict(request_id=uuid.uuid4().hex, task='math', spec=dict(
        program_code='def f():\n    return [1.0]\n', function_name='f', eval_timeout_seconds=60))


OWNER = dict(lease_id=None, owner_run='run', token='t')


def test_drain_finishes_in_time_grades_and_hands_the_rest_back_as_draining():
    async def run():
        grading, futures = service(slots=2)
        try:
            bodies = [math_body() for _ in range(3)]
            for body in bodies:
                assert (await grading.submit(OWNER, body))[0] == 202
            await asyncio.sleep(0)
            assert len(futures) == 2  # the third waits in the queue
            drain = asyncio.ensure_future(grading.shut_down(time.monotonic() + .5, linger=2))
            await asyncio.sleep(.1)
            status, refused = await grading.submit(OWNER, math_body())
            assert status == 429 and refused['draining'] and not grading.capacity(None)['ready']
            futures[0].set_result({'result': [1.0], 'error': None, 'metrics': {}})
            await asyncio.sleep(.05)
            views = await grading.results(OWNER, [b['request_id'] for b in bodies], wait=0)
            assert not drain.done()  # lingers until every handed-back grade is fetched
            done, late, queued = (views[b['request_id']] for b in bodies)
            assert done['state'] == 'done'
            assert queued['state'] == 'cancelled' and queued['error']['class'] == 'draining'
            assert late['state'] == 'running'
            await asyncio.sleep(.6)
            late = (await grading.results(OWNER, [bodies[1]['request_id']]))[bodies[1]['request_id']]
            assert late['state'] == 'cancelled' and late['error']['class'] == 'draining'
            assert await asyncio.wait_for(drain, 2) == 2
            # An owner's own cancel stays an ordinary (charged) cancellation.
            other, _ = service()
            body = math_body()
            await other.submit(OWNER, body)
            assert (await other.cancel(body['request_id'], 'owner'))['error']['class'] == 'cancelled'
            await other.close()
        finally:
            await grading.close()
    asyncio.run(run())


def test_client_moves_a_drained_grade_without_charging_a_retry():
    from tests.tpu_swarm.test_grading_transport import GradingFarm, done, request, transport
    from tpu.science import grading_transport as gt

    async def run():
        farm = GradingFarm()
        farm.outcomes['a'] = lambda request_id: {'request_id': request_id, 'state': 'cancelled', 'result': None,
                                                 'error': {'class': 'draining', 'detail': 'cancelled'}}
        farm.outcomes['b'] = done({'result': [2.0], 'error': None, 'metrics': {}})
        t = transport(farm, ['a', 'b'], max_infra_retries=0)
        try:
            await t._ensure_started()
            t.directory.pools['b'].unhealthy_until = time.monotonic() + .3
            result = await asyncio.wait_for(t.grade(request()), 10)
            assert result['result'] == [2.0] and [h for h, _ in farm.submits] == ['a', 'b']
            reroute, = [f for e, f in t.reports if e == 'grading_reroute']
            assert reroute['kind'] == 'draining' and reroute['charged'] is False
            assert not any(e in ('grading_retry', 'grading_infra_failure') for e, _ in t.reports)
            # Any other server-side cancel is still charged (cap 0 -> dropped).
            farm.outcomes['b'] = farm.outcomes['a'] = lambda request_id: {
                'request_id': request_id, 'state': 'cancelled', 'result': None,
                'error': {'class': 'cancelled', 'detail': 'cancelled'}}
            with pytest.raises(gt.GradingInfrastructureFailed):
                await asyncio.wait_for(t.grade(request()), 10)
        finally:
            await t.close()
    asyncio.run(run())


# -- the same Ingress, zero engines; discovery ----------------------------------------------------

def test_zero_engine_ingress_serves_the_grading_contract_and_drains(tmp_path, monkeypatch):
    import inspect
    from tpu.swarm.ray_train import serving
    from tests.tpu_swarm.test_farm_leases import Remote
    cfg = Config.load(PROFILE).for_host(16, 32 * GIB).to_dict()
    cfg['root'] = str(tmp_path)

    async def run():
        catalog = serving.Catalog.__ray_metadata__.modified_class([], 0)
        cls = serving.Ingress.func_or_class.__mro__[1]
        gateway = cls(cfg, [], SimpleNamespace(commit=Remote(catalog.commit), snapshot=Remote(catalog.snapshot),
                                               quarantine=Remote(lambda reason: None)), ['10.1.2.3'])
        grading = gateway.grading
        grading.warmup_task.cancel()
        futures = []

        def dispatch(task, spec):
            future = asyncio.get_running_loop().create_future()
            futures.append(future)
            return future
        grading.dispatch, grading.cancel_ref = dispatch, lambda ref: ref.cancel()
        monkeypatch.setattr(serving.serve, 'get_replica_context', lambda: SimpleNamespace(servable_object=gateway))
        app = inspect.getclosurevars(serving.Ingress.func_or_class.__init__).nonlocals['frozen_app_or_func']
        headers = {'x-grading-run': 'run', 'x-grading-token': 't'}
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://farm') as client:
                assert (await client.get('/health')).status_code == 503  # warming up
                grading.ready = True
                assert (await client.get('/health')).json() == dict(status='ok', engines=0)
                status = (await client.get('/status')).json()
                assert status['expected_engines'] == 0 and status['replicas'] == [] and status['state'] == 'unleased'
                assert status['url'] == 'http://10.1.2.3:24800' and status['grading']['ready']
                assert (await client.get('/v1/models')).json()['data'] == []
                capacity = (await client.get('/skyrl/v1/grading/capacity', headers=headers)).json()
                assert capacity['open'] and capacity['ready'] and capacity['families']['placement']['total'] == 7
                assert capacity['families']['placement']['open_limit'] == 10  # 1.5x, the same open-grading contract
                body = math_body()
                assert (await client.post('/skyrl/v1/grading/submit', json=body, headers=headers)).status_code == 202
                await asyncio.sleep(0)
                drain = asyncio.ensure_future(gateway.drain_grading(.2))
                await asyncio.sleep(.05)
                assert (await client.get('/health')).status_code == 503
                assert (await client.post('/skyrl/v1/grading/submit', json=math_body(), headers=headers)).status_code == 429
                await asyncio.sleep(.3)
                views = (await client.post('/skyrl/v1/grading/results', json=dict(ids=[body['request_id']]),
                                           headers=headers)).json()['views']
                assert views[body['request_id']]['error']['class'] == 'draining'
                assert await asyncio.wait_for(drain, 5) == dict(handed_back=1)
        finally:
            await grading.close()
            await gateway.http.aclose()
    asyncio.run(run())


def remote_probe(monkeypatch, responses):
    def urlopen(request, timeout=None):
        url = request.full_url
        if 'metadata.google.internal' in url:
            value = responses.get('metadata')
        else:
            value = responses[url.split(':24800', 1)[1]]
        if isinstance(value, Exception):
            raise value
        return io.BytesIO(value.encode() if isinstance(value, str) else json.dumps(value).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', output)
    try:
        exec('PARAMS = %r\n' % dict(action='farm', port=24800) + REMOTE, {})
    finally:
        monkeypatch.setattr(sys, 'stdout', sys.__stdout__)
    return json.loads(output.getvalue())


def test_discovery_lists_a_zero_engine_farm_at_its_own_address(monkeypatch, tmp_path):
    import socket
    import urllib.error
    status = dict(expected_engines=0, replicas=[], state='unleased', instance='i1', url='http://127.0.0.1:24800',
                  expected=[], grading=dict(ready=True, families=['placement']))
    responses = {'/health': dict(status='ok', engines=0), '/status': status, '/v1/models': dict(data=[]),
                 'metadata': urllib.error.URLError(socket.gaierror(-2, 'Name or service not known'))}
    probe = remote_probe(monkeypatch, responses)
    assert probe['ok'], probe
    result = probe['result']
    assert result['engines'] == 0 and result['direct'] and result['models'] == []
    assert result['url'] == 'http://127.0.0.1:24800'
    # A GCE farm keeps its metadata address (and its tunnel); a farm with no engines and no grading is not a farm.
    gce = remote_probe(monkeypatch, dict(responses, metadata='127.0.0.1'))
    assert gce['ok'] and not gce['result']['direct']
    empty = remote_probe(monkeypatch, dict(responses, **{'/status': dict(status, grading=None)}))
    assert not empty['ok']
    # Published ready and directly reachable; generation never selects it (no models).
    monkeypatch.setattr(discovery, 'rpc', lambda *a, **k: probe)
    args = SimpleNamespace(registry=tmp_path / 'farms.json', ssh_config_dir=tmp_path, name_contains=['inference-farm'],
                           trainer_pool=[], trainer_job_id=[], discovery_group='', run_scoped_only=False,
                           farm_port=24800, tunnels=True, port_base=29000, push=False)
    svc = discovery.Service(args, dict(farms=[], trainers=[]), now=lambda: 0)
    svc.tunnel = lambda farm: pytest.fail('a directly reachable farm needs no tunnel')
    row = dict(job_id=7, status='RUNNING', cluster='della-cpu-7', run_id='inference-farm-grading-della-cpu-001', pool='p')
    farm, = svc.tick([row])['farms']
    assert farm['url'] == 'http://127.0.0.1:24800' and farm['tunnel_ok'] and farm['engines'] == 0
    from tpu.science.grading_transport import OpenFarmDirectory
    assert OpenFarmDirectory.listable(farm)
    assert not discovery.is_ready(dict(ok=True, health='healthy', capabilities=None, engines=0))


# -- several farms per host (non-exclusive Slurm workers) ----------------------------------------

def test_ports_are_placed_by_the_managed_job_and_tpu_defaults_stay():
    from dataclasses import asdict
    from tpu.swarm.ray_train import bootstrap
    from tpu.swarm.ray_train.config import farm_ports
    assert farm_ports(1234) == (21234, 25000 + 1234 % 37 * 200) and farm_ports(5999) == (21999, 25000 + 5999 % 37 * 200)
    for job in range(0, 20000, 7):
        http, base = farm_ports(job)
        assert 20000 <= http < 24000 and 25000 <= base and base + 199 < 32768
    config = Config.load(PROFILE)
    assert config.job_ports and bootstrap.moved_ports(config, {}) == (None, None)
    http, base = bootstrap.moved_ports(config, dict(SKYPILOT_MANAGED_JOB_ID='1234'))
    moved = config.with_ports(http, base)
    assert moved.ports.inference == 21234 and moved.ports.ray == base and moved.ports.metrics == base + 8
    assert (moved.ports.worker_min, moved.ports.worker_max) == (base + 10, base + 199)
    assert bootstrap.moved_ports(config, dict(SKYPILOT_MANAGED_JOB_ID='1234', SKYRL_FARM_PORT='31000')) == (31000, base)
    tpu = Config.load(ROOT / 'tpu/swarm/ray_train/profiles/inference-farm-rq4-gemma-east5-v6e8-001.json')
    assert not tpu.job_ports and asdict(tpu.ports) == asdict(type(tpu.ports)())
    assert bootstrap.moved_ports(tpu, dict(SKYPILOT_MANAGED_JOB_ID='1234')) == (None, None)  # TPU: fixed ports


def test_production_della_profile_at_64_cpus_128_gib():
    config = Config.load(ROOT / 'tpu/swarm/ray_train/profiles/inference-farm-rq4-grading-della.json')
    assert 'inference-farm' in config.run_id and config.job_ports and config.engines_per_host == 0
    host = config.for_host(64, 128 * GIB)
    assert {n: f['slots_per_host'] for n, f in host.grading_families.items()} == {'placement': 30, 'routing': 6, 'math': 28}
    assert host.grading_memory_gib == 112 and host.grading_service_cpus == 4 and host.grading.time_limit_lead_seconds == 900


def test_sixteen_cpus_sixty_four_gib_sizing():
    host = Config.load(PROFILE).for_host(16, 64 * GIB)
    assert {n: f['slots_per_host'] for n, f in host.grading_families.items()} == {'placement': 7, 'routing': 1, 'math': 7}
    assert host.grading_memory_gib == 58 and host.grading_service_cpus == 2 and host.ray_service_cpus(0) == 1


def test_discovery_reaches_slurm_pool_farms_at_their_job_placed_port(monkeypatch, tmp_path):
    from tpu.swarm.ray_train.config import farm_ports
    calls = []
    # SkyPilot's Slurm job is "<cluster>-<user hash>"; the pool row names only the cluster.
    squeue = 'gradefarm-della-test-3-7bfcb694 della-h14n13\nother-job della-h1n1\ngradefarm-della-test-30-7bfcb694 della-h2n2\n'
    run = lambda argv, **kw: calls.append(argv) or SimpleNamespace(stdout=squeue, returncode=0)
    url = f'http://della-h14n13:{farm_ports(1234)[0]}'
    assert discovery.slurm_farm_url(1234, 'gradefarm-della-test-3', run=run) == url
    assert discovery.slurm_farm_url(1234, 'gradefarm-della-test-3-7bfcb694', run=run) == url
    assert discovery.slurm_farm_url(1234, 'gradefarm-della-test-9', run=run) is None
    assert calls[-1][:2] == ['squeue', '-h']
    probed = []
    monkeypatch.setattr(discovery, 'slurm_farm_url', lambda job, cluster: f'http://della-h14n13:{farm_ports(job)[0]}')
    monkeypatch.setattr(discovery, 'local_rpc', lambda url: probed.append(url) or dict(ok=True, result=dict(
        models=[], url=url, engines=0, direct=True, grading=dict(ready=True), instance='i')))
    monkeypatch.setattr(discovery, 'rpc', lambda *a, **k: pytest.fail('no SSH probe for a Slurm pool farm'))
    args = SimpleNamespace(registry=tmp_path / 'farms.json', ssh_config_dir=tmp_path, name_contains=['gradetest'],
                           trainer_pool=[], trainer_job_id=[], discovery_group='', run_scoped_only=False, farm_port=24800,
                           tunnels=True, port_base=29000, push=False, slurm_pool=['gradefarm-della-test'])
    svc = discovery.Service(args, dict(farms=[], trainers=[]), now=time.time)
    svc.tunnel = lambda farm: pytest.fail('no tunnel for a directly reachable farm')
    rows = [dict(job_id=1234, status='RUNNING', cluster='gradefarm-della-test-3-7bfcb694',
                 run_id='gradetest-della-cpu-001', pool='gradefarm-della-test', priority=0),
            dict(job_id=99, status='RUNNING', cluster='c', run_id='inference-farm-x', pool='other', priority=0)]
    listed, = svc.tick(rows)['farms']
    assert probed == [f'http://della-h14n13:{farm_ports(1234)[0]}'] and listed['source'] == 'slurm'
    assert listed['url'] == probed[0] and listed['tunnel_ok'] and listed['engines'] == 0


def test_time_limit_stop_idles_so_managed_job_recovery_replaces_the_farm(monkeypatch, tmp_path):
    import signal
    import threading
    from tpu.swarm.ray_train import controller, lifecycle
    farm = controller.Controller.__new__(controller.Controller)
    farm.config, farm.failure, farm.stopping = Config.load(PROFILE), None, threading.Event()
    farm.report = lambda *a, **k: None
    monkeypatch.setenv('SKYRL_FARM_STOP_AT', str(time.time() + 900))  # lead 900 s: the end is already near
    assert farm.serve_until_stopped(poll=.01) == lifecycle.HOST_ENDING_EXIT
    monkeypatch.setenv('SKYRL_FARM_STOP_AT', str(time.time() + 3600))
    farm.stopping.set()
    assert farm.serve_until_stopped(poll=.01) == 143  # an explicit stop exits normally
    # After the drain, bootstrap idles (SIGTERM ignored) instead of ending the job as succeeded.
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        if len(sleeps) == 3:
            raise KeyboardInterrupt  # stands in for the allocation's SIGKILL
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(KeyboardInterrupt):
            lifecycle.idle_until_killed(tmp_path / 'log.jsonl', sleep=sleep)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    assert len(sleeps) == 3 and 'host_ending_idle' in (tmp_path / 'log.jsonl').read_text()
