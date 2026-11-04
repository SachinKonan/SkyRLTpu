"""A remote-only trainer on workers 0,1 of the on-demand v4-64 next to the pool job: nothing shared with that job."""
import json
import os
import subprocess
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train import assigned_launch, bootstrap, direct_discovery, discovery, lifecycle, runtime_service
from tpu.swarm.ray_train.commands import maxtext_kwargs, trainer_command, trainer_environment
from tpu.swarm.ray_train.config import Config, Ports

PROFILE = 'tpu/swarm/ray_train/profiles/rl28k-qwen35-erdos-s2-od16.json'
POOL_JOB = 'tpu/swarm/ray_train/profiles/rl28k-qwen35-erdos-s2-od.json'
IPS = ['10.130.0.68', '10.130.0.62']
TASK = 'skyrl-od-q35'


def raw(**changes):
    return dict(Config.load(PROFILE).to_dict(), **changes)


def test_profile_is_the_tp8_partition_of_the_pool_run():
    config, pool = Config.load(PROFILE), Config.load(POOL_JOB)
    assert config.od_v4_64_partition and not pool.od_v4_64_partition and not config.od_v4_64_subset
    assert config.assigned_host_ips == IPS and config.active_ranks == [0, 1] and config.idle_ranks == ()
    t = config.trainer
    assert (t.tp, t.fsdp, t.process_bounds, t.chip_bounds, t.token_budget) == (8, 1, '1,2,1', '2,2,1', 28672)
    assert t.tp * t.fsdp == 8 and config.inference_hosts == 0
    assert config.inference.external_pool_discovery_group == 'rl28k-qwen35'
    assert config.inference.external_pool_required_compatibility
    # The science recipe is the pool run's: only the layout and its host budget change.
    a, b = config.client_env, pool.client_env
    assert {k: v for k, v in a.items() if k != 'TTD_SICK_MARKER'} == {k: v for k, v in b.items() if k != 'TTD_SICK_MARKER'}
    assert config.cache.trainer_gib + config.cache.reserve_gib + assigned_launch.PARTITION_MARGIN_GIB <= 308


def test_two_buckets_warm_both_shapes_as_singleton_rows(tmp_path):
    from tpu.swarm.ray_train.warmup_contract import shapes
    config, pool = Config.load(PROFILE), Config.load(POOL_JOB)
    assert pool.trainer.seq_buckets == '28672' and config.trainer.backward_warmup
    env = trainer_environment(config, tmp_path, tmp_path / 'run', IPS, 0)
    assert env['TUNIX_SEQ_BUCKETS'] == '24576,28672' and env['TUNIX_UNIFORM_SEQ_LEN'] == '0'
    assert env['TUNIX_BACKWARD_WARMUP'] == '1' and env['TUNIX_WARMUP_MAX_LENGTH'] == '28672'
    # The same call backward_warmup.run makes: one row per tile, shorter bucket first.
    cases = shapes(int(env['TUNIX_WARMUP_MAX_LENGTH']), config.trainer.token_budget, int(env['TUNIX_ROW_SHARD']),
                   uniform=int(env['TUNIX_UNIFORM_SEQ_LEN']), buckets=[int(n) for n in env['TUNIX_SEQ_BUCKETS'].split(',')])
    assert cases == [(1, 24576), (1, 28672)]
    # Dropping the top bucket would leave long rows without a warmed shape.
    d = raw()
    d['trainer'] = dict(d['trainer'], seq_buckets='24576')
    with pytest.raises(ValueError, match='maximum training length'):
        Config.from_dict(d)


def test_nothing_of_the_pool_job_is_reused():
    config, pool = Config.load(PROFILE), Config.load(POOL_JOB)
    ours, theirs = asdict(config.ports), asdict(pool.ports)
    ranges = lambda p: set(range(p.pop('worker_min'), p.pop('worker_max') + 1)) | set(p.values())
    assert not (ranges(ours) | {config.systemd_port}) & (ranges(theirs) | {pool.systemd_port})
    assert config.root != pool.root and config.run_gcs != pool.run_gcs
    assert config.cache.trainer_compile != pool.cache.trainer_compile
    assert config.storage_min_free_gib == 0 and config.checkpoint_cleanup_timeout == 0
    assert max(asdict(config.ports).values()) < 32768


@pytest.mark.parametrize('change,match', [
    (dict(assigned_host_ips=['10.130.0.62', '10.130.0.68']), 'workers 0,1'),
    (dict(storage_min_free_gib=30), 'no storage cleanup'),
    (dict(checkpoint_cleanup_timeout=600), 'no storage cleanup'),
    (dict(ports=asdict(Ports())), 'ports apart'),
    (dict(systemd_port=24900), 'ports apart'),
    (dict(root='~/.cache/rl28k-qwen35-erdos-s1-od'), 'own root'),
    (dict(physical_node=''), 'assigned hosts'),
])
def test_partition_rejects_anything_that_could_touch_the_pool_job(change, match):
    with pytest.raises(ValueError, match=match):
        Config.from_dict(raw(**change))


@pytest.mark.parametrize('bounds', ['2,1,1', '1,1,2'])
def test_partition_rejects_the_shapes_that_failed_on_hardware(bounds):
    d = raw()
    d['trainer'] = dict(d['trainer'], process_bounds=bounds)
    with pytest.raises(ValueError, match='bounds 1,2,1'):
        Config.from_dict(d)


def test_worker_range_overlap_is_refused():
    d = raw()
    d['ports'] = dict(d['ports'], worker_min=22500, worker_max=23499)
    with pytest.raises(ValueError, match='ports apart'):
        Config.from_dict(d)


def test_assigned_v4_64_stays_two_host_remote_only():
    d = raw()
    d['inference'] = dict(d['inference'], remote_only=False)
    with pytest.raises(ValueError):
        Config.from_dict(d)


def test_v5p_assigned_and_od_box_paths_are_unchanged(tmp_path):
    assert not Config.load('tpu/swarm/ray_train/profiles/rl28k-qwen35-erdos-s1-od.json').od_v4_64_partition
    from tests.tpu_swarm.test_assigned_v5p_host import raw as v5p
    config = Config.from_dict(v5p())
    assert config.assigned_host_ips and not config.od_v4_64_partition
    assert bootstrap.partition_runtime(config, tmp_path) == {}


def test_trainer_runs_as_process_0_on_worker_0(tmp_path):
    config = Config.load(PROFILE)
    env = trainer_environment(config, tmp_path, tmp_path / 'run', IPS, 1)
    assert env['TPU_PROCESS_BOUNDS'] == '1,2,1' and env['TPU_CHIPS_PER_PROCESS_BOUNDS'] == '2,2,1'
    assert env['TPU_PROCESS_ADDRESSES'] == '10.130.0.68:25804,10.130.0.62:25804'
    assert env['TPU_PROCESS_PORT'] == '25804' and env['CLOUD_TPU_TASK_ID'] == '1' and env['TUNIX_ROW_SHARD'] == '1'
    command = trainer_command(config, tmp_path, tmp_path, IPS[0], IPS, 1)
    assert command[command.index('--coordinator-address') + 1] == '10.130.0.68:25803'
    assert maxtext_kwargs(config, tmp_path)['ici_tensor_parallelism'] == 8


def test_runtime_is_confined_to_spare_cpus_private_tmp_and_a_memory_cap(tmp_path, monkeypatch):
    from tpu.science import core_pool
    config = Config.load(PROFILE)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: set(range(240)))
    nodes = tmp_path / 'node'
    for index, cpus in enumerate(('0-59,120-179', '60-119,180-239')):
        (nodes / f'node{index}').mkdir(parents=True)
        (nodes / f'node{index}/cpulist').write_text(cpus)
    layout = core_pool.layout
    monkeypatch.setattr(core_pool, 'layout', lambda count: layout(count, topology_root=nodes))
    settings = bootstrap.partition_runtime(config, tmp_path)
    assert settings['lease'] == tmp_path / '.systemd-runtime.lock'
    properties = dict(p.removeprefix('--property=').split('=', 1) for p in settings['properties'])
    assert properties['PrivateTmp'] == 'yes' and properties['MemoryMax'] == '176G'
    spare = set(map(int, properties['AllowedCPUs'].split(',')))
    pool_service = set(layout(48, topology_root=nodes)[1])
    assert len(spare) == 192 and not spare & pool_service
    # Inside the unit our pool lays itself out on those CPUs only.
    grading, service, _ = layout(config.grading_service_cpus, affinity=spare, topology_root=nodes)
    assert set(grading) | set(service) == spare and len(grading) >= 2 * 16


def test_memory_cap_leaves_cache_and_grades_inside_the_budget():
    config = Config.load(PROFILE)
    assert config.grading_memory_gib == 16 * 4
    assert config.partition_memory_max_gib + config.grading_memory_gib == config.cache.trainer_gib + config.cache.reserve_gib


def launched(tmp_path, monkeypatch, **kwargs):
    """What runtime_service.launch hands systemd-run: its argv plus the payload it points at."""
    seen = {}

    class Done:
        returncode = 0

        def poll(self):
            return 0

    def popen(args):
        seen.update(json.loads(open(args[-1]).read()), args=args)
        return Done()
    monkeypatch.setattr(runtime_service.subprocess, 'Popen', popen)
    monkeypatch.setattr(runtime_service.subprocess, 'check_output', lambda *a, **k: 'ActiveState=inactive\n')
    monkeypatch.setattr(lifecycle, 'finish', lambda *a, **k: {})
    assert runtime_service.launch(['x'], tmp_path, IPS, 0, TASK, **kwargs) == 0
    return seen


def test_systemd_run_carries_the_partition_settings(tmp_path, monkeypatch):
    seen = launched(tmp_path, monkeypatch, lease=tmp_path / 'lease', properties=('--property=PrivateTmp=yes',))
    assert '--property=PrivateTmp=yes' in seen['args'] and seen['lease'] == str(tmp_path / 'lease')


def test_default_runtime_keeps_the_home_lease(tmp_path, monkeypatch):
    seen = launched(tmp_path, monkeypatch)
    assert seen['lease'] is None and not any('PrivateTmp' in a for a in seen['args'])


def test_supervisors_of_two_roots_hold_separate_leases(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setenv('HOME', str(tmp_path))
    held = (tmp_path / '.skyrl-systemd-runtime.lock').open('a')
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # the pool job's runtime
    run = tmp_path / 'run'
    run.mkdir()
    payload = run / 'payload.json'
    payload.write_text(json.dumps(dict(directory=str(run), parent_pid=-1, parent_start='x', rank=1, ips=['127.0.0.1'],
                                       port=1, token='t', peer_timeout=1, lease=str(tmp_path / 'ours.lock'))))
    # Reaches the parent check past the lease; with the home lease it would raise BlockingIOError.
    assert runtime_service.supervise(payload) == 143
    payload.write_text(json.dumps(dict(directory=str(run), parent_pid=-1, parent_start='x', rank=1, ips=['127.0.0.1'],
                                       port=1, token='t', peer_timeout=1)))
    with pytest.raises(BlockingIOError):
        runtime_service.supervise(payload)


def systemctl(units, owners=''):
    def run(argv, **kwargs):
        if argv[:2] == ['systemctl', 'list-units']:
            return SimpleNamespace(returncode=0, stderr='', stdout=''.join('%s loaded active running %s\n' % u for u in units))
        if 'fuser' in argv:
            return SimpleNamespace(returncode=0 if owners else 1, stdout=owners, stderr='')
        raise AssertionError('nothing may be stopped: %s' % argv)
    return run


def pool_job_proc(tmp_path):
    entry = tmp_path / 'proc/11'
    entry.mkdir(parents=True, exist_ok=True)
    (entry / 'cmdline').write_bytes(b'python3\0-m\0skyrl.tinker.api\0')
    (entry / 'cgroup').write_text('0::/user.slice/skypilot-launcher.scope\n')
    return tmp_path / 'proc'


def gates(monkeypatch, foreign=False):
    monkeypatch.setattr(lifecycle, 'tpu_gates', lambda *, shared, run, home_min_gib=30: dict(device_owners=run(['fuser'], capture_output=True).stdout.split(), foreign_tpu_lock=foreign))


def test_coexist_starts_beside_the_pool_job_and_stops_nothing(tmp_path, monkeypatch):
    gates(monkeypatch)
    run = systemctl([('skyrl-runtime-aaaa.service', 'sky-managed-pool-job')])
    kills = []
    report = lifecycle.prepare(TASK, shared=True, coexist=True, run=run, proc=pool_job_proc(tmp_path), kill=lambda *a: kills.append(a))
    assert report['units'] and report['processes'] and not kills
    with pytest.raises(RuntimeError, match='refusing'):
        lifecycle.prepare(TASK, shared=True, run=run, proc=pool_job_proc(tmp_path), kill=lambda *a: kills.append(a))


@pytest.mark.parametrize('owners,foreign', [(' 4242', False), ('', True)])
def test_coexist_refuses_a_held_tpu(tmp_path, monkeypatch, owners, foreign):
    gates(monkeypatch, foreign)
    with pytest.raises(RuntimeError, match='refusing'):
        lifecycle.prepare(TASK, shared=True, coexist=True, run=systemctl([], owners), proc=pool_job_proc(tmp_path))


def test_finish_on_this_host_stops_only_our_task(tmp_path):
    calls = []
    units = [('skyrl-runtime-aaaa.service', 'sky-managed-pool-job'), ('skyrl-runtime-bbbb.service', TASK)]

    def run(argv, **kwargs):
        calls.append(argv)
        out = ''.join('%s loaded active running %s\n' % u for u in units) if argv[:2] == ['systemctl', 'list-units'] else ''
        return SimpleNamespace(returncode=0, stdout=out, stderr='')
    report = lifecycle.finish(TASK, 'exit', shared=True, run=run, proc=pool_job_proc(tmp_path), kill=lambda *a: pytest.fail('killed'))
    assert report['units'] == ['skyrl-runtime-bbbb.service'] and not report['processes']
    assert all('skyrl-runtime-aaaa.service' not in c for c in calls)


def test_bootstrap_starts_coexisting_with_the_partition_runtime(tmp_path, monkeypatch):
    order = {}
    monkeypatch.setattr(lifecycle, 'prepare', lambda task, **kw: order.update(prepare=kw))
    monkeypatch.setattr(runtime_service, 'launch', lambda *a, **kw: order.update(launch=kw) or 0)
    config = Config.from_dict(raw(root=str(tmp_path / 'rl28k-qwen35-erdos-s2-od16')))
    monkeypatch.setattr(bootstrap.Config, 'load', staticmethod(lambda path: config))
    monkeypatch.setattr(bootstrap.subprocess, 'check_output', lambda *a, **k: IPS[0] + '\n')
    monkeypatch.setattr(bootstrap, 'partition_runtime', lambda config, root: dict(lease='L', properties=('P',)))
    monkeypatch.setattr('sys.argv', ['bootstrap', 'profile.json'])
    monkeypatch.setenv('SKYPILOT_NODE_IPS', ' '.join(IPS))
    monkeypatch.setenv('SKYPILOT_NODE_RANK', '0')
    monkeypatch.setenv('SKYPILOT_TASK_ID', TASK)
    monkeypatch.delenv('SKYRL_SYSTEMD_OWNED', raising=False)
    assert bootstrap.main() == 0
    assert order['prepare']['shared'] is True and order['prepare']['coexist'] is True
    assert order['launch']['shared'] is True and order['launch']['lease'] == 'L' and order['launch']['properties'] == ('P',)
    assert order['launch']['port'] == 25900


GOOD = dict(available_gib=300.0, disk_free_gib=120.0, devices=4, device_owners=[], ports_in_use=[], cpus=240)


@pytest.mark.parametrize('change,word', [
    (dict(device_owners=['123']), 'held'), (dict(ports_in_use=[25800]), 'ports'),
    (dict(available_gib=271.9), 'available'), (dict(disk_free_gib=24.0), 'home'), (dict(devices=0), 'no TPU')])
def test_launch_gate(change, word):
    assert assigned_launch.gate_verdict(GOOD, 272) is None
    assert word in assigned_launch.gate_verdict(dict(GOOD, **change), 272)


def test_launch_is_refused_at_the_gate_before_anything_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(assigned_launch, 'external_ips', lambda: {ip: '1.2.3.4' for ip in IPS})
    monkeypatch.setattr(assigned_launch, 'partition_gate', lambda config: ({IPS[0]: 'held', IPS[1]: None}, {}))

    def refuse(*a, **k):
        raise AssertionError('claimed or uploaded through a closed gate')
    monkeypatch.setattr(assigned_launch.host_claim, 'claim_hosts', refuse)
    monkeypatch.setattr(assigned_launch.subprocess, 'run', refuse)
    with pytest.raises(SystemExit, match='refusing'):
        assigned_launch.launch(PROFILE, 'skyrl-od-t', tmp_path)


def test_dry_run_writes_two_rank_scripts_and_touches_no_host(tmp_path, monkeypatch):
    monkeypatch.setattr(assigned_launch, 'external_ips', lambda: {ip: '1.2.3.4' for ip in IPS})

    def no_ssh(*a, **k):
        raise AssertionError('dry run reached a host')
    monkeypatch.setattr(assigned_launch, 'ssh', no_ssh)
    record = assigned_launch.launch(PROFILE, 'skyrl-od-t', tmp_path, dry_run=True)
    assert record['hosts'] == IPS and record['uri'].startswith('gs://sk7524-tinker-tpu-us-central2/code-bundles/ray-training-')
    for rank in (0, 1):
        text = (tmp_path / f'launch-{rank}.sh').read_text()
        assert f'SKYPILOT_NODE_RANK={rank}' in text and "SKYPILOT_NODE_IPS='10.130.0.68 10.130.0.62'" in text
        assert 'SKYPILOT_TASK_ID=skyrl-od-t' in text and 'tpu.swarm.ray_train.bootstrap' in text
        assert 'profiles/rl28k-qwen35-erdos-s2-od16.json' in text and 'apt-get' not in text


def test_science_trainer_is_not_a_partition():
    circuit = Config.load('tpu/swarm/ray_train/profiles/rl28k-qwen35-circuit-s2-od.json')
    assert circuit.science_task
    with pytest.raises(ValueError):
        Config.from_dict(dict(circuit.to_dict(), hosts=2, assigned_host_ips=IPS, physical_node='n'))


def test_build_refuses_a_managed_launch(tmp_path):
    from tpu.swarm.ray_train.build import build
    with pytest.raises(ValueError, match='direct deployment'):
        build(PROFILE, tmp_path)


def test_two_chip_blocks_pass_the_topology_check_with_worker_0_on_y_2():
    from tpu.swarm.ray_train.assigned_topology import verify_partition
    rows = [dict(process_id=0, coords=[[x, y, 0] for y in (2, 3) for x in (0, 1)]),
            dict(process_id=1, coords=[[x, y, 0] for y in (0, 1) for x in (0, 1)])]
    assert verify_partition(rows, '1,2,1', '2,2,1')
    with pytest.raises(ValueError, match='wrong chip coordinates'):
        verify_partition([rows[0], dict(rows[1], coords=rows[0]['coords'])], '1,2,1', '2,2,1')


class Chip:
    device_kind, core_on_chip, platform = 'TPU v4', 0, 'tpu'

    def __init__(self, id, coords, process):
        self.id, self.coords, self.process_index = id, coords, process


def test_tp8_ring_is_connected_on_the_real_process_layout():
    from jax.experimental import mesh_utils
    # Worker 0 = process 0 on y 2-3, worker 1 = process 1 on y 0-1, as probed.
    chips = []
    for process, ys in enumerate(((2, 3), (0, 1))):
        for y in ys:
            for x in (0, 1):
                chips.append(Chip(len(chips), (x, y, 0), process))
    # MaxText's mesh: ici_parallelism with the tensor axis only.
    full = (1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1)
    mesh = mesh_utils.create_device_mesh(full, chips, contiguous_submeshes=False, allow_split_physical_axes=True).reshape(-1)
    assert {c.coords for c in mesh} == {(x, y, 0) for x in range(2) for y in range(4)}
    positions = {c.coords for c in mesh}
    seen, stack = {mesh[0].coords}, [mesh[0].coords]
    while stack:
        a = stack.pop()
        for b in positions - seen:
            if sum(abs(i - j) for i, j in zip(a, b)) == 1:
                seen.add(b)
                stack.append(b)
    assert seen == positions


def test_discovery_offers_group_farms_to_the_private_trainer(tmp_path, monkeypatch):
    config = Config.load(PROFILE)
    posts = []
    hash_ = config.inference.external_pool_required_compatibility[0]

    def private_rpc(entry, action, port, body=None, group=''):
        assert port == config.ports.inference and group == 'rl28k-qwen35'
        if body is not None:
            posts.append(body)
            return dict(ok=True, result=dict(run_id=config.run_id, instance='i', urls=body['urls']))
        return dict(ok=True, result=dict(run_id=config.run_id, model=config.model, instance='i', enabled=True,
                                         lease_scope='run', urls=[], accepted_compatibility=[hash_], candidate_limit=16))
    monkeypatch.setattr(discovery, 'private_rpc', private_rpc)
    monkeypatch.setattr(discovery, 'rpc', lambda *a, **k: dict(ok=True, result=dict(
        models=[config.model], url='http://10.202.0.44:24800', owner_run=None, state='unleased', active=0, engines=2,
        instance='f', capabilities=dict(compatibility_sha256=hash_), grading=dict(ready=True))))
    monkeypatch.setattr(discovery, 'write', lambda *a: None)
    private = dict(farms=[], trainers=[dict(run_id=config.run_id, model=config.model, group='rl28k-qwen35', external_ip='35.186.57.88',
                                            ssh_key='k', port=config.ports.inference, stable_urls=[], allow_spot=True,
                                            farm_pools=['tpuswarm-v6e8-east5b'])])
    args = SimpleNamespace(name_contains=['inference-farm-rl28k-qwen35'], registry=tmp_path / 'r', ssh_config_dir=tmp_path, farm_port=24800,
                           tunnels=False, port_base=29000, push=True, trainer_pool=[], trainer_job_id=[], discovery_group='',
                           run_scoped_only=True, slurm_pool=[])
    document = discovery.Service(args, private, now=lambda: 0).tick([dict(
        job_id=3244, status='RUNNING', run_id='inference-farm-rl28k-qwen35-east5b-v6e8-007', cluster='c', pool='tpuswarm-v6e8-east5b')])
    assert posts == [dict(run_id=config.run_id, instance='i', model=config.model, group='rl28k-qwen35', urls=['http://10.202.0.44:24800'])]
    assert document['trainers']['direct'][0]['state'] == 'updated'
    # Without a required hash a fresh trainer would never be offered a farm.
    assert direct_discovery.candidates(dict(model=config.model, accepted_compatibility=[]), private['trainers'][0],
                                       [dict(models=[config.model], capabilities=dict(compatibility_sha256=hash_), url='u')]) == []


def test_partition_home_gate_is_25_gib_and_pool_workers_keep_30(tmp_path):
    from shutil import disk_usage
    free = disk_usage(tmp_path).free / 1024**3
    if free < 30:
        pytest.skip('test host has under 30 GiB free')
    (tmp_path / 'accel0').touch()
    lifecycle.tpu_gates(shared=True, home=tmp_path, tmp=tmp_path, dev=tmp_path, home_min_gib=25,
                        run=lambda *a, **k: subprocess.CompletedProcess(a, 0, '', ''))
    with pytest.raises(RuntimeError, match='insufficient storage on home'):
        lifecycle.tpu_gates(shared=True, home=tmp_path, tmp=tmp_path, dev=tmp_path, home_min_gib=int(free) + 1000,
                            run=lambda *a, **k: subprocess.CompletedProcess(a, 0, '', ''))
