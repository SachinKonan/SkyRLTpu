"""Remote-only trainer on the 2x4x2 box of the on-demand v4-64: ranks 5,3,2,4 train, rank 0 is a chipless head."""
import pytest

from tpu.swarm.ray_train.bootstrap import check_worker_ids, workload_resources
from tpu.swarm.ray_train.commands import client_environment
from tpu.swarm.ray_train.config import Config

PROFILE = 'tpu/swarm/ray_train/profiles/remote-only-v432-qwen-ac2-pilot-20260922.json'
IPS = [f'10.0.0.{worker}' for worker in range(6)]
# Measured by job 3275 under bounds 1,2,2, the same in both address orders: worker -> first y, z.
BOX = {5: (0, 0), 3: (2, 0), 2: (0, 1), 4: (2, 1)}


def od_raw(tmp_path, **changes):
    raw = Config.load(PROFILE).to_dict()
    raw.update(root=str(tmp_path), accelerator='tpu-v4-64', hosts=6, zone='us-central2-b')
    raw['trainer'] = dict(raw['trainer'], process_bounds='1,2,2')
    raw.update(changes)
    return raw


def test_od_trainer_is_the_box_with_two_chipless_ranks(tmp_path):
    config = Config.from_dict(od_raw(tmp_path))
    assert config.od_v4_64_subset and config.idle_ranks == (0, 1) and config.active_ranks == [2, 3, 4, 5]
    assert config.trainer.hosts == 4 and config.trainer.tp * config.trainer.fsdp == 16
    assert config.inference_hosts == 0 and config.engine_groups([]) == []


def test_other_profiles_have_no_idle_ranks(tmp_path):
    config = Config.load(PROFILE)
    assert not config.od_v4_64_subset and config.idle_ranks == ()
    assert config.active_ranks == list(range(config.hosts))
    farm = Config.load('tpu/swarm/ray_train/profiles/inference-farm-scaf-qwen38-central2-v464od-001.json')
    assert farm.od_v4_64_subset and farm.idle_ranks == () and farm.active_ranks == list(range(6))


@pytest.mark.parametrize('field,value,match', [
    ('hosts', 6, 'every host'), ('tp', 4, 'TP x FSDP'), ('process_bounds', '1,1,4', '2x4x2 box'),
    ('process_bounds', '2,2,1', '2x4x2 box'), ('chip_bounds', '2,1,1', '2x4x2 box')])
def test_od_trainer_rejects_any_other_layout(tmp_path, field, value, match):
    raw = od_raw(tmp_path)
    raw['trainer'] = dict(raw['trainer'], **{field: value})
    with pytest.raises(ValueError, match=match):
        Config.from_dict(raw)


def test_six_hosts_need_the_on_demand_pool_and_remote_only(tmp_path):
    raw = od_raw(tmp_path, hosts=5)
    with pytest.raises(ValueError):
        Config.from_dict(raw)
    raw = od_raw(tmp_path)
    raw['inference'] = dict(raw['inference'], remote_only=False)
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_worker_ids_must_equal_ranks(tmp_path):
    config = Config.from_dict(od_raw(tmp_path))
    check_worker_ids(config, {'SKYPILOT_TPU_WORKER_IDS': '0,1,2,3,4,5'})
    for value in ({}, {'SKYPILOT_TPU_WORKER_IDS': '0,1,2,3,4,6'}):
        with pytest.raises(SystemExit):
            check_worker_ids(config, value)
    check_worker_ids(Config.load(PROFILE), {})  # no fixed ranks, no requirement


def test_idle_ranks_advertise_nothing(tmp_path):
    raw = od_raw(tmp_path)
    raw['grading'] = dict(raw.get('grading') or {}, families={'math': dict(slots_per_host=16, cpus=2, memory_gib=4)})
    config = Config.from_dict(raw)
    for rank in config.idle_ranks:
        assert workload_resources(config, rank) == {'TPU': 0}
    for rank in config.active_ranks:
        assert workload_resources(config, rank) == {'TPU': 4, 'grading_math': 16}


def test_local_grading_capacity_counts_working_hosts_only(tmp_path):
    raw = od_raw(tmp_path)
    raw['grading'] = dict(raw.get('grading') or {}, families={'math': dict(slots_per_host=16, cpus=2, memory_gib=4)})
    config = Config.from_dict(raw)
    env = client_environment(config, tmp_path, IPS[0], trainer_head=IPS[1])
    assert env['SKYRL_GRADING_LOCAL_HOSTS'] == '4' and env['SKYRL_GRADING_LOCAL_SLOTS'] == '64'


class Ref:
    def __init__(self, value):
        self.value = value


class Method:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def remote(self, *args):
        self.owner.calls.append((self.name, args))
        return Ref(getattr(self.owner, self.name)(*args))


class FakeHost:
    """A Host actor handle: answers the controller's probes with this rank's chip coordinates."""

    def __init__(self, rank, rows, calls):
        self.rank, self.rows, self.calls = rank, rows, calls

    def __getattr__(self, name):
        if name in ('preflight', 'topology_ready', 'probe', 'set_trainer_leader', 'prepare'):
            return Method(self, '_' + name)
        raise AttributeError(name)

    def _preflight(self):
        return True

    _topology_ready = _preflight

    def _prepare(self, role):
        return dict(rank=self.rank, role=role)

    def _set_trainer_leader(self, rank):
        return rank

    def _probe(self, ranks, port, subset=False):
        return dict(process_id=ranks.index(self.rank), coords=self.rows[self.rank])


class Reached(Exception):
    pass


def run_setup(tmp_path, monkeypatch, config, ips, rows, stop='topology_validated'):
    """Drive Controller.setup through topology selection against fake host actors."""
    from tpu.swarm.ray_train import controller
    owner = controller.Controller(config, ips)
    fakes, events = [], {}

    class Actor:
        def options(self, **kw):
            return self

        def remote(self, config_dict, rank, ips):
            calls = []
            fakes.append((rank, calls))
            return FakeHost(rank, rows, calls)

    def report(event, **fields):
        events[event] = fields
        if event == stop:
            raise Reached

    owner.report = report
    monkeypatch.setattr(owner, '_monitor', lambda: None)
    monkeypatch.setattr(owner, 'checked_get', lambda refs, timeout: [ref.value for ref in refs])
    monkeypatch.setattr(controller, 'NodeAffinitySchedulingStrategy', lambda *a, **kw: None)
    monkeypatch.setattr(controller.ray, 'nodes', lambda: [dict(NodeManagerAddress=ip, NodeID=ip, Alive=True) for ip in ips])
    monkeypatch.setattr(controller.ray, 'remote', lambda **kw: lambda cls: Actor())
    with pytest.raises(Reached):
        owner.setup()
    return events, dict(fakes)


def probes(calls):
    return [args for name, args in calls if name == '_probe']


def box_coords(blocks=BOX):
    return {rank: [[x, y0 + y, z] for y in range(2) for x in range(2)] for rank, (y0, z) in blocks.items()}


def row_coords(rows):
    return {rank: [[x, y, z] for x in range(2) for y in range(2)] for rank, z in rows.items()}


def test_setup_trains_on_the_box_and_keeps_the_head_on_rank_0(tmp_path, monkeypatch):
    config = Config.from_dict(od_raw(tmp_path))
    events, calls = run_setup(tmp_path, monkeypatch, config, IPS, box_coords(), stop='cache_barrier_complete')
    assert events['topology_validated']['train_ranks'] == [5, 3, 2, 4]
    assert events['topology_validated']['inference_ranks'] == []
    assert events['topology_validated']['trainer_leader'] == 5
    for rank in (2, 3, 4, 5):
        # Only the subset probe under the real bounds; a full probe of six hosts is no box.
        assert probes(calls[rank]) == [([5, 3, 2, 4], config.ports.topology_subset, True)]
        assert ('_prepare', ('trainer',)) in calls[rank]
    # Rank 0 installs the client without a chip; rank 1 does nothing at all.
    assert not probes(calls[0]) and ('_topology_ready', ()) in calls[0] and ('_prepare', ('trainer',)) in calls[0]
    assert not probes(calls[1]) and {name for name, _ in calls[1]} == {'_preflight', '_set_trainer_leader'}


def test_setup_accepts_libtpu_placing_hosts_in_another_frame(tmp_path, monkeypatch):
    # Address order does not bind a process to a block.
    config = Config.from_dict(od_raw(tmp_path))
    flipped = {5: (2, 1), 3: (0, 1), 2: (2, 0), 4: (0, 0)}
    events, _ = run_setup(tmp_path, monkeypatch, config, IPS, box_coords(flipped))
    assert events['remote_only_topology_verified']['train_ranks'] == [5, 3, 2, 4]


@pytest.mark.parametrize('broken', [
    {5: (0, 0), 3: (2, 0), 2: (0, 1), 4: (0, 1)},  # two hosts on one block
    {5: (0, 0), 3: (1, 0), 2: (0, 1), 4: (2, 1)}])  # a block off its 2-chip grid
def test_setup_refuses_a_broken_box(tmp_path, monkeypatch, broken):
    config = Config.from_dict(od_raw(tmp_path))
    with pytest.raises(ValueError, match='wrong chip coordinates'):
        run_setup(tmp_path, monkeypatch, config, IPS, box_coords(broken))


class Chip:
    device_kind, core_on_chip, platform = 'TPU v4', 0, 'tpu'

    def __init__(self, id, coords, process):
        self.id, self.coords, self.process_index = id, coords, process


@pytest.mark.parametrize('order', [(5, 3, 2, 4), (2, 3, 4, 5), (5, 4, 3, 2)])
def test_tp8_groups_are_the_two_z_planes_for_any_process_order(order):
    from jax.experimental import mesh_utils
    coords = box_coords()
    chips = [Chip(0, tuple(c), process) for process, rank in enumerate(order) for c in coords[rank]]
    # MaxText: create_device_mesh(ici_parallelism, contiguous_submeshes=False, allow_split_physical_axes=True).
    mesh = mesh_utils.create_device_mesh((2, 8), chips, contiguous_submeshes=False, allow_split_physical_axes=True)
    for row in mesh:
        assert len({chip.coords[2] for chip in row}) == 1
        assert {chip.coords[:2] for chip in row} == {(x, y) for x in range(2) for y in range(4)}
    assert {tuple(sorted({c.process_index for c in row})) for row in mesh} == (
        {(0, 1), (2, 3)} if order == (5, 3, 2, 4) else {tuple(sorted(order.index(r) for r in pair)) for pair in ((5, 3), (2, 4))})


def test_science_references_and_preparation_skip_the_idle_ranks(tmp_path, monkeypatch):
    from tpu.science import training_setup
    raw = Config.load('tpu/swarm/ray_train/profiles/gradetest-ro-v5p32-qwen35-circuit-001.json').to_dict()
    raw.update(root=str(tmp_path), accelerator='tpu-v4-64', hosts=6, zone='us-central2-b', single_bucket=False)
    raw['trainer'] = dict(raw['trainer'], hosts=4, tp=8, fsdp=2, process_bounds='1,2,2', chip_bounds='2,2,1')
    raw['inference'] = dict(raw['inference'], ragged_conv1d=True)
    config = Config.from_dict(raw)
    seen = {}

    def prepare(config, ips, nodes, grading_rank):
        seen['ips'] = ips
        return None, []
    monkeypatch.setattr(training_setup, 'prepare', prepare)
    monkeypatch.setattr(training_setup, 'check_references', lambda task, results, **kw: seen.update(kw) or {})
    _, calls = run_setup(tmp_path, monkeypatch, config, IPS, box_coords(), stop='science_reference_passed')
    assert seen['ips'] == IPS[2:] and seen['expected_hosts'] == 4
    assert all(('_prepare', ('trainer',)) in calls[rank] for rank in (0, 2, 3, 4, 5))
    assert not any(name == '_prepare' for name, _ in calls[1])


def test_v4_32_setup_keeps_the_full_probe_and_z_order(tmp_path, monkeypatch):
    raw = Config.load(PROFILE).to_dict()
    raw['root'] = str(tmp_path)
    config = Config.from_dict(raw)
    ips = IPS[:4]
    rows = {0: 1, 1: 2, 2: 3, 3: 0}
    # The subset probe of a v4-32 reports the process's own row after reordering.
    events, calls = run_setup(tmp_path, monkeypatch, config, ips, row_coords(rows))
    assert events['topology_validated']['train_ranks'] == [3, 0, 1, 2]
    assert events['remote_only_topology_verified']['rank_to_z'] == [1, 2, 3, 0]
    assert probes(calls[0]) == [([0, 1, 2, 3], config.ports.topology_jax),
                                ([3, 0, 1, 2], config.ports.topology_subset, True)]
    assert all(('_topology_ready', ()) in calls[rank] for rank in range(4))
    assert events['topology_validated']['trainer_leader'] == 3
