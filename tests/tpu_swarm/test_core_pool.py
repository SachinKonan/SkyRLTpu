"""Shared grading core pool: count-based admission, pinned cores, any family mix."""
import json
import multiprocessing as mp

import pytest

from tpu.science import core_pool


def numa(tmp_path, nodes):
    for index, cpus in enumerate(nodes):
        folder = tmp_path / f'node{index}'
        folder.mkdir(parents=True)
        (folder / 'cpulist').write_text(cpus)
    return str(tmp_path)


V5P = '0-51,104-155', '52-103,156-207'


def test_layout_spreads_services_over_numa_nodes(tmp_path):
    topology = numa(tmp_path / 'topo', V5P)
    grading, service, nodes = core_pool.layout(48, set(range(208)), topology)
    assert len(service) == 48 and len(grading) == 160
    assert sum(1 for c in service if c in range(0, 52) or c in range(104, 156)) == 24
    assert set(grading).isdisjoint(service)
    assert set(nodes) == set(grading) and set(nodes.values()) == {0, 1}
    with pytest.raises(ValueError):
        core_pool.layout(8, set(range(208)), topology)


def install(tmp_path, service=48, memory=200, cpus=208):
    topology = numa(tmp_path / 'topo', V5P) if cpus == 208 else str(tmp_path / 'none')
    root = tmp_path / 'locks'
    document = core_pool.install(service, memory, root=root, affinity=set(range(cpus)), topology_root=topology)
    return root, document


def test_mixed_families_take_disjoint_cores_on_one_node(tmp_path):
    root, document = install(tmp_path)
    nodes = {int(c): n for c, n in document['nodes'].items()}
    routing, r_lease = core_pool.try_acquire(10, 20, root=root)
    placement, p_lease = core_pool.try_acquire(4, 4, root=root)
    ac2, a_lease = core_pool.try_acquire(2, 4, root=root)
    held = routing + placement + ac2
    assert len(set(held)) == 16 and set(held) <= set(document['grading'])
    for cpus in (routing, placement, ac2):
        assert len({nodes[c] for c in cpus}) == 1
    for lease in (r_lease, p_lease, a_lease):
        lease.close()
    again, lease = core_pool.try_acquire(10, 20, root=root)
    assert again == routing
    lease.close()


def test_cores_and_memory_both_bound_admission(tmp_path):
    root, document = install(tmp_path, memory=24)
    first = core_pool.try_acquire(2, 20, root=root)
    assert first is not None
    assert core_pool.try_acquire(2, 8, root=root) is None  # 4 GiB left
    second = core_pool.try_acquire(2, 4, root=root)
    assert second is not None
    first[1].close()
    second[1].close()
    leases = []
    while (held := core_pool.try_acquire(10, 1, root=root)) is not None:
        leases.append(held)
    assert len(leases) == 16  # 160 cores / 10, then none left
    with pytest.raises(TimeoutError):
        core_pool.acquire(10, 1, deadline_seconds=.3, root=root)
    for _, lease in leases:
        lease.close()


def test_rejects_impossible_requests_and_missing_pool(tmp_path):
    root, _ = install(tmp_path, memory=24)
    with pytest.raises(ValueError):
        core_pool.try_acquire(161, 1, root=root)
    with pytest.raises(ValueError):
        core_pool.try_acquire(2, 25, root=root)
    with pytest.raises(core_pool.PoolUnavailable):
        core_pool.try_acquire(2, 4, root=tmp_path / 'empty')


def _hold_and_die(root, out):
    import os
    cpus, _lease = core_pool.try_acquire(10, 20, root=root)
    out.write_text(json.dumps(cpus))
    os._exit(0)  # never closes the lease explicitly


def test_dead_holder_releases_its_cores(tmp_path):
    root, _ = install(tmp_path)
    out = tmp_path / 'held.json'
    process = mp.get_context('fork').Process(target=_hold_and_die, args=(root, out))
    process.start()
    process.join(10)
    cpus = json.loads(out.read_text())
    again, lease = core_pool.try_acquire(10, 20, root=root)
    assert again == cpus
    lease.close()


def test_install_refuses_to_change_a_pool_in_use(tmp_path):
    root, document = install(tmp_path)
    cpus, lease = core_pool.try_acquire(2, 4, root=root)
    topology = str(tmp_path / 'topo')
    with pytest.raises(RuntimeError, match='different core pool'):
        core_pool.install(64, 200, root=root, affinity=set(range(208)), topology_root=topology)
    assert core_pool.install(48, 200, root=root, affinity=set(range(208)), topology_root=topology) == document
    lease.close()
    changed = core_pool.install(64, 200, root=root, affinity=set(range(208)), topology_root=topology)
    assert len(changed['grading']) == 144
    assert json.loads((root / core_pool.POOL_FILE).read_text()) == changed


def test_v6e8_farm_defaults(tmp_path):
    from tpu.swarm.ray_train.config import Config
    from pathlib import Path
    profile = Path(__file__).parents[2] / 'tpu/swarm/ray_train/profiles/farm-v6e8-east5b-qwen-grading-ac2-20260924.json'
    config = Config.load(profile)
    assert config.grading_pool and config.engines_per_host == 2
    assert config.grading_service_cpus == 48
    grading, service, _ = core_pool.layout(config.grading_service_cpus, set(range(180)))
    assert len(grading) == 132


def test_remote_only_trainer_may_bootstrap_one_bounded_layer_through_farms(tmp_path):
    from pathlib import Path
    from tpu.swarm.ray_train.config import Config
    profile = Path(__file__).parents[2] / 'tpu/swarm/ray_train/profiles/remote-only-v6e8-east5b-qwen-ac2-pilot-20261005.json'
    raw = Config.load(profile).to_dict()
    raw.update(root=str(tmp_path), client_env=dict(raw['client_env'], TTD_ENV='erdos_min_overlap', TTD_PROBLEM_TYPE=''),
               bootstrap_layers=1, bootstrap_max_drafts=1024, bootstrap_target_valid=512, bootstrap_group_size=16,
               bootstrap_max_groups=32, bootstrap_fixed_budget=True, bootstrap_require_full_pool=True, bootstrap_seed=0)
    config = Config.from_dict(raw)
    assert config.bootstrap_module == 'tpu.swarm.ray_train.seed_bootstrap'
    for change in (dict(bootstrap_all_hosts=True), dict(bootstrap_max_drafts=0, bootstrap_fixed_budget=False,
                                                        bootstrap_require_full_pool=False, bootstrap_seed=None),
                   dict(bootstrap_layers=2)):
        with pytest.raises(ValueError):
            Config.from_dict(dict(raw, **change))
