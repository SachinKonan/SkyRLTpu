"""Topology-aware science admission and reference gates before training."""
import os
from pathlib import Path
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy


def split_roles(task, train_ranks, inference_ranks, *, placement_backend='tpu', accelerator=None):
    train, inference = list(train_ranks), list(inference_ranks)
    v5p_cpu = accelerator == 'tpu-v5p-32' and task == 'placement' and placement_backend == 'cpu'
    expected = (1, 3) if v5p_cpu else (4, 4)
    if ((len(train), len(inference)) != expected or len(set(train + inference)) != sum(expected)):
        raise ValueError(f'science topology requires disjoint host blocks {expected}')
    grading = inference.pop() if task == 'placement' and placement_backend == 'tpu' else None
    return train, inference, grading


def prepare(config, ips, nodes, grading_rank):
    root = os.environ['SCIENCE_WORKER_ROOT']
    refs = []
    group = None
    if config.science_task == 'placement' and config.science_placement_backend == 'cpu':
        from .placement_ray import grade_cpu_case
        from .placement_task import CASES
        for ip in ips:
            helper = config.client_env.get('SCIENCE_PLACEMENT_HELPER', 'none')
            for name in ('challenge_seed.py', 'challenge_seed_fast_proxy.py' if helper == 'fast_proxy_v1' else 'challenge_seed_jax.py'):
                source = (Path(root) / 'tpu/science' / name).read_text()
                for case in CASES:
                    refs.append(grade_cpu_case.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                        nodes[ip]['NodeID'], soft=False)).remote(source, case, root,
                            slots_per_host=config.science_placement_slots_per_host, helper=helper))
    elif config.science_task == 'placement':
        from .placement_slots import grading_bundles
        from .placement_ray import grade_case
        from .placement_task import CASES
        node = nodes[ips[grading_rank]]
        group = placement_group(grading_bundles(node), strategy='STRICT_PACK',
                                name='science-placement-' + config.run_id)
        ray.get(group.ready(), timeout=120)
        for name in ('challenge_seed.py', 'challenge_seed_jax.py'):
            source = (Path(root) / 'tpu/science' / name).read_text()
            for case in CASES:
                refs.append(grade_case.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=group, placement_group_bundle_index=-1)).remote(
                        source, case, root, accelerator=config.accelerator))
    else:
        from .ray_cpu import grade
        source = (Path(root) / 'tpu/science/seed_routing.py').read_text()
        for ip in ips:
            refs.append(grade.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                nodes[ip]['NodeID'], soft=False)).remote('routing', source, root,
                    slots_per_host=config.science_routing_slots_per_host,
                    routing_suite=config.client_env.get("SCIENCE_ROUTING_SUITE", "full")))
    return group, refs


def check_references(task, results, *, expected_hosts=8, placement_backend='tpu', routing_suite='full', placement_helper='none'):
    from .routing_suite import validate_suite
    validate_suite(routing_suite)
    cpu_placement = task == 'placement' and placement_backend == 'cpu'
    from .challenge_contract import CASES
    expected = 2 * len(CASES) * expected_hosts if cpu_placement else 2 * len(CASES) if task == 'placement' else expected_hosts
    if len(results) != expected or any(r['correctness'] != 1 for r in results):
        raise RuntimeError('science reference failed; refusing to train on a broken grader')
    if cpu_placement:
        from collections import Counter
        from .challenge_contract import CASES
        by_host = {}
        for row in results:
            m = row['metrics']
            if (m.get('physical_tpu_chips') != 0 or m.get('hard_memory_gib') != 8
                    or len(m.get('hard_cpus', [])) != 4
                    or m.get('candidate_device', {}).get('device_kind') != 'cpu'):
                raise RuntimeError('CPU placement reference escaped its resource contract')
            if placement_helper == 'fast_proxy_v1' and (m.get('helper') != placement_helper
                    or m.get('candidate_device', {}).get('helper') != placement_helper
                    or set(m.get('helper_hashes', {})) != {'__init__.py', 'congestion.c', 'libproxy.so'}):
                raise RuntimeError('CPU placement reference did not use the configured helper')
            by_host.setdefault(m['ray_node_id'], []).append(m['case'])
        if len(by_host) != expected_hosts or any(Counter(v) != Counter({c: 2 for c in CASES}) for v in by_host.values()):
            raise RuntimeError('CPU placement references did not cover every host and case')
    elif task == 'placement':
        if {r['metrics']['physical_chip_id'] for r in results} != {0, 1, 2, 3}:
            raise RuntimeError('placement references did not exercise all four chips')
        if len({r['metrics']['ray_node_id'] for r in results}) != 1:
            raise RuntimeError('placement grading escaped the selected host')
    elif any(r['metrics'].get('case_count') != (24 if routing_suite == 'q20' else 72)
             or r['metrics'].get('routing_suite', 'full') != routing_suite for r in results):
        raise RuntimeError('routing reference did not verify the selected suite')
    return dict(task=task, evaluations=len(results), rewards=[r['reward'] for r in results])
