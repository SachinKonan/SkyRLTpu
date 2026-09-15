"""Topology-aware science admission and reference gates before training."""
import os
from pathlib import Path
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy


def split_roles(task, train_ranks, inference_ranks):
    train, inference = list(train_ranks), list(inference_ranks)
    if len(train) != 4 or len(inference) != 4 or set(train) & set(inference):
        raise ValueError('science topology requires disjoint four-host blocks')
    grading = inference.pop() if task == 'placement' else None
    return train, inference, grading


def prepare(config, ips, nodes, grading_rank):
    root = os.environ['SCIENCE_WORKER_ROOT']
    refs = []
    group = None
    if config.science_task == 'placement':
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
                nodes[ip]['NodeID'], soft=False)).remote('routing', source, root))
    return group, refs


def check_references(task, results):
    if len(results) != 8 or any(r['correctness'] != 1 for r in results):
        raise RuntimeError('science reference failed; refusing to train on a broken grader')
    if task == 'placement':
        if {r['metrics']['physical_chip_id'] for r in results} != {0, 1, 2, 3}:
            raise RuntimeError('placement references did not exercise all four chips')
        if len({r['metrics']['ray_node_id'] for r in results}) != 1:
            raise RuntimeError('placement grading escaped the selected host')
    elif any(r['metrics'].get('case_count') != 72 for r in results):
        raise RuntimeError('routing reference did not verify all 72 cases')
    return dict(task=task, evaluations=len(results), rewards=[r['reward'] for r in results])
