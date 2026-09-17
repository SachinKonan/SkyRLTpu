"""Verify and combine completed routing seed shards using normal PUCT admission."""
import hashlib
import math
from pathlib import Path

from .bootstrap import commit_rows, identity, read, save

# The sixteen-slot update changes only grading admission in bootstrap.py.
# Preserve import of the already-running frozen Muse shards, whose sampling,
# repair selection, grading budgets and PUCT admission are identical.
PRE_SLOT_UPDATE_BOOTSTRAP_SHA256 = '095a117a734d00c913db453eee541e5a543f7aade39e1e5908b4c3511d79c990'


def compatible_bootstrap_implementation(digest):
    current = hashlib.sha256(Path(__file__).with_name('bootstrap.py').read_bytes()).hexdigest()
    return digest in (current, PRE_SLOT_UPDATE_BOOTSTRAP_SHA256)


def verify_pool(path, expected):
    pool = read(path)
    if identity(pool) != expected or pool.get('step') != 0:
        raise ValueError('imported seed pool checksum or step mismatch')
    seeds = [s for s in pool['states'] if s.get('code')]
    if not seeds or any(not math.isfinite(s['value']) or not 0 < s['value'] <= 1 for s in seeds):
        raise ValueError('imported seed pool has no valid positive-reward seeds')
    return pool


def sampling_contract(config):
    """Fields that must agree across shards and their training destination."""
    return dict(model=config.model, model_preset=config.model_preset,
                member=config.client_member_spec, base_bundle_sha256=config.base_bundle_sha256,
                hf=config.cache.hf, task=config.science_task,
                context=config.client_context_window, phase1=config.client_phase1_max_tokens,
                inference=config.to_dict()['inference'],
                env={k: v for k, v in config.client_env.items()
                     if k not in ('GROUPS_PER_BATCH', 'TTD_SICK_MARKER')})


def merge(clients, target, destination):
    """clients are complete downloaded client/ directories, not partial pools."""
    from ttt_discover import State
    from ttt_discover.tinker_utils.sampler import PUCTSampler
    from tpu.swarm.ray_train.config import Config
    from .training_env import RoutingTrainingEnv, candidate_prompt

    target.validate()
    if target.science_task != 'routing' or target.inference_only or target.bootstrap_layers:
        raise ValueError('seed destination must be normal routing training')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('merge destination already exists; do not overwrite a promoted pool')
    roots, rows, sources, run_ids = {}, [], [], set()
    signature = sampling_contract(target)
    tokenizer = None
    for client in map(Path, clients):
        folder = client / 'bootstrap'
        record = read(folder / 'contract.json')
        contract = record['contract']
        summary = read(folder / 'complete.json')
        config = Config.from_dict(contract['config'])
        config.validate()
        if (not config.bootstrap_only or config.bootstrap_layers != 2
                or config.run_id in run_ids or sampling_contract(config) != signature
                or contract['prompt'] != candidate_prompt('routing')
                or contract['policy'] != 'frozen-base-no-adapter' or contract['repair'] != 'invalid-only'):
            raise ValueError('incompatible or duplicate seed shard')
        if (identity(contract) != record['sha256'] or summary['contract_sha256'] != record['sha256']
                or not compatible_bootstrap_implementation(contract['implementation_sha256'])
                or summary['optimizer_steps'] != 0 or summary['layers'] != 2):
            raise ValueError('seed contract checksum or implementation mismatch')
        if tokenizer is not None and tokenizer != contract['tokenizer']:
            raise ValueError('shards used different tokenizers')
        tokenizer = contract['tokenizer']
        run_ids.add(config.run_id)
        verify_pool(client / 'tinker_log' / config.run_id / 'puct_sampler_step_000000.json',
                    summary['pool_sha256'])
        shard_roots = {s.id: s for s in map(State.from_dict, read(folder / 'roots.json'))}
        groups = int(config.client_env['GROUPS_PER_BATCH'])
        size = int(config.client_env['GROUP_SIZE'])
        if len(shard_roots) != groups or set(roots) & set(shard_roots):
            raise ValueError('missing or duplicate task roots')
        roots.update(shard_roots)
        shard_rows = []
        for layer in range(2):
            plan = read(folder / f'layer-{layer}-plan.json')
            layer_rows = []
            if len(plan) > groups or (layer == 0 and len(plan) != groups):
                raise ValueError('incorrect seed group budget')
            prior = {r['id']: r for r in shard_rows}
            for index, parent in enumerate(plan):
                if parent['root_id'] not in shard_roots:
                    raise ValueError('unknown task root')
                repair = parent['repair_parent_id']
                if layer == 0:
                    if repair is not None:
                        raise ValueError('draft unexpectedly has a repair parent')
                elif (repair not in prior or prior[repair]['correctness'] != 0
                        or prior[repair]['root_id'] != parent['root_id']):
                    raise ValueError('repair parent was not an invalid draft')
                directory = folder / f'layer-{layer}' / f'group-{index:03d}'
                paths = sorted(directory.glob('grade-*.json'))
                if len(paths) != size:
                    raise ValueError('incomplete graded seed group')
                for path in paths:
                    row = read(path)
                    if (row['root_id'] != parent['root_id'] or row['repair_parent_id'] != repair
                            or row['layer'] != layer or row['correctness'] not in (0, 1)
                            or not math.isfinite(row['reward'])
                            or (row['correctness'] == 1 and not (row['code'] and 0 < row['reward'] <= 1))
                            or (row['correctness'] == 0 and row['reward'] != 0)):
                        raise ValueError('invalid graded seed record')
                    layer_rows.append(row)
            declared = read(folder / f'layer-{layer}-summary.json')
            if declared != dict(groups=len(plan), total=len(layer_rows),
                                valid=sum(r['correctness'] == 1 for r in layer_rows)):
                raise ValueError('layer totals disagree with grade journal')
            shard_rows.extend(layer_rows)
        if (summary['total'] != len(shard_rows)
                or summary['valid'] != sum(r['correctness'] == 1 for r in shard_rows)):
            raise ValueError('completion totals disagree with grade journal')
        rows.extend(shard_rows)
        sources.append(dict(run_id=config.run_id, **summary))
    if len(roots) != int(target.client_env['GROUPS_PER_BATCH']):
        raise ValueError('combined seed roots do not match the training batch size')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate candidate IDs')
    path = destination / 'puct_sampler_step_000000.json'
    save(path, dict(step=0, states=[s.to_dict() for s in roots.values()],
                   initial_states=[s.to_dict() for s in roots.values()], puct_n={}, puct_m={}, puct_T=0))
    sampler = PUCTSampler(str(destination / 'puct_sampler.json'), env_type=RoutingTrainingEnv,
                          problem_type='routing', batch_size=len(roots), resume_step=0)
    commit_rows(sampler, roots, rows)
    pool = read(path)
    digest = identity(pool)
    verify_pool(path, digest)
    report = dict(pool_sha256=digest, roots=len(roots), total=len(rows),
                  valid=sum(r['correctness'] == 1 for r in rows),
                  retained=sum(bool(s['code']) for s in pool['states']), optimizer_steps=0,
                  sources=sources, repair_selection='invalid-only within each shard')
    save(destination / 'seed-import.json', report)
    return report
