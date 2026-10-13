"""Pin an immutable step-zero pool for independent training seeds."""
from dataclasses import replace
import math
import json

from .bootstrap_reuse import identity, validate_reuse
from .config import Config


def capped_draft_contract(document, target, implementation_sha256):
    """Continue an unfinished journal under a fresh run ID and corrected cap.

    Root, tokenizer, prompts, requests, implementation and every scientific
    setting remain fixed. The caller copies the saved root and draft journal.
    """
    import copy
    contract = document['contract']
    if identity(contract) != document.get('sha256'):
        raise ValueError('bootstrap journal contract checksum mismatch')
    if contract.get('implementation_sha256') != implementation_sha256:
        raise ValueError('cannot migrate drafts across bootstrap implementations')
    old = Config.from_dict(contract['config'])
    if old.run_id == target.run_id or target.bootstrap_require_full_pool:
        raise ValueError('draft migration needs a fresh identity and capped retention')
    expected = json.loads(json.dumps(old.to_dict()).replace(old.run_id, target.run_id))
    expected['bootstrap_require_full_pool'] = False
    if expected != target.to_dict():
        raise ValueError('draft migration changes sampling or training settings')
    amended = copy.deepcopy(contract)
    amended['config'] = target.to_dict()
    return dict(contract=amended, sha256=identity(amended))


def finalize_capped_journal(document, rows):
    """Promote a fully graded frozen journal without regenerating or regrading.

    Preserve the original generation contract, including its historical strict
    completion flag, and explicitly record the corrected retention policy.
    """
    contract = document['contract']
    if identity(contract) != document.get('sha256'):
        raise ValueError('bootstrap journal contract checksum mismatch')
    source = Config.from_dict(contract['config'])
    if not source.bootstrap_fixed_budget or source.bootstrap_max_drafts < 1:
        raise ValueError('capped journal promotion requires a fixed sampling budget')
    expected = {f'bootstrap-{g:03d}-{i:03d}'
                for g in range(source.bootstrap_max_drafts // source.bootstrap_group_size)
                for i in range(source.bootstrap_group_size)}
    if len(rows) != source.bootstrap_max_drafts or {r['id'] for r in rows} != expected:
        raise ValueError('bootstrap journal must contain every graded draft exactly once')
    if any(r.get('correctness') not in (0, 1) for r in rows):
        raise ValueError('invalid grading correctness value')
    from .seed_bootstrap import selected_states
    selected = selected_states(rows, source.bootstrap_target_valid)
    if not selected:
        raise ValueError('bootstrap journal has no valid programs')
    pool = dict(step=0, states=selected, initial_states=[], puct_n={}, puct_m={}, puct_T=0)
    summary = dict(contract_sha256=document['sha256'], pool_sha256=identity(pool),
                   total=len(rows), drafted=len(rows), valid=sum(r['correctness'] == 1 for r in rows),
                   retained=len(selected), layers=1, optimizer_steps=0, stop_reason='fixed_budget',
                   completion_policy='retain_at_most_target')
    return summary, pool


def prepare(config, source_run_id, document, summary, pool):
    if source_run_id == config.run_id:
        raise ValueError('shared consumer must have a new run identity')
    if identity(pool) != summary.get('pool_sha256'):
        raise ValueError('shared pool checksum mismatch')
    states = pool.get('states', [])
    if (pool.get('step') != 0 or len(states) != summary.get('retained')
            or not 1 <= len(states) <= config.bootstrap_target_valid
            or pool.get('initial_states') or pool.get('puct_n') or pool.get('puct_m')
            or pool.get('puct_T') != 0 or any(s.get('parents') for s in states)):
        raise ValueError('shared pool must contain only independent step-zero roots')
    codes = [s.get('code', '').strip() for s in states]
    if (not all(codes) or len(set(codes)) != len(states)
            or any(not math.isfinite(s.get('value', float('nan'))) for s in states)):
        raise ValueError('shared pool must contain unique programs with finite rewards')
    pinned = replace(config, bootstrap_reuse_source_run_id=source_run_id,
                     bootstrap_reuse_contract_sha256=document['sha256'],
                     bootstrap_reuse_pool_sha256=summary['pool_sha256'])
    pinned.validate()
    validate_reuse(pinned, document, summary)
    return pinned
