from dataclasses import replace
import json
from pathlib import Path

import pytest

from tpu.swarm.ray_train.bootstrap_reuse import identity
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.shared_bootstrap import prepare

MANIFEST = json.loads(Path('tpu/swarm/ray_train/campaigns/qwen35-qwen38-20261006.json').read_text())
ROWS = {r['run_id']: r for r in MANIFEST['runs'] + MANIFEST['bootstrap_runs']}
CONSUMERS = [r for r in MANIFEST['runs'] if r.get('requires_shared_pool')]


def inputs(row):
    source = Config.load(ROWS[row['bootstrap_pool_source']]['profile'])
    target = Config.load(row['profile'])
    contract = {'config': source.to_dict()}
    document = {'contract': contract, 'sha256': identity(contract)}
    pool = dict(step=0, states=[dict(id=str(i), parents=[], code='program '+str(i), value=0.5) for i in range(512)],
                initial_states=[], puct_n={}, puct_m={}, puct_T=0)
    summary = dict(contract_sha256=identity(contract), pool_sha256=identity(pool),
                   drafted=1024, total=1024, retained=512, optimizer_steps=0)
    return target, source.run_id, document, summary, pool


@pytest.mark.parametrize('row', CONSUMERS, ids=lambda r: r['run_id'])
def test_source_pool_reused_across_regions_and_training_seeds(row):
    args = inputs(row)
    pinned = prepare(*args)
    assert pinned.bootstrap_reuse_source_run_id == args[1]
    assert pinned.bootstrap_reuse_pool_sha256 == identity(args[4])
    assert pinned.client_env['TTD_M0_LORA_SEED'] == str(row['seed'])


@pytest.mark.parametrize('damage', ['pool', 'counts', 'parents', 'step', 'shortfall', 'optimizer', 'source', 'model', 'loss', 'thinking', 'reward'])
def test_reuse_rejects_changed_science_or_noninitial_state(damage):
    target, source, document, summary, pool = inputs(CONSUMERS[0])
    if damage == 'pool': pool['states'].pop()
    elif damage == 'counts': pool['puct_n'] = {'a': 1}
    elif damage == 'parents': pool['states'][0]['parents'] = ['other']
    elif damage == 'step': pool['step'] = 1
    elif damage == 'shortfall': summary['drafted'] = 1000
    elif damage == 'optimizer': summary['optimizer_steps'] = 1
    elif damage == 'source': source = 'wrong-source'
    elif damage == 'model': target = replace(target, base_bundle_sha256='f'*64)
    elif damage == 'loss': target = replace(target, client_env={**target.client_env, 'TTD_LOSS_FN': 'cispo'})
    elif damage == 'thinking': target = replace(target, client_phase1_max_tokens=12000)
    elif damage == 'reward': target = replace(target, client_env={**target.client_env, 'TTD_ADV_PIECEWISE_RHO': '0.1'})
    # Even a correctly hashed later search tree must be rejected.
    summary['pool_sha256'] = identity(pool)
    with pytest.raises((ValueError, RuntimeError)):
        prepare(target, source, document, summary, pool)


@pytest.mark.parametrize('count', [1, 170, 428, 512])
def test_retention_is_a_cap_not_a_minimum(count):
    target, source, document, summary, pool = inputs(CONSUMERS[0])
    # Historical source collectors had the erroneous strict completion flag.
    document['contract']['config']['bootstrap_require_full_pool'] = True
    document['sha256'] = identity(document['contract'])
    summary['contract_sha256'] = document['sha256']
    pool['states'] = pool['states'][:count]
    summary.update(retained=count, pool_sha256=identity(pool))
    assert prepare(target, source, document, summary, pool).bootstrap_target_valid == 512


@pytest.mark.parametrize('valid, retained', [(170, 170), (428, 428), (700, 512)])
def test_completed_journal_ranks_up_to_cap(valid, retained):
    from tpu.swarm.ray_train.shared_bootstrap import finalize_capped_journal
    target, source, document, _, __ = inputs(CONSUMERS[0])
    rows = [dict(id=f'bootstrap-{i//16:03d}-{i%16:03d}', correctness=int(i<valid),
                 state=dict(id=str(i), code='program '+str(i), value=float(i), parents=['old']))
            for i in range(1024)]
    summary, pool = finalize_capped_journal(document, rows)
    assert summary['retained'] == retained
    assert pool['states'][0]['value'] == valid-1
    assert all(not s['parents'] for s in pool['states'])
    with pytest.raises(ValueError, match='every graded draft'):
        finalize_capped_journal(document, rows[:-1])


def test_completed_journal_with_zero_valid_programs_still_stops():
    from tpu.swarm.ray_train.shared_bootstrap import finalize_capped_journal
    _, __, document, ___, ____ = inputs(CONSUMERS[0])
    rows=[dict(id=f'bootstrap-{i//16:03d}-{i%16:03d}', correctness=0, state=None) for i in range(1024)]
    with pytest.raises(ValueError, match='no valid programs'):
        finalize_capped_journal(document, rows)


def test_partial_journal_migration_preserves_science_and_requires_same_code():
    from tpu.swarm.ray_train.shared_bootstrap import capped_draft_contract
    original=Config.load('tpu/swarm/ray_train/profiles/qcmp-qwen38-ac1-pwc05-s0-20261006.json')
    original=replace(original,bootstrap_require_full_pool=True)
    contract=dict(config=original.to_dict(),implementation_sha256='fixed-code',prompt='saved question',root_sha256='saved root')
    document=dict(contract=contract,sha256=identity(contract))
    d=json.loads(json.dumps(original.to_dict()).replace(original.run_id,original.run_id+'-cap'))
    d['bootstrap_require_full_pool']=False;target=Config.from_dict(d)
    corrected=capped_draft_contract(document,target,'fixed-code')
    assert corrected['contract']['prompt']=='saved question'
    assert document['contract']['config']['bootstrap_require_full_pool'] is True
    with pytest.raises(ValueError,match='implementations'):
        capped_draft_contract(document,target,'different-code')
    with pytest.raises(ValueError,match='settings'):
        capped_draft_contract(document,replace(target,client_phase1_max_tokens=12000),'fixed-code')
