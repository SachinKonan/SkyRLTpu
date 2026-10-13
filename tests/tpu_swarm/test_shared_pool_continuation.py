from dataclasses import replace

import pytest

from tpu.swarm.ray_train.bootstrap_reuse import identity, validate_reuse
from tpu.swarm.ray_train.config import Config


def case():
    original = Config.load('tpu/swarm/ray_train/profiles/qcmp-qwen38-qubit-pwc05-s1-shared-v464-wd-20261007-pinned.json')
    original = replace(original, bootstrap_reuse_source_run_id='',
                       bootstrap_reuse_contract_sha256='', bootstrap_reuse_pool_sha256='')
    contract = {'config': original.to_dict()}
    digest = identity(contract)
    cfg = replace(original, bootstrap_reuse_source_run_id=original.run_id,
                  bootstrap_reuse_contract_sha256=digest, bootstrap_reuse_pool_sha256='a'*64,
                  resume_min_checkpoint_step=8,
                  client_env={**original.client_env, 'NUM_EPOCHS': '16'})
    summary = dict(contract_sha256=digest, pool_sha256='a'*64, optimizer_steps=0,
                   drafted=1024, total=1024, retained=170)
    return cfg, {'contract': contract, 'sha256': digest}, summary


def test_checkpointed_extension_preserves_seed_contract():
    cfg, doc, summary = case()
    before = identity(doc)
    validate_reuse(cfg, doc, summary)
    assert identity(doc) == before


@pytest.mark.parametrize('floor,cap,resume', [(0, '16', True), (8, '9', True),
                                           (16, '16', True), (8, '16', False),
                                           (8, 'bad', True)])
def test_invalid_extension_rejected(floor, cap, resume):
    cfg, doc, summary = case()
    cfg = replace(cfg, resume_min_checkpoint_step=floor, checkpoint_resume=resume,
                  client_env={**cfg.client_env, 'NUM_EPOCHS': cap})
    with pytest.raises(RuntimeError, match='checkpointed continuation'):
        validate_reuse(cfg, doc, summary)


def test_extension_still_rejects_learning_rate_change():
    cfg, doc, summary = case()
    with pytest.raises(RuntimeError, match='scientific settings'):
        validate_reuse(replace(cfg, client_learning_rate='9e-4'), doc, summary)


def placement_case():
    original = Config.load('tpu/swarm/ray_train/profiles/qcmp-qwen38-circuit-pwc05-s0-shared-v464-ram96-wd-20261007-pinned.json')
    original = replace(original, bootstrap_reuse_source_run_id='',
                       bootstrap_reuse_contract_sha256='', bootstrap_reuse_pool_sha256='')
    contract = {'config': original.to_dict()}
    digest = identity(contract)
    raw = original.to_dict()
    raw.update(run_id=original.run_id + '-resume', bootstrap_reuse_source_run_id=original.run_id,
               bootstrap_reuse_contract_sha256=digest, bootstrap_reuse_pool_sha256='a'*64,
               resume_min_checkpoint_step=2, science_placement_slots_per_host=32)
    raw['grading']['families']['placement']['slots_per_host'] = 32
    raw['client_env']['NUM_EPOCHS'] = '15'
    summary = dict(contract_sha256=digest, pool_sha256='a'*64, optimizer_steps=0,
                   drafted=1024, total=1024, retained=428)
    return raw, {'contract': contract, 'sha256': digest}, summary


def test_placement_continuation_allows_smaller_host_concurrency():
    raw, doc, summary = placement_case()
    validate_reuse(Config.from_dict(raw), doc, summary)


@pytest.mark.parametrize('field', ['cpus', 'memory_gib', 'runtime', 'helper'])
def test_placement_concurrency_does_not_allow_changed_case_budgets(field):
    raw, doc, summary = placement_case()
    if field in ('cpus', 'memory_gib'):
        raw['grading']['families']['placement'][field] = 2
    elif field == 'runtime':
        raw['science_placement_runtime'] = 'legacy'
    else:
        raw['client_env']['SCIENCE_PLACEMENT_HELPER'] = 'none'
    with pytest.raises((RuntimeError, ValueError)):
        validate_reuse(Config.from_dict(raw), doc, summary)
