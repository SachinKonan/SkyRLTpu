from dataclasses import replace
import hashlib
import json

import pytest

from tpu.swarm.ray_train.bootstrap_reuse import identity
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.host import Host
from tpu.swarm.ray_train import seed_bootstrap


@pytest.fixture
def saved(tmp_path):
    original = Config.load('tpu/swarm/ray_train/profiles/fresh-v6e-gemma-rglru-grpo-lr4e5-s1-20260919-fix1.json')
    original = replace(original, checkpoint_resume=True)
    contract = dict(config=original.to_dict(), implementation_sha256='old-implementation')
    pool = {'seeds': [{'id': 'seed-a', 'value': 0.5}]}
    cfg = replace(original, bootstrap_reuse_contract_sha256=identity(contract),
                  bootstrap_reuse_pool_sha256=identity(pool),
                  client_env={**original.client_env, 'NUM_EPOCHS': '10'},
                  inference=replace(original.inference, external_pool_updates=True,
                                    external_pool_lease_scope='run', external_pool_scheduler=True),
                  cache=replace(original.cache, trainer_compile='gs://test/new-trainer'))
    cfg.validate()
    host = Host.__new__(Host)
    host.config, host.run, host.log = cfg, tmp_path, tmp_path/'host.jsonl'
    folder = tmp_path/'client/bootstrap'
    folder.mkdir(parents=True)
    document = {'contract': contract, 'sha256': identity(contract)}
    summary = dict(contract_sha256=identity(contract), pool_sha256=identity(pool), retained=1, optimizer_steps=0)
    (folder/'contract.json').write_text(json.dumps(document))
    (folder/'complete.json').write_text(json.dumps(summary))
    snapshot = tmp_path/'client/tinker_log'/cfg.run_id/'puct_sampler_step_000000.json'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(pool))
    return host, folder, snapshot, original


def test_explicit_reuse_preserves_completed_pool_and_provenance(saved):
    host, folder, snapshot, _ = saved
    before = {p: p.read_bytes() for p in [folder/'contract.json', folder/'complete.json', snapshot]}
    assert host.bootstrap_status()['retained'] == 1
    assert all(p.read_bytes() == content for p, content in before.items())
    assert json.loads(host.log.read_text())['event'] == 'bootstrap_reuse_verified'


@pytest.mark.parametrize('damage', ['contract', 'document_hash', 'summary_contract', 'summary_pool',
                                  'pool', 'missing_pool', 'missing_complete', 'no_seeds', 'optimizer_steps'])
def test_reuse_rejects_corrupt_or_incomplete_state(saved, damage):
    host, folder, snapshot, _ = saved
    if damage == 'missing_complete':
        (folder/'complete.json').unlink()
    elif damage == 'missing_pool':
        snapshot.unlink()
    elif damage == 'pool':
        snapshot.write_text('{}')
    elif damage in ('contract', 'document_hash'):
        p = folder/'contract.json'; d = json.loads(p.read_text())
        if damage == 'contract': d['contract']['implementation_sha256'] = 'modified'
        else: d['sha256'] = 'f'*64
        p.write_text(json.dumps(d))
    else:
        p = folder/'complete.json'; d = json.loads(p.read_text())
        key, value = {'summary_contract': ('contract_sha256', 'f'*64),
                      'summary_pool': ('pool_sha256', 'f'*64), 'no_seeds': ('retained', 0),
                      'optimizer_steps': ('optimizer_steps', 1)}[damage]
        d[key] = value; p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError): host.bootstrap_status()


@pytest.mark.parametrize('change', ['run_id', 'temperature', 'loss', 'bootstrap_limit', 'context', 'base_bundle'])
def test_reuse_rejects_unreviewed_semantic_changes(saved, change):
    host, _, __, ___ = saved
    cfg = host.config
    if change == 'run_id': cfg = replace(cfg, run_id='different-run')
    elif change in ('temperature', 'loss'):
        cfg = replace(cfg, client_env={**cfg.client_env, change.upper(): 'changed'})
    elif change == 'bootstrap_limit': cfg = replace(cfg, bootstrap_max_drafts=2048)
    elif change == 'context': cfg = replace(cfg, inference=replace(cfg.inference, max_model_length=10240))
    else: cfg = replace(cfg, base_bundle_sha256='f'*64)
    host.config = cfg
    with pytest.raises(RuntimeError, match='outside the allowed'): host.bootstrap_status()


def test_unpinned_config_and_implementation_changes_still_fail(saved):
    host, folder, _, original = saved
    host.config = replace(host.config, bootstrap_reuse_contract_sha256='', bootstrap_reuse_pool_sha256='')
    with pytest.raises(RuntimeError, match='bootstrap config changed'): host.bootstrap_status()
    host.config = original
    with pytest.raises(RuntimeError, match='bootstrap implementation changed'): host.bootstrap_status()
    p = folder/'contract.json'; d = json.loads(p.read_text())
    d['contract']['implementation_sha256'] = hashlib.sha256(open(seed_bootstrap.__file__, 'rb').read()).hexdigest()
    p.write_text(json.dumps(d))
    assert host.bootstrap_status()['retained'] == 1


@pytest.mark.parametrize('changes', [dict(bootstrap_reuse_pool_sha256=''),
    dict(bootstrap_reuse_contract_sha256='invalid'), dict(checkpoint_resume=False),
    dict(bootstrap_layers=0), dict(bootstrap_only=True)])
def test_reuse_configuration_requires_complete_pins_and_resume(saved, changes):
    host, *_ = saved
    with pytest.raises(ValueError, match='bootstrap reuse'):
        replace(host.config, **changes).validate()
