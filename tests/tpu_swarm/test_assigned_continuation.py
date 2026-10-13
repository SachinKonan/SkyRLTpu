from copy import deepcopy
from dataclasses import replace

import pytest

from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.bootstrap_reuse import validate_assigned_migration


def pair():
    old = Config.load('tpu/swarm/ray_train/profiles/qcmp-qwen38-erdos-s0-resume1-v6e8-2farms-20261008.json')
    new = old.to_dict()
    import json
    new = json.loads(json.dumps(new).replace(old.bucket, 'gs://sk7524-tinker-tpu-us-central1'))
    new.update(accelerator='tpu-v5p-64', zone='us-central1-a', assigned_host_ips=['10.128.1.93'],
               physical_node='test-node', resume_min_checkpoint_step=10)
    new['trainer'].update(tp=1, fsdp=4, chip_bounds='2,2,1')
    new['cache']['reserve_gib'] = 192
    new['client_env']['NUM_EPOCHS'] = '15'
    return old, Config.from_dict(new)


def test_checkpoint_continuation_changes_only_placement_and_cap():
    old, new = pair()
    before = deepcopy(old.to_dict())
    validate_assigned_migration(old, new)
    assert old.to_dict() == before


@pytest.mark.parametrize('change', ['seed', 'loss', 'budget', 'weights', 'lr', 'root', 'run'])
def test_scientific_changes_rejected(change):
    old, new = pair()
    if change in ('seed', 'loss'):
        key = 'TTD_M0_LORA_SEED' if change == 'seed' else 'TTD_LOSS_FN'
        new = replace(new, client_env={**new.client_env, key: 'different'})
    elif change == 'budget':
        new = replace(new, bootstrap_max_drafts=2048)
    elif change == 'weights':
        new = replace(new, cache=replace(new.cache, hf=new.cache.hf + '-wrong'))
    elif change == 'lr':
        new = replace(new, client_learning_rate='9e-4')
    elif change == 'root':
        new = replace(new, root='~/.cache/different')
    else:
        new = replace(new, run_id='different')
    with pytest.raises((RuntimeError, ValueError)):
        validate_assigned_migration(old, new)


def test_step_extension_requires_checkpoint():
    old, new = pair()
    with pytest.raises(RuntimeError, match='saved checkpoint'):
        validate_assigned_migration(old, replace(new, resume_min_checkpoint_step=0))


def test_tp1_preserves_tp8_checkpoint_kv_shape(tmp_path):
    from tpu.swarm.ray_train.commands import maxtext_kwargs
    old, new = pair()
    new = replace(new, trainer=replace(new.trainer, maxtext_kwargs={
        **new.trainer.maxtext_kwargs, 'override_model_config': True, 'base_num_kv_heads': 8}))
    validate_assigned_migration(old, new)
    assert maxtext_kwargs(old, tmp_path)['base_num_kv_heads'] == maxtext_kwargs(new, tmp_path)['base_num_kv_heads'] == 8
    with pytest.raises(RuntimeError, match='scientific'):
        bad = replace(new, trainer=replace(new.trainer, maxtext_kwargs={
            **new.trainer.maxtext_kwargs, 'base_num_kv_heads': 16}))
        validate_assigned_migration(old, bad)


def test_two_host_tp4_resume_preserves_kv_and_scientific_recipe(tmp_path):
    from tpu.swarm.ray_train.commands import maxtext_kwargs
    old, single = pair()
    single = replace(single, trainer=replace(single.trainer, maxtext_kwargs={
        **single.trainer.maxtext_kwargs, 'override_model_config': True, 'base_num_kv_heads': 8}))
    dual = replace(single, hosts=2, assigned_host_ips=('10.128.1.93', '10.128.1.94'),
                   trainer=replace(single.trainer, hosts=2, tp=4, fsdp=2,
                                   process_bounds='1,1,2'))
    validate_assigned_migration(old, dual)
    validate_assigned_migration(single, dual)
    assert maxtext_kwargs(old, tmp_path)['base_num_kv_heads'] == maxtext_kwargs(dual, tmp_path)['base_num_kv_heads'] == 8
    with pytest.raises(RuntimeError, match='scientific'):
        validate_assigned_migration(single, replace(dual, client_learning_rate='1e-2'))
    with pytest.raises(RuntimeError, match='scientific'):
        validate_assigned_migration(single, replace(dual, trainer=replace(dual.trainer, maxtext_kwargs={
            **dual.trainer.maxtext_kwargs, 'base_num_kv_heads': 16})))
