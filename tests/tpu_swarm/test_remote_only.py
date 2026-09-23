"""Remote-only trainer: every host trains, every completion comes from leased farms."""
import pytest

from tpu.swarm.ray_train.config import Config

PROFILE = 'tpu/swarm/ray_train/profiles/remote-only-v432-qwen-ac2-pilot-20260922.json'


def remote_only_config(tmp_path, **inference):
    raw = Config.load(PROFILE).to_dict()
    raw['root'] = str(tmp_path)
    raw['inference'].update(inference)
    return Config.from_dict(raw)


def test_remote_only_profile_contract(tmp_path):
    config = remote_only_config(tmp_path)
    assert config.inference.remote_only
    assert config.trainer.hosts == config.hosts == 4
    assert config.trainer.tp * config.trainer.fsdp == 16
    assert config.inference_hosts == 0
    assert config.engine_slots([]) == []
    assert config.borrows_inference
    assert config.inference.external_pool_lease_scope == 'run'
    assert config.inference.external_pool_target_leases == 2
    assert config.inference.external_pool_candidate_limit == 16
    assert config.inference.external_pool_required_compatibility == []
    assert config.client_env['NUM_EPOCHS'] == '1'
    assert not config.bootstrap_layers and not config.bootstrap_max_drafts


def test_candidate_limit_raises_the_two_url_cap(tmp_path):
    urls = [f'http://farm{i}' for i in range(5)]
    config = remote_only_config(tmp_path, external_pool_urls={'Qwen/Qwen3.5-27B': urls})
    assert config.inference.external_pool_urls['Qwen/Qwen3.5-27B'] == urls
    with pytest.raises(ValueError, match='at most 2'):
        remote_only_config(tmp_path, external_pool_candidate_limit=2, external_pool_target_leases=1,
                           external_pool_urls={'Qwen/Qwen3.5-27B': urls[:3]})


@pytest.mark.parametrize('change', [
    dict(external_pool_lease_scope='phase', external_pool_require_initial=False),
    dict(external_pool_updates=False),
    dict(external_pool_scheduler=False),
    dict(external_pool_attestation=False),
    dict(external_pool_target_leases=32),
    dict(external_pool_candidate_limit=0),
    dict(external_pool_required_compatibility=['not-a-digest']),
    dict(external_pool_required_compatibility=['a' * 64, 'a' * 64]),
    dict(request_timeout=7200),
    dict(farm_cancel_grace_seconds=120),
    dict(remote_only='yes'),
])
def test_reject_invalid_remote_only_inference_settings(tmp_path, change):
    with pytest.raises(ValueError):
        remote_only_config(tmp_path, **change)


@pytest.mark.parametrize('change', [
    {'trainer': {'hosts': 2, 'tp': 4, 'fsdp': 2, 'process_bounds': '1,1,2'}},
    {'bootstrap_layers': 1, 'bootstrap_all_hosts': True},
    {'bootstrap_max_drafts': 64},
    {'trainer_env': {'SKYRL_EXTERNAL_WATCHDOG_INFLIGHT_SEC': '3600'}},
    {'client_env': {'TTD_SAMPLING_PROGRESS_TIMEOUT': '900'}},
    {'inference_only': True, 'inference_only_ranks': [0, 1, 2, 3], 'trainer': {'hosts': 0}},
])
def test_reject_invalid_remote_only_topology(tmp_path, change):
    raw = Config.load(PROFILE).to_dict()
    raw['root'] = str(tmp_path)
    for key, value in change.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = dict(raw[key], **value)
        else:
            raw[key] = value
    with pytest.raises(ValueError):
        Config.from_dict(raw)


def test_existing_profiles_keep_single_lease_defaults(tmp_path):
    raw = Config.load('tpu/swarm/ray_train/profiles/science-q20-v4-qwen-grpo-clean-20260918.json').to_dict()
    raw['root'] = str(tmp_path)
    config = Config.from_dict(raw)
    assert not config.inference.remote_only
    assert config.inference.external_pool_target_leases == 1
    assert config.inference.external_pool_candidate_limit == 2
    raw['inference']['external_pool_urls'] = {raw['model']: ['http://a', 'http://b', 'http://c']}
    with pytest.raises(ValueError, match='at most 2'):
        Config.from_dict(raw)
