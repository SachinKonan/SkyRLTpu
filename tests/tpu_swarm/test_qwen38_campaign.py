from pathlib import Path

import pytest

from tpu.swarm.ray_train.config import Config, PRESETS
from tpu.swarm.ray_train.overlay import manifest

PROFILES = sorted(Path('tpu/swarm/ray_train/profiles').glob('qwen38-v464-*.json'))
FARMS = sorted(Path('tpu/swarm/ray_train/profiles').glob('inference-farm-v6e8-*-qwen38-*.json'))


def test_matrix():
    assert len(PROFILES) == 12
    assert {p.name.split('-')[2] for p in PROFILES} == {'erdos', 'ac1', 'circuit', 'qubit'}
    assert PRESETS['qwen3.8-27b'].trainer == PRESETS['qwen3.5-27b'].trainer
    assert PRESETS['qwen3.8-27b'].maxtext_model == 'qwen3.5-27b'


@pytest.mark.parametrize('profile', PROFILES, ids=lambda p: p.stem)
def test_run_contract(profile):
    config = Config.load(profile)
    config.validate()
    assert config.model == 'Qwen/Qwen3.8-27B'
    assert config.native_thinking_format == 'qwen3.5-27b'
    from tpu.swarm.ray_train.thinking_budget.contract import check
    assert check(Path.cwd(), model=config.native_thinking_format)
    assert config.bootstrap_fixed_budget and config.bootstrap_require_full_pool
    assert (config.bootstrap_max_drafts, config.bootstrap_target_valid) == (1024, 512)
    assert config.bootstrap_seed in (0, 1, 2)
    assert config.client_env['TTD_M0_LORA_SEED'] == str(config.bootstrap_seed)
    assert config.client_env['TTD_EXPLICIT_LORA_SEED'] == '1'
    assert config.client_env['TTD_ADV_ESTIMATOR'] == 'piecewise_valid_entropic_centered_adaptive'
    assert config.client_env['TTD_ADV_PIECEWISE_RHO'] == '0.5'
    assert (config.client_env['GROUPS_PER_BATCH'], config.client_env['GROUP_SIZE']) == ('16', '32')
    assert config.client_env['NUM_EPOCHS'] == '10'
    assert not config.seed_pool_sha256 and not config.bootstrap_reuse_pool_sha256
    assert config.cache.orbax != 'gs://sk7524-tinker-tpu-us-central2/skyrl-maxtext-ckpts'
    for path in vars(config.cache).values():
        if isinstance(path, str) and path.startswith('gs://'):
            assert path.startswith('gs://sk7524-tinker-tpu-us-central2/')
    sources = manifest(Path.cwd(), config)
    assert 'third_party/discover/ttt_discover/rl/ensemble.py' in sources
    if config.has_math_environment:
        assert f"third_party/discover/examples/{config.client_env['TTD_ENV']}/env.py" in sources


@pytest.mark.parametrize('profile', FARMS, ids=lambda p: p.stem)
def test_regional_farm_contract(profile):
    config = Config.load(profile)
    assert config.model == 'Qwen/Qwen3.8-27B'
    assert config.inference_only and config.trainer.hosts == 0
    assert config.inference.tp == 4 and config.engines_per_host == 2
    assert config.native_thinking_format == 'qwen3.5-27b'
    assert config.inference.require_lease and config.inference.max_loras == 1
    assert config.inference.external_pool_attestation
    assert not config.inference.prefix_caching
    assert config.cache.hf_layout == 'snapshot'
    region = config.zone.rsplit('-', 1)[0]
    assert config.bucket == 'gs://sk7524-tinker-tpu-' + region
    assert config.cache.hf == config.bucket + '/hf-cache-qwen38-1d4bf0f2'
