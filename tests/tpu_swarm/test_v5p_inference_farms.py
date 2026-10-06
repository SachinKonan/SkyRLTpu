from pathlib import Path

import pytest

from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.borrowing_service import unit


@pytest.mark.parametrize('model', ['qwen', 'gemma', 'muse'])
def test_v5p_farm_roles_and_isolated_cache(model):
    cfg = Config.load(Path('tpu/swarm/ray_train/profiles') /
                      f'inference-farm-v5p32-{model}-20260921.json')
    assert cfg.accelerator == 'tpu-v5p-32' and cfg.effective_zone == 'us-east5-a'
    assert cfg.inference_only and cfg.trainer.hosts == 0
    assert cfg.inference_only_ranks == [0, 1, 2, 3] and cfg.inference_hosts == 4
    assert cfg.inference.tp == 4 and cfg.inference.hosts_per_engine == 1
    assert cfg.inference.max_sequences == 16 and cfg.inference.max_model_length == 22528
    assert cfg.inference.require_lease and cfg.inference.external_pool_attestation
    assert cfg.inference.native_thinking_budget and cfg.inference.max_loras == 1
    assert cfg.systemd_runtime
    assert cfg.run_id in cfg.cache.inference_compile
    # Model weights are portable; Gemma's established manifest lives in central2.
    assert 'us-east5' in cfg.cache.inference_compile
    assert 'us-central2' not in cfg.cache.inference_compile_seed


def test_service_supports_name_only_and_legacy_pool():
    args = ('/repo', '/venv/python', '/config/env', '/ssh', None, ['train'], '/state/lock')
    rendered = unit(*args)
    assert '"--farm-name-contains" "inference-farm"' in rendered
    assert '--farm-pool' not in rendered
    assert '--run-scoped-only' in rendered
    rendered = unit(*args[:4], 'legacy-v4', *args[5:])
    assert '"--farm-pool" "legacy-v4"' in rendered
    with pytest.raises(ValueError):
        unit(*args, farm_name_contains='')
