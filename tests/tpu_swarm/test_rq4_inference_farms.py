import json
from pathlib import Path
import re

import pytest

from tpu.swarm.ray_train.config import Config

PROFILES = Path('tpu/swarm/ray_train/profiles')
CASES = {
    'inference-farm-rq4-qwen-asia-v6e8-001': ('qwen3.5-27b', 'asia-northeast1-b', 'asia-northeast1', 4, 2, 32),
    'inference-farm-rq4-gemma-east5-v6e8-001': ('gemma4-31b', 'us-east5-b', 'us-east5', 8, 1, 128),
    'inference-farm-rq4-muse-east5-v6e8-001': ('muse-glimmer-30b', 'us-east5-b', 'us-east5', 8, 1, 128),
}


@pytest.mark.parametrize('run,expected', CASES.items())
def test_rq4_farm_contract_and_topology(run, expected):
    preset, zone, _, tp, engines, max_sequences = expected
    config = Config.load(PROFILES / f'{run}.json')
    assert config.run_id == run and config.model_preset == preset
    assert config.zone == zone and config.accelerator == 'tpu-v6e-8' and config.hosts == 1
    assert config.inference_only and config.trainer.hosts == 0
    assert config.inference.tp == tp and config.engines_per_host == engines
    assert config.inference.max_sequences == max_sequences
    assert config.inference.max_model_length == 22528
    assert config.inference.require_lease and config.inference.external_pool_attestation
    assert config.inference.max_loras == 1 and not config.inference.prefix_caching


@pytest.mark.parametrize('run,expected', CASES.items())
def test_rq4_farm_reads_only_its_own_region(run, expected):
    region = expected[2]
    raw = json.loads((PROFILES / f'{run}.json').read_text())
    paths = [raw['bucket'], raw['base_bundle']] + [
        value for value in raw['cache'].values()
        if isinstance(value, str) and value.startswith('gs://')]
    assert paths
    assert {re.match(r'gs://sk7524-tinker-tpu-([^/]+)', path).group(1) for path in paths} == {region}
