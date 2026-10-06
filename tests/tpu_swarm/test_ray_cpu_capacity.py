"""Explicit Ray admission capacity must not alter candidate CPU budgets."""
from dataclasses import replace
from pathlib import Path

import pytest

from tpu.swarm.ray_train.commands import client_environment
from tpu.swarm.ray_train.config import Config


def test_capacity_override_preserves_candidate_budget():
    old = Config.load('tpu/swarm/ray_train/profiles/science-routing-v4-qwen-bootstrap-l2-001.json')
    new = replace(old, ray_cpu_capacity=64)
    new.validate()
    assert old.ray_cpus_per_host == 32
    assert Config.from_dict(new.to_dict()).ray_cpus_per_host == 64
    before = client_environment(old, Path('/runtime'), 'head')
    after = client_environment(new, Path('/runtime'), 'head')
    assert before == after
    assert replace(new, ray_cpu_capacity=0).ray_cpus_per_host == old.ray_cpus_per_host


@pytest.mark.parametrize('value', [-1, True, 64.5, '64'])
def test_capacity_rejects_invalid_values(value):
    old = Config.load('tpu/swarm/ray_train/profiles/science-routing-v4-qwen-bootstrap-l2-001.json')
    with pytest.raises(ValueError, match='ray_cpu_capacity'):
        replace(old, ray_cpu_capacity=value).validate()
