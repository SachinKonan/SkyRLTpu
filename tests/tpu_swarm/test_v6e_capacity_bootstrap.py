"""v6e pool readiness must not depend on model caches or engine startup."""

import os
from pathlib import Path
import subprocess
import tarfile

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
CONFIGS = [REPO / 'tpu/swarm/examples' / name for name in (
    'v6e32-qwen35-grpo-pool.yaml',
    'v6e32-qwen35-grpo-single-pool.yaml',
)]


@pytest.mark.parametrize('path', CONFIGS, ids=lambda path: path.name)
def test_capacity_bootstrap_is_independent_of_model_services(path, tmp_path):
    config = yaml.safe_load(path.read_text())
    setup = config['setup']
    for forbidden in ('HF_CACHE_COMPLETE', 'ensure_orbax_ckpt.sh',
                      'bash "$root/tpu/swarm/prepare_qwen35_v6e32.sh"',
                      'bash "$root/tpu/jobman/cell_worker.sh"', 'ray stop'):
        assert forbidden not in setup

    home = tmp_path / 'home'
    home.mkdir()
    commands = tmp_path / 'bin'
    commands.mkdir()
    bundle = tmp_path / 'bundle'
    for relative in ('tpu/swarm/run_qwen35_v6e32_grpo.sh',
                     'tpu/swarm/prepare_qwen35_v6e32.sh',
                     'tpu/jobman/cell_worker.sh',
                     'third_party/TPUSwarm/pyproject.toml',
                     '.tpuswarm-bundle-manifest'):
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('exit 97\n')
    archive = tmp_path / 'bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(bundle, arcname='.')

    for name in ('git', 'tmux', 'g++', 'uv', 'sleep'):
        target = commands / name
        target.write_text('#!/bin/bash\nexit 0\n')
        target.chmod(0o755)
    gcloud = commands / 'gcloud'
    gcloud.write_text('''#!/bin/bash
set -eu
if [[ "$*" == "storage objects describe gs://test/code.tar.gz --format=value(generation)" ]]; then
  echo 123
elif [[ "$1 $2" == "storage cp" && "$3" == "gs://test/code.tar.gz" ]]; then
  echo copy >> "$TEST_TRANSFERS"
  cp "$TEST_ARCHIVE" "$4"
  # First transfer reports failure despite leaving bytes: setup must retry.
  [[ $(wc -l < "$TEST_TRANSFERS") -gt 1 ]]
else
  echo "Unexpected cloud/cache operation: $*" >&2
  exit 98
fi
''')
    gcloud.chmod(0o755)
    env = {
        **os.environ,
        'HOME': str(home), 'PATH': f'{commands}:{os.environ["PATH"]}',
        'TPUSWARM_SKYRL_BUNDLE_URL': 'gs://test/code.tar.gz',
        'SKYPILOT_SETUP_NODE_RANK': '0',
        'SKYPILOT_SETUP_NODE_IPS': '\n'.join(f'10.0.0.{n}' for n in range(8)),
        'TEST_ARCHIVE': str(archive),
        'TEST_TRANSFERS': str(tmp_path / 'transfers'),
    }
    # Keep the transient shell script inside the isolated test directory.
    setup = setup.replace('/tmp/tpuswarm-pool-setup.',
                          str(tmp_path / 'tpuswarm-pool-setup.'))
    for rank in ('0', '7'):
        result = subprocess.run(['bash', '-c', setup], env={**env,
                                'SKYPILOT_SETUP_NODE_RANK': rank},
                                text=True, capture_output=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f'pool rank {rank} ready' in result.stdout
    assert (home / 'SkyRLTpu-tpuswarm').is_symlink()
    # First attempt fails, second succeeds, then the complete bundle is reused.
    assert (tmp_path / 'transfers').read_text().splitlines() == ['copy', 'copy']
    assert not list((home / '.cache/tpuswarm/bundles').glob('.extract.*'))
