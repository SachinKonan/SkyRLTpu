from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
POOL_CONFIGS = sorted((REPO / "tpu/swarm/examples").glob("*pool.yaml"))
SWARM_CONFIGS = sorted((REPO / "tpu/swarm/examples").glob("*.yaml"))


@pytest.mark.parametrize("path", POOL_CONFIGS, ids=lambda path: path.name)
def test_pool_setup_retains_allocated_tpu_on_setup_failure(path: Path):
    config = yaml.safe_load(path.read_text())
    setup = config["setup"]

    assert "TPUSWARM_POOL_SETUP" in setup
    assert "TPUSWARM_SETUP_RETRY_SECONDS:-60" in setup
    assert "retaining the TPU and retrying" in setup
    assert 'if bash "$setup_script"; then' in setup

    syntax = subprocess.run(
        ["bash", "-n"], input=setup, text=True, capture_output=True
    )
    assert syntax.returncode == 0, syntax.stderr


@pytest.mark.parametrize("path", POOL_CONFIGS, ids=lambda path: path.name)
def test_pool_archive_contract_uses_interpreter_compatible_permissions(path: Path):
    config = yaml.safe_load(path.read_text())
    setup = config["setup"]

    assert 'test -x "$staging/' not in setup
    assert 'test -f "$staging/.tpuswarm-bundle-manifest"' in setup

    staged_paths = re.findall(r'test -[fr] "\$staging/([^"$]+)"', setup)
    assert staged_paths
    for staged_path in staged_paths:
        if staged_path == ".tpuswarm-bundle-manifest":
            continue
        assert (REPO / staged_path).is_file(), staged_path


def test_pool_configs_do_not_reference_retired_missing_bundle():
    for path in POOL_CONFIGS:
        config = yaml.safe_load(path.read_text())
        bundle_url = config["envs"]["TPUSWARM_SKYRL_BUNDLE_URL"]
        assert not bundle_url.endswith("/tpuswarm-skyrl-v1.tar.gz"), path.name


@pytest.mark.parametrize("path", SWARM_CONFIGS, ids=lambda path: path.name)
def test_archive_entrypoints_do_not_depend_on_execute_mode(path: Path):
    config_text = path.read_text()
    assert 'test -x "$staging/' not in config_text


def test_bundle_builder_validates_the_published_archive():
    builder = (REPO / "tpu/swarm/build_skyrl_bundle.sh").read_text()

    assert 'usage: $0 gs://BUCKET/code-bundles/BUNDLE.tar.gz' in builder
    assert 'tar -xzf "$ARCHIVE" -C "$VERIFY_ROOT"' in builder
    assert 'if [ ! -r "$VERIFY_ROOT/$required" ]' in builder
