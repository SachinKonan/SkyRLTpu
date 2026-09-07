import os
import shutil
import subprocess
from types import SimpleNamespace
import tarfile
import tomllib

import pytest

from tpu.swarm.ray_train import client_dependencies, host
from tpu.swarm.ray_train.build import build


def fake_host(tmp_path, monkeypatch):
    project = tmp_path / "client_env"
    project.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).write_bytes((host.CLIENT_ENV / name).read_bytes())
    monkeypatch.setattr(host, "CLIENT_ENV", project)
    calls = []
    root = tmp_path / "runtime"
    marker = root / "envs/client/.complete"

    def checked(name, command, **kwargs):
        calls.append((name, command, kwargs))
        marker.parent.mkdir(parents=True, exist_ok=True)

    obj = SimpleNamespace(root=root, source=tmp_path / "source",
                          config=SimpleNamespace(base_bundle_sha256="frozen-source"), checked=checked)
    return obj, calls, marker, project


def test_client_uses_frozen_lock_and_no_dependency_resolution(tmp_path, monkeypatch):
    obj, calls, marker, project = fake_host(tmp_path, monkeypatch)
    marker.parent.mkdir(parents=True)
    marker.write_text(obj.config.base_bundle_sha256)
    host.Host.install_client(obj)
    commands = {name: (command, kwargs) for name, command, kwargs in calls}
    sync, options = commands["client-sync"]
    assert "--frozen" in sync and str(project) in sync
    assert options["env"]["UV_PROJECT_ENVIRONMENT"] == str(marker.parent)
    install = commands["client-install"][0]
    assert "--no-deps" in install and "--no-build-isolation" in install
    assert not any("[math]" in arg for arg in install)
    assert calls[-1][0] == "client-import"
    assert marker.read_text() != obj.config.base_bundle_sha256
    identity = marker.read_text()
    calls.clear()
    host.Host.install_client(obj)
    assert [c[0] for c in calls] == ["client-lock-check"]
    (project / "uv.lock").write_text((project / "uv.lock").read_text() + "\n# revised lock\n")
    calls.clear()
    host.Host.install_client(obj)
    assert "client-sync" in [c[0] for c in calls]
    assert marker.read_text() != identity


def test_failed_client_install_never_marks_complete(tmp_path, monkeypatch):
    obj, calls, marker, _ = fake_host(tmp_path, monkeypatch)
    original = obj.checked

    def fail(name, command, **kwargs):
        original(name, command, **kwargs)
        if name == "client-lock-check":
            raise RuntimeError("dependency drift")

    obj.checked = fail
    with pytest.raises(RuntimeError, match="drift"):
        host.Host.install_client(obj)
    assert not marker.exists()


def test_missing_lock_does_not_destroy_existing_environment(tmp_path, monkeypatch):
    obj, _, marker, project = fake_host(tmp_path, monkeypatch)
    marker.parent.mkdir(parents=True)
    marker.write_text("old")
    (project / "uv.lock").unlink()
    with pytest.raises(FileNotFoundError):
        host.Host.install_client(obj)
    assert marker.read_text() == "old"


@pytest.mark.parametrize("change", [None, ("tinker", "0.27.1"), ("unlocked-package", "1.0")])
def test_dependency_verifier_rejects_drift(monkeypatch, change):
    lock = tomllib.loads((host.CLIENT_ENV / "uv.lock").read_text())
    versions = {p["name"]: p["version"] for p in lock["package"] if "registry" in p.get("source", {})}
    if change:
        versions[change[0]] = change[1]
    monkeypatch.setattr(client_dependencies.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": name}, version=version) for name, version in versions.items()])
    if change:
        with pytest.raises(RuntimeError, match="dependency drift"):
            client_dependencies.verify_environment(host.CLIENT_ENV / "uv.lock")
    else:
        client_dependencies.verify_environment(host.CLIENT_ENV / "uv.lock")


def test_lock_preserves_client_contract_and_cpu_torch():
    lock = tomllib.loads((host.CLIENT_ENV / "uv.lock").read_text())
    packages = {p["name"]: p for p in lock["package"]}
    assert packages["tinker"]["version"] == "0.22.7"
    assert packages["ray"]["version"] == "2.58.0"
    assert packages["transformers"]["version"] == "5.8.0"
    assert packages["torch"]["version"] == "2.10.0+cpu"
    assert packages["torch"]["source"]["registry"] == "https://download.pytorch.org/whl/cpu"
    assert not any(name.startswith("nvidia-") for name in packages)


def test_bundle_includes_client_lock(tmp_path):
    profile = host.CLIENT_ENV.parent / "profiles/qwen_v5p_32.json"
    archive, _, _ = build(profile, tmp_path)
    with tarfile.open(archive) as bundle:
        for name in ("pyproject.toml", "uv.lock"):
            member = bundle.extractfile("tpu/swarm/ray_train/client_env/" + name)
            assert member.read() == (host.CLIENT_ENV / name).read_bytes()


@pytest.mark.skipif(os.environ.get("RAY_TRAIN_CLIENT_INSTALL_SMOKE") != "1",
                    reason="opt-in CPU install smoke with predownloaded uv cache")
def test_real_frozen_client_install(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_OFFLINE", "1")
    source = tmp_path / "source"
    discover = host.CLIENT_ENV.parents[3] / "third_party/discover"
    shutil.copytree(discover, source / "third_party/discover", ignore=shutil.ignore_patterns(
        ".git", ".pytest_cache", "__pycache__", "*.egg-info", "results", "tests", "docs"))

    def checked(name, command, **kwargs):
        subprocess.run(command, env=kwargs.get("env"), cwd=tmp_path, check=True)

    obj = SimpleNamespace(root=tmp_path / "runtime", source=source,
                          config=SimpleNamespace(base_bundle_sha256="smoke-source"), checked=checked)
    host.Host.install_client(obj)
    host.Host.install_client(obj)
    assert (obj.root / "envs/client/.complete").exists()
