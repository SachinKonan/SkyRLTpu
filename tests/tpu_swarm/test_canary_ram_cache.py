import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "tpu/swarm/ray_serve_canary"
spec = importlib.util.spec_from_file_location("canary_ram", ROOT / "ram_cache.py")
ram = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ram)


def test_budget():
    ram.check_budget({"MemAvailable": 394 * ram.GIB, "SwapTotal": 0},
                     96 * ram.GIB, 128 * ram.GIB)
    with pytest.raises(RuntimeError, match="insufficient"):
        ram.check_budget({"MemAvailable": 200 * ram.GIB, "SwapTotal": 0},
                         96 * ram.GIB, 128 * ram.GIB)
    with pytest.raises(RuntimeError, match="swap"):
        ram.check_budget({"MemAvailable": 394 * ram.GIB, "SwapTotal": ram.GIB},
                         96 * ram.GIB, 128 * ram.GIB)


def test_redirect_preserves_disk_cache_and_is_idempotent(tmp_path):
    mount = tmp_path / "ram-cache"
    mount.mkdir()
    for name in ram.CACHE_NAMES:
        (tmp_path / name).mkdir()
        (tmp_path / name / "old").write_bytes(b"old")
    (tmp_path / "model-ready").touch()
    (tmp_path / "xla-ready").touch()
    ram.redirect_cache_dirs(tmp_path, mount)
    ram.redirect_cache_dirs(tmp_path, mount)
    for name in ram.CACHE_NAMES:
        assert (tmp_path / name).resolve() == mount / name
        assert (tmp_path / "disk-cache-before-ram" / name / "old").read_bytes() == b"old"
    assert not (tmp_path / "model-ready").exists()
    assert not (tmp_path / "xla-ready").exists()


def test_refuses_unrelated_symlink(tmp_path):
    mount = tmp_path / "ram-cache"
    mount.mkdir()
    (tmp_path / "model").symlink_to(tmp_path / "unrelated")
    with pytest.raises(RuntimeError, match="unexpected"):
        ram.redirect_cache_dirs(tmp_path, mount)


def test_only_inference_role_mounts_ram():
    shell = (ROOT / "run.sh").read_text()
    infer = shell.split('elif [[ "$SKYPILOT_NODE_RANK" == 1')[1].split('\nfi\n')[0]
    assert '"$CANARY_ROOT/code/ram_cache.py"' in infer
