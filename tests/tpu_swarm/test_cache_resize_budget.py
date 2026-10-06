from pathlib import Path
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train import cache


@pytest.mark.parametrize('used,available,allowed', [
    (52, 329, True),   # Occupied 64-GiB cache can safely grow to 128 GiB.
    (52, 315, False),  # Filling it would breach the 240-GiB reserve.
    (4, 320, False),   # Empty capacity is not already allocated memory.
])
def test_resize_accounts_for_occupied_pages(tmp_path, monkeypatch, used, available, allowed):
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == '/proc/meminfo':
            return f'MemAvailable: {available * 1024**2} kB\nSwapTotal: 0 kB\n'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(cache.os.path, 'ismount', lambda _: True)
    monkeypatch.setattr(cache.subprocess, 'check_output',
                        lambda *a, **k: b'{"filesystems":[{"fstype":"tmpfs"}]}')
    capacity = [64]
    monkeypatch.setattr(cache.os, 'statvfs', lambda _: SimpleNamespace(
        f_blocks=capacity[0], f_bfree=capacity[0] - used, f_frsize=cache.GIB))
    def resize(args, **kwargs):
        assert 'remount,size=128G' in args
        capacity[0] = 128
    monkeypatch.setattr(cache.subprocess, 'run', resize)
    if allowed:
        assert cache.mount_cache(tmp_path, 128, 240) == tmp_path
        assert capacity[0] == 128
    else:
        with pytest.raises(RuntimeError, match='not enough available memory to grow'):
            cache.mount_cache(tmp_path, 128, 240)
        assert capacity[0] == 64
