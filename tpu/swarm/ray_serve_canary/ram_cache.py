"""Bounded, inference-only RAM cache; preserve prior disk caches on migration."""
import json
import os
from pathlib import Path
import subprocess

GIB = 1024**3
CACHE_NAMES = ("model", "model-downloads", "xla")


def check_budget(memory, cap, reserve):
    if cap < 64 * GIB or reserve < 128 * GIB:
        raise ValueError("require at least 64 GiB cache and 128 GiB runtime reserve")
    if memory["SwapTotal"]:
        raise RuntimeError("RAM-cache canary requires swap disabled")
    if memory["MemAvailable"] < cap + reserve:
        raise RuntimeError("insufficient available memory for cache plus runtime reserve")


def redirect_cache_dirs(root, mount):
    backup = root / "disk-cache-before-ram"
    backup.mkdir(exist_ok=True)
    for name in CACHE_NAMES:
        target = mount / name
        target.mkdir(exist_ok=True)
        path = root / name
        if path.is_symlink():
            if path.resolve() != target.resolve():
                raise RuntimeError(f"unexpected cache symlink: {path}")
            continue
        if path.exists():
            if (backup / name).exists():
                raise RuntimeError(f"refusing to overwrite preserved cache: {backup / name}")
            path.rename(backup / name)
        path.symlink_to(target, target_is_directory=True)
    # These markers live on disk, so cannot establish readiness of a new tmpfs.
    for name in ("model-ready", "xla-ready"):
        (root / name).unlink(missing_ok=True)


def setup(root):
    root = Path(root).resolve()
    cap = int(os.environ.get("CANARY_RAM_CACHE_GIB", "96")) * GIB
    reserve = int(os.environ.get("CANARY_RAM_RESERVE_GIB", "128")) * GIB
    mount = root / "ram-cache"
    if mount.is_symlink():
        raise RuntimeError("RAM mount path must not be a symlink")
    mount.mkdir(mode=0o700, exist_ok=True)
    if not os.path.ismount(mount):
        if any(mount.iterdir()):
            raise RuntimeError("refusing to hide existing files under a mount")
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines()}
        check_budget(memory, cap, reserve)
        options = f"size={cap},mode=0700,uid={os.getuid()},gid={os.getgid()},nosuid,nodev"
        subprocess.run(["sudo", "-n", "mount", "-t", "tmpfs", "-o", options,
                        "tmpfs", str(mount)], check=True)
    info = json.loads(subprocess.check_output([
        "findmnt", "--json", "--mountpoint", str(mount), "--output", "FSTYPE,TARGET"]))
    fs = info["filesystems"][0]
    actual = os.statvfs(mount)
    if fs["fstype"] != "tmpfs" or actual.f_blocks * actual.f_frsize > cap:
        raise RuntimeError("cache mount is not the expected bounded tmpfs")
    redirect_cache_dirs(root, mount)
    print(json.dumps({"event": "ram_cache_ready", "mount": str(mount),
                      "capacity_bytes": actual.f_blocks * actual.f_frsize}), flush=True)


if __name__ == "__main__":
    setup(os.environ["CANARY_ROOT"])
