"""Restore completed flat JAX cache objects, excluding uploaded transfer debris."""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

from transfer import run_transfer


def cache_objects(listing, prefix):
    selected = []
    for entry in listing:
        if entry.get("type") != "cloud_object":
            continue
        meta = entry["metadata"]
        uri = "gs://" + meta["bucket"] + "/" + meta["name"]
        if not uri.startswith(prefix + "/"):
            raise ValueError("cache object outside expected prefix")
        name = uri[len(prefix) + 1:]
        if not name.endswith("-cache"):
            continue
        if PurePosixPath(name).name != name:
            raise ValueError("expected a flat compilation-cache prefix")
        selected.append((uri + "#" + str(meta["generation"]), name,
                         int(meta["size"]), meta["md5Hash"]))
    if not selected:
        raise ValueError("no completed compilation-cache objects found")
    return selected


def valid(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    md5 = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            md5.update(chunk)
    return base64.b64encode(md5.digest()).decode() == digest


def main():
    root = Path(os.environ["CANARY_ROOT"])
    prefix = os.environ["XLA_CACHE_GCS"].rstrip("/")
    cache = root / "xla"
    cache.mkdir(exist_ok=True)
    entries = cache_objects(json.loads(subprocess.check_output(
        ["gcloud", "storage", "ls", "--json", prefix + "/**"])), prefix)
    missing = [e for e in entries if not valid(cache / e[1], e[2], e[3])]
    print(json.dumps({"event": "xla_plan", "valid_reused": len(entries) - len(missing),
                      "missing": len(missing), "missing_bytes": sum(e[2] for e in missing)}), flush=True)
    for attempt in range(3):
        if not missing:
            return
        for _, name, _, _ in missing:
            (cache / name).unlink(missing_ok=True)
            (cache / (name + "_.gstmp")).unlink(missing_ok=True)
        run_transfer(["gcloud", "storage", "cp", "--read-paths-from-stdin", str(cache) + "/"],
                     root, cache, "xla", "".join(e[0] + "\n" for e in missing), check=False)
        missing = [e for e in missing if not valid(cache / e[1], e[2], e[3])]
    if missing:
        raise RuntimeError(f"compilation cache validation failed: {[e[1] for e in missing]}")


if __name__ == "__main__":
    main()
