"""Capture/compare installed environments without importing JAX or touching TPU.

Run with EACH role's Python, on both the working legacy host and its proposed
replacement. An installer recipe is not evidence of installed-package parity.
No credentials, environment dump, or pip configuration are collected.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
from urllib.parse import urlsplit, urlunsplit


def clean_url(url):
    parts = urlsplit(url)
    # urlunsplit canonicalizes SQLite's significant slash count. These local
    # database URLs contain no authority credentials: retain their path form.
    if parts.scheme == "sqlite" and not parts.netloc:
        return url.split("?", 1)[0].split("#", 1)[0]
    # Local editable paths differ between launchers and are covered by hashes.
    if parts.scheme == "file":
        return "file:<local-source>"
    host = parts.hostname or ""
    if parts.port:
        host += ":" + str(parts.port)
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def collect(source=None):
    packages = {}
    installed_sources = {}
    for dist in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        record = {"version": dist.version}
        raw = dist.read_text("direct_url.json")
        if raw:
            direct = json.loads(raw)
            record["origin"] = clean_url(direct["url"])
            if "vcs_info" in direct:
                record["commit"] = direct["vcs_info"].get("commit_id")
            if "archive_info" in direct:
                record["archive_hashes"] = direct["archive_info"].get("hashes", {})
        if name in packages and packages[name] != record:
            raise RuntimeError("duplicate installed distribution: " + name)
        packages[name] = record
        # Version metadata alone misses locally patched MaxText, inference and
        # SDK modules. Hash their Python sources without importing the packages.
        if name in {"maxtext", "tpu-inference", "tinker", "ttt-discover", "transformers", "tokenizers"}:
            for member in dist.files or ():
                if str(member).endswith(".py"):
                    path = Path(dist.locate_file(member))
                    if path.is_file():
                        installed_sources[name + "/" + str(member)] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {}
    if source:
        root = Path(source)
        for name in ("skyrl/backends/tunix_backend.py", "skyrl/tinker/loss_fns.py",
                     "tpu/vllm_tpu_server.py", "tpu/thinking_budget/server.py",
                     "third_party/discover/ttt_discover/tinker_utils/completers.py"):
            path = root / name
            if path.is_file():
                sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema": 1, "python": platform.python_version(), "machine": platform.machine(),
            "packages": dict(sorted(packages.items())), "sources": sources,
            "installed_sources": dict(sorted(installed_sources.items()))}


def differences(expected, actual):
    delta = {}
    for name in ("schema", "python", "machine", "packages", "sources", "installed_sources"):
        if expected.get(name) != actual.get(name):
            if name in ("packages", "sources", "installed_sources"):
                left, right = expected.get(name, {}), actual.get(name, {})
                delta[name] = {k: {"expected": left.get(k), "actual": right.get(k)}
                               for k in sorted(left.keys() | right.keys()) if left.get(k) != right.get(k)}
            else:
                delta[name] = {"expected": expected.get(name), "actual": actual.get(name)}
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--expect", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inventory = collect(args.source)
    if args.output:
        partial = args.output.with_suffix(".partial")
        partial.write_text(json.dumps(inventory, sort_keys=True, indent=2) + "\n")
        partial.replace(args.output)
    else:
        print(json.dumps(inventory, sort_keys=True, indent=2))
    if args.expect:
        delta = differences(json.loads(args.expect.read_text()), inventory)
        if delta:
            raise SystemExit("Runtime inventory differs: " + json.dumps(delta, sort_keys=True))


if __name__ == "__main__":
    main()
