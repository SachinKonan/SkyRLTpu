"""Reject dependency drift before starting the training client."""
from importlib import metadata
from pathlib import Path
import re
import sys
import tomllib


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def verify_environment(lock_path):
    lock = tomllib.loads(Path(lock_path).read_text())
    expected = {canonical(p["name"]): p["version"] for p in lock["package"]
                if "registry" in p.get("source", {})}
    installed = {canonical(d.metadata["Name"]): d.version for d in metadata.distributions()}
    for name, version in installed.items():
        if name == "ttt-discover":
            continue
        if expected.get(name) != version:
            raise RuntimeError(f"client dependency drift: {name}=={version}; locked {expected.get(name)}")
    for name in ("tinker", "ray", "torch", "transformers"):
        if name not in installed:
            raise RuntimeError(f"missing locked client dependency: {name}")
    print("client dependency lock verified: " + ", ".join(
        f"{name}=={installed[name]}" for name in ("tinker", "ray", "torch", "transformers")))


if __name__ == "__main__":
    verify_environment(sys.argv[1])
