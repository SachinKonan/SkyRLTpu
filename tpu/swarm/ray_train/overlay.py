"""Explicit source overlay for opt-in multi-LoRA builds, isolated by digest."""
import hashlib
import json
from pathlib import Path
import shutil

FILES = (
    "skyrl/backends/tunix_backend.py",
    "skyrl/backends/lora_init.py",
    "skyrl/tinker/loss_fns.py",
    "skyrl/tinker/types.py",
    "skyrl/tinker/api.py",
    "skyrl/tinker/engine.py",
    "tpu/run_ttd_ensemble.py",
    "tpu/vllm_tpu_server.py",
    "third_party/discover/ttt_discover/rl/ensemble.py",
    "third_party/discover/ttt_discover/rl/multi_lora.py",
    "third_party/discover/ttt_discover/rl/multi_lora_request.py",
    "third_party/discover/ttt_discover/tinker_utils/completers.py",
)


def manifest(repo):
    return {name: hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest() for name in FILES}


def identity(directory):
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        raise RuntimeError("multi-LoRA requires a built, pinned source overlay")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(directory, destination):
    directory, destination = Path(directory), Path(destination)
    records = json.loads((directory / "manifest.json").read_text())
    if set(records) != set(FILES):
        raise RuntimeError("unexpected source overlay file set")
    for name, expected in records.items():
        source = directory / name
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source overlay checksum mismatch: {name}")
    for name in records:
        target = destination / name
        if target.is_symlink():
            raise RuntimeError(f"refusing source overlay through symlink: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / name, target)
