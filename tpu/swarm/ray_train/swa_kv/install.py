"""Ship sliding-window KV (tpu-inference agent/swa-kv a4fb045bf) to farms that run the pinned base bundle.

Farms install tpu-inference from the base bundle's third_party/tpu-inference, not from the submodule, so the
three changed files travel as one exact patch (acfe1f056..a4fb045bf), applied with no fuzz like the native
thinking-budget patch. The patch is inert unless SKYRL_SWA_KV=1 is set for the engine.
"""
import hashlib
from pathlib import Path
import subprocess

PATCH = Path(__file__).with_name("swa_kv.patch")


def enabled(config):
    return (config.inference.engine_env or {}).get("SKYRL_SWA_KV") == "1"


def identity():
    return hashlib.sha256(PATCH.read_bytes()).hexdigest()


def install(source):
    core = Path(source) / "third_party/tpu-inference"
    command = ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(PATCH)]
    subprocess.run(command + ["--dry-run"], cwd=core, check=True, capture_output=True, text=True)
    subprocess.run(command, cwd=core, check=True, capture_output=True, text=True)
