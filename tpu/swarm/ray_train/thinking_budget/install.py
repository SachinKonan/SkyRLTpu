"""Apply a focused native-budget patch to an isolated, pinned serving source."""
import hashlib
from pathlib import Path
import shutil
import subprocess

ASSETS = ("runner.patch", "thinking_budget.py", "thinking_budget_api.py",
          "thinking_budget_device.py", "server.py", "install.py", "contract.py")


def identity():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ASSETS:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return digest.hexdigest()


def install(source, model=None, require_client=True):
    source, assets = Path(source), Path(__file__).parent
    from .contract import check
    check(source, model=model, require_client=require_client)
    core = source / "third_party/tpu-inference"
    # No fuzzy application, replacement of the full core, or shared checkout edits.
    command = ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(assets / "runner.patch")]
    subprocess.run(command + ["--dry-run"], cwd=core, check=True, capture_output=True, text=True)
    subprocess.run(command, cwd=core, check=True, capture_output=True, text=True)
    for name in ("thinking_budget.py", "thinking_budget_api.py", "thinking_budget_device.py"):
        shutil.copyfile(assets / name, core / "tpu_inference/runner" / name)
    wrapper = source / "tpu/thinking_budget"
    wrapper.mkdir(exist_ok=True)
    shutil.copyfile(assets / "server.py", wrapper / "server.py")
