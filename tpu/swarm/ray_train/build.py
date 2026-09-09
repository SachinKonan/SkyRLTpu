"""Package only this implementation and generate a pinned SkyPilot task."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
import tempfile

import yaml

from .config import ACCELERATOR_RUNTIME, Config


def build(profile, output):
    config = Config.load(profile)
    package = Path(__file__).resolve().parent
    repo = package.parents[2]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "ray-training.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and (
                path.suffix in (".py", ".json", ".md")
                or path.parent == package / "client_env" and path.name in ("pyproject.toml", "uv.lock")
            ):
                bundle.add(path, arcname=str(path.relative_to(repo)), recursive=False)
        for name in ("select_v4_64_topology.py", "select_v6e_32_topology.py", "select_v5p_32_topology.py"):
            selector = repo / "tpu/swarm" / name
            bundle.add(selector, arcname=str(selector.relative_to(repo)))
    with archive.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
    uri = config.bucket.rstrip("/") + "/code-bundles/ray-training-" + digest + ".tar.gz"
    source = "tpu/swarm/ray_train/profiles/" + Path(profile).name
    if Path(profile).resolve().parent != package / "profiles":
        raise ValueError("profile must be in this package's profiles directory")
    is_v4 = config.accelerator.startswith("tpu-v4-")
    zone = config.effective_zone
    runtime = ACCELERATOR_RUNTIME[config.accelerator]
    # Each archive is immutable and has no shared-tree credentials or unrelated
    # agent edits. Existing training code comes from the SHA-pinned base bundle.
    task = dict(name=config.run_id, resources=dict(cloud="gcp", zone=zone,
        accelerators=config.accelerator, accelerator_args=dict(gcp_queued_resource=True, runtime_version=runtime),
        use_spot=True, disk_size=300 if is_v4 else 150,
        job_recovery=dict(strategy="FAILOVER", max_restarts_on_errors=0)),
        envs=dict(RAY_TRAIN_CODE=uri, RAY_TRAIN_CODE_SHA256=digest),
        run=f'''set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
code="$HOME/.cache/skyrl-ray-code/$RAY_TRAIN_CODE_SHA256"
if [ ! -f "$code/.complete" ]; then
  mkdir -p "$code"
  gcloud storage cp "$RAY_TRAIN_CODE" "$code/code.tar.gz"
  printf '%s  %s\\n' "$RAY_TRAIN_CODE_SHA256" "$code/code.tar.gz" | sha256sum -c -
  tar -xzf "$code/code.tar.gz" -C "$code"
  touch "$code/.complete"
fi
export PYTHONPATH="$code${{PYTHONPATH:+:$PYTHONPATH}}"
exec python3 -m tpu.swarm.ray_train.bootstrap "$code/{source}"
''')
    path = output / (config.run_id + ".yaml")
    path.write_text(yaml.safe_dump(task, sort_keys=False))
    return archive, uri, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--output", required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    archive, uri, task = build(args.profile, args.output)
    if args.upload:
        subprocess.run(["gcloud", "storage", "cp", "--no-clobber", str(archive), uri], check=True)
    print(f"archive={archive}\nuri={uri}\ntask={task}")


if __name__ == "__main__":
    main()
