"""Prepare single-adapter Gemma and Muse farm replacements.

The replacements preserve the already deployed, attested serving bundle and
change only the embedded profile.  This avoids pulling unrelated worktree
changes into a live inference service.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tpu.swarm.ray_train.config import Config


HERE = Path(__file__).resolve().parent
PROFILES = ROOT / "tpu/swarm/ray_train/profiles"


JOBS = (
    {
        "model": "gemma",
        "source_archive": ROOT / ".science/reallocation-10step/packages/farm10-gemma-2-20260921-10step-deployment/ray-training.tar.gz",
        "source_task": ROOT / ".science/reallocation-10step/packages/farm10-gemma-2-20260921-10step-deployment/farm10-gemma-2-20260921.yaml",
        "source_profile": PROFILES / "farm10-gemma-2-20260921-10step-deployment.json",
        "profile": PROFILES / "farm-parity-gemma-prefix-20260922.json",
        "run_id": "farm-parity-gemma-prefix-20260922",
    },
    {
        "model": "muse",
        "source_archive": ROOT / ".science/muse-farm2-prefix16-20260922/ray-training.tar.gz",
        "source_task": ROOT / ".science/muse-farm2-prefix16-20260922/inference-farm-v4-32-muse-2-prefix16-20260922.yaml",
        "source_profile": PROFILES / "farm10-muse-2-20260922-10step-deployment.json",
        "profile": PROFILES / "farm-parity-muse-rpa-seq16-20260922.json",
        "run_id": "farm-parity-muse-rpa-seq16-20260922",
    },
)


def updated_profile(job: dict) -> bytes:
    profile = json.loads(job["source_profile"].read_text())
    profile["run_id"] = job["run_id"]
    profile["root"] = "~/.cache/" + job["run_id"]
    profile["inference"]["max_loras"] = 1
    if job["model"] == "gemma":
        profile["inference"]["prefix_caching"] = True
    else:
        profile["inference"].update(
            max_sequences=16,
            memory_utilization=0.7,
            prefix_caching=True,
            batched_rpa_kernel=False,
        )
    for phase in ("trainer", "inference"):
        profile["cache"][phase + "_compile"] = (
            profile["bucket"].rstrip("/")
            + "/farm-parity-20260922/"
            + job["run_id"]
            + "/"
            + phase
        )
    job["profile"].write_text(json.dumps(profile, indent=2) + "\n")
    Config.load(job["profile"])
    return job["profile"].read_bytes()


def patch_archive(job: dict, profile_data: bytes, output: Path) -> str:
    embedded = "tpu/swarm/ray_train/profiles/" + job["source_profile"].name
    changed = []
    with tarfile.open(job["source_archive"], "r:gz") as old, tarfile.open(output, "w:gz") as new:
        for member in old:
            data = old.extractfile(member).read() if member.isfile() else None
            if member.name == embedded:
                data = profile_data
                member = copy.copy(member)
                member.size = len(data)
                changed.append(member.name)
            new.addfile(member, io.BytesIO(data) if data is not None else None)
    if changed != [embedded]:
        raise RuntimeError(f"expected one embedded profile, changed={changed}")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main() -> None:
    manifests = []
    for job in JOBS:
        folder = HERE / job["run_id"]
        folder.mkdir(parents=True, exist_ok=True)
        profile_data = updated_profile(job)
        archive = folder / "ray-training.tar.gz"
        digest = patch_archive(job, profile_data, archive)
        profile = json.loads(profile_data)
        uri = profile["bucket"].rstrip("/") + "/code-bundles/ray-training-" + digest + ".tar.gz"
        task = yaml.safe_load(job["source_task"].read_text())
        task["name"] = job["run_id"]
        task["envs"]["RAY_TRAIN_CODE"] = uri
        task["envs"]["RAY_TRAIN_CODE_SHA256"] = digest
        task_path = folder / (job["run_id"] + ".yaml")
        task_path.write_text(yaml.safe_dump(task, sort_keys=False))
        manifests.append(
            {
                "model": job["model"],
                "run_id": job["run_id"],
                "profile": str(job["profile"].relative_to(ROOT)),
                "archive": str(archive.relative_to(ROOT)),
                "archive_sha256": digest,
                "code_uri": uri,
                "task": str(task_path.relative_to(ROOT)),
                "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            }
        )
    (HERE / "prepared.json").write_text(json.dumps({"jobs": manifests}, indent=2) + "\n")
    print(json.dumps({"jobs": manifests}, indent=2))


if __name__ == "__main__":
    main()
