"""Synchronous GCS write-through for locally materialized checkpoints."""

import subprocess
from pathlib import Path


def mirror_checkpoint_to_gcs(
    local_path: Path,
    mirror_base: str,
    model_id: str,
    family: str = "",
) -> str:
    """Upload one checkpoint and verify the remote object has the same size."""
    if not local_path.is_file():
        raise FileNotFoundError(f"checkpoint mirror source is absent: {local_path}")

    destination_parts = [mirror_base.rstrip("/"), model_id]
    if family:
        destination_parts.append(family.strip("/"))
    destination_parts.append(local_path.name)
    destination = "/".join(destination_parts)

    copied = subprocess.run(
        ["gcloud", "storage", "cp", str(local_path), destination],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if copied.returncode != 0:
        detail = (copied.stderr or copied.stdout).strip()[-2000:]
        raise RuntimeError(f"checkpoint mirror upload failed for {destination}: {detail}")

    described = subprocess.run(
        [
            "gcloud",
            "storage",
            "objects",
            "describe",
            destination,
            "--format=value(size)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    remote_size = described.stdout.strip()
    local_size = local_path.stat().st_size
    if described.returncode != 0 or remote_size != str(local_size):
        detail = (described.stderr or described.stdout).strip()[-2000:]
        raise RuntimeError(
            f"checkpoint mirror verification failed for {destination}: "
            f"local_size={local_size} remote_size={remote_size or 'unknown'} {detail}"
        )
    return destination
