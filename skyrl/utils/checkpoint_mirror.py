"""Synchronous GCS durability for locally materialized checkpoints."""

import os
import subprocess
from pathlib import Path


def _checkpoint_uri(
    mirror_base: str,
    model_id: str,
    filename: str,
    family: str = "",
) -> str:
    parts = [mirror_base.rstrip("/"), model_id]
    if family:
        parts.append(family.strip("/"))
    parts.append(filename)
    return "/".join(parts)


def mirror_checkpoint_to_gcs(
    local_path: Path,
    mirror_base: str,
    model_id: str,
    family: str = "",
) -> str:
    """Upload one checkpoint and verify the remote object has the same size."""
    if not local_path.is_file():
        raise FileNotFoundError(f"checkpoint mirror source is absent: {local_path}")

    destination = _checkpoint_uri(mirror_base, model_id, local_path.name, family)

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


def restore_checkpoint_from_gcs(
    local_path: Path,
    mirror_base: str,
    model_id: str,
    family: str = "",
) -> str:
    """Atomically restore one missing checkpoint and verify its object size."""
    source = _checkpoint_uri(mirror_base, model_id, local_path.name, family)
    if local_path.is_file():
        return source

    described = subprocess.run(
        [
            "gcloud",
            "storage",
            "objects",
            "describe",
            source,
            "--format=value(size)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    remote_size = described.stdout.strip()
    if described.returncode != 0 or not remote_size.isdigit():
        detail = (described.stderr or described.stdout).strip()[-2000:]
        raise RuntimeError(
            f"checkpoint mirror lookup failed for {source}: "
            f"remote_size={remote_size or 'unknown'} {detail}"
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_name(f".{local_path.name}.{os.getpid()}.partial")
    try:
        copied = subprocess.run(
            ["gcloud", "storage", "cp", source, str(temporary)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if copied.returncode != 0:
            detail = (copied.stderr or copied.stdout).strip()[-2000:]
            raise RuntimeError(f"checkpoint mirror download failed for {source}: {detail}")
        downloaded_size = temporary.stat().st_size
        if downloaded_size != int(remote_size):
            raise RuntimeError(
                f"checkpoint mirror download verification failed for {source}: "
                f"downloaded_size={downloaded_size} remote_size={remote_size}"
            )
        os.replace(temporary, local_path)
    finally:
        temporary.unlink(missing_ok=True)
    return source
