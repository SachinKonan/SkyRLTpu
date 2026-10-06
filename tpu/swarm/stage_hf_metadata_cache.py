#!/usr/bin/env python3
"""Stage a Hugging Face snapshot without model weight payloads."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


_OBJECT_ID = re.compile(r"^[0-9A-Za-z._-]+$")


def _copy_source(source: str, relative: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.startswith("gs://"):
        subprocess.run(
            ["gcloud", "storage", "cp", f"{source.rstrip('/')}/{relative}", str(destination)],
            check=True,
        )
    else:
        shutil.copy2(Path(source) / relative, destination)


def _copy_source_if_present(source: str, relative: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.startswith("gs://"):
        result = subprocess.run(
            ["gcloud", "storage", "cp", f"{source.rstrip('/')}/{relative}", str(destination)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return True
        destination.unlink(missing_ok=True)
        if "matched no objects or files" in result.stderr:
            return False
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            stderr=result.stderr,
        )

    source_path = Path(source) / relative
    if not source_path.is_file():
        return False
    shutil.copy2(source_path, destination)
    return True


def _flat_snapshot_files(source: str, revision: str) -> list[str]:
    relative_prefix = f"snapshots/{revision}/"
    if source.startswith("gs://"):
        source_prefix = f"{source.rstrip('/')}/{relative_prefix}"
        result = subprocess.run(
            ["gcloud", "storage", "ls", "--recursive", f"{source_prefix}**"],
            check=True,
            capture_output=True,
            text=True,
        )
        files = []
        for url in result.stdout.splitlines():
            url = url.strip()
            if not url or url.endswith("/"):
                continue
            if not url.startswith(source_prefix):
                raise ValueError(f"unexpected GCS object outside snapshot: {url!r}")
            files.append(url[len(source_prefix) :])
        return files

    snapshot_dir = Path(source) / relative_prefix
    if not snapshot_dir.is_dir():
        return []
    return [str(path.relative_to(snapshot_dir)) for path in snapshot_dir.rglob("*") if path.is_file()]


def _stage_flat_snapshot_metadata(
    source: str, revision: str, snapshot_dir: Path
) -> tuple[int, int]:
    copied_files = 0
    copied_bytes = 0
    for relative in _flat_snapshot_files(source, revision):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"invalid snapshot path: {relative!r}")
        if relative.endswith(".safetensors"):
            continue
        destination = snapshot_dir / relative_path
        _copy_source(source, f"snapshots/{revision}/{relative}", destination)
        copied_files += 1
        copied_bytes += destination.stat().st_size
    if copied_files == 0:
        raise FileNotFoundError(f"no metadata files found in snapshot {revision!r}")
    return copied_files, copied_bytes


def stage_metadata_cache(source: str, model_dir: Path) -> tuple[int, int]:
    """Copy non-safetensor objects and materialize an offline HF snapshot."""
    model_dir = model_dir.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="hf-metadata-") as temp:
        temp_dir = Path(temp)
        ref_path = temp_dir / "main"
        _copy_source(source, "refs/main", ref_path)
        revision = ref_path.read_text().strip()
        if not _OBJECT_ID.fullmatch(revision):
            raise ValueError(f"invalid HF revision: {revision!r}")

        copied_files = 0
        copied_bytes = 0
        snapshot_dir = model_dir / "snapshots" / revision
        blob_dir = model_dir / "blobs"
        manifest_path = temp_dir / f"{revision}.json"
        has_manifest = _copy_source_if_present(
            source, f"trees/{revision}.json", manifest_path
        )
        if has_manifest:
            manifest = json.loads(manifest_path.read_text())
            for name, metadata in manifest.get("files", {}).items():
                if name.endswith(".safetensors"):
                    continue
                blob_id = metadata.get("lfs_sha256") or metadata.get("blob_id")
                if not blob_id or not _OBJECT_ID.fullmatch(blob_id):
                    raise ValueError(f"invalid blob id for {name!r}: {blob_id!r}")
                size = int(metadata["size"])
                blob_path = blob_dir / blob_id
                if not blob_path.is_file() or blob_path.stat().st_size != size:
                    temporary_blob = blob_path.with_name(f".{blob_path.name}.tmp")
                    _copy_source(source, f"blobs/{blob_id}", temporary_blob)
                    if temporary_blob.stat().st_size != size:
                        temporary_blob.unlink(missing_ok=True)
                        raise ValueError(f"wrong size for HF metadata blob {name!r}")
                    os.replace(temporary_blob, blob_path)
                    copied_files += 1
                    copied_bytes += size

                snapshot_path = snapshot_dir / name
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_path.unlink(missing_ok=True)
                snapshot_path.symlink_to(os.path.relpath(blob_path, snapshot_path.parent))
        else:
            copied_files, copied_bytes = _stage_flat_snapshot_metadata(
                source, revision, snapshot_dir
            )

        refs_dir = model_dir / "refs"
        trees_dir = model_dir / "trees"
        refs_dir.mkdir(parents=True, exist_ok=True)
        trees_dir.mkdir(parents=True, exist_ok=True)
        # huggingface_hub reads refs verbatim when resolving offline snapshots.
        # Its own cache format stores the commit without a trailing newline.
        (refs_dir / "main").write_text(revision)
        if has_manifest:
            shutil.copy2(manifest_path, trees_dir / manifest_path.name)
    return copied_files, copied_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    copied_files, copied_bytes = stage_metadata_cache(args.source, args.model_dir)
    print(
        f"HF metadata cache staged: files={copied_files} "
        f"bytes={copied_bytes} model_dir={args.model_dir}"
    )


if __name__ == "__main__":
    main()
