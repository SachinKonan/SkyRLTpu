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

        manifest_path = temp_dir / f"{revision}.json"
        _copy_source(source, f"trees/{revision}.json", manifest_path)
        manifest = json.loads(manifest_path.read_text())

        copied_files = 0
        copied_bytes = 0
        snapshot_dir = model_dir / "snapshots" / revision
        blob_dir = model_dir / "blobs"
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

        refs_dir = model_dir / "refs"
        trees_dir = model_dir / "trees"
        refs_dir.mkdir(parents=True, exist_ok=True)
        trees_dir.mkdir(parents=True, exist_ok=True)
        # huggingface_hub reads refs verbatim when resolving offline snapshots.
        # Its own cache format stores the commit without a trailing newline.
        (refs_dir / "main").write_text(revision)
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
