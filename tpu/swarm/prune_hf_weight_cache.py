#!/usr/bin/env python3
"""Remove model weight payloads while preserving HF tokenizer/config metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def prune_weight_cache(model_dir: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Delete safetensor snapshot entries and their model-scoped blobs."""
    model_dir = model_dir.expanduser().resolve()
    blob_dir = model_dir / "blobs"
    candidates: set[Path] = set()

    for manifest_path in (model_dir / "trees").glob("*.json"):
        manifest = json.loads(manifest_path.read_text())
        for name, metadata in manifest.get("files", {}).items():
            if not name.endswith(".safetensors"):
                continue
            blob_id = metadata.get("lfs_sha256") or metadata.get("blob_id")
            if blob_id:
                candidate = (blob_dir / blob_id).resolve()
                if not _inside(candidate, blob_dir.resolve()):
                    raise ValueError(f"manifest blob escapes cache: {candidate}")
                candidates.add(candidate)

    snapshot_weights = list((model_dir / "snapshots").glob("*/*.safetensors"))
    for weight_path in snapshot_weights:
        if weight_path.is_symlink():
            candidate = weight_path.resolve(strict=False)
            if _inside(candidate, blob_dir.resolve()):
                candidates.add(candidate)

    removed_files = 0
    removed_bytes = 0
    for path in sorted(candidates | set(snapshot_weights)):
        if not path.is_file() and not path.is_symlink():
            continue
        if path.is_file() and not path.is_symlink():
            removed_bytes += path.stat().st_size
        removed_files += 1
        if not dry_run:
            path.unlink()

    for pattern in ("*.incomplete", "*_.gstmp"):
        for partial in model_dir.rglob(pattern):
            if partial.is_file():
                removed_bytes += partial.stat().st_size
                removed_files += 1
                if not dry_run:
                    partial.unlink()

    return removed_files, removed_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    removed_files, removed_bytes = prune_weight_cache(
        args.model_dir, dry_run=args.dry_run
    )
    print(
        f"HF weight cache pruned: files={removed_files} "
        f"bytes={removed_bytes} model_dir={args.model_dir}"
    )


if __name__ == "__main__":
    main()
