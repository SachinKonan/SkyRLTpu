"""Restore manifest-selected files using one native gcloud transfer batch."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

from transfer import run_transfer


def selected_files(manifest):
    selected = []
    for name, meta in manifest["files"].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe manifest path")
        if rel.suffix not in (".json", ".safetensors", ".model", ".txt", ".tiktoken", ".jinja"):
            continue
        blob = meta.get("lfs_sha256") or meta["blob_id"]
        if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", blob):
            raise ValueError("invalid manifest blob ID")
        if type(meta["size"]) is not int or meta["size"] < 0:
            raise ValueError("invalid manifest size")
        selected.append((name, blob, meta["size"]))
    return selected


def matches(path, size):
    return path.is_file() and path.stat().st_size == size


def restore(root, prefix, manifest):
    model = root / "model"
    model.mkdir(exist_ok=True)
    stage = root / "model-downloads"
    stage.mkdir(exist_ok=True)
    files = selected_files(manifest)
    missing = [(name, blob, size) for name, blob, size in files
               if not matches(model / name, size)]
    print(json.dumps({"event": "cache_plan", "reused_files": len(files) - len(missing),
                      "missing_files": len(missing),
                      "missing_bytes": sum(size for _, _, size in missing)}), flush=True)
    for attempt in range(3):
        blobs = sorted({blob for _, blob, size in missing if not matches(stage / blob, size)})
        for blob in blobs:
            (stage / blob).unlink(missing_ok=True)
        if blobs:
            # Only gcloud's completed destination names can be promoted. Its
            # resumable temporary files remain under its own management.
            run_transfer(["gcloud", "storage", "cp", "--read-paths-from-stdin", str(stage) + "/"],
                         root, stage, "hf", "".join(prefix + "/blobs/" + b + "\n" for b in blobs),
                         check=False)
        remaining = []
        for name, blob, size in missing:
            source = stage / blob
            if not matches(source, size):
                remaining.append((name, blob, size))
                continue
            target = model / name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".promoting")
            partial.unlink(missing_ok=True)
            os.link(source, partial)
            partial.replace(target)
            print("cache ready", name, size, flush=True)
        missing = remaining
        if not missing:
            break
    if missing:
        raise RuntimeError(f"incomplete cache after three batches: {missing}")
    for _, blob, _ in files:
        (stage / blob).unlink(missing_ok=True)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    required = set(index["weight_map"].values()) | {
        "config.json", "tokenizer.json", "preprocessor_config.json", "video_preprocessor_config.json"}
    expected = {name: size for name, _, size in files}
    for name in required:
        if name not in expected or not matches(model / name, expected[name]):
            raise RuntimeError(f"missing or invalid required model file {name}")


def main():
    root = Path(os.environ["CANARY_ROOT"])
    prefix = os.environ["HF_CACHE_GCS"] + "/models--Qwen--Qwen3.5-27B"
    manifests = subprocess.check_output(
        ["gcloud", "storage", "ls", prefix + "/trees/*.json"], text=True).splitlines()
    if len(manifests) != 1:
        raise RuntimeError(f"expected one pinned HF manifest, got {manifests}")
    manifest = json.loads(subprocess.check_output(["gcloud", "storage", "cat", manifests[0]]))
    (root / "model-ready").unlink(missing_ok=True)
    restore(root, prefix, manifest)
    (root / "model-ready").write_text(manifests[0])


if __name__ == "__main__":
    main()
