#!/usr/bin/env python3
"""Convert a Tunix training checkpoint from repeated (logical) K/V heads to the model's native K/V heads,
keeping its optimizer state.

A TP8 trainer (v6e-128 pair) runs Qwen3.5/Muse with 8 logical K/V heads: MaxText repeat-interleaves each
native head, and the trainer ties the copies (RepeatedKVHeads: summed gradients, identical Adam moments). A
single-host v5p trainer (TP1/FSDP4) builds the native layout (Qwen3.5: 4 K/V heads x 256, Muse: 2 x 128), so
the K/V LoRA B tensors -- and the Adam mu/nu of exactly those tensors -- must be collapsed to one replica per
native head. Like tpu/science/weights_only_checkpoint.collapse_repeated_kv (which handles weights-only
archives), this:
  * requires every replica group to be bit-identical, and verifies that re-expanding the collapsed tensor
    reproduces the saved tensor exactly; anything else is refused;
  * drops the saved mesh placements (lora_layouts, optimizer_layouts), so the destination trainer places every
    restored array with its own template (TunixBackend._state_from_flat uses the template sharding when a
    layout is absent);
  * keeps every other tensor byte-identical and the optimizer counters (count, step) and hyperparameters as
    they are, so the destination resumes the SAME Adam state, not a fresh one.
The source archive is never modified; the destination must not exist.

    python tpu/cosearch_collapse_kv.py SRC.tar.gz DST.tar.gz --model qwen3.5-27b
    python tpu/cosearch_collapse_kv.py SRC.tar.gz DST.tar.gz --model muse-glimmer-30b
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

KV_B = re.compile(r"\['(?:key|value)'\]\['kernel_lora_b'\]")
# Native K/V heads and head_dim of the HF models (what a non-TP8 MaxText trainer instantiates).
NATIVE = {"qwen3.5-27b": (4, 256), "qwen3.8-27b": (4, 256), "muse-glimmer-30b": (2, 128)}
FILES = ("lora_weights.npz", "optimizer_state.npz", "tunix_checkpoint_meta.json")


def _read(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:gz") as archive:
        members = {Path(m.name).name: m for m in archive if m.isfile()}
        if set(members) != set(FILES):
            raise ValueError(f"expected {FILES}, archive has {sorted(members)}")
        return {name: archive.extractfile(m).read() for name, m in members.items()}


def _npz(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes() + str((arr.dtype, arr.shape)).encode()).hexdigest()


def collapse(value: np.ndarray, native: int, head_dim: int) -> tuple[np.ndarray, int]:
    """(collapsed tensor, replicas per native head). Refuses divergent replicas or a lossy round trip."""
    width = value.shape[-1]
    if width % head_dim or (width // head_dim) % native:
        raise ValueError(f"K/V width {width} is not a multiple of {native} native heads x {head_dim}")
    logical = width // head_dim
    rep = logical // native
    if rep == 1:
        return value, 1
    grouped = value.reshape(*value.shape[:-1], native, rep, head_dim)
    canonical = grouped[..., :1, :]
    if not np.array_equal(grouped, np.broadcast_to(canonical, grouped.shape)):
        raise ValueError("K/V replicas are not bit-identical")
    out = np.ascontiguousarray(canonical.reshape(*value.shape[:-1], native * head_dim))
    expanded = np.repeat(out.reshape(*value.shape[:-1], native, 1, head_dim), rep, axis=-2).reshape(value.shape)
    if not np.array_equal(expanded, value):
        raise ValueError("K/V round trip failed")
    return out, rep


def convert(source: Path, destination: Path, native: int, head_dim: int) -> dict:
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("destination must be a new file")
    raw = _read(source)
    meta = json.loads(raw["tunix_checkpoint_meta.json"])
    if meta.get("format") != "tunix_backend_v2" or "lora_mix" in meta:
        raise ValueError("expected a plain Tunix v2 adapter checkpoint")
    report = dict(source=str(source), destination=str(destination), native_kv_heads=native, head_dim=head_dim,
                  lora_config=meta.get("lora_config"), converted={}, unchanged={})
    out = {}
    for name in ("lora_weights.npz", "optimizer_state.npz"):
        arrays = _npz(raw[name])
        kv = [k for k in arrays if KV_B.search(k)]
        if not kv:
            raise ValueError(f"no K/V LoRA B tensors in {name}")
        converted = {}
        for key, value in arrays.items():
            if KV_B.search(key):
                new, rep = collapse(value, native, head_dim)
                converted[key] = new
                report["converted"][f"{name}:{key}"] = dict(
                    before=list(value.shape), after=list(new.shape), replicas_per_native_head=rep,
                    replicas_bit_identical=True, round_trip_exact=True)
            else:
                converted[key] = value
        out[name] = converted
        report["unchanged"][name] = len(arrays) - len(kv)
    # Counters and hyperparameters stay exactly as saved: the destination resumes this Adam state.
    report["optimizer_counters"] = {k: out["optimizer_state.npz"][k].item() for k in out["optimizer_state.npz"]
                                    if "lora" not in k}
    meta = dict(meta)
    meta.pop("lora_layouts", None)
    meta.pop("optimizer_layouts", None)
    payloads = {}
    for name in ("lora_weights.npz", "optimizer_state.npz"):
        buf = io.BytesIO()
        np.savez(buf, **out[name])
        payloads[name] = buf.getvalue()
    payloads["tunix_checkpoint_meta.json"] = json.dumps(meta, sort_keys=True).encode()
    partial = destination.with_name(destination.name + ".partial")
    try:
        with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
            for name in FILES:
                info = tarfile.TarInfo(name)
                info.size = len(payloads[name])
                archive.addfile(info, io.BytesIO(payloads[name]))
        verify(source, partial, native, head_dim)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    report.update(
        source_archive_sha256=_file_sha(source), archive_sha256=_file_sha(destination),
        source_lora_npz_sha256=hashlib.sha256(raw["lora_weights.npz"]).hexdigest(),
        lora_npz_sha256=hashlib.sha256(payloads["lora_weights.npz"]).hexdigest(),
        optimizer_npz_sha256=hashlib.sha256(payloads["optimizer_state.npz"]).hexdigest(),
        bytes=destination.stat().st_size, layouts_dropped=True, optimizer_present=True)
    return report


def verify(source: Path, destination: Path, native: int, head_dim: int) -> None:
    """Re-read both archives: every non-K/V array byte-identical, every K/V array re-expands exactly, no
    layouts, same LoRA config and optimizer counters."""
    src, dst = _read(source), _read(destination)
    smeta, dmeta = json.loads(src["tunix_checkpoint_meta.json"]), json.loads(dst["tunix_checkpoint_meta.json"])
    if "lora_layouts" in dmeta or "optimizer_layouts" in dmeta or dmeta.get("lora_config") != smeta.get("lora_config"):
        raise ValueError("destination metadata is not the source's minus its layouts")
    for name in ("lora_weights.npz", "optimizer_state.npz"):
        a, b = _npz(src[name]), _npz(dst[name])
        if set(a) != set(b):
            raise ValueError(f"{name}: key sets differ")
        for key, old in a.items():
            new = b[key]
            if KV_B.search(key):
                rep = old.shape[-1] // (native * head_dim)
                back = np.repeat(new.reshape(*new.shape[:-1], native, 1, head_dim), rep, axis=-2).reshape(old.shape)
                if not np.array_equal(back, old):
                    raise ValueError(f"{name}:{key}: collapsed tensor does not re-expand to the source")
            elif _sha(old) != _sha(new):
                raise ValueError(f"{name}:{key}: tensor changed")


def inspect(source: Path, native: int, head_dim: int) -> dict:
    """Read-only: the K/V layout an archive was saved in (logical heads per K/V LoRA B tensor, replicas
    bit-identical) and whether it carries Adam state, so a resume on another mesh can be checked to need no
    conversion (same logical_kv_heads)."""
    raw = _read(Path(source))
    meta = json.loads(raw["tunix_checkpoint_meta.json"])
    report = dict(source=str(source), native_kv_heads=native, head_dim=head_dim, format=meta.get("format"),
                  lora_config=meta.get("lora_config"), layouts_present="lora_layouts" in meta, kv={})
    for name in ("lora_weights.npz", "optimizer_state.npz"):
        arrays = _npz(raw[name])
        for key, value in arrays.items():
            if KV_B.search(key):
                _, rep = collapse(value, native, head_dim)
                report["kv"][f"{name}:{key}"] = dict(shape=list(value.shape), logical_kv_heads=native * rep,
                                                     replicas_bit_identical=True)
        if name == "optimizer_state.npz":
            report["optimizer_present"] = bool(arrays)
            report["optimizer_counters"] = {k: arrays[k].item() for k in arrays if "lora" not in k}
    report["logical_kv_heads"] = sorted({v["logical_kv_heads"] for v in report["kv"].values()})
    report["archive_sha256"] = _file_sha(Path(source))
    report["bytes"] = Path(source).stat().st_size
    return report


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("destination", nargs="?", help="omit with --inspect")
    ap.add_argument("--model", required=True, choices=sorted(NATIVE))
    ap.add_argument("--report", help="write the JSON report here (default: <destination>.json)")
    ap.add_argument("--inspect", action="store_true", help="only report the saved K/V layout (writes no archive)")
    x = ap.parse_args(argv)
    native, head_dim = NATIVE[x.model]
    if x.inspect:
        report = inspect(Path(x.source), native, head_dim)
        report["model"] = x.model
        text = json.dumps(report, indent=1)
        if x.report:
            Path(x.report).write_text(text + "\n")
        print(text)
        return 0
    if not x.destination:
        ap.error("destination is required unless --inspect")
    report = convert(Path(x.source), Path(x.destination), native, head_dim)
    report["model"] = x.model
    text = json.dumps(report, indent=1)
    Path(x.report or x.destination + ".json").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
