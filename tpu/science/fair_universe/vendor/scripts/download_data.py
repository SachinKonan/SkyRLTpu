"""Optional Zenodo public TRAIN download (no official test/hidden labels).

Use `--sample` first; full release is ~6.8 GB. Supports resuming via HTTP Range
when the server permits, verifies published MD5 checksums after downloading.
"""
from __future__ import annotations
import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE = "https://zenodo.org/records/20056065/files/"
MD5 = {
    "WIDE12H_bin2_2arcmin_mask.npy": "29b8bbb8a9ad41d3dacfaf61008a8268",
    "sampled_WIDE12H_bin2_2arcmin_kappa.npy": "63388a13ed245208a05a4f66c8f188c1",
    "sampled_label.npy": "9a08befb4c635966cf61ced1136e7144",
    "WIDE12H_bin2_2arcmin_kappa_newrealization.npy": "330ccc9ced8288b1d4d9bd81e74228b0",
    "label_newrealization.npy": "2b88e3ccf8c8e66b445ee649886586c7",
}


def md5(path):
    digest = hashlib.md5()  # noqa: S324 -- checksum supplied by Zenodo; not for security.
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(filename, destination):
    dest = Path(destination) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and md5(dest) == MD5[filename]:
        print(f"Verified existing {dest}")
        return
    partial = dest.with_suffix(dest.suffix + ".partial")
    old = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(BASE + filename + "?download=1")
    if old:
        request.add_header("Range", f"bytes={old}-")
    with urllib.request.urlopen(request, timeout=90) as response:
        resumable = response.status == 206 and old > 0
        mode = "ab" if resumable else "wb"
        with open(partial, mode) as target:
            while True:
                chunk = response.read(2**20)
                if not chunk:
                    break
                target.write(chunk)
    if md5(partial) != MD5[filename]:
        raise ValueError(f"Checksum mismatch for {partial}; retry/remove partial")
    partial.replace(dest)
    print(f"Verified {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--sample", action="store_true")
    choice.add_argument("--full", action="store_true")
    a = p.parse_args()
    files = ["WIDE12H_bin2_2arcmin_mask.npy"] + (
        ["sampled_WIDE12H_bin2_2arcmin_kappa.npy", "sampled_label.npy"] if a.sample
        else ["WIDE12H_bin2_2arcmin_kappa_newrealization.npy", "label_newrealization.npy"])
    for name in files:
        download(name, a.out)


if __name__ == "__main__":
    main()
