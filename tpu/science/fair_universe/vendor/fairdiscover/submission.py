"""Produce official Phase-2 Codabench result.json ZIP from trained JAX model.

Official repo's scoring_program/score.py loads predictions/result.json and reads
"ood_scores". This package uses result.json at ZIP root (result submission).
The official test maps ALREADY INCLUDE NOISE; no augmentation here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
from .data import MapStore
from .worker import checked_predict, load_candidate, code_hash


def load_test_batch(test_array, first, last, mask):
    batch = np.asarray(test_array[first:last], dtype=np.float32)
    h, w = mask.shape
    if batch.ndim == 3 and batch.shape[1:] == (h, w):
        return batch * mask[None]
    if batch.ndim == 2 and batch.shape[1] == int(mask.sum()):
        out = np.zeros((len(batch), h*w), dtype=np.float32)
        out[:, mask.reshape(-1)] = batch
        return out.reshape(len(batch), h, w)
    if batch.ndim == 2 and batch.shape[1] == h * w:
        return batch.reshape(len(batch), h, w) * mask[None]
    raise ValueError(f"Unexpected test batch shape: {batch.shape}; mask {mask.shape}")


def export_submission(candidate, run_dir, stage, test_file, out_zip, batch_size=32,
                      expected_count=None):
    run = Path(run_dir)
    cfg = json.loads((run / "trial_config.json").read_text())
    meta = json.loads((run / "train_meta.json").read_text())
    if meta["candidate_sha256"] != code_hash(candidate):
        raise ValueError("Candidate has changed since checkpoint was trained")
    mod = load_candidate(candidate)
    model = mod.load_model(run / "artifacts", cfg)
    store = MapStore(stage)
    test = np.load(test_file, mmap_mode="r")
    count = len(test)
    if expected_count is not None and count != int(expected_count):
        raise ValueError(f"Test has {count} examples, expected {expected_count}")
    if count == 0:
        raise ValueError("Empty test dataset")
    all_scores = []
    for start in range(0, count, batch_size):
        maps = load_test_batch(test, start, min(start+batch_size, count), store.mask)
        # DO NOT add shape noise: official test inputs are already noisy.
        all_scores.extend(checked_predict(mod, model, maps, store.mask, cfg).tolist())
    if not np.all(np.isfinite(all_scores)) or len(all_scores) != count:
        raise ValueError("Invalid predictions")
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"ood_scores": all_scores}, separators=(",", ":"))
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("result.json", payload)
    # Also save plain JSON alongside the ZIP for inspection.
    (out_zip.parent / (out_zip.stem + ".json")).write_text(payload)
    return {"zip": str(out_zip), "count": count, "score_min": float(min(all_scores)),
            "score_max": float(max(all_scores))}


def main():
    p = argparse.ArgumentParser(description="Export Phase 2 result.json ZIP")
    p.add_argument("--candidate", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--test-file", required=True,
                   help="Official noisy Phase 2 test .npy, NOT locally synthesized reward data")
    p.add_argument("--out-zip", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--expected-count", type=int)
    a = p.parse_args()
    print(json.dumps(export_submission(a.candidate, a.run_dir, a.stage,
                                       a.test_file, a.out_zip,
                                       a.batch_size, a.expected_count), indent=2))


if __name__ == "__main__":
    main()
