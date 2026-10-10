"""Lazy FAIR convergence-map IO and split staging.

Official public Zenodo release stores only observed pixels: [cosmology, realization,
132019]. Reconstruct the 2D footprint just-in-time, not as a 26 GB array.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .config import read_config, noise_sigma

MODES = {
    "full": ("WIDE12H_bin2_2arcmin_kappa_newrealization.npy", "label_newrealization.npy"),
    "sample": ("sampled_WIDE12H_bin2_2arcmin_kappa.npy", "sampled_label.npy"),
}
MASK_FILE = "WIDE12H_bin2_2arcmin_mask.npy"


def stage(raw_dir: str, stage_dir: str, mode="full", seed=2026,
          train_ratio=.75, val_ratio=.125, map_file=None, label_file=None,
          mask_file=None):
    """Create split indices and a reference manifest without duplicating the full maps.

    Splits are along realization axis, SHARED across cosmologies. This avoids
    reusing correlated realization IDs across train / val / reward.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Need nonempty train, val, reward fractions")
    raw = Path(raw_dir).resolve()
    out = Path(stage_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    maps_path = (raw / (map_file or MODES[mode][0])).resolve()
    labels_path = (raw / (label_file or MODES[mode][1])).resolve()
    mask_path = (raw / (mask_file or MASK_FILE)).resolve()
    for path in (maps_path, labels_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; see README download instructions")
    maps = np.load(maps_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    mask = np.load(mask_path, mmap_mode="r")
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError(f"Expected 2D boolean survey mask, got {mask.shape}, {mask.dtype}")
    if maps.ndim not in (3, 4) or labels.shape != (*maps.shape[:2], 5):
        raise ValueError(f"Bad map/label shapes: {maps.shape} / {labels.shape}")
    if maps.ndim == 3 and maps.shape[-1] != int(mask.sum()):
        raise ValueError("Compressed map length does not equal mask sum")
    if maps.ndim == 4 and maps.shape[2:] != mask.shape:
        raise ValueError("Full map dimensions differ from mask")
    ncosmo, nreal = maps.shape[:2]
    if nreal < 3:
        raise ValueError("Need at least 3 realizations for train/val/reward")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nreal)
    cut1 = min(nreal - 2, max(1, int(nreal * train_ratio)))
    cut2 = min(nreal - 1, cut1 + max(1, int(nreal * val_ratio)))
    for split, realization_ids in [("train", perm[:cut1]),
                                   ("val", perm[cut1:cut2]),
                                   ("reward", perm[cut2:])]:
        indices = np.array([(c, int(r)) for c in range(ncosmo)
                            for r in realization_ids], dtype=np.int32)
        np.save(out / f"{split}_ids.npy", indices)
    manifest = {
        "version": 1, "mode": mode,
        "maps": str(maps_path), "labels": str(labels_path), "mask": str(mask_path),
        "n_cosmologies": ncosmo, "n_realizations": nreal,
        "map_shape": list(mask.shape), "observed_pixels": int(mask.sum()),
        "split_seed": int(seed), "train_ratio": train_ratio,
        "val_ratio": val_ratio, "source_dtype": str(maps.dtype),
        "counts": {split: int(len(np.load(out / f"{split}_ids.npy")))
                   for split in ("train", "val", "reward")},
        "warning": "Indices are split across shared realization IDs, not random images."
    }
    with open(out / "manifest.json", "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


class MapStore:
    """Memory-mapped compressed or full maps; reconstruct only requested batches."""
    def __init__(self, staged_dir):
        self.staged_dir = Path(staged_dir)
        self.manifest = json.loads((self.staged_dir / "manifest.json").read_text())
        self.maps = np.load(self.manifest["maps"], mmap_mode="r")
        self.labels = np.load(self.manifest["labels"], mmap_mode="r")
        self.mask = np.asarray(np.load(self.manifest["mask"]), dtype=bool)
        self.mask_flat = self.mask.reshape(-1)
        self.height, self.width = self.mask.shape

    def ids(self, split):
        if split not in ("train", "val", "reward"):
            raise ValueError("Invalid split")
        return np.load(self.staged_dir / f"{split}_ids.npy")

    def get_clean(self, ids):
        ids = np.asarray(ids)
        if len(ids) == 0:
            return np.zeros((0, self.height, self.width), dtype=np.float32)
        signal = np.asarray(self.maps[ids[:, 0], ids[:, 1]], dtype=np.float32)
        if self.maps.ndim == 4:
            return signal * self.mask[None]
        out = np.zeros((len(ids), self.height * self.width), dtype=np.float32)
        out[:, self.mask_flat] = signal
        return out.reshape(len(ids), self.height, self.width)

    def get_labels(self, ids):
        ids = np.asarray(ids)
        return np.asarray(self.labels[ids[:, 0], ids[:, 1]], dtype=np.float32)

    def apply_noise(self, clean, rng, config):
        # Training/public maps are noiseless; official Phase 2 test is already noisy.
        sigma = noise_sigma(config)
        noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
        return (clean + noise) * self.mask[None]

    def get_noisy(self, ids, rng, config):
        return self.apply_noise(self.get_clean(ids), rng, config)


@dataclass
class DataView:
    """The only dataset abstraction passed into candidate.fit.

    The model sees in-distribution train and validation views, with labels;
    reward sample IDs and synthetic anomaly labels are never passed to fit.
    """
    store: MapStore
    split: str
    config: dict

    @property
    def mask(self):
        return self.store.mask

    def __len__(self):
        return len(self.store.ids(self.split))

    def iter_batches(self, batch_size=16, epochs=1, shuffle=True, seed=0,
                     add_noise=True, drop_last=False, limit=None):
        ids = self.store.ids(self.split)
        rng = np.random.default_rng(seed)
        if limit is not None:
            limit = max(0, min(int(limit), len(ids)))
            ids = ids[rng.choice(len(ids), size=limit, replace=False)] if shuffle else ids[:limit]
        for _ in range(epochs):
            order = rng.permutation(len(ids)) if shuffle else np.arange(len(ids))
            for start in range(0, len(ids), batch_size):
                selected = ids[order[start:start + batch_size]]
                if drop_last and len(selected) < batch_size:
                    continue
                maps = (self.store.get_noisy(selected, rng, self.config) if add_noise
                        else self.store.get_clean(selected))
                yield {"maps": maps, "labels": self.store.get_labels(selected),
                       "mask": self.store.mask}


def main():
    p = argparse.ArgumentParser(description="Index official FAIR public dataset")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=MODES, default="sample")
    p.add_argument("--map-file")
    p.add_argument("--label-file")
    p.add_argument("--mask-file")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    print(json.dumps(stage(a.raw_dir, a.out, a.mode, a.seed,
                           map_file=a.map_file, label_file=a.label_file,
                           mask_file=a.mask_file), indent=2))


if __name__ == "__main__":
    main()
