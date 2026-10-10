"""Materialize candidate-visible splits separately from private judge inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from .contract import VERSION, SPLIT_COUNTS, NOISE_SIGMA, digest


def prepare(raw, output, seed=2026):
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new immutable dataset directory: {output}')
    maps_path = raw / 'WIDE12H_bin2_2arcmin_kappa_newrealization.npy'
    labels_path = raw / 'label_newrealization.npy'
    mask_path = raw / 'WIDE12H_bin2_2arcmin_mask.npy'
    maps = np.load(maps_path, mmap_mode='r', allow_pickle=False)
    labels = np.load(labels_path, mmap_mode='r', allow_pickle=False)
    mask = np.load(mask_path, allow_pickle=False)
    if mask.dtype != bool or mask.shape != (1424, 176):
        raise ValueError('Unexpected survey mask')
    if maps.shape != (101, 256, int(mask.sum())) or labels.shape != (101, 256, 5):
        raise ValueError('Expected the independent-realization Phase 2 training release')
    order = np.random.default_rng(seed).permutation(256)
    manifest = dict(version=VERSION, split_seed=seed, noise_sigma=NOISE_SIGMA,
                    sources={p.name: digest(p) for p in (maps_path, labels_path, mask_path)},
                    counts={}, realization_ids={})
    output.mkdir(parents=True)
    start = 0
    for name, count in SPLIT_COUNTS.items():
        folder = output / ('fit' if name in ('train', 'val') else 'judge') / name
        folder.mkdir(parents=True)
        ids = order[start:start+count]; start += count
        out = np.lib.format.open_memmap(folder / 'maps.npy', mode='w+', dtype=maps.dtype,
                                        shape=(101 * count, maps.shape[-1]))
        ys = np.empty((101 * count, 5), labels.dtype)
        for c in range(101):
            out[c*count:(c+1)*count] = maps[c, ids]
            ys[c*count:(c+1)*count] = labels[c, ids]
        out.flush(); del out
        np.save(folder / 'labels.npy', ys)
        np.save(folder / 'mask.npy', mask)
        manifest['counts'][name] = 101 * count
        manifest['realization_ids'][name] = ids.tolist()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


class DataView:
    """Only a physically separate fitting or normal-calibration directory."""
    def __init__(self, folder):
        folder = Path(folder)
        self.maps = np.load(folder / 'maps.npy', mmap_mode='r', allow_pickle=False)
        self.labels = np.load(folder / 'labels.npy', mmap_mode='r', allow_pickle=False)
        self.mask = np.load(folder / 'mask.npy', allow_pickle=False)

    def __len__(self):
        return len(self.maps)

    def clean(self, ids):
        result = np.zeros((len(ids), *self.mask.shape), dtype=np.float32)
        result[:, self.mask] = self.maps[ids]
        return result

    def iter_batches(self, batch_size=16, epochs=1, shuffle=True, seed=0,
                     add_noise=True, drop_last=False, limit=None):
        rng = np.random.default_rng(seed)
        ids = np.arange(len(self))
        if limit is not None:
            n = max(0, min(len(ids), int(limit)))
            ids = rng.choice(ids, n, replace=False) if shuffle else ids[:n]
        for _ in range(epochs):
            order = rng.permutation(ids) if shuffle else ids
            for start in range(0, len(order), batch_size):
                selected = order[start:start+batch_size]
                if drop_last and len(selected) != batch_size:
                    continue
                maps = self.clean(selected)
                if add_noise:
                    maps += rng.normal(0, NOISE_SIGMA, maps.shape).astype(np.float32) * self.mask
                yield dict(maps=maps, labels=np.asarray(self.labels[selected]), mask=self.mask)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.raw, a.output), indent=2))
