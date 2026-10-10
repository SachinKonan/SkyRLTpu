import json
import sys
from pathlib import Path
import numpy as np
from fairdiscover.data import stage, MapStore, DataView
from fairdiscover.config import read_config, noise_sigma

# Test generator is a script (no implicit package import via scripts namespace).
def create_toy(raw):
    rng = np.random.default_rng(42)
    raw.mkdir()
    mask = np.ones((14, 8), dtype=bool)
    mask[0] = False
    mask[:, -1] = False
    maps = rng.normal(0, .01, (4, 16, int(mask.sum()))).astype(np.float16)
    labels = rng.normal(size=(4,16,5))
    np.save(raw / 'WIDE12H_bin2_2arcmin_mask.npy', mask)
    np.save(raw / 'sampled_WIDE12H_bin2_2arcmin_kappa.npy', maps)
    np.save(raw / 'sampled_label.npy', labels)
    return mask


def test_stage_lazily_reconstructs_and_prevents_overlap(tmp_path):
    mask = create_toy(tmp_path / 'raw')
    m = stage(str(tmp_path / 'raw'), str(tmp_path / 'stage'), mode='sample')
    store = MapStore(tmp_path / 'stage')
    splits = [store.ids(k) for k in ('train','val','reward')]
    assert m['counts']['train'] + m['counts']['val'] + m['counts']['reward'] == 4*16
    keys = [set(map(tuple, rows)) for rows in splits]
    assert not(keys[0]&keys[1] or keys[0]&keys[2] or keys[1]&keys[2])
    # shared realization IDs for every cosmology means no spurious overlap
    assert set(splits[0][:,1]).isdisjoint(set(splits[1][:,1]))
    batch = next(DataView(store, 'train', read_config()).iter_batches(batch_size=3, seed=1))
    assert batch['maps'].shape == (3, 14, 8)
    assert np.all(batch['maps'][:,~mask] == 0)
    assert batch['labels'].shape == (3,5)
    assert abs(noise_sigma(read_config()) - .025819888974716113) < 1e-6
