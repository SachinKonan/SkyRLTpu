"""Tests for the new candidate and the budgeted-sampling correction."""
from pathlib import Path
import importlib.util
import numpy as np
from fairdiscover.config import read_config
from fairdiscover.data import stage, MapStore, DataView
from fairdiscover.worker import load_candidate, checked_predict

ROOT = Path(__file__).resolve().parents[1]


def make_store(tmp_path):
    spec = importlib.util.spec_from_file_location('toy', ROOT/'scripts/make_toy_data.py')
    toy = importlib.util.module_from_spec(spec); spec.loader.exec_module(toy)
    toy.make_toy(tmp_path/'raw', n_cosmo=8, n_real=16)
    stage(str(tmp_path/'raw'), str(tmp_path/'stage'), mode='sample')
    return MapStore(tmp_path/'stage')


def test_limited_shuffle_samples_across_cosmologies(tmp_path):
    store = make_store(tmp_path)
    cfg = read_config()
    batches = list(DataView(store, 'train', cfg).iter_batches(
        batch_size=12, epochs=1, shuffle=True, seed=31415, limit=12))
    labels = np.concatenate([b['labels'] for b in batches])
    assert len(labels) == 12
    assert len(np.unique(labels[:, 0])) > 1
    repeat = list(DataView(store, 'train', cfg).iter_batches(
        batch_size=12, epochs=1, shuffle=True, seed=31415, limit=12))
    np.testing.assert_array_equal(batches[0]['maps'], repeat[0]['maps'])


def test_multiscale_reload_mask_and_batch_contract(tmp_path):
    store = make_store(tmp_path)
    mod = load_candidate(ROOT/'candidates/multiscale_jax.py')
    cfg = read_config(); cfg.update(batch_size=16, steps=8, evolved_mode='conditional')
    info = mod.fit(DataView(store,'train',cfg), DataView(store,'val',cfg), cfg, tmp_path/'artifacts')
    assert info['uses_proxy_labels'] is False
    assert info['features'] == 54
    model = mod.load_model(tmp_path/'artifacts',cfg)
    maps = store.get_noisy(store.ids('val')[:6],np.random.default_rng(331),cfg)
    all_at_once = checked_predict(mod,model,maps,store.mask,cfg)
    separate = np.concatenate([checked_predict(mod,model,maps[i:i+1],store.mask,cfg) for i in range(6)])
    np.testing.assert_allclose(all_at_once,separate,rtol=2.e-4,atol=2.e-3)
    changed = maps.copy(); changed[:,~store.mask] = 9999.
    np.testing.assert_allclose(all_at_once,checked_predict(mod,model,changed,store.mask,cfg),atol=1.e-5)
    for mode in ['global','conditional','neighbor','ensemble','max_ensemble']:
        cfg['evolved_mode'] = mode
        result = checked_predict(mod,model,maps,store.mask,cfg)
        assert result.shape == (6,)
