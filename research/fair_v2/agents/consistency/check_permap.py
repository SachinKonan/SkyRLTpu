"""Rule-1 check: re-score 100 test maps in a different batch (alone, reversed order) and compare
family theta/copula energies and the small<->large cross energies to the stored full-batch values."""
import pickle

import numpy as np
import torch

import lib
from fit_families import run_model

sel = np.arange(100)[::-1].copy()
out = {}
for fam in ('small', 'large', 'dog', 'cnn', 'psS'):
    with open(lib.OUT / 'fam' / f'model_{fam}.pkl', 'rb') as fh:
        lk = pickle.load(fh)
    f = lib.cols(fam, 'test')[sel, 0]
    r = run_model(lk, f)
    with np.load(lib.OUT / 'fam' / f'{fam}.npz') as d:
        ref_cop, ref_th = d['test/cop'][sel], d['test/th'][sel]
    out[fam] = (lk, torch.as_tensor(r['th'], device=lk.dev), f)
    print(fam, 'max|dcop|', float(np.abs(r['cop'] - ref_cop).max()), 'max|dth|', float(np.abs(r['th'] - ref_th).max()),
          'cop scale', float(np.abs(ref_cop).mean()), flush=True)
with np.load(lib.OUT / 'scale' / 'scale.npz') as d:
    ref = d['test/cop_small@large'][sel]
lk, _, f = out['small']
with torch.no_grad():
    v = lk.copula_energy(lk.standardize(f), out['large'][1]).cpu().numpy()
print('cop_small@large max|diff|', float(np.abs(v - ref).max()))
