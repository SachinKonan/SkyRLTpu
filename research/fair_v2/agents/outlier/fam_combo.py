"""Per-family phase2 scores of OE-based detectors and their combinations with the
current detectors, on the SAME synthetic audit3 maps as fam_baseline.py (same
seeds; run on della-gh like fam_baseline so noise is identical).

Scores: L (OE logit standardized on calib3), O (fit_v3 conditional likelihood on
the OE features, pickled model from --fit), Sd, C (fam_baseline z), and combos.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import data as D
from gens import Library, ALL_FAMILIES
from model import phase2_score
from train_oe import load_net, setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--src', required=True)           # v3 source dir name (for calib standardization of L)
    ap.add_argument('--fit', required=True)           # fit_v3 out dir holding model_O.pkl
    ap.add_argument('--tag', default='O')
    ap.add_argument('--base', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier/fam_baseline')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    dev, mask, m = setup()
    lab = D.labels(); train = np.load(D.TRAIN, mmap_mode='r')
    net, st = load_net(args.ckpt, dev)
    lib = Library(m, dev)
    mo = pickle.load(open(Path(args.fit) / f'model_{args.tag}.pkl', 'rb'))
    cal = np.load(Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3') / args.src / 'calib.npz')['feats'][..., 0].reshape(-1)
    mu, sd = cal.mean(), cal.std()
    base = {}
    for p in sorted(Path(args.base).glob('fam_baseline*.npz'), key=lambda p: p.stat().st_mtime):
        with np.load(p) as f:
            base.update({k: f[k] for k in f.files})
    idx = D.index('audit3'); labs = lab[idx[:, 0], idx[:, 1]]
    gen = torch.Generator(device=dev)

    @torch.no_grad()
    def feats(make):
        out = []
        for lo in range(0, len(idx), 64):
            sl = idx[lo:lo + 64]
            full = torch.zeros((len(sl), 1424, 176), device=dev)
            full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
            x = make(full, labs[lo:lo + 64])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred, logit, flog, e = net(x, m)
            out.append(torch.cat([logit.float()[:, None], flog.float(), pred.float(), e.float()], 1).cpu().numpy())
        return np.concatenate(out)

    def zs(f, key):
        raw, th = mo['lk'].profile(f.astype(np.float32))
        raw = mo['lk'].last_copula if mo['stat'] == 'copula' else raw
        z = dict(L=(f[:, 0] - mu) / sd, O=mo['std'](raw, th), Sd=base[f'{key}_Sd'], C=base[f'{key}_C'])
        z['Sd+C'] = 0.5 * (z['Sd'] + z['C'])
        z['mean:L+Sd+C'] = (z['L'] + z['Sd'] + z['C']) / 3
        z['mean:O+Sd+C'] = (z['O'] + z['Sd'] + z['C']) / 3
        z['mean:L+O+Sd+C'] = (z['L'] + z['O'] + z['Sd'] + z['C']) / 4
        return z

    t0 = time.time()
    gen.manual_seed(777)
    zi = zs(feats(lambda full, l: D.add_noise(full, m, gen)), 'ind')
    res = dict(trained=st['fams'], normal_q999={k: round(float(np.quantile(v, .999)), 3) for k, v in zi.items()})
    for i, fam in enumerate(ALL_FAMILIES):
        if f'{fam}_Sd' not in base:
            continue
        gen.manual_seed(5000 + i)
        zf = zs(feats(lambda full, l: lib.make(fam, full, l, gen)[0]), fam)
        res[fam] = {k: round(phase2_score(zi[k], zf[k]), 4) for k in zi}
        print(fam, 'trained' if fam in st['fams'] else 'HELD-OUT', res[fam], f'{time.time()-t0:.0f}s', flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
