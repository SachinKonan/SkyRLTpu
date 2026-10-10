"""How do the CURRENT detectors (fit_v3 run r1: Sd=stats:dog, C=cnn0:raw, mean:Sd+C)
score the synthetic generator families? Uses the same audit3 maps, seeds and
generator draws as train_oe.py --stage eval (seed 777 InD, 5000+i family i).
Tells us which families are in the blind spot of the current pipeline."""
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
from cnn import Net
from gens import Library, ALL_FAMILIES
from groups import groups
from model import phase2_score
from stats import Stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/runs/r1')
    ap.add_argument('--out', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier/fam_baseline')
    ap.add_argument('--edges', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2/edges.npz')
    ap.add_argument('--cnn', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/cnn_b/net.pt')
    ap.add_argument('--fams', default=','.join(ALL_FAMILIES))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    lab = D.labels(); train = np.load(D.TRAIN, mmap_mode='r')
    stats = Stats(mask, dev, edges=dict(np.load(args.edges)))
    st = torch.load(args.cnn, map_location=dev, weights_only=False)
    net = Net().to(dev).to(memory_format=torch.channels_last); net.load_state_dict(st['net']); net.eval()
    lib = Library(m, dev)
    G, _ = groups()
    idx = D.index('audit3'); labs = lab[idx[:, 0], idx[:, 1]]
    gen = torch.Generator(device=dev)
    models = {t: pickle.load(open(Path(args.run) / f'model_{t}.pkl', 'rb')) for t in ('Sd', 'C')}

    @torch.no_grad()
    def feats(make):
        S, C = [], []
        for lo in range(0, len(idx), 64):
            sl = idx[lo:lo + 64]
            full = torch.zeros((len(sl), 1424, 176), device=dev)
            full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
            x = make(full, labs[lo:lo + 64])
            S.append(stats(x).cpu().numpy())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred, e = net(x, m)
            C.append(torch.cat([e.float(), pred.float()], 1).cpu().numpy())
        return dict(Sd=np.concatenate(S)[:, G['dog']], C=np.concatenate(C))

    def zscore(f):
        z = {}
        for t, mo in models.items():
            raw, th = mo['lk'].profile(f[t].astype(np.float32))
            raw = mo['lk'].last_copula if mo['stat'] == 'copula' else raw
            z[t] = mo['std'](raw, th)
        z['Sd+C'] = 0.5 * (z['Sd'] + z['C'])
        return z

    t0 = time.time()
    gen.manual_seed(777)
    zi = zscore(feats(lambda full, l: D.add_noise(full, m, gen)))
    jp = out / 'fam_baseline.json'
    res = json.loads(jp.read_text()) if jp.exists() else {}
    res['normal_q999'] = {k: round(float(np.quantile(v, .999)), 3) for k, v in zi.items()}
    store = {f'ind_{k}': v for k, v in zi.items()}
    for i, fam in enumerate(ALL_FAMILIES):
        if fam not in args.fams.split(','):
            continue
        gen.manual_seed(5000 + i)
        zf = zscore(feats(lambda full, l: lib.make(fam, full, l, gen)[0]))
        res[fam] = {k: round(phase2_score(zi[k], zf[k]), 4) for k in zi}
        store.update({f'{fam}_{k}': v for k, v in zf.items()})
        print(fam, res[fam], f'{time.time()-t0:.0f}s', flush=True)
        jp.write_text(json.dumps(res, indent=1))
    np.savez(out / f'fam_baseline_{len(store)}.npz', **store)


if __name__ == '__main__':
    main()
