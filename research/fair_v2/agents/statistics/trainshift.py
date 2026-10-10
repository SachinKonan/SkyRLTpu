"""Synthetic training negatives on a FIT3 subset (every 4th fit map) for directional detectors.
Writes fair-data/agents/statistics/tshift/<src>/<shift>.npz (+ normal.npz with fresh noise), sources:
stats, cnn0 (cnn_b checkpoint == cnn0), stx_{scat,prof,pkcl,ipow,bisp,nois,tail}. Resumable per shift."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import data as D  # noqa: E402
from stats import Stats  # noqa: E402
from cnn import Net  # noqa: E402
from famfeats import Fam, NoiseFam  # noqa: E402
from dshift import apply as dapply, DSHIFTS  # noqa: E402

OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/tshift')
SHIFTS = ['normal', 'bary9.0', 'baryDMO', 'spread0.6_t2', 'concen0.6_t3', 'ia0.2', 'peakcut0.3', 'psfpat',
          'srcclust0.15', 'supp0.15', 'boost0.15', 'bary6.5', 'spread0.3_t2']
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 480


@torch.no_grad()
def main():
    t_start = time.time()
    dev = torch.device('cuda')
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    lab = D.labels(); train = np.load(D.TRAIN, mmap_mode='r')
    stats = Stats(mask, dev, edges=dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2/edges.npz')))
    st = torch.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/cnn_b/net.pt', map_location=dev, weights_only=False)
    net = Net(widths=tuple(st.get('widths', (32, 64, 128, 192, 256, 256)))).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(st['net']); net.eval()
    fam = Fam(mask, dev, dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/edges_fam.npz')))
    nfam = NoiseFam(mask, dev, dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/edges_nois.npz')))
    idx = D.index('fit3')[::4]
    gen = torch.Generator(device=dev)
    for j, name in enumerate(SHIFTS):
        if (OUT / 'stats' / f'{name}.npz').exists():
            continue
        if time.time() - t_start > BUDGET:
            print('budget', flush=True); return
        t0 = time.time()
        gen.manual_seed((700 + j) * 1000)
        acc = {}
        for lo in range(0, len(idx), 64):
            sl = idx[lo:lo + 64]
            full = torch.zeros((len(sl), 1424, 176), device=dev)
            full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
            if name == 'normal':
                x = D.add_noise(full, m, gen)
            elif name in DSHIFTS:
                x = D.add_noise(dapply(full, m, *DSHIFTS[name]), m, gen)
            else:
                x = D.noisy_shift(name, full, m, gen, lab[sl[:, 0], sl[:, 1]])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred, e = net(x, m)
            o = dict(stats=stats(x), cnn0=torch.cat([e.float(), pred.float()], 1))
            o.update({'stx_' + k: v for k, v in fam(x, ('prof', 'pkcl', 'ipow', 'bisp')).items()})
            o.update({'stx_' + k: v for k, v in nfam(x).items()})
            for k, v in o.items():
                acc.setdefault(k, []).append(v.float().cpu().numpy())
        for k, v in acc.items():
            (OUT / k).mkdir(parents=True, exist_ok=True)
            np.savez(OUT / k / f'{name}.npz', feats=np.concatenate(v).astype(np.float32), index=idx,
                     labels=lab[idx[:, 0], idx[:, 1]])
        print(f'{name} {time.time() - t0:.0f}s', flush=True)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
