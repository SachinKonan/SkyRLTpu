"""Private diagnostic shifts that target the halo-PROFILE hypothesis (BCM-like, mass conserving), built
from audit3 TRAINING maps only (allowed: synthetic OoD). Not in data.SHIFTS, so stats/cnn0 features are
computed here on the same noisy maps and written to fair-data/agents/statistics/dshift/<src>/shift_<name>.npz.

  spread{a}_t{t}  : excess E = relu(G1*k - t*sd) around high peaks; k -> k - a*G1(E) + a*G3(E)
                    (core mass moved to the outskirts, total mass conserved; BCM-like gas expulsion)
  concen{a}_t{t}  : the opposite (more concentrated halos, DMO-like vs baryons)
Noise: 1 view, gen seed (900+i)*1000, batches of 64 (no relation to the v3 seeds).
"""
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

OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/dshift')
DSHIFTS = {'spread0.3_t2': (0.3, 2.0), 'spread0.6_t2': (0.6, 2.0), 'spread0.6_t3': (0.6, 3.0),
           'concen0.3_t2': (-0.3, 2.0), 'concen0.6_t3': (-0.6, 3.0)}
CNN = sys.argv[1] if len(sys.argv) > 1 else '/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/cnn_b/net.pt'


def apply(clean, m, a, t):
    sm = D._smooth(clean, m, 1.0, 1.0)
    sd = sm[:, m > 0].std(1, keepdim=True)[:, :, None]
    E = torch.relu(sm - t * sd) * m
    return clean + a * (D._smooth(E, m, 3.0, 3.0) - D._smooth(E, m, 1.0, 1.0)) * m


@torch.no_grad()
def main():
    dev = torch.device('cuda')
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    lab = D.labels(); train = np.load(D.TRAIN, mmap_mode='r')
    stats = Stats(mask, dev, edges=dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2/edges.npz')))
    st = torch.load(CNN, map_location=dev, weights_only=False)
    net = Net(widths=tuple(st.get('widths', (32, 64, 128, 192, 256, 256)))).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(st['net']); net.eval()
    fam = Fam(mask, dev, dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/edges_fam.npz')))
    nfam = NoiseFam(mask, dev, dict(np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/edges_nois.npz')))
    idx = D.index('audit3')
    gen = torch.Generator(device=dev)
    # check the CNN checkpoint reproduces cnn0 on audit batch 0 (seed 33)
    gen.manual_seed(33 * 1000)
    sl = idx[:64]
    full = torch.zeros((64, 1424, 176), device=dev)
    full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
    x = D.add_noise(full, m, gen)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred, e = net(x, m)
    ref = np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/cnn0/audit.npz')['feats'][:64, 0]
    got = torch.cat([e.float(), pred.float()], 1).cpu().numpy()
    print('cnn0 check rel err', np.abs(got - ref).max() / np.abs(ref).max(), flush=True)
    for i, (name, (a, t)) in enumerate(DSHIFTS.items()):
        if (OUT / 'stats' / f'shift_{name}.npz').exists():
            continue
        t0 = time.time()
        gen.manual_seed((900 + i) * 1000)
        acc = {}
        for lo in range(0, len(idx), 64):
            sl = idx[lo:lo + 64]
            full = torch.zeros((len(sl), 1424, 176), device=dev)
            full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
            x = D.add_noise(apply(full, m, a, t), m, gen)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred, e = net(x, m)
            o = dict(stats=stats(x), cnn0=torch.cat([e.float(), pred.float()], 1))
            o.update({'stx_' + k: v for k, v in fam(x).items()})
            o.update({'stx_' + k: v for k, v in nfam(x).items()})
            for k, v in o.items():
                acc.setdefault(k, []).append(v.float().cpu().numpy())
        for k, v in acc.items():
            (OUT / k).mkdir(parents=True, exist_ok=True)
            np.savez(OUT / k / f'shift_{name}.npz', feats=np.concatenate(v)[:, None].astype(np.float32),
                     index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        print(f'{name} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
