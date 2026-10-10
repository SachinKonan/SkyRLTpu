"""Per-map independence check: rescore selected test maps ONE AT A TIME from
raw pixels (features + CNN + likelihood profile + standardization) and compare
with the batch-produced test scores."""
import argparse
import json
import pickle

import numpy as np
import torch

import data as D
from cnn import Net
from stats import Stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--models', default='Sd,C')
    ap.add_argument('--cnn', required=True)
    ap.add_argument('--edges', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2/edges.npz')
    ap.add_argument('--indices', default='0,1,17,101,1007,4095,7001,9999')
    args = ap.parse_args()
    dev = torch.device('cuda')
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    stats = Stats(mask, dev, edges=dict(np.load(args.edges)))
    st = torch.load(args.cnn, map_location=dev, weights_only=False)
    net = Net(widths=tuple(st.get('widths', (32, 64, 128, 192, 256, 256)))).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(st['net']); net.eval()
    test = np.load(D.TEST, mmap_mode='r')
    idx = [int(i) for i in args.indices.split(',')]
    feats = {'stats': [], 'cnn0': []}
    with torch.no_grad():
        for i in idx:
            full = torch.zeros((64, 1424, 176), device=dev)
            full[:1, m > 0] = torch.as_tensor(np.asarray(test[i:i + 1]), device=dev).float()
            feats['stats'].append(stats(full)[:1].cpu().numpy())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred, e = net(full, m)
            feats['cnn0'].append(torch.cat([e.float(), pred.float()], 1)[:1].cpu().numpy())
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    from groups import groups
    G, _ = groups()
    out = {}
    for tag in args.models.split(','):
        mdl = pickle.load(open(f'{args.run}/model_{tag}.pkl', 'rb'))
        parts = []
        for item in mdl['spec'].split('|'):
            src, subset = item.split(':')
            f = feats[src]
            cols = np.arange(f.shape[-1]) if subset == 'raw' else np.concatenate([G[g] for g in subset.split('+')])
            parts.append(f[:, cols])
        x = np.concatenate(parts, 1)
        single = []
        for r in range(len(idx)):
            raw, th = mdl['lk'].profile(x[r:r + 1])
            if mdl['stat'] == 'copula':
                raw = mdl['lk'].last_copula
            single.append(mdl['std'](raw, th)[0])
        batch = np.load(f'{args.run}/z_{tag}.npz')['test'][idx]
        out[tag] = dict(max_abs_diff=float(np.max(np.abs(np.array(single) - batch))),
                        single=np.round(single, 4).tolist(), batch=np.round(batch, 4).tolist())
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
