"""Low-FPR fine-tuning of the OE OoD head on frozen run_all14 embeddings (CPU).

  python ft/train_head.py --lam 3 --mu 1e-3 --tag l3

Head: s(e) = w . silu(e) + b, initialized from run_all14's ood head. Loss = balanced BCE (positives = synthetic
maps from the run's 14 generator families, normals = fit3 maps) + lam * a ranking loss that only looks at the
normals whose current upper-tail rank u_j lies in [0.001, 0.05] (the band the leaderboard metric integrates),
weighted ~ 1/u_j: mean_j,p w_j softplus((s(n_j) - s(p)) / tau), + mu ||w - w0||^2. Ranks are refreshed every
--refresh steps over the whole normal bank. Validation: phase2 metric (mean TPR at 100 log-spaced FPR in
[0.001, 0.05]) of held-out positives (pos_900) vs calib3 normals (never seen by the backbone or the head).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

POOL = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier/ft_pool')
V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
CK = '/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier/run_all14/ckpt.pt'
FPRS = np.logspace(np.log10(0.001), np.log10(0.05), 100)


def phase2(normal, anom):
    t = np.quantile(normal, 1 - FPRS)
    return float(np.mean([(anom > x).mean() for x in t]))


def silu(e):
    e = torch.as_tensor(np.asarray(e), dtype=torch.float32)
    return e * torch.sigmoid(e)


def cached(name):
    f = np.load(V3 / 'oe' / f'{name}.npz')['feats']
    return f.reshape(-1, f.shape[-1])[:, 23:279]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, default=3.0)
    ap.add_argument('--mu', type=float, default=1e-3)
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--refresh', type=int, default=50)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--surr', default='softplus', choices=['softplus', 'sigmoid'])
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(16)
    st = torch.load(CK, map_location='cpu', weights_only=False)
    w0 = st['net']['ood.1.weight'].float()[0].clone()
    b0 = st['net']['ood.1.bias'].float().clone()
    neg = torch.cat([silu(cached('fit'))] + [silu(np.load(p)['e']) for p in sorted(POOL.glob('neg_1.npz'))])
    pos = torch.cat([silu(np.load(p)['e']) for p in sorted(POOL.glob('pos_1.npz'))])
    vpos = silu(np.load(POOL / 'pos_900.npz')['e'])
    vneg = silu(cached('calib'))
    print(f'train normals {len(neg)} positives {len(pos)}; val positives {len(vpos)} calib normals {len(vneg)}', flush=True)
    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = torch.optim.Adam([w, b], lr=args.lr)

    def score(x):
        return x @ w + b

    def val():
        with torch.no_grad():
            return phase2(score(vneg).numpy(), score(vpos).numpy())

    hist = [dict(step=0, val=val())]
    print(f'step 0 val {hist[0]["val"]:.4f} (original head)', flush=True)
    best = (hist[0]['val'], w.detach().clone(), b.detach().clone(), 0)
    nn_, npos = len(neg), len(pos)
    for step in range(1, args.steps + 1):
        if step == 1 or step % args.refresh == 0:
            with torch.no_grad():
                sn = score(neg)
                order = torch.argsort(sn, descending=True)
                u = torch.empty(nn_)
                u[order] = (torch.arange(nn_, dtype=torch.float32) + 1) / nn_
                band = torch.nonzero((u >= 0.001) & (u <= 0.05))[:, 0]
                bw = 1.0 / u[band]
                bw = bw / bw.sum()
        pi = torch.randint(0, npos, (4096,))
        ni = torch.randint(0, nn_, (4096,))
        sp, sn_b = score(pos[pi]), score(neg[ni])
        bce = 0.5 * (F.binary_cross_entropy_with_logits(sp, torch.ones_like(sp))
                     + F.binary_cross_entropy_with_logits(sn_b, torch.zeros_like(sn_b)))
        loss = bce + args.mu * ((w - w0) ** 2).sum()
        if args.lam > 0:
            j = band[torch.multinomial(bw, 1024, replacement=True)]
            snj = score(neg[j])
            pp = score(pos[pi[:1024]])
            d = (snj[:, None] - pp[None, :]) / args.tau
            # sigmoid = smooth misordering indicator: hopeless and easy positives saturate (no gradient)
            rank = torch.sigmoid(d).mean() if args.surr == 'sigmoid' else F.softplus(d).mean() * args.tau
            loss = loss + args.lam * rank
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            v = val()
            hist.append(dict(step=step, val=v, loss=float(loss)))
            print(f'step {step} loss {float(loss):.4f} val {v:.4f}', flush=True)
            if v > best[0]:
                best = (v, w.detach().clone(), b.detach().clone(), step)
    out = POOL / f'head_{args.tag}.pt'
    torch.save(dict(w=best[1], b=best[2], step=best[3], val=best[0], val0=hist[0]['val'], args=vars(args), hist=hist), out)
    print(json.dumps(dict(tag=args.tag, best_val=round(best[0], 4), at=best[3], original=round(hist[0]['val'], 4))), flush=True)


if __name__ == '__main__':
    main()
