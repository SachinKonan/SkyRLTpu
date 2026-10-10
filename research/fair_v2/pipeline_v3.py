"""v3 pipeline: map-level split (all 256 baryon models in the fit set).

  --stage train : train the CNN on fit3 normal maps (fresh noise every step)
  --stage embed : for fit3 (2 views), calib3 (2 views), audit3 (1 view),
                  synthetic shifts of audit3, and the public test, compute the
                  summary statistics AND the CNN embedding on the SAME noisy
                  map. Writes OUT/stats/<name>.npz and OUT/cnn/<name>.npz.

The public test is only per-map inference input.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data as D
from cnn import Net
from stats import Stats


def load_fit_gpu(train, idx, dev):
    x = torch.empty((len(idx), train.shape[-1]), dtype=torch.float16, device=dev)
    for lo in range(0, len(idx), 2048):
        sl = idx[lo:lo + 2048]
        x[lo:lo + len(sl)] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev)
    return x


def train_cnn(args, dev, train, lab, m):
    idx = D.index(args.fit_part)
    t0 = time.time()
    xfit = load_fit_gpu(train, idx, dev)
    theta = lab[idx[:, 0], idx[:, 1]]
    t_mean, t_std = theta.mean(0), theta.std(0)
    tfit = torch.as_tensor((theta - t_mean) / t_std, dtype=torch.float32, device=dev)
    print(f'fit maps {len(idx)} loaded in {time.time()-t0:.0f}s', flush=True)
    torch.manual_seed(args.seed)
    net = Net(widths=tuple(args.widths)).to(dev).to(memory_format=torch.channels_last)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    steps = args.epochs * (len(idx) // args.batch)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    for ep in range(args.epochs):
        perm = torch.randperm(len(idx), generator=gen, device=dev)
        tl, tn = 0.0, 0
        net.train()
        for lo in range(0, len(perm) - args.batch + 1, args.batch):
            rows = perm[lo:lo + args.batch]
            full = torch.zeros((len(rows), 1424, 176), device=dev)
            full[:, m > 0] = xfit[rows].float()
            x = D.add_noise(full, m, gen)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred, _ = net(x, m)
            loss = F.mse_loss(pred.float(), tfit[rows])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item() * len(rows); tn += len(rows)
        print(f'epoch {ep} train mse {tl/tn:.4f} {time.time()-t0:.0f}s', flush=True)
        torch.save(dict(net=net.state_dict(), t_mean=t_mean, t_std=t_std, widths=args.widths), args.out / 'net.pt')


@torch.no_grad()
def embed(args, dev, train, lab, m, mask):
    stats = Stats(mask, dev, edges=dict(np.load(args.edges)))
    nets = []
    for ck in args.cnn:
        st = torch.load(ck, map_location=dev, weights_only=False)
        net = Net(widths=tuple(st.get('widths', (32, 64, 128, 192, 256, 256)))).to(dev).to(memory_format=torch.channels_last)
        net.load_state_dict(st['net']); net.eval(); nets.append(net)
    gen = torch.Generator(device=dev)
    (args.out / 'stats').mkdir(parents=True, exist_ok=True)
    for k in range(len(nets)):
        (args.out / f'cnn{k}').mkdir(parents=True, exist_ok=True)

    def features(x):
        s = stats(x)
        outs = []
        for net in nets:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred, e = net(x, m)
            outs.append(torch.cat([e.float(), pred.float()], 1).cpu().numpy())
        return s.cpu().numpy(), outs

    def save(name, s, outs, idx=None):
        extra = {} if idx is None else dict(index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        np.savez(args.out / 'stats' / f'{name}.npz', feats=s, **extra)
        for k, o in enumerate(outs):
            np.savez(args.out / f'cnn{k}' / f'{name}.npz', feats=o, **extra)

    def run(name, part, views, seed, shift=None):
        if (args.out / 'stats' / f'{name}.npz').exists():
            return
        t = time.time()
        idx = D.index(part)
        S, O = [], [[] for _ in nets]
        for v in range(views):
            gen.manual_seed(seed * 1000 + v)
            ps, po = [], [[] for _ in nets]
            for lo in range(0, len(idx), args.batch):
                sl = idx[lo:lo + args.batch]
                full = torch.zeros((len(sl), 1424, 176), device=dev)
                full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
                x = (D.noisy_shift(shift, full, m, gen, lab[sl[:, 0], sl[:, 1]]) if shift
                     else D.add_noise(full, m, gen))
                s, outs = features(x)
                ps.append(s)
                for k, o in enumerate(outs):
                    po[k].append(o)
            S.append(np.concatenate(ps))
            for k in range(len(nets)):
                O[k].append(np.concatenate(po[k]))
        save(name, np.stack(S, 1).astype(np.float32), [np.stack(o, 1).astype(np.float32) for o in O], idx)
        print(f'{name}: {len(idx)}x{views} in {time.time()-t:.0f}s', flush=True)

    run('fit', 'fit3', 2, 31)
    run('calib', 'calib3', 2, 32)
    run('audit', 'audit3', 1, 33)
    for i, s in enumerate(D.SHIFTS):
        run('shift_' + s, 'audit3', 1, 300 + i, shift=s)
    if not (args.out / 'stats' / 'test.npz').exists():
        t = time.time()
        test = np.load(D.TEST, mmap_mode='r')
        ps, po = [], [[] for _ in nets]
        for lo in range(0, len(test), args.batch):
            n = min(args.batch, len(test) - lo)
            # Fixed batch shape (zero padding) so a map's features never depend on others.
            full = torch.zeros((args.batch, 1424, 176), device=dev)
            full[:n, m > 0] = torch.as_tensor(np.asarray(test[lo:lo + n]), device=dev).float()
            s, outs = features(full)
            ps.append(s[:n])
            for k, o in enumerate(outs):
                po[k].append(o[:n])
        save('test', np.concatenate(ps)[:, None].astype(np.float32),
             [np.concatenate(o)[:, None].astype(np.float32) for o in po])
        print(f'test in {time.time()-t:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['train', 'embed'], required=True)
    ap.add_argument('--out', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3'))
    ap.add_argument('--fit-part', default='fit3')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--widths', type=int, nargs='+', default=[32, 64, 128, 192, 256, 256])
    ap.add_argument('--cnn', nargs='*', default=[])
    ap.add_argument('--edges', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/feats-v2/edges.npz')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    mask = D.mask(); m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    lab = D.labels()
    train = np.load(D.TRAIN, mmap_mode='r')
    if args.stage == 'train':
        train_cnn(args, dev, train, lab, m)
    else:
        if args.batch < 64:
            args.batch = 64
        embed(args, dev, train, lab, m, mask)


if __name__ == '__main__':
    main()
