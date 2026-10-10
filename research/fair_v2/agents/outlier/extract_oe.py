"""Write the `oe` feature source in the exact v3 format (bit-identical noisy maps).

Copies the pipeline_v3.embed loop: fit3 2 views seed 31, calib3 2 views seed 32,
audit3 1 view seed 33, shifts 1 view seed 300+i via data.noisy_shift, batches of
64 with noise drawn per batch from gen.manual_seed(seed*1000+v); test in fixed
batches of 64 with zero padding; eval mode. Must run on della-gh (GH200) like
the existing `stats`/`cnn0` sources. Resumable: finished files are skipped.

feats (N, V, 1+nf+5+256) = [ood_logit(1), family-head logits(nf = 1 + head size),
                     parameter predictions(5, normalized), pooled embedding(256)].

--verify CKPT: recompute cnn0 features for the first audit batch with CKPT and
compare with v3/cnn0/audit.npz (checks noise bit-identity).
"""
import argparse
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
from train_oe import load_net, setup

BATCH = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/oe')
    ap.add_argument('--verify', default='')
    ap.add_argument('--budget-min', type=float, default=8.0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev, mask, m = setup()
    lab = D.labels()
    train = np.load(D.TRAIN, mmap_mode='r')
    net, st = load_net(args.ckpt, dev)
    gen = torch.Generator(device=dev)
    t_start = time.time()

    @torch.no_grad()
    def features(x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, logit, flog, e = net(x, m)
        return torch.cat([logit.float()[:, None], flog.float(), pred.float(), e.float()], 1).cpu().numpy()

    if args.verify:
        cst = torch.load(args.verify, map_location=dev, weights_only=False)
        cnet = Net(widths=tuple(cst.get('widths', (32, 64, 128, 192, 256, 256)))).to(dev).to(memory_format=torch.channels_last)
        cnet.load_state_dict(cst['net']); cnet.eval()
        idx = D.index('audit3')
        gen.manual_seed(33 * 1000)
        ref = np.load('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/cnn0/audit.npz')['feats'][:, 0]
        for b in range(2):
            sl = idx[b * BATCH:(b + 1) * BATCH]
            full = torch.zeros((len(sl), 1424, 176), device=dev)
            full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
            x = D.add_noise(full, m, gen)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                pred, e = cnet(x, m)
            f = torch.cat([e.float(), pred.float()], 1).cpu().numpy()
            r = ref[b * BATCH:(b + 1) * BATCH]
            print(f'verify batch {b}: max|diff| {np.abs(f - r).max():.3e} (ref scale {np.abs(r).mean():.3e})', flush=True)
        return

    def run(name, part, views, seed, shift=None):
        if (out / f'{name}.npz').exists():
            return
        if time.time() - t_start > args.budget_min * 60:
            print('budget reached before', name, flush=True)
            sys.exit(3)
        t = time.time()
        idx = D.index(part)
        O = []
        for v in range(views):
            gen.manual_seed(seed * 1000 + v)
            po = []
            for lo in range(0, len(idx), BATCH):
                sl = idx[lo:lo + BATCH]
                full = torch.zeros((len(sl), 1424, 176), device=dev)
                full[:, m > 0] = torch.as_tensor(np.asarray(train[sl[:, 0], sl[:, 1]]), device=dev).float()
                x = (D.noisy_shift(shift, full, m, gen, lab[sl[:, 0], sl[:, 1]]) if shift
                     else D.add_noise(full, m, gen))
                po.append(features(x))
            O.append(np.concatenate(po))
        tmp = out / f'{name}.tmp.npz'
        np.savez(tmp, feats=np.stack(O, 1).astype(np.float32), index=idx, labels=lab[idx[:, 0], idx[:, 1]])
        tmp.replace(out / f'{name}.npz')
        print(f'{name}: {len(idx)}x{views} in {time.time()-t:.0f}s', flush=True)

    run('fit', 'fit3', 2, 31)
    run('calib', 'calib3', 2, 32)
    run('audit', 'audit3', 1, 33)
    for i, s in enumerate(D.SHIFTS):
        run('shift_' + s, 'audit3', 1, 300 + i, shift=s)
    if not (out / 'test.npz').exists():
        t = time.time()
        test = np.load(D.TEST, mmap_mode='r')
        po = []
        for lo in range(0, len(test), BATCH):
            n = min(BATCH, len(test) - lo)
            full = torch.zeros((BATCH, 1424, 176), device=dev)
            full[:n, m > 0] = torch.as_tensor(np.asarray(test[lo:lo + n]), device=dev).float()
            po.append(features(full)[:n])
        tmp = out / 'test.tmp.npz'
        np.savez(tmp, feats=np.concatenate(po)[:, None].astype(np.float32))
        tmp.replace(out / 'test.npz')
        print(f'test in {time.time()-t:.0f}s', flush=True)
    nfh = net.fam[1].out_features
    fams = __import__('gens').ALL_FAMILIES
    (out / 'SOURCE.txt').write_text(
        f'outlier-exposure CNN features from {args.ckpt} (step {st["step"]}, trained families {st["fams"]}); '
        f'columns: 0 ood_logit, 1..{nfh} family-head logits (InD, then {fams[:nfh - 1]}), '
        f'{nfh + 1}..{nfh + 5} normalized param preds, {nfh + 6}..{nfh + 261} pooled embedding\n')
    print('done', flush=True)


if __name__ == '__main__':
    main()
