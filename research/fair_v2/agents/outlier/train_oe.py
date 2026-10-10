"""Outlier-exposure CNN: InD fit3 maps vs synthetic alternative-physics maps.

  --stage train : resumable training (checkpoint every --ckpt-every steps and at
                  the time budget). Each step: B_in InD fit3 maps with fresh noise
                  (data.add_noise) + B_ood synthetic maps built from OTHER random
                  fit3 clean maps with a generator family from gens.py, then the
                  identical noise + float16 quantization.
                  Loss = BCE(InD vs OoD) + w_fam * CE(family) + w_reg * MSE(params, InD only).
  --stage eval  : audit3 maps only (never trained on): InD logits vs every
                  generator family (random amplitudes, fixed seed) and every
                  data.SHIFTS key. Reports phase2_score per family / shift.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data as D
from cnn import Net
from model import phase2_score
from gens import Library, ALL_FAMILIES, HELDOUT_DEFAULT, NFAM_HEAD

WIDTHS = (32, 64, 128, 192, 256, 256)


class OENet(nn.Module):
    def __init__(self, nfam=NFAM_HEAD, widths=WIDTHS):
        super().__init__()
        self.base = Net(widths)
        self.ood = nn.Sequential(nn.SiLU(), nn.Linear(256, 1))
        self.fam = nn.Sequential(nn.SiLU(), nn.Linear(256, nfam + 1))

    def forward(self, x, m):
        e = self.base.embed(x, m)
        return self.base.head(e), self.ood(e)[:, 0], self.fam(e), e


def load_net(path, dev):
    st = torch.load(path, map_location=dev, weights_only=False)
    nfam = st['net']['fam.1.weight'].shape[0] - 1
    net = OENet(nfam).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(st['net'])
    net.eval()
    return net, st


def setup():
    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    mask = D.mask()
    m = torch.as_tensor(mask, device=dev, dtype=torch.float32)
    return dev, mask, m


def train(args):
    dev, mask, m = setup()
    lab = D.labels()
    train_np = np.load(D.TRAIN, mmap_mode='r')
    idx = D.index('fit3')
    t0 = time.time()
    xfit = torch.empty((len(idx), int(mask.sum())), dtype=torch.float16, device=dev)
    for lo in range(0, len(idx), 2048):
        sl = idx[lo:lo + 2048]
        xfit[lo:lo + len(sl)] = torch.as_tensor(np.asarray(train_np[sl[:, 0], sl[:, 1]]), device=dev)
    theta = lab[idx[:, 0], idx[:, 1]]
    print(f'fit3 maps {len(idx)} loaded in {time.time()-t0:.0f}s', flush=True)
    holdout = [f for f in args.holdout.split(',') if f]
    fams = [f for f in ALL_FAMILIES if f not in holdout]
    wmap = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in args.fam_weights.split(',') if kv}
    fw = torch.tensor([wmap.get(f, 1.0) for f in fams], device=dev)
    print('train families', fams, 'weights', fw.tolist(), 'held out', holdout, flush=True)
    lib = Library(m, dev)
    torch.manual_seed(args.seed)
    net = OENet().to(dev).to(memory_format=torch.channels_last)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck = out / 'ckpt.pt'
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 300

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, args.steps - warm)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    step = 0
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        gen.set_state(st['gen'].cpu() if hasattr(st['gen'], 'cpu') else st['gen'])
        step = st['step']; t_mean, t_std = st['t_mean'], st['t_std']
        print(f'resumed at step {step}', flush=True)
    else:
        t_mean, t_std = theta.mean(0), theta.std(0)
        if args.init:
            ini = torch.load(args.init, map_location=dev, weights_only=False)
            if any(k.startswith('base.') for k in ini['net']):     # warm start from an OE checkpoint
                own = net.state_dict()
                sd = {k: v for k, v in ini['net'].items() if k in own and own[k].shape == v.shape}
                net.load_state_dict(sd, strict=False)
            else:
                net.base.load_state_dict(ini['net'])
            t_mean, t_std = ini['t_mean'], ini['t_std']
            print('initialized backbone from', args.init, flush=True)
    tfit = torch.as_tensor((theta - t_mean) / t_std, dtype=torch.float32, device=dev)

    def save(step):
        tmp = out / 'ckpt.tmp'
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                        step=step, t_mean=t_mean, t_std=t_std, holdout=holdout, fams=fams, args=vars(args)), tmp)
        tmp.replace(ck)

    N = len(idx)
    bi, bo, k = args.batch_in, args.batch_ood, args.fam_per_step
    cb = bo // k
    net.train()
    hist = {f: [] for f in fams}; hist['ind'] = []
    acc = dict(bce=0.0, ce=0.0, mse=0.0, n=0)
    tstart = time.time()
    while step < args.steps:
        rows = torch.randint(0, N, (bi + k * cb,), generator=gen, device=dev)
        full = torch.zeros((len(rows), 1424, 176), device=dev)
        full[:, m > 0] = xfit[rows].float()
        x_in = D.add_noise(full[:bi], m, gen)
        xs, fl = [x_in], [torch.zeros(bi, dtype=torch.long, device=dev)]
        fsel = torch.multinomial(fw, k, replacement=True, generator=gen).tolist()
        r_ood = rows[bi:].cpu().numpy()
        lab_ood = theta[r_ood]
        for j, fi in enumerate(fsel):
            sl = slice(bi + j * cb, bi + (j + 1) * cb)
            xo, _ = lib.make(fams[fi], full[sl], lab_ood[j * cb:(j + 1) * cb], gen)
            xs.append(xo)
            fl.append(torch.full((cb,), 1 + ALL_FAMILIES.index(fams[fi]), dtype=torch.long, device=dev))
        x = torch.cat(xs)
        famlab = torch.cat(fl)
        y = (famlab > 0).float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, logit, flog, _ = net(x, m)
        bce = F.binary_cross_entropy_with_logits(logit.float(), y)
        ce = F.cross_entropy(flog.float(), famlab)
        mse = F.mse_loss(pred[:bi].float(), tfit[rows[:bi]])
        loss = bce + args.w_fam * ce + args.w_reg * mse
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        acc['bce'] += bce.item(); acc['ce'] += ce.item(); acc['mse'] += mse.item(); acc['n'] += 1
        with torch.no_grad():
            lg = logit.float().detach()
            hist['ind'].append(lg[:bi].mean().item())
            for j, fi in enumerate(fsel):
                hist[fams[fi]].append(lg[bi + j * cb: bi + (j + 1) * cb].mean().item())
        if step % args.log_every == 0:
            n = acc['n']
            fm = ' '.join(f'{f}={np.mean(v[-200:]):+.2f}' for f, v in hist.items() if v)
            print(f'step {step} bce {acc["bce"]/n:.4f} ce {acc["ce"]/n:.4f} mse {acc["mse"]/n:.4f} '
                  f'lr {sched.get_last_lr()[0]:.2e} {time.time()-tstart:.0f}s | mean logit {fm}', flush=True)
            acc = dict(bce=0.0, ce=0.0, mse=0.0, n=0)
        if step % args.ckpt_every == 0:
            save(step)
        if time.time() - tstart > args.budget_min * 60:
            print(f'time budget reached at step {step}', flush=True)
            break
    save(step)
    print(f'saved step {step}', flush=True)


@torch.no_grad()
def evaluate(args):
    dev, mask, m = setup()
    lab = D.labels()
    train_np = np.load(D.TRAIN, mmap_mode='r')
    out = Path(args.out)
    net, st = load_net(out / 'ckpt.pt', dev)
    lib = Library(m, dev)
    idx = D.index('audit3')
    if args.eval_n:
        idx = idx[:args.eval_n]
    labs = lab[idx[:, 0], idx[:, 1]]
    gen = torch.Generator(device=dev)
    bs = 64

    def run(make):
        L, Fm, S = [], [], []
        for lo in range(0, len(idx), bs):
            sl = idx[lo:lo + bs]
            full = torch.zeros((len(sl), 1424, 176), device=dev)
            full[:, m > 0] = torch.as_tensor(np.asarray(train_np[sl[:, 0], sl[:, 1]]), device=dev).float()
            x, s = make(full, labs[lo:lo + bs])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, logit, flog, _ = net(x, m)
            flog = flog.float()
            L.append(logit.float().cpu().numpy())
            Fm.append((torch.logsumexp(flog[:, 1:], 1) - flog[:, 0]).cpu().numpy())  # any-OoD vs InD
            S.append(np.zeros(len(sl)) if s is None else s.float().cpu().numpy())
        return np.concatenate(L), np.concatenate(Fm), np.concatenate(S)

    t0 = time.time()
    gen.manual_seed(777)
    ind, ind_f, _ = run(lambda full, l: (D.add_noise(full, m, gen), None))
    res = dict(step=int(st['step']), holdout=st['holdout'], families={}, shifts={})
    store = dict(ind=ind, ind_f=ind_f)
    for i, fam in enumerate(ALL_FAMILIES):
        gen.manual_seed(5000 + i)
        lo_, fo, s = run(lambda full, l: lib.make(fam, full, l, gen))
        q = np.quantile(s, [1 / 3, 2 / 3])
        terc = [phase2_score(ind, lo_[(s >= a) & (s <= b)]) for a, b in ((0, q[0]), (q[0], q[1]), (q[1], 1))]
        res['families'][fam] = dict(heldout=fam not in st['fams'], logit=round(phase2_score(ind, lo_), 4),
                                    famhead=round(phase2_score(ind_f, fo), 4),
                                    by_strength_tercile=[round(v, 4) for v in terc])
        store['fam_' + fam] = lo_
        print(fam, res['families'][fam], f'{time.time()-t0:.0f}s', flush=True)
    for i, s in enumerate(D.SHIFTS):
        gen.manual_seed(300 + i)
        lo_, fo, _ = run(lambda full, l: (D.noisy_shift(s, full, m, gen, l), None))
        res['shifts'][s] = dict(logit=round(phase2_score(ind, lo_), 4), famhead=round(phase2_score(ind_f, fo), 4))
        store['shift_' + s] = lo_
        print(s, res['shifts'][s], flush=True)
    res['normal_logit_quantiles'] = np.quantile((ind - ind.mean()) / ind.std(), [.5, .9, .99, .999]).round(3).tolist()
    for key in ('families', 'shifts'):
        for sc in ('logit', 'famhead'):
            v = [d[sc] for d in res[key].values()]
            res[f'{key}_mean_{sc}'] = round(float(np.mean(v)), 4)
    ho = [f for f in ALL_FAMILIES if f not in st['fams']]
    if ho:
        res['heldout_mean_logit'] = round(float(np.mean([res['families'][f]['logit'] for f in ho])), 4)
        res['trained_mean_logit'] = round(float(np.mean([res['families'][f]['logit'] for f in ALL_FAMILIES
                                                         if f not in ho])), 4)
    res['heldout_families'] = ho
    tag = f'{st["step"]}_{len(ALL_FAMILIES)}fam'
    (out / f'eval_step{tag}.json').write_text(json.dumps(res, indent=1))
    np.savez(out / f'eval_step{tag}.npz', **store)
    print(json.dumps({k: v for k, v in res.items() if k not in ('families', 'shifts')}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['train', 'eval', 'both'], default='both')
    ap.add_argument('--out', required=True)
    ap.add_argument('--holdout', default=','.join(HELDOUT_DEFAULT))
    ap.add_argument('--init', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/cnn_b/net.pt')
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch-in', type=int, default=24)
    ap.add_argument('--batch-ood', type=int, default=24)
    ap.add_argument('--fam-per-step', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--w-fam', type=float, default=0.3)
    ap.add_argument('--w-reg', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--budget-min', type=float, default=46)
    ap.add_argument('--ckpt-every', type=int, default=500)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--eval-n', type=int, default=0)
    ap.add_argument('--fam-weights', default='')
    args = ap.parse_args()
    if args.stage in ('train', 'both'):
        train(args)
    if args.stage in ('eval', 'both'):
        evaluate(args)


if __name__ == '__main__':
    main()
