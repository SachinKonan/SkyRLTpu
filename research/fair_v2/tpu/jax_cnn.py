"""JAX/Flax CNN parameter regressor mirroring research/fair_v2/cnn.py (torch Net) for TPU training.

One process per TPU chip trains one model on fit3 noisy maps (fresh noise every step, float16
quantized) to predict the 5 normalized parameters. Saves params.npz (+ batch stats, t_mean/t_std)
and ref.npz (pred/emb for 4 fixed maps) for conversion checks (cnn_to_torch.py on della).
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

SCALE = 0.03
NOISE_SIGMA = 0.4 / math.sqrt(2 * 30 * 4)
H, W = 1424, 176


def split_fit3():
    u = np.random.default_rng(2027).random((101, 256))
    c, r = np.nonzero(u < 0.8)
    return np.stack([c, r], 1)


def adaptive_pool_matrix(n_in, n_out):
    """Torch adaptive_avg_pool1d as a (n_out, n_in) averaging matrix."""
    P = np.zeros((n_out, n_in))
    for i in range(n_out):
        s = (i * n_in) // n_out
        e = -((-(i + 1) * n_in) // n_out)
        P[i, s:e] = 1.0 / (e - s)
    return P


def out_size(n, strides):
    for s in strides:
        n = (n + 2 - 3) // s + 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/dev/shm/fair/data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--widths', type=int, nargs='+', default=[32, 64, 128, 192, 256, 256])
    ap.add_argument('--epochs', type=float, default=20)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--bench', type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax
    from flax.traverse_util import flatten_dict

    widths = tuple(args.widths)
    strides = [1] + [2] * (len(widths) - 1)
    hh, ww = out_size(H, strides), out_size(W, strides)
    mask = np.load(Path(args.data) / 'WIDE12H_bin2_2arcmin_mask.npy')
    mm = adaptive_pool_matrix(H, hh) @ mask.astype(np.float64) @ adaptive_pool_matrix(W, ww).T
    pool_w = jnp.asarray(mm / mm.sum(), jnp.float32)

    def bn(name, train):
        return nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5, name=name)

    class Block(nn.Module):
        cout: int
        stride: int

        @nn.compact
        def __call__(self, x, train):
            s = self.stride
            h = nn.Conv(self.cout, (3, 3), strides=(s, s), padding=((1, 1), (1, 1)), use_bias=False,
                        dtype=jnp.bfloat16, name='c1')(x)
            h = nn.silu(bn('b1', train)(h))
            h = nn.Conv(self.cout, (3, 3), padding=((1, 1), (1, 1)), use_bias=False, dtype=jnp.bfloat16, name='c2')(h)
            h = bn('b2', train)(h)
            if x.shape[-1] == self.cout and s == 1:
                sk = x
            else:
                sk = nn.Conv(self.cout, (1, 1), strides=(s, s), padding='VALID', use_bias=False, dtype=jnp.bfloat16,
                             name='skip_conv')(x)
                sk = bn('skip_bn', train)(sk)
            return nn.silu(h + sk)

    class Net(nn.Module):
        @nn.compact
        def __call__(self, x, m, train):
            inp = jnp.stack([x / SCALE, jnp.broadcast_to(m, x.shape)], -1)
            h = nn.Conv(widths[0], (3, 3), padding=((1, 1), (1, 1)), use_bias=False, dtype=jnp.bfloat16, name='stem_conv')(inp)
            h = nn.silu(bn('stem_bn', train)(h))
            k = 0
            for i, w in enumerate(widths):
                h = Block(w, 1 if i == 0 else 2, name=f'blk{k}')(h, train); k += 1
                h = Block(w, 1, name=f'blk{k}')(h, train); k += 1
            h = h.astype(jnp.float32)
            mean = jnp.einsum('bhwc,hw->bc', h, pool_w)
            sd = jnp.sqrt(jnp.clip(jnp.einsum('bhwc,hw->bc', (h - mean[:, None, None, :]) ** 2, pool_w), 1e-8))
            e = nn.Dense(256, name='emb0')(jnp.concatenate([mean, sd], -1))
            e = nn.Dense(256, name='emb2')(nn.silu(e))
            return nn.Dense(5, name='head1')(nn.silu(e)), e

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = Path(args.data)
    train = np.load(data / 'WIDE12H_bin2_2arcmin_kappa_newrealization.npy', mmap_mode='r')
    lab = np.load(data / 'label_newrealization.npy')
    idx = split_fit3()
    xfit = np.ascontiguousarray(train[idx[:, 0], idx[:, 1]])
    theta = lab[idx[:, 0], idx[:, 1]]
    t_mean, t_std = theta.mean(0), theta.std(0)
    tfit = ((theta - t_mean) / t_std).astype(np.float32)
    flat = np.nonzero(mask.ravel())[0]
    print(f'[{args.seed}] loaded {xfit.shape} in {time.time()-t0:.0f}s devices {jax.devices()} pooled {hh}x{ww}', flush=True)
    m = jnp.asarray(mask, jnp.float32)
    flat_j = jnp.asarray(flat)
    model = Net()
    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)
    variables = model.init(k_init, jnp.zeros((2, H, W)), m, train=False)
    params, batch_stats = variables['params'], variables['batch_stats']
    steps_per_epoch = len(xfit) // args.batch
    total = int(args.epochs * steps_per_epoch)
    sched = optax.cosine_onecycle_schedule(total, args.lr, pct_start=0.05, div_factor=25.0, final_div_factor=1e4)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    opt_state = tx.init(params)

    def to_full(rows):
        full = jnp.zeros((rows.shape[0], H * W), jnp.float32).at[:, flat_j].set(rows.astype(jnp.float32))
        return full.reshape(rows.shape[0], H, W)

    def noisy(rows, key):
        clean = to_full(rows)
        x = clean + jax.random.normal(key, clean.shape) * NOISE_SIGMA
        return (x * m).astype(jnp.float16).astype(jnp.float32) * m

    @jax.jit
    def step(params, batch_stats, opt_state, rows, t, key):
        x = noisy(rows, key)

        def loss_fn(p):
            (pred, _), upd = model.apply({'params': p, 'batch_stats': batch_stats}, x, m, train=True,
                                         mutable=['batch_stats'])
            return ((pred - t) ** 2).mean(), upd['batch_stats']
        (loss, new_bs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_bs, opt_state, loss

    rng = np.random.default_rng(args.seed)
    log, done, tl, tn, te = [], 0, 0.0, 0, time.time()
    for ep in range(math.ceil(args.epochs)):
        perm = rng.permutation(len(xfit))
        for k in range(steps_per_epoch):
            if done >= total:
                break
            sel = np.sort(perm[k * args.batch:(k + 1) * args.batch])
            key, sk = jax.random.split(key)
            params, batch_stats, opt_state, loss = step(params, batch_stats, opt_state, jnp.asarray(xfit[sel]),
                                                        jnp.asarray(tfit[sel]), sk)
            done += 1
            if done % 100 == 0:
                tl += float(loss); tn += 1
            if args.bench and done == 10:
                jax.block_until_ready(loss); tb = time.time()
            if args.bench and done == args.bench + 10:
                jax.block_until_ready(loss)
                print(f'BENCH step_time {(time.time()-tb)/args.bench:.4f}s steps/epoch {steps_per_epoch}', flush=True)
                return
        flat_p = {k: np.asarray(v, np.float32) for k, v in flatten_dict(jax.device_get(params), sep='/').items()}
        flat_s = {k: np.asarray(v, np.float32) for k, v in flatten_dict(jax.device_get(batch_stats), sep='/').items()}
        np.savez(out / 'params.tmp.npz', **{'p/' + k: v for k, v in flat_p.items()},
                 **{'s/' + k: v for k, v in flat_s.items()}, t_mean=t_mean, t_std=t_std)
        os.replace(out / 'params.tmp.npz', out / 'params.npz')
        log.append(dict(epoch=ep, step=done, loss=tl / max(tn, 1), seconds=time.time() - t0))
        (out / 'log.json').write_text(json.dumps(dict(args=vars(args), log=log)))
        print(f'[{args.seed}] epoch {ep} done mse {tl/max(tn,1):.4f} epoch {time.time()-te:.0f}s total {time.time()-t0:.0f}s', flush=True)
        tl, tn, te = 0.0, 0, time.time()
        if done >= total:
            break
    x = noisy(jnp.asarray(xfit[:4]), jax.random.PRNGKey(12345))
    pred, e = model.apply({'params': params, 'batch_stats': batch_stats}, x, m, train=False)
    np.savez(out / 'ref.npz', x=np.asarray(x), pred=np.asarray(pred), emb=np.asarray(e))
    print(f'[{args.seed}] finished {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
