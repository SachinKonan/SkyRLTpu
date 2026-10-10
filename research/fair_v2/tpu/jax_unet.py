"""JAX/Flax U-Net denoiser mirroring agents/denoiser/unet.py (torch) for TPU training.

One process per TPU chip trains one model on fit3 maps (fresh shape noise every step,
float16-quantized like the test files). Parameters are saved as npz with torch-compatible
layouts (see to_torch.py on della) so feature extraction reuses the torch pipeline.
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


def split_fit3(n_cos=101, n_real=256):
    u = np.random.default_rng(2027).random((n_cos, n_real))
    c, r = np.nonzero(u < 0.8)
    return np.stack([c, r], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/dev/shm/fair/data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--widths', type=int, nargs='+', default=[32, 64, 128, 256, 256])
    ap.add_argument('--epochs', type=float, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bench', type=int, default=0, help='only time this many steps')
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    import optax

    class Block(nn.Module):
        cout: int

        @nn.compact
        def __call__(self, x, train):
            bn = lambda name: nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5, name=name)
            h = nn.Conv(self.cout, (3, 3), padding=((1, 1), (1, 1)), use_bias=False, dtype=jnp.bfloat16, name='c1')(x)
            h = nn.silu(bn('b1')(h))
            h = nn.Conv(self.cout, (3, 3), padding=((1, 1), (1, 1)), use_bias=False, dtype=jnp.bfloat16, name='c2')(h)
            h = bn('b2')(h)
            s = x if x.shape[-1] == self.cout else nn.Conv(self.cout, (1, 1), use_bias=False, dtype=jnp.bfloat16,
                                                            name='skip')(x)
            return nn.silu(h + s)

    class Up(nn.Module):
        """Torch ConvTranspose2d(cin, cout, 2, 2): weight (cin, cout, 2, 2) + bias, no overlap."""
        cin: int
        cout: int

        @nn.compact
        def __call__(self, x):
            k = 1.0 / math.sqrt(self.cout * 4)
            w = self.param('weight', lambda key: jax.random.uniform(key, (self.cin, self.cout, 2, 2), minval=-k, maxval=k))
            b = self.param('bias', lambda key: jax.random.uniform(key, (self.cout,), minval=-k, maxval=k))
            y = jnp.einsum('bhwc,coij->bhiwjo', x.astype(jnp.bfloat16), w.astype(jnp.bfloat16))
            B, h, _, w_, _, o = y.shape
            return y.reshape(B, 2 * h, 2 * w_, o) + b.astype(y.dtype)

    class UNet(nn.Module):
        widths: tuple

        @nn.compact
        def __call__(self, x, m, train):
            w = self.widths
            inp = jnp.stack([x / SCALE, jnp.broadcast_to(m, x.shape)], -1)
            h = Block(w[0], name='inc')(inp, train)
            skips = [h]
            for i in range(1, len(w)):
                h = nn.Conv(w[i], (2, 2), strides=(2, 2), padding='VALID', use_bias=False, dtype=jnp.bfloat16,
                            name=f'down{i - 1}_conv')(h)
                h = nn.silu(nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5,
                                         name=f'down{i - 1}_bn')(h))
                h = Block(w[i], name=f'enc{i - 1}')(h, train)
                skips.append(h)
            h = Block(w[-1], name='mid')(h, train)
            for j, i in enumerate(range(len(w) - 1, 0, -1)):
                u = Up(w[i], w[i - 1], name=f'up{j}')(h)
                h = Block(w[i - 1], name=f'dec{j}')(jnp.concatenate([u.astype(jnp.float32), skips[i - 1].astype(jnp.float32)], -1), train)
            y = nn.Conv(1, (1, 1), dtype=jnp.float32, name='out')(h.astype(jnp.float32))[..., 0] * SCALE
            return y * m

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = Path(args.data)
    train = np.load(data / 'WIDE12H_bin2_2arcmin_kappa_newrealization.npy', mmap_mode='r')
    mask = np.load(data / 'WIDE12H_bin2_2arcmin_mask.npy')
    idx = split_fit3()
    xfit = np.ascontiguousarray(train[idx[:, 0], idx[:, 1]])  # (N, 132019) float16
    flat = np.nonzero(mask.ravel())[0]
    print(f'[{args.seed}] loaded {xfit.shape} in {time.time()-t0:.0f}s devices {jax.devices()}', flush=True)

    m = jnp.asarray(mask, jnp.float32)
    flat_j = jnp.asarray(flat)
    npix = float(mask.sum())
    model = UNet(tuple(args.widths))
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

    @jax.jit
    def step(params, batch_stats, opt_state, rows, key):
        clean = to_full(rows)
        noise = jax.random.normal(key, clean.shape) * NOISE_SIGMA
        x = ((clean + noise) * m).astype(jnp.float16).astype(jnp.float32) * m

        def loss_fn(p):
            y, upd = model.apply({'params': p, 'batch_stats': batch_stats}, x, m, train=True, mutable=['batch_stats'])
            loss = (((y - clean) ** 2) * m).sum() / (rows.shape[0] * npix) / NOISE_SIGMA ** 2
            return loss, upd['batch_stats']
        (loss, new_bs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_bs, opt_state, loss

    rng = np.random.default_rng(args.seed)
    log = []
    done = 0
    tl, tn, te = 0.0, 0, time.time()
    for ep in range(math.ceil(args.epochs)):
        perm = rng.permutation(len(xfit))
        for k in range(steps_per_epoch):
            if done >= total:
                break
            rows = jnp.asarray(xfit[np.sort(perm[k * args.batch:(k + 1) * args.batch])])
            key, sk = jax.random.split(key)
            params, batch_stats, opt_state, loss = step(params, batch_stats, opt_state, rows, sk)
            done += 1
            if done % 200 == 0:
                tl += float(loss); tn += 1
                print(f'[{args.seed}] ep {ep} step {done}/{total} loss {float(loss):.4f} {time.time()-t0:.0f}s', flush=True)
            if args.bench and done == 10:
                jax.block_until_ready(loss); tb = time.time()
            if args.bench and done == args.bench + 10:
                jax.block_until_ready(loss)
                print(f'BENCH step_time {(time.time()-tb)/args.bench:.4f}s steps/epoch {steps_per_epoch}', flush=True)
                return
        from flax.traverse_util import flatten_dict
        flat_p = {k: np.asarray(v, np.float32) for k, v in flatten_dict(jax.device_get(params), sep='/').items()}
        flat_s = {k: np.asarray(v, np.float32) for k, v in flatten_dict(jax.device_get(batch_stats), sep='/').items()}
        np.savez(out / 'params.tmp.npz', **{'p/' + k: v for k, v in flat_p.items()}, **{'s/' + k: v for k, v in flat_s.items()})
        os.replace(out / 'params.tmp.npz', out / 'params.npz')
        log.append(dict(epoch=ep, step=done, loss=tl / max(tn, 1), seconds=time.time() - t0))
        (out / 'log.json').write_text(json.dumps(dict(args=vars(args), log=log)))
        print(f'[{args.seed}] epoch {ep} done loss {tl/max(tn,1):.4f} epoch {time.time()-te:.0f}s total {time.time()-t0:.0f}s', flush=True)
        tl, tn, te = 0.0, 0, time.time()
        if done >= total:
            break
    # reference outputs for conversion checks: 4 fit maps with fixed noise
    ref_rows = jnp.asarray(xfit[:4])
    clean = to_full(ref_rows)
    noise = jax.random.normal(jax.random.PRNGKey(12345), clean.shape) * NOISE_SIGMA
    x = ((clean + noise) * m).astype(jnp.float16).astype(jnp.float32) * m
    y = model.apply({'params': params, 'batch_stats': batch_stats}, x, m, train=False)
    np.savez(out / 'ref.npz', x=np.asarray(x), y=np.asarray(y))
    print(f'[{args.seed}] finished {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
