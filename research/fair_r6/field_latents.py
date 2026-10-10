"""Terminal noise extraction from the unchanged R2 field network."""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
from flax import serialization

sys.path.insert(1, str(Path(__file__).resolve().parents[1]/'fair_r2'))
from field_flow import Velocity


def ode_latent(velocity, x, mask, steps):
    dt = 1./steps
    def vf(x, t): return velocity(x, j.full((len(x),), t))*mask
    def step(i, h):
        v0 = vf(h, i*dt); v1 = vf(h+dt*v0, (i+1)*dt)
        return (h+.5*dt*(v0+v1))*mask
    return jax.lax.fori_loop(0, steps, step, x*mask)


class FrozenNoiseField:
    def __init__(self, run, steps=32, batch=8):
        run = Path(run); cfg = json.loads((run/'config.json').read_text())
        with np.load(run/'normalization.npz', allow_pickle=False) as f:
            self.norm = {k: f[k] for k in f.files}
        self.support = self.norm['support'].astype(bool); self.batch = batch
        mask = j.asarray(self.support, j.float32)
        coverage = j.asarray(self.norm['count']/4*self.support, j.float32)
        params = jax.tree.map(j.asarray, serialization.msgpack_restore((run/'best.msgpack').read_bytes()))
        net = Velocity(cfg['width'])
        velocity = lambda x,t: net.apply({'params': params}, x, t, coverage)
        self.apply = jax.jit(lambda x: ode_latent(velocity, x, mask, steps))

    def __call__(self, observed):
        x = np.asarray(observed)
        if x.ndim != 3 or x.shape[1:] != self.support.shape:
            raise ValueError('Wrong pooled observation shape')
        if not np.isfinite(x[:, self.support]).all():
            raise ValueError('Nonfinite observed pixel')
        if not len(x): return np.empty(x.shape, np.float32)
        result = []
        for lo in range(0,len(x),self.batch):
            b = x[lo:lo+self.batch]; n = len(b)
            if n < self.batch: b = np.pad(b, ((0,self.batch-n),(0,0),(0,0)))
            q = np.where(self.support, b, self.norm['mean'])
            q = (q-self.norm['mean'])/self.norm['scale']*self.support
            result.append(np.asarray(self.apply(j.asarray(q,j.float32)))[:n])
        result = np.concatenate(result)
        if not np.isfinite(result).all(): raise ValueError('Nonfinite terminal latent field')
        return result
