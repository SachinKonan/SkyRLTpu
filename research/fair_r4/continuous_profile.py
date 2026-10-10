"""Bounded per-map inference of physical conditioning parameters in frozen flows."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import jax
import jax.numpy as j
from flax import serialization

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'fair_r2'))
from flow import AffineFlow


def project_polygon(c, vertices, equations):
    """Exact Euclidean projection onto a convex 2D polygon, within FP error."""
    inside = j.all(c @ equations[:, :2].T + equations[:, 2] <= 1e-6, axis=-1)
    a = vertices; edge = j.roll(vertices, -1, axis=0) - a
    t = j.sum((c[:, None, :] - a[None]) * edge[None], axis=-1) / j.sum(edge * edge, axis=-1)
    point = a[None] + j.clip(t, 0, 1)[..., None] * edge[None]
    distance = j.sum(j.square(c[:, None, :] - point), axis=-1)
    index = j.argmin(distance, axis=1)
    closest = j.take_along_axis(point, index[:, None, None], axis=1)[:, 0, :]
    return j.where(inside[:, None], c, closest)


class Profile:
    def __init__(self, run, steps=128, starts=4, step_size=.04, batch=32):
        run = Path(run); self.cfg = json.loads((run / 'config.json').read_text())
        assert self.cfg['condition'] == 'cosmology' and self.cfg['transform'] == 'asinh'
        with np.load(run / 'normalization.npz', allow_pickle=False) as f:
            self.norm = {k: f[k] for k in f.files}
        n = self.norm; self.grid = n['cosmology_grid']
        standardized = (self.grid - n['condition_mean']) / n['condition_std']
        hull = ConvexHull(standardized)
        self.vertices = standardized[hull.vertices]
        self.equations = hull.equations
        params = jax.tree.map(j.asarray, serialization.msgpack_restore((run / 'best.msgpack').read_bytes()))
        net = AffineFlow(self.cfg['dim'], self.cfg['layers'], self.cfg['width'], self.cfg['scale_bound'])
        self.steps, self.starts, self.step_size, self.batch = steps, starts, step_size, batch
        grid = j.asarray(standardized, j.float32)
        vertices = j.asarray(self.vertices, j.float32); equations = j.asarray(self.equations, j.float32)

        def energy(x, c): return net.apply({'params': params}, x, c)[1]
        self.energy = jax.jit(energy)
        self.project = jax.jit(lambda c: project_polygon(c, vertices, equations))

        @jax.jit
        def infer(x):
            count = len(x)
            eg = energy(j.repeat(x, len(grid), axis=0), j.tile(grid, (count, 1))).reshape(count, len(grid))
            negative, indices = jax.lax.top_k(-eg, starts)
            c = grid[indices].reshape(-1, 2); xx = j.repeat(x, starts, axis=0)
            initial = (-negative).reshape(-1)
            state = (c, j.zeros_like(c), j.zeros_like(c), initial, c)
            grad = jax.grad(lambda cc: energy(xx, cc).sum())

            def body(state, step):
                c, m, v, best, best_c = state
                g = grad(c)
                g = g / j.maximum(j.linalg.norm(g, axis=-1, keepdims=True) / 100, 1)
                m = .9*m + .1*g; v = .999*v + .001*g*g
                mh = m/(1-.9**(step+1)); vh = v/(1-.999**(step+1))
                rate = step_size*(.2 + .8*.5*(1+j.cos(j.pi*step/steps)))
                c = project_polygon(c-rate*mh/(j.sqrt(vh)+1e-8), vertices, equations)
                e = energy(xx, c); improved = e < best
                best_c = j.where(improved[:, None], c, best_c); best = j.minimum(best, e)
                return (c, m, v, best, best_c), j.min(best.reshape(count, starts), axis=1)

            state, history = jax.lax.scan(body, state, j.arange(steps, dtype=j.float32))
            best = state[3].reshape(count, starts); best_c = state[4].reshape(count, starts, 2)
            chosen = j.argmin(best, axis=1)
            c = j.take_along_axis(best_c, chosen[:, None, None], axis=1)[:, 0]
            return dict(grid_energy=eg.min(1), continuous_energy=best.min(1),
                        grid_theta=grid[indices[:, 0]], continuous_theta=c,
                        halfway_energy=history[steps//2-1], grid_argmin=indices[:, 0])
        self._infer = infer

    def features(self, raw):
        n = self.norm
        value = np.arcsinh((np.asarray(raw, np.float64)-n['target_offset'])/n['target_scale'])
        return ((value-n['mean']) @ n['whiten']).astype(np.float32)

    def __call__(self, raw, true_theta=None):
        x = self.features(raw); result = {}
        for lo in range(0, len(x), self.batch):
            b = x[lo:lo+self.batch]; count = len(b)
            if count < self.batch: b = np.pad(b, ((0, self.batch-count),(0,0)))
            values = {k:np.asarray(v)[:count] for k,v in self._infer(j.asarray(b)).items()}
            if true_theta is not None:
                c = (np.asarray(true_theta[lo:lo+count])-self.norm['condition_mean'])/self.norm['condition_std']
                values['true_energy'] = np.asarray(self.energy(j.asarray(b[:count]),j.asarray(c,j.float32)))
            for name, value in values.items(): result.setdefault(name, []).append(value)
        return {k:np.concatenate(v) for k,v in result.items()}

    def physical_theta(self, standardized):
        return standardized*self.norm['condition_std']+self.norm['condition_mean']

    def dense_reference(self, raw, resolution=51):
        lo, hi = self.vertices.min(0), self.vertices.max(0)
        c = np.stack(np.meshgrid(*(np.linspace(a,b,resolution) for a,b in zip(lo,hi))),axis=-1).reshape(-1,2)
        c = c[np.all(c @ self.equations[:,:2].T+self.equations[:,2] <= 1e-9,axis=1)]
        original = (self.grid-self.norm['condition_mean'])/self.norm['condition_std']
        c = np.concatenate([c,original]); x = self.features(raw); values=[]
        for b in x:
            e = np.asarray(self.energy(j.asarray(np.repeat(b[None],len(c),axis=0)),j.asarray(c,j.float32)))
            values.append(float(e.min()))
        return np.asarray(values), len(c)
