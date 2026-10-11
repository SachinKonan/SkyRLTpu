"""Inference-only per-family conditional likelihoods and the score components built on them.

Family (mirrors model.Likelihood.profile / copula_energy, as used by agents/consistency/fit_families.py):
  x = standardized family columns; profile -2 log p + prior over theta (3 restarts x 150 Adam steps, box clamp);
  returns best theta (normalized) and the copula energy at it.
Components (mirror agents/consistency/export3.py; all parameters frozen from calib3):
  lik  : z = Std(copula energy at the family's best fit, theta)
  cons : z = Std(log(1 + Mahalanobis^2 of the family-estimate deviations), mean estimate); the estimates are
         family profiles, CNN predictions and ridge theta-probes of a feature source
  xsl  : z = Std(cross-scale copula energy differences, mean of small/large estimates)
Every operation is per map (row); profiles run in fixed-size chunks.
"""
import numpy as np
import torch
import torch.nn as nn

DZ_SIGMA = 0.022


class Net(nn.Module):
    def __init__(self, d, width, depth):
        super().__init__()
        layers, k = [], 5
        for _ in range(depth):
            layers += [nn.Linear(k, width), nn.SiLU()]
            k = width
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(width, d)
        self.logvar = nn.Linear(width, d)

    def forward(self, t):
        h = self.body(t)
        return self.mu(h), self.logvar(h).clamp(-12, 6)


class Regressor(nn.Module):
    def __init__(self, d, width=512):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 5))

    def forward(self, s):
        return self.f(s)


class Family:
    def __init__(self, blob, dev):
        self.b = blob
        self.dev = dev
        g = lambda k: blob[k].to(dev).float()
        self.sources = blob['sources']            # list of [source, list of column indices]
        self.f_mean = blob['f_mean'].double().numpy()
        self.f_std = blob['f_std'].double().numpy()
        self.keep = blob['keep'].numpy().astype(bool)
        self.t_lo, self.t_hi = g('t_lo'), g('t_hi')
        self.t_mean, self.t_std = blob['t_mean'].double().numpy(), blob['t_std'].double().numpy()
        self.cosmo, self.hull_r = g('cosmo'), float(blob['hull_r'])
        self.offsets = g('offsets')               # (restarts, 5) frozen restart offsets
        self.net = Net(int(blob['dk']), int(blob['width']), int(blob['depth'])).to(dev)
        self.net.load_state_dict(blob['net']); self.net.eval()
        self.reg = Regressor(int(blob['dk']), int(blob['reg_width'])).to(dev)
        self.reg.load_state_dict(blob['reg']); self.reg.eval()
        for p in list(self.net.parameters()) + list(self.reg.parameters()):
            p.requires_grad_(False)
        self.Rinv, self.Rc_inv, self.cq, self.cg = g('Rinv'), g('Rc_inv'), g('cq'), g('cg')

    def columns(self, feats):
        return np.concatenate([feats[src][:, np.asarray(cols, dtype=np.int64)] for src, cols in self.sources], 1)

    def standardize(self, feats):
        f = self.columns(feats)
        return torch.as_tensor(((f - self.f_mean) / self.f_std)[:, self.keep], dtype=torch.float32, device=self.dev)

    def copula(self, u):
        q, gk = self.cq, self.cg
        K = q.shape[1]
        ut = u.T.contiguous()
        i = torch.searchsorted(q, ut).clamp(1, K - 1)
        lo = torch.gather(q, 1, i - 1); hi = torch.gather(q, 1, i)
        w = ((ut - lo) / (hi - lo).clamp(min=1e-9)).clamp(0, 1)
        g = gk[i - 1] + w * (gk[i] - gk[i - 1])
        slope_hi = (gk[-1] - gk[-11]) / (q[:, -1] - q[:, -11]).clamp(min=1e-6)
        slope_lo = (gk[10] - gk[0]) / (q[:, 10] - q[:, 0]).clamp(min=1e-6)
        above = ut > q[:, -1:]
        below = ut < q[:, :1]
        g = torch.where(above, gk[-1] + slope_hi[:, None] * (ut - q[:, -1:]), g)
        g = torch.where(below, gk[0] + slope_lo[:, None] * (ut - q[:, :1]), g)
        return g.clamp(-8, 8).T

    def copula_energy(self, x, t):
        mu, lv = self.net(t)
        g = self.copula((x - mu) * torch.exp(-0.5 * lv))
        return (g @ self.Rc_inv * g).sum(1)

    def energy(self, x, t):
        mu, lv = self.net(t)
        u = (x - mu) * torch.exp(-0.5 * lv)
        return (u @ self.Rinv * u).sum(1) + lv.sum(1)

    def prior(self, t):
        dz = t[:, 4] * float(self.t_std[4]) + float(self.t_mean[4])
        pen = (dz / DZ_SIGMA) ** 2
        dmin = torch.cdist(t[:, :2], self.cosmo).min(1).values
        return pen + 50.0 * torch.relu(dmin - self.hull_r) ** 2

    def _profile_chunk(self, x, steps=150, lr=0.05):
        lo, hi = self.t_lo, self.t_hi
        with torch.no_grad():
            t0 = self.reg(x).clamp(lo, hi)
        best_l = torch.full((len(x),), float('inf'), device=self.dev)
        best_t = t0.clone()
        for r in range(self.offsets.shape[0]):
            t = (t0 if r == 0 else (t0 + self.offsets[r:r + 1])).clamp(lo, hi)
            t = t.clone().requires_grad_(True)
            opt = torch.optim.Adam([t], lr=lr)
            for _ in range(steps):
                with torch.enable_grad():
                    l = self.energy(x, t) + self.prior(t)
                    opt.zero_grad(set_to_none=True); l.sum().backward()
                opt.step()
                with torch.no_grad():
                    t.clamp_(lo, hi)
            with torch.no_grad():
                l = self.energy(x, t) + self.prior(t)
                better = l < best_l
                best_l = torch.where(better, l, best_l)
                best_t = torch.where(better[:, None], t, best_t)
        best_t = best_t.detach()
        with torch.no_grad():
            cop = self.copula_energy(x, best_t)
        return best_t, cop

    def profile(self, feats, chunk):
        """Returns (x, theta_hat normalized (N,5) float32 tensor, copula energy (N,) float64 numpy)."""
        x = self.standardize(feats)
        N = len(x)
        ts, cs = [], []
        for lo in range(0, N, chunk):
            n = min(chunk, N - lo)
            xc = torch.zeros((chunk, x.shape[1]), device=self.dev)   # fixed shape; zero rows = mean features
            xc[:n] = x[lo:lo + n]
            t, c = self._profile_chunk(xc)
            ts.append(t[:n]); cs.append(c[:n])
        return x, torch.cat(ts), torch.cat(cs).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def copula_energy_at(self, x, t, chunk):
        out = []
        for lo in range(0, len(x), chunk):
            out.append(self.copula_energy(x[lo:lo + chunk], t[lo:lo + chunk]))
        return torch.cat(out).cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------------- components (numpy, per row)
def ridge_view(r, F):
    """Ridge theta-probe view (agents/consistency/probes.ridge_fit): normalized theta from one feature source."""
    X = np.asarray(F, dtype=np.float64)
    return (((X - r['mu']) / r['sd']) @ r['W'] + r['ym']).astype(np.float32)


def design(theta, t_mean, t_std):
    z = (theta - t_mean) / t_std
    cols = [np.ones(len(z))] + [z[:, i] for i in range(5)] + [z[:, i] * z[:, j] for i in range(5) for j in range(i, 5)]
    return np.stack(cols, 1)


def std_apply(p, raw, th_norm):
    """Standardizer (model.Standardizer.__call__) with theta given in normalized units."""
    tm, ts = p['t_mean'], p['t_std']
    X = design(th_norm * ts + tm, tm, ts)
    return (raw - X @ p['loc']) / np.exp(0.5 * (X @ p['scale'])) / p['norm']


def component_z(name, c, th, cop, preds, xsl_cross=None):
    """c: component params; th: family -> normalized theta (N,5) float32 numpy; cop: family -> copula energy;
    preds: non-family estimate views (N,5) normalized theta: 'pred0'/'pred1' CNN predictions and ridge views
    (e.g. 'pden_b', 'pdirpower'); xsl_cross: dict with cop_small@large, cop_large@small."""
    kind = c['kind']
    if kind == 'lik':
        f = c['family']
        return std_apply(c['std'], cop[f], th[f].astype(np.float32))
    if kind == 'cons':
        A = np.stack([(preds[f] if f in preds else th[f]).astype(np.float32) for f in c['inputs']], 1)
        vec = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
        d = vec - c['mu']
        m2 = np.einsum('ij,jk,ik->i', d, c['Ci'], d)
        return std_apply(c['std'], np.log(m2 + 1.0), A.mean(1))
    if kind == 'xsl':
        raw = (xsl_cross['small@large'] - cop['small']) + (xsl_cross['large@small'] - cop['large'])
        return std_apply(c['std'], raw, 0.5 * (th['small'].astype(np.float32) + th['large'].astype(np.float32)))
    raise ValueError(f'unknown component kind {kind} for {name}')
