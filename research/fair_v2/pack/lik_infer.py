"""Inference-only conditional-likelihood detector (mirrors model.Likelihood/Standardizer)."""
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


class Detector:
    def __init__(self, blob, dev):
        self.b = blob
        self.dev = dev
        g = lambda k: blob[k].to(dev).float() if torch.is_tensor(blob[k]) else blob[k]
        self.f_mean, self.f_std = blob['f_mean'].double().numpy(), blob['f_std'].double().numpy()
        self.keep = blob['keep'].numpy().astype(bool)
        self.t_mean, self.t_std = blob['t_mean'].double().numpy(), blob['t_std'].double().numpy()
        self.t_lo, self.t_hi = g('t_lo'), g('t_hi')
        self.cosmo, self.hull_r = g('cosmo'), blob['hull_r']
        self.net = Net(blob['dk'], blob['width'], blob['depth']).to(dev)
        self.net.load_state_dict(blob['net']); self.net.eval()
        self.reg = Regressor(blob['dk']).to(dev)
        self.reg.load_state_dict(blob['reg']); self.reg.eval()
        self.Rinv, self.Rc_inv, self.cq, self.cg = g('Rinv'), g('Rc_inv'), g('cq'), g('cg')
        self.loc, self.scale = blob['std_loc'].double().numpy(), blob['std_scale'].double().numpy()
        self.norm = blob['std_norm']
        for p in list(self.net.parameters()) + list(self.reg.parameters()):
            p.requires_grad_(False)

    def columns(self, feats):
        parts = []
        for src, cols in self.b['sources']:
            f = feats[src]
            parts.append(f if cols == 'raw' else f[:, cols])
        return np.concatenate(parts, 1)

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
        g = torch.where(ut > q[:, -1:], gk[-1] + slope_hi[:, None] * (ut - q[:, -1:]), g)
        g = torch.where(ut < q[:, :1], gk[0] + slope_lo[:, None] * (ut - q[:, :1]), g)
        return g.clamp(-8, 8).T

    def energy(self, x, t):
        mu, lv = self.net(t)
        u = (x - mu) * torch.exp(-0.5 * lv)
        return (u @ self.Rinv * u).sum(1) + lv.sum(1)

    def prior(self, t):
        dz = t[:, 4] * float(self.t_std[4]) + float(self.t_mean[4])
        dmin = torch.cdist(t[:, :2], self.cosmo).min(1).values
        return (dz / DZ_SIGMA) ** 2 + 50.0 * torch.relu(dmin - self.hull_r) ** 2

    def score(self, feats, steps=150, lr=0.05, restarts=3):
        f = self.columns(feats)
        x = torch.as_tensor(((f - self.f_mean) / self.f_std)[:, self.keep], dtype=torch.float32, device=self.dev)
        g = torch.Generator(device=self.dev); g.manual_seed(0)
        with torch.no_grad():
            t0 = self.reg(x).clamp(self.t_lo, self.t_hi)
        best_l = torch.full((len(x),), float('inf'), device=self.dev)
        best_t = t0.clone()
        for r in range(restarts):
            offset = 0.5 * torch.randn((1, 5), generator=g, device=self.dev)
            t = (t0 if r == 0 else t0 + offset).clamp(self.t_lo, self.t_hi).clone().requires_grad_(True)
            opt = torch.optim.Adam([t], lr=lr)
            for _ in range(steps):
                l = self.energy(x, t) + self.prior(t)
                opt.zero_grad(set_to_none=True); l.sum().backward(); opt.step()
                with torch.no_grad():
                    t.clamp_(self.t_lo, self.t_hi)
            with torch.no_grad():
                l = self.energy(x, t) + self.prior(t)
                better = l < best_l
                best_l = torch.where(better, l, best_l)
                best_t = torch.where(better[:, None], t, best_t)
        with torch.no_grad():
            if self.b['stat'] == 'copula':
                mu, lv = self.net(best_t)
                gc = self.copula((x - mu) * torch.exp(-0.5 * lv))
                raw = (gc @ self.Rc_inv * gc).sum(1)
            else:
                raw = best_l
        raw = raw.cpu().numpy().astype(np.float64)
        theta = best_t.cpu().numpy().astype(np.float64) * self.t_std + self.t_mean
        z = (theta - self.t_mean) / self.t_std
        cols = [np.ones(len(z))] + [z[:, i] for i in range(5)] + [z[:, i] * z[:, j] for i in range(5) for j in range(i, 5)]
        X = np.stack(cols, 1)
        return (raw - X @ self.loc) / np.exp(0.5 * (X @ self.scale)) / self.norm
