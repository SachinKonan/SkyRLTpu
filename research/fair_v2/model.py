"""Conditional Gaussian likelihood of summaries given physical parameters.

    s | theta ~ N(mu(theta), D(theta) R D(theta))

mu and log-diagonal D come from one MLP trained by Gaussian NLL on normal maps
(fresh noise views). R is the shrunk correlation of standardized residuals.

Anomaly score for one map: profile -2 log-likelihood
    L(s) = min_theta  u^T R^-1 u + 2 sum log D(theta) + prior(theta)
over the training parameter domain (box for T_AGN, f0; Gaussian prior for
Delta_z; soft hull penalty keeping (Omega_m, S8) near training cosmologies),
standardized by its normal-map location/scale as a smooth function of the
fitted parameters. Every operation is per map.
"""
import math

import numpy as np
import torch
import torch.nn as nn

DZ_SIGMA = 0.022


class Net(nn.Module):
    def __init__(self, d, width=512, depth=3):
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
    """Initial parameter estimate from standardized summaries."""
    def __init__(self, d, width=512):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                               nn.Linear(width, 5))

    def forward(self, s):
        return self.f(s)


def oas_correlation(u):
    """Oracle-approximating shrinkage of a correlation matrix (u standardized)."""
    n, p = u.shape
    s = (u.T @ u) / n
    d = torch.sqrt(torch.diag(s)).clamp(min=1e-6)
    s = s / d[:, None] / d[None, :]
    tr = p
    tr2 = (s * s).sum()
    rho = ((1 - 2 / p) * tr2 + tr ** 2) / ((n + 1 - 2 / p) * (tr2 - tr ** 2 / p))
    rho = float(rho.clamp(0, 1))
    return (1 - rho) * s + rho * torch.eye(p, device=u.device), rho


class Likelihood:
    def __init__(self, dev='cpu', eig_floor=0.02):
        self.dev = torch.device(dev)
        self.eig_floor = eig_floor

    def copula(self, u):
        """Map standardized residuals (N, d) to normal scores via frozen marginals."""
        q, gk = self.cq, self.cg
        K = q.shape[1]
        ut = u.T.contiguous()
        i = torch.searchsorted(q, ut).clamp(1, K - 1)
        lo = torch.gather(q, 1, i - 1); hi = torch.gather(q, 1, i)
        w = ((ut - lo) / (hi - lo).clamp(min=1e-9)).clamp(0, 1)
        g = gk[i - 1] + w * (gk[i] - gk[i - 1])
        # linear tails in normal-score space beyond the observed range, capped
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

    # ------------------------------------------------------------------ fit
    def fit(self, feats, theta, *, steps=6000, batch=2048, lr=2e-3, width=512, depth=3, seed=0, log=print,
            weight_decay=1e-5, jitter=0.0):
        """feats (N, V, d) float32 normal summaries; theta (N, 5)."""
        torch.manual_seed(seed)
        N, V, d = feats.shape
        flat = feats.reshape(N * V, d).astype(np.float64)
        self.f_mean = flat.mean(0)
        self.f_std = flat.std(0) + 1e-8
        keep = self.f_std > 1e-6
        self.keep = keep
        self.t_mean = theta.mean(0)
        self.t_std = theta.std(0)
        self.t_lo = (theta.min(0) - self.t_mean) / self.t_std
        self.t_hi = (theta.max(0) - self.t_mean) / self.t_std
        cos = np.unique(theta[:, :2], axis=0)
        self.cosmo = torch.as_tensor((cos - self.t_mean[:2]) / self.t_std[:2], dtype=torch.float32, device=self.dev)
        # nearest-neighbour spacing of training cosmologies (normalized)
        dd = torch.cdist(self.cosmo, self.cosmo) + 1e9 * torch.eye(len(self.cosmo), device=self.dev)
        self.hull_r = float(dd.min(1).values.median())
        x = torch.as_tensor(((feats - self.f_mean) / self.f_std)[..., keep], dtype=torch.float32, device=self.dev)
        t = torch.as_tensor((theta - self.t_mean) / self.t_std, dtype=torch.float32, device=self.dev)
        dk = int(keep.sum())
        self.jmask = torch.tensor([0., 0., 1., 1., 0.5], device=self.dev)
        self.net = Net(dk, width, depth).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
        g = torch.Generator(device=self.dev); g.manual_seed(seed)
        for step in range(steps):
            i = torch.randint(0, N, (batch,), generator=g, device=self.dev)
            v = torch.randint(0, V, (batch,), generator=g, device=self.dev)
            ti = t[i]
            if jitter > 0:
                # Smoothness in the nuisance directions: jitter baryon (T_AGN, f0)
                # and Delta_z inputs so the model cannot key on exact per-model values.
                ti = ti + jitter * torch.randn(ti.shape, generator=g, device=self.dev) * self.jmask
            mu, lv = self.net(ti)
            r = x[i, v] - mu
            loss = 0.5 * (r * r * torch.exp(-lv) + lv).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if step % 1000 == 0 or step == steps - 1:
                log(f'  nll step {step}: {loss.item():.4f}')
        with torch.no_grad():
            us = []
            for lo in range(0, N, 4096):
                mu, lv = self.net(t[lo:lo + 4096])
                us.append(((x[lo:lo + 4096] - mu[:, None]) * torch.exp(-0.5 * lv)[:, None]).reshape(-1, dk))
            u = torch.cat(us)
            R, rho = oas_correlation(u.double())
            self.Rinv = torch.linalg.inv(R).float()
            self.rho = rho
            log(f'  correlation shrinkage rho={rho:.4f}; residual std mean={float(u.std(0).mean()):.3f}')
            # Gaussian copula: per-feature empirical marginal of standardized residuals
            probs = torch.linspace(0.0, 1.0, 1001, device=self.dev, dtype=torch.float64)
            probs = (probs * (u.shape[0] - 1) + 0.5) / u.shape[0]
            self.cq = torch.quantile(u.double().T.contiguous(), probs, dim=1).T.float().contiguous()  # (d, K)
            self.cg = torch.distributions.Normal(0., 1.).icdf(probs).float()                        # (K,)
            gc = self.copula(u)
            Rc, rho_c = oas_correlation(gc.double())
            ev, evec = torch.linalg.eigh(Rc)
            ev = ev.clamp(min=self.eig_floor)
            self.Rc_inv = ((evec / ev) @ evec.T).float()
            log(f'  copula correlation rho={rho_c:.4f}; eig floor {self.eig_floor} hit {(ev <= self.eig_floor).sum().item()}')
        # parameter regressor for initialization
        self.reg = Regressor(dk).to(self.dev)
        opt = torch.optim.AdamW(self.reg.parameters(), lr=1e-3, weight_decay=1e-5)
        for step in range(3000):
            i = torch.randint(0, N, (batch,), generator=g, device=self.dev)
            v = torch.randint(0, V, (batch,), generator=g, device=self.dev)
            loss = ((self.reg(x[i, v]) - t[i]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        log(f'  regressor mse (normalized) {loss.item():.4f}')
        return self

    # ------------------------------------------------------------ scoring
    def standardize(self, feats):
        return torch.as_tensor(((feats - self.f_mean) / self.f_std)[..., self.keep], dtype=torch.float32,
                               device=self.dev)

    def energy(self, x, t):
        mu, lv = self.net(t)
        u = (x - mu) * torch.exp(-0.5 * lv)
        quad = (u @ self.Rinv * u).sum(1)
        return quad + lv.sum(1)

    def prior(self, t):
        dz = t[:, 4] * float(self.t_std[4]) + float(self.t_mean[4])
        pen = (dz / DZ_SIGMA) ** 2
        dmin = torch.cdist(t[:, :2], self.cosmo).min(1).values
        pen = pen + 50.0 * torch.relu(dmin - self.hull_r) ** 2
        return pen

    def profile(self, feats, *, steps=150, lr=0.05, restarts=3, seed=0):
        """Minimize energy+prior over theta, batched; returns (L, theta_hat_raw)."""
        x = self.standardize(feats)
        lo = torch.as_tensor(self.t_lo, dtype=torch.float32, device=self.dev)
        hi = torch.as_tensor(self.t_hi, dtype=torch.float32, device=self.dev)
        g = torch.Generator(device=self.dev); g.manual_seed(seed)
        with torch.no_grad():
            t0 = self.reg(x).clamp(lo, hi)
        best_l = torch.full((len(x),), float('inf'), device=self.dev)
        best_t = t0.clone()
        for r in range(restarts):
            # Restart offsets are shared by every map (no dependence on batch position).
            offset = 0.5 * torch.randn((1, 5), generator=g, device=self.dev)
            t = (t0 if r == 0 else (t0 + offset)).clamp(lo, hi)
            t = t.clone().requires_grad_(True)
            opt = torch.optim.Adam([t], lr=lr)
            for _ in range(steps):
                l = self.energy(x, t) + self.prior(t)
                opt.zero_grad(set_to_none=True); l.sum().backward(); opt.step()
                with torch.no_grad():
                    t.clamp_(lo, hi)
            with torch.no_grad():
                l = self.energy(x, t) + self.prior(t)
                better = l < best_l
                best_l = torch.where(better, l, best_l)
                best_t = torch.where(better[:, None], t, best_t)
        theta = best_t.detach().cpu().numpy() * self.t_std + self.t_mean
        with torch.no_grad():
            cop = self.copula_energy(x, best_t).cpu().numpy().astype(np.float64)
        self.last_copula = cop
        return best_l.detach().cpu().numpy().astype(np.float64), theta

    def energy_at(self, feats, theta):
        x = self.standardize(feats)
        t = torch.as_tensor((theta - self.t_mean) / self.t_std, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            return self.energy(x, t).cpu().numpy().astype(np.float64)


def design(theta, t_mean, t_std):
    z = (theta - t_mean) / t_std
    cols = [np.ones(len(z))] + [z[:, i] for i in range(5)] + [z[:, i] * z[:, j] for i in range(5) for j in range(i, 5)]
    return np.stack(cols, 1)


class Standardizer:
    """Normal-map location/scale of a raw score as a quadratic function of theta_hat."""
    def fit(self, raw, theta, t_mean, t_std, ridge=1.0):
        self.t_mean, self.t_std = t_mean, t_std
        X = design(theta, t_mean, t_std)
        A = X.T @ X + ridge * np.eye(X.shape[1])
        self.loc = np.linalg.solve(A, X.T @ raw)
        res = raw - X @ self.loc
        self.scale = np.linalg.solve(A, X.T @ np.log(res ** 2 + 1e-12))
        # correct the log-chi2(1) bias of E[log res^2]
        self.norm = 1.0
        z = self(raw, theta)
        self.norm = float(z.std())
        return self

    def __call__(self, raw, theta):
        X = design(theta, self.t_mean, self.t_std)
        sd = np.exp(0.5 * (X @ self.scale))
        return (raw - X @ self.loc) / sd / self.norm


def phase2_score(normal, anomalous):
    """Official functional: mean TPR at 100 log-spaced FPRs in [0.001, 0.05]."""
    from sklearn.metrics import roc_curve
    y = np.r_[np.zeros(len(normal)), np.ones(len(anomalous))]
    s = np.r_[normal, anomalous]
    fpr, tpr, _ = roc_curve(y, s)
    grid = np.logspace(-3, math.log10(0.05), 100)
    return float(np.mean(np.interp(grid, fpr, tpr)))
