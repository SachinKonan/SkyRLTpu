"""New per-map statistic families (GPU torch). Every feature is a fixed function of ONE map.

Families (each written as its own feature source):
  scat : Morlet scattering S1 (J=5 x L=4), orientation-reduced S2/S1, wavelet phase-harmonic
         cross-scale phase coupling and modulus scale-scale correlation.
  prof : stacked radial kappa profiles around peaks and minima in frozen height bins
         (detection on Gaussian-smoothed maps, profile measured on the raw noisy map).
  pkcl : peak clustering (high peaks x all peaks / minima, deep minima x minima) and Hessian
         shape of extrema (ellipticity, curvature/height) per height bin.
  ipow : position-dependent power: response of local small-scale band power to the local
         large-scale kappa (integrated bispectrum), local power variance, band-band coupling.
  bisp : binned reduced bispectrum Q(b1,b2,b3) of band-filtered fields (all closed triangles).

Maps are zero padded to (1536, 240) for FFTs so nothing wraps around the footprint. Frozen
constants (extremum height bin edges) are fitted on fit3 normal noisy maps with `fit_edges`.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

H, W = 1424, 176
HP, WP = 1536, 240
FAMILIES = ('scat', 'prof', 'pkcl', 'ipow', 'bisp')

# scattering
J, L = 5, 4
# profiles / clustering
DET = (1.5, 3.0)                       # detection smoothing (px)
EXT_Q = (0.5, 0.8, 0.93, 0.98)         # quantiles of extremum heights -> 5 bins
PROF_R = (0, 0.5, 1.5, 2.5, 3.5, 5, 7, 10, 14, 20)
CL_R = (1.5, 3, 5, 7, 10, 14, 20, 28)
# ipow
IP_SCALES = (0.0, 1.0, 2.0, 4.0, 8.0)
IP_BANDS = ((0, 1), (1, 2), (2, 3), (3, 4))
IP_L = (8.0, 16.0)
# bispectrum bands (cycles / pixel)
BS_EDGES = (0.006, 0.012, 0.024, 0.04, 0.065, 0.1, 0.15, 0.23, 0.35, 0.5)


def _triangles():
    e = BS_EDGES
    nb = len(e) - 1
    out = []
    for a in range(nb):
        for b in range(a, nb):
            for c in range(b, nb):
                if e[c] <= e[a + 1] + e[b + 1]:      # a closed triangle is possible
                    out.append((a, b, c))
    return out


TRI = _triangles()


class Fam:
    def __init__(self, mask, dev, edges=None):
        self.dev = dev
        m = torch.as_tensor(np.asarray(mask), device=dev, dtype=torch.float32)
        self.m = m
        self.mp = F.pad(m, (0, WP - W, 0, HP - H))
        fy = torch.fft.fftfreq(HP, device=dev)[:, None]
        fx = torch.fft.rfftfreq(WP, device=dev)[None, :]
        self.r2 = fy ** 2 + fx ** 2                      # rfft grid, cycles/px squared
        self.Mr = torch.fft.rfft2(self.mp)
        self._smask = {}
        self._inter = {}
        # ---- scattering filters (full fft grid, rad/px)
        ky = 2 * math.pi * torch.fft.fftfreq(HP, device=dev)[:, None]
        kx = 2 * math.pi * torch.fft.fftfreq(WP, device=dev)[None, :]
        self.psi = []
        for j in range(J):
            sig = 0.8 * 2 ** j
            xi = 0.75 * math.pi / 2 ** j
            row = []
            for l in range(L):
                th = math.pi * l / L
                g = torch.exp(-0.5 * sig ** 2 * ((ky - xi * math.sin(th)) ** 2 + (kx - xi * math.cos(th)) ** 2))
                b = torch.exp(-0.5 * sig ** 2 * (ky ** 2 + kx ** 2 + xi ** 2))
                row.append((g - b).contiguous())
            self.psi.append(row)
        self.scat_int = [self.interior(0.8 * 2 ** j) for j in range(J)]
        # second-order fields at scale j2 are band-limited: evaluate them on a grid subsampled by
        # sc = 2^(j2-1) via Fourier cropping (fixed function; aliasing < 1% of the filter peak)
        self.crop, self.psi_c, self.scat_int_c = [], [], []
        for j in range(J):
            sc = max(1, 2 ** (j - 1))
            hy, hx = HP // (2 * sc), WP // (2 * sc)
            iy = torch.cat([torch.arange(0, hy, device=dev), torch.arange(HP - hy, HP, device=dev)])
            ix = torch.cat([torch.arange(0, hx, device=dev), torch.arange(WP - hx, WP, device=dev)])
            self.crop.append((iy, ix, sc))
            self.psi_c.append([p[iy][:, ix].contiguous() if sc > 1 else p for p in self.psi[j]])
            self.scat_int_c.append(self.scat_int[j][::sc, ::sc])
        # ---- profile windows
        R = int(max(PROF_R[-1], CL_R[-1]))
        self.R = R
        d = torch.arange(-R, R + 1, device=dev, dtype=torch.float32)
        rr = torch.sqrt(d[:, None] ** 2 + d[None, :] ** 2)
        self.ry = torch.remainder(torch.arange(-R, R + 1, device=dev), HP)
        self.rx = torch.remainder(torch.arange(-R, R + 1, device=dev), WP)
        def onehot(edges):
            e = torch.as_tensor(edges, device=dev, dtype=torch.float32)
            oh = torch.stack([(rr >= e[i]) & (rr < e[i + 1]) for i in range(len(edges) - 1)], -1).float()
            return oh                                   # (2R+1, 2R+1, nr)
        self.prof_oh = onehot(PROF_R)
        self.cl_oh = onehot(CL_R)
        self.det_int = [self.interior(s) for s in DET]
        # ---- ipow
        self.ip_int = [self.interior(Lsm, 0.8) for Lsm in IP_L]
        self.ip_int0 = [self.interior(s) for s in IP_SCALES]
        # ---- bispectrum band masks (rfft grid)
        r = torch.sqrt(self.r2)
        self.bands = torch.stack([((r >= BS_EDGES[i]) & (r < BS_EDGES[i + 1])).float()
                                  for i in range(len(BS_EDGES) - 1)])
        self.edges = None if edges is None else {k: torch.as_tensor(v, device=dev, dtype=torch.float32)
                                                 for k, v in edges.items()}

    # ------------------------------------------------------------- helpers
    def gk(self, s):
        return torch.exp(-2 * math.pi ** 2 * s * s * self.r2)

    def smask(self, s):
        if s not in self._smask:
            self._smask[s] = torch.fft.irfft2(self.Mr * self.gk(s), s=(HP, WP)) if s > 0 else self.mp.clone()
        return self._smask[s]

    def interior(self, s, thr=0.95):
        key = (s, thr)
        if key not in self._inter:
            inside = (self.mp > 0) if s == 0 else ((self.smask(s) > thr) & (self.mp > 0))
            padded = F.pad(inside.float()[None, None], (1, 1, 1, 1), value=0.)
            self._inter[key] = -F.max_pool2d(-padded, 3, 1, 0)[0, 0] > 0.5
        return self._inter[key]

    def smooth_f(self, xf, s):
        """xf = rfft2 of a padded masked field -> mask-normalized smoothed field (masked)."""
        if s == 0:
            return torch.fft.irfft2(xf, s=(HP, WP)) * self.mp
        return torch.fft.irfft2(xf * self.gk(s), s=(HP, WP)) / self.smask(s).clamp(min=0.05) * self.mp

    def pad(self, maps):
        return F.pad(maps * self.m, (0, WP - W, 0, HP - H))

    @staticmethod
    def mmean(x, inter):
        w = inter.float()
        return (x * w).sum((-2, -1)) / w.sum()

    def extrema(self, D, inter, sign):
        z = sign * D
        z = torch.where(inter[None], z, torch.full_like(z, -1e9))
        mx = F.max_pool2d(z[:, None], 3, 1, 1)[:, 0]
        return (z >= mx) & inter[None]

    # ------------------------------------------------------------- edges
    @torch.no_grad()
    def fit_edges(self, maps):
        kp = self.pad(maps)
        Kr = torch.fft.rfft2(kp)
        edges = {}
        for i, s in enumerate(DET):
            D = self.smooth_f(Kr, s)
            for sign, nm in ((1, 'pk'), (-1, 'mn')):
                ext = self.extrema(D, self.det_int[i], sign)
                v = (sign * D)[ext]
                edges[f'{nm}{i}'] = torch.quantile(v[torch.randperm(v.numel(), device=v.device)[:4_000_000]],
                                                   torch.as_tensor(EXT_Q, device=v.device))
        self.edges = edges
        return {k: v.cpu().numpy() for k, v in edges.items()}

    # ------------------------------------------------------------- families
    @torch.no_grad()
    def __call__(self, maps, fams=FAMILIES):
        kp = self.pad(maps.float())
        out = {}
        if 'scat' in fams:
            out['scat'] = self.scat(kp)
        if 'prof' in fams or 'pkcl' in fams:
            p, c = self.prof(kp)
            out['prof'], out['pkcl'] = p, c
        if 'ipow' in fams:
            out['ipow'] = self.ipow(kp)
        if 'bisp' in fams:
            out['bisp'] = self.bisp(kp)
        return {k: v for k, v in out.items() if k in fams}

    def scat(self, kp):
        B = kp.shape[0]
        Kf = torch.fft.fft2(kp)
        S1 = torch.zeros((B, J, L), device=self.dev)
        S1q2 = torch.zeros((B, J, L), device=self.dev)
        S2 = torch.zeros((B, J, L, J, L), device=self.dev)
        wph1 = torch.zeros((B, J - 1, L), device=self.dev)
        wph2 = torch.zeros((B, J - 2, L), device=self.dev)
        mc = torch.zeros((B, J - 1, L), device=self.dev)
        for l1 in range(L):
            zs = [torch.fft.ifft2(Kf * self.psi[j][l1]) for j in range(J)]
            mods = [z.abs() * self.mp for z in zs]
            for j in range(J):
                S1[:, j, l1] = self.mmean(mods[j], self.scat_int[j])
                S1q2[:, j, l1] = self.mmean(mods[j] ** 2, self.scat_int[j])
            # phase harmonics: z_j vs [z_{j+1}]^2 and [z_{j+2}]^4 (same central frequency)
            for j in range(J - 1):
                it = self.scat_int[j + 1]
                a = zs[j]
                ph = torch.sgn(zs[j + 1])
                bq = mods[j + 1] * ph ** 2
                num = self.mmean((a * bq.conj()).real, it)
                wph1[:, j, l1] = num / torch.sqrt(self.mmean(mods[j] ** 2, it) * self.mmean(mods[j + 1] ** 2, it))
                # modulus scale-scale correlation
                x1 = mods[j]; x2 = mods[j + 1]
                m1 = self.mmean(x1, it)[:, None, None]; m2 = self.mmean(x2, it)[:, None, None]
                cov = self.mmean((x1 - m1) * (x2 - m2), it)
                mc[:, j, l1] = cov / torch.sqrt(self.mmean((x1 - m1) ** 2, it) * self.mmean((x2 - m2) ** 2, it))
                if j < J - 2:
                    it2 = self.scat_int[j + 2]
                    ph2 = torch.sgn(zs[j + 2])
                    bq2 = mods[j + 2] * ph2 ** 4
                    num2 = self.mmean((a * bq2.conj()).real, it2)
                    wph2[:, j, l1] = num2 / torch.sqrt(self.mmean(mods[j] ** 2, it2) * self.mmean(mods[j + 2] ** 2, it2))
            del zs
            for j1 in range(J - 1):
                Uf = torch.fft.fft2(mods[j1])
                for j2 in range(j1 + 1, J):
                    iy, ix, sc = self.crop[j2]
                    Ufc = Uf[:, iy][:, :, ix] if sc > 1 else Uf
                    for l2 in range(L):
                        fld = torch.fft.ifft2(Ufc * self.psi_c[j2][l2]).abs()
                        S2[:, j1, l1, j2, l2] = self.mmean(fld, self.scat_int_c[j2]) / sc ** 2
            del mods
        feats = [torch.log(S1.reshape(B, -1)), torch.log(S1q2.mean(2))]
        # reduced S2: orientation-averaged S2/S1 per (j1, j2, |dl|)
        red = []
        for j1 in range(J - 1):
            for j2 in range(j1 + 1, J):
                acc = {0: [], 1: [], 2: []}
                for l1 in range(L):
                    for l2 in range(L):
                        dl = (l2 - l1) % L
                        dl = min(dl, L - dl)
                        acc[dl].append(S2[:, j1, l1, j2, l2] / S1[:, j1, l1])
                for dl in (0, 1, 2):
                    red.append(torch.log(torch.stack(acc[dl], 1).mean(1)))
        feats.append(torch.stack(red, 1))
        feats.append(wph1.mean(2)); feats.append(wph2.mean(2)); feats.append(mc.mean(2))
        return torch.cat(feats, 1)

    def prof(self, kp):
        B = kp.shape[0]
        Kr = torch.fft.rfft2(kp)
        e = self.edges
        R = self.R
        nr_p = len(PROF_R) - 1
        prof_feats, cl_feats = [], []
        for i, s in enumerate(DET):
            D = self.smooth_f(Kr, s)
            inter = self.det_int[i]
            # Hessian of detection field (central differences)
            Dx = (torch.roll(D, -1, 2) - torch.roll(D, 1, 2)) / 2
            Dy = (torch.roll(D, -1, 1) - torch.roll(D, 1, 1)) / 2
            Dxx = torch.roll(D, -1, 2) - 2 * D + torch.roll(D, 1, 2)
            Dyy = torch.roll(D, -1, 1) - 2 * D + torch.roll(D, 1, 1)
            Dxy = (torch.roll(torch.roll(D, -1, 1), -1, 2) - torch.roll(torch.roll(D, -1, 1), 1, 2)
                   - torch.roll(torch.roll(D, 1, 1), -1, 2) + torch.roll(torch.roll(D, 1, 1), 1, 2)) / 4
            del Dx, Dy
            Pf_all, Pf_hi, n_all, n_hi = {}, {}, {}, {}
            for sign, nm in ((1, 'pk'), (-1, 'mn')):
                ext = self.extrema(D, inter, sign)
                idx = torch.bucketize(sign * D, e[f'{nm}{i}'])          # 0..4
                P = torch.stack([(ext & (idx == b)).float() for b in range(len(EXT_Q) + 1)], 1)   # (B,5,HP,WP)
                cnt = P.sum((-2, -1))                                    # (B,5)
                Pf = torch.fft.rfft2(P)
                C = torch.fft.irfft2(Pf.conj() * Kr[:, None], s=(HP, WP))
                Cw = C[:, :, self.ry][:, :, :, self.rx]
                del C
                N = torch.fft.irfft2(Pf.conj() * self.Mr, s=(HP, WP))
                Nw = N[:, :, self.ry][:, :, :, self.rx]
                del N
                num = torch.einsum('bkyx,yxr->bkr', Cw, self.prof_oh)
                den = torch.einsum('bkyx,yxr->bkr', Nw, self.prof_oh)
                prof = torch.where(den > 0.5, num / den.clamp(min=0.5), torch.zeros_like(num))
                height = (P * D[:, None]).sum((-2, -1)) / cnt.clamp(min=1)
                prof_feats += [prof.reshape(B, -1), height]
                # Hessian shape at extrema per bin: ellipticity and curvature/height
                tr = -sign * (Dxx + Dyy)
                ell = torch.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2) / tr.abs().clamp(min=1e-8)
                ch = tr / (sign * D).abs().clamp(min=1e-6)
                cl_feats.append((P * ell[:, None]).sum((-2, -1)) / cnt.clamp(min=1))
                cl_feats.append((P * ch[:, None]).sum((-2, -1)) / cnt.clamp(min=1))
                Pf_all[nm] = Pf.sum(1); n_all[nm] = cnt.sum(1)
                Pf_hi[nm] = Pf[:, 3:].sum(1); n_hi[nm] = cnt[:, 3:].sum(1)
                del P, Pf, Cw, Nw
            # clustering: xi(r) = <pairs> / <expected under uniform density inside the interior> - 1
            If = torch.fft.rfft2(inter.float())
            area = inter.float().sum()
            for a, b in (('pk', 'pk'), ('pk', 'mn'), ('mn', 'mn')):
                pairs = torch.fft.irfft2(Pf_hi[a].conj() * Pf_all[b], s=(HP, WP))[:, self.ry][:, :, self.rx]
                expd = torch.fft.irfft2(Pf_hi[a].conj() * If, s=(HP, WP))[:, self.ry][:, :, self.rx]
                pr = torch.einsum('byx,yxr->br', pairs, self.cl_oh)
                ex = torch.einsum('byx,yxr->br', expd, self.cl_oh) * (n_all[b] / area)[:, None]
                cl_feats.append(pr / ex.clamp(min=1e-6) - 1)
            del Dxx, Dyy, Dxy, D
        return torch.cat(prof_feats, 1), torch.cat(cl_feats, 1)

    def ipow(self, kp):
        Kr = torch.fft.rfft2(kp)
        G = [self.smooth_f(Kr, s) for s in IP_SCALES]
        ws = [(G[a] - G[b]) * self.ip_int0[b] for a, b in IP_BANDS]
        del G
        w2f = [torch.fft.rfft2(w * w) for w in ws]
        w3f = [torch.fft.rfft2(w * w * w) for w in ws]
        del ws
        feats = []
        for li, Ls in enumerate(IP_L):
            it = self.ip_int[li]
            d = self.smooth_f(Kr, Ls)
            dm = d - self.mmean(d, it)[:, None, None]
            dv = self.mmean(dm * dm, it)
            d2 = dm * dm - dv[:, None, None]
            d2v = self.mmean(d2 * d2, it)
            Ps = []
            for bi in range(len(IP_BANDS)):
                # local power normalized only by the smoothed mask (band field zero outside its interior)
                P = self.smooth_f(w2f[bi], Ls)
                Pm = self.mmean(P, it)
                Pc = P - Pm[:, None, None]
                Pv = self.mmean(Pc * Pc, it)
                feats.append(self.mmean(Pc * dm, it) / torch.sqrt(Pv * dv))
                feats.append(self.mmean(Pc * d2, it) / torch.sqrt(Pv * d2v))
                feats.append(torch.log(Pv / Pm ** 2))
                T = self.smooth_f(w3f[bi], Ls)
                Tc = T - self.mmean(T, it)[:, None, None]
                feats.append(self.mmean(Tc * dm, it) / torch.sqrt(self.mmean(Tc * Tc, it) * dv))
                Ps.append((Pc, Pv))
            for a in range(len(IP_BANDS)):
                for b in range(a + 1, len(IP_BANDS)):
                    feats.append(self.mmean(Ps[a][0] * Ps[b][0], it) / torch.sqrt(Ps[a][1] * Ps[b][1]))
        return torch.stack(feats, 1)

    def bisp(self, kp):
        Kr = torch.fft.rfft2(kp)
        sel = self.mp > 0
        I = torch.stack([torch.fft.irfft2(Kr * self.bands[b], s=(HP, WP))[:, sel] for b in range(len(self.bands))], 1)
        I = I / torch.sqrt((I * I).mean(-1, keepdim=True)).clamp(min=1e-12)      # unit rms per band
        Q = [(I[:, a] * I[:, b] * I[:, c]).mean(-1) for a, b, c in TRI]
        return torch.stack(Q, 1) * math.sqrt(sel.sum().item())


# ---------------------------------------------------------------------------------------------
# R1 noise-structure battery ('nois') and R3 high-S/N tails ('tail').
# ---------------------------------------------------------------------------------------------
NS_HP = (1.0, 2.0)                 # high-pass h = x - G_s x
NS_TILE = 16
TL_SCALES = (1.0, 2.0, 3.0)
TL_POS = (3.0, 4.0, 5.0)
TL_NEG = (3.0, 4.0)
AN_EDGES = (0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.5)


class NoiseFam(Fam):
    """Adds `nois` and `tail` on top of Fam's helpers; frozen scales in self.nedges."""

    def __init__(self, mask, dev, nedges=None):
        super().__init__(mask, dev, edges=None)
        self.nedges = None if nedges is None else {k: torch.as_tensor(v, device=dev, dtype=torch.float32)
                                                   for k, v in nedges.items()}
        T = NS_TILE
        self.hp_int = [self.interior(s) for s in NS_HP]
        self.g4_int = self.interior(4.0, 0.8)
        self.tiles = []
        for it in self.hp_int:
            cnt = F.avg_pool2d(it.float()[None, None], T, T)[0, 0] * T * T          # (HP/T, WP/T)
            valid = cnt >= 0.75 * T * T
            iy, ix = torch.nonzero(valid, as_tuple=True)
            self.tiles.append((cnt, valid, iy.float(), ix.float()))
        self.tl_int = [self.interior(s) for s in TL_SCALES]
        fy = torch.fft.fftfreq(HP, device=dev)[:, None].expand(HP, WP // 2 + 1)
        fx = torch.fft.rfftfreq(WP, device=dev)[None, :].expand(HP, WP // 2 + 1)
        ang = torch.atan2(fy.abs(), fx.abs())
        r = torch.sqrt(fy ** 2 + fx ** 2)
        wedge = torch.where(ang < math.pi / 8, 0, torch.where(ang > 3 * math.pi / 8, 1, 2))
        band = torch.bucketize(r, torch.as_tensor(AN_EDGES, device=dev)) - 1
        nbd = len(AN_EDGES) - 1
        ok = (band >= 0) & (band < nbd)
        self.an_bin = torch.where(ok, band * 3 + wedge, torch.full_like(band, nbd * 3)).flatten()
        self.an_n = nbd

    @torch.no_grad()
    def fit_nedges(self, maps):
        kp = self.pad(maps.float())
        Kr = torch.fft.rfft2(kp)
        e = {}
        for i, s in enumerate(NS_HP):
            h = kp - self.smooth_f(Kr, s)
            e[f'hsd{i}'] = h[:, self.hp_int[i]].std().reshape(1)
        for i, s in enumerate(TL_SCALES):
            k = self.smooth_f(Kr, s)
            e[f'ksd{i}'] = k[:, self.tl_int[i]].std().reshape(1)
        self.nedges = e
        return {k: v.cpu().numpy() for k, v in e.items()}

    @staticmethod
    def _corr(a, b):
        a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
        return (a * b).mean(1) / torch.sqrt((a * a).mean(1) * (b * b).mean(1)).clamp(min=1e-30)

    @staticmethod
    def _rank(a):
        return torch.argsort(torch.argsort(a, 1), 1).float()

    def nois(self, kp, Kr):
        B = kp.shape[0]
        T = NS_TILE
        e = self.nedges
        feats = []
        G4 = self.smooth_f(Kr, 4.0)
        for i, s in enumerate(NS_HP):
            it = self.hp_int[i]
            h = (kp - self.smooth_f(Kr, s)) * it
            hv = h[:, it] / e[f'hsd{i}']
            mu = hv.mean(1, keepdim=True); c = hv - mu
            var = (c * c).mean(1)
            feats += [torch.log(var), (c ** 3).mean(1) / var ** 1.5, (c ** 4).mean(1) / var ** 2]
            for k in (3.0, 4.0, 5.0):
                feats.append(torch.log1p((hv > k).float().sum(1)))
                feats.append(torch.log1p((hv < -k).float().sum(1)))
            cnt, valid, iy, ix = self.tiles[i]
            v = (F.avg_pool2d((h * h)[:, None], T, T)[:, 0] * T * T / cnt.clamp(min=1))[:, valid] / e[f'hsd{i}'] ** 2
            mt = (F.avg_pool2d((G4 * it)[:, None], T, T)[:, 0] * T * T / cnt.clamp(min=1))[:, valid]
            vm = v.mean(1)
            feats += [torch.log(vm), v.std(1) / vm]
            feats += [self._corr(v, mt), self._corr(self._rank(v), self._rank(mt)),
                      self._corr(v, (mt - mt.mean(1, keepdim=True)).abs())]
            feats += [self._corr(v, iy[None].expand_as(v)), self._corr(v, ix[None].expand_as(v))]
            feats += [torch.log(v.max(1).values / vm), torch.log(v.min(1).values.clamp(min=1e-6) / vm)]
            # spatial coherence of tile variances: lag-1 correlation along the long axis and
            # fraction of variance surviving 4x4-tile block averaging
            vg = torch.zeros((B,) + valid.shape, device=self.dev)
            vg[:, valid] = v - vm[:, None]
            pair = valid[1:] & valid[:-1]
            a = vg[:, 1:][:, pair]; b = vg[:, :-1][:, pair]
            feats.append((a * b).mean(1) / (v.var(1) + 1e-30))
            blk_n = F.avg_pool2d(valid.float()[None, None], 4, 4)[0, 0] * 16
            blk = F.avg_pool2d(vg[:, None], 4, 4)[:, 0] * 16 / blk_n.clamp(min=1)
            bsel = blk_n >= 8
            feats.append(torch.log((blk[:, bsel] ** 2).mean(1) / (v.var(1) + 1e-30)))
        # 2D power wedges: horizontal / vertical / diagonal per band
        power = (Kr.real ** 2 + Kr.imag ** 2).reshape(B, -1)
        nb3 = self.an_n * 3
        pw = torch.zeros((B, nb3 + 1), device=self.dev).index_add_(1, self.an_bin, power)[:, :nb3].reshape(B, self.an_n, 3)
        cntw = torch.bincount(self.an_bin, minlength=nb3 + 1)[:nb3].reshape(self.an_n, 3).float()
        pw = pw / cntw
        feats.append(torch.log(pw[:, :, 0] / pw[:, :, 1]))
        feats.append(torch.log(pw[:, :, 2] / (0.5 * (pw[:, :, 0] + pw[:, :, 1]))))
        # exactly-zero observed pixels (holes / extra masks)
        z = ((kp == 0) & (self.mp > 0)).float()
        feats.append(torch.log1p(z.sum((-2, -1))))
        z4 = F.avg_pool2d(z[:, None], 2, 1)[:, 0] > 0.99
        feats.append(torch.log1p(z4.float().sum((-2, -1))))
        return torch.cat([f.reshape(B, -1) for f in feats], 1)

    def tail(self, kp, Kr):
        e = self.nedges
        feats = []
        for i, s in enumerate(TL_SCALES):
            it = self.tl_int[i]
            k = self.smooth_f(Kr, s)
            nu = k / e[f'ksd{i}']
            v = nu[:, it]
            for t in TL_POS:
                feats.append(torch.log1p((v > t).float().sum(1)))
            for t in TL_NEG:
                feats.append(torch.log1p((v < -t).float().sum(1)))
            pk = self.extrema(k, it, +1)
            for t in TL_POS:
                feats.append(torch.log1p((pk & (nu > t)).float().sum((-2, -1))))
            mn = self.extrema(k, it, -1)
            for t in TL_NEG:
                feats.append(torch.log1p((mn & (nu < -t)).float().sum((-2, -1))))
            top = torch.topk(v, 20, 1).values
            feats += [top[:, 0], top[:, 4], top[:, 19]]
            bot = torch.topk(-v, 5, 1).values
            feats += [bot[:, 0], bot[:, 4]]
        return torch.stack(feats, 1)

    @torch.no_grad()
    def __call__(self, maps, fams=('nois', 'tail')):
        kp = self.pad(maps.float())
        Kr = torch.fft.rfft2(kp)
        out = {}
        if 'nois' in fams:
            out['nois'] = self.nois(kp, Kr)
        if 'tail' in fams:
            out['tail'] = self.tail(kp, Kr)
        return out
