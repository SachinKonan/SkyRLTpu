"""FAIR Universe Phase-2 OoD detector (code submission, full member set).

Per map (fixed-shape batches, eval mode, nothing shared across test maps):
  1. feature sources: summary statistics, two CNN embeddings (+ parameter predictions), U-Net denoiser
     residual statistics and statistics of the denoised map, noise-battery and bispectrum statistics,
     power-spectrum anisotropy, an outlier-exposure CNN head, field-level diffusion (denoising ELBO) features;
  2. per-family conditional-Gaussian likelihoods profiled over the 5 physical parameters (fixed-size chunks);
  3. components: likelihood z-scores, parameter-consistency scores across families, CNN predictions and
     ridge theta-probes, cross-scale score;
  4. score = weighted mean of component z-scores (weights in config.json).
All normalizations were frozen on training-simulation calibration maps; no test statistics are used.
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # deterministic cuBLAS (must precede CUDA init)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)
from fwl_lik import Family, component_z, ridge_view  # noqa: E402

H, W = 1424, 176


def _load(fn):
    path = os.path.join(MODEL_DIR, fn)
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:                     # very old torch without weights_only
        return torch.load(path, map_location='cpu')


def _np(d):
    return {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in d.items()}


def _log(msg):
    print(f'[fwl {time.strftime("%H:%M:%S")}] {msg}', flush=True)


class Model:
    def __init__(self):
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = json.load(open(os.path.join(MODEL_DIR, 'config.json')))
        self.weights = {k: float(v) for k, v in self.cfg['weights'].items() if float(v) != 0.0}
        meta = _load('components.pt')
        self.comps = {}
        for name in self.weights:
            c = dict(meta[name])
            if 'std' in c:
                c['std'] = _np(c['std'])
            for k in ('mu', 'Ci'):
                if k in c:
                    c[k] = c[k].numpy()
            if 'ridge' in c:
                c['ridge'] = {v: {k: (x.numpy() if torch.is_tensor(x) else x) for k, x in r.items()}
                              for v, r in c['ridge'].items()}
            self.comps[name] = c
        fams = sorted({f for c in self.comps.values() for f in c['families']})
        self.fams = {f: Family(_load(f'fam_{f}.pt'), self.dev) for f in fams}
        srcs = {s for f in self.fams.values() for s, _ in f.sources}
        # feat_z components read one raw feature column (e.g. the OE classifier's own OoD logit)
        srcs |= {c['src'] for c in self.comps.values() if c.get('kind') == 'feat_z'}
        for c in self.comps.values():
            for p in c.get('preds', []):
                srcs.add({'pred0': 'cnn0', 'pred1': 'cnn1'}.get(p) or c['ridge'][p]['src'])
        self.sources = sorted(srcs)
        self.mask = np.load(os.path.join(MODEL_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
        self.m = torch.as_tensor(self.mask, device=self.dev, dtype=torch.float32)
        self.sel = self.m > 0
        self.amp = torch.bfloat16          # the networks' features were extracted under bf16 autocast
        if os.environ.get('FWL_AMP') == '0':      # testing hook only
            self.amp = None
        self.timing = {} if os.environ.get('FWL_TIMING') else None   # per-stage timing (debug only)
        self._build_extractors()
        self._check_amp()
        _log(f'device {self.dev} amp {self.amp} components {self.weights} families {fams} sources {self.sources}')

    def _check_amp(self):
        """bf16 autocast matters: some features (e.g. the denoiser's finite-difference divergence) were extracted
        under bf16 and fp32 changes them. Keep bf16 wherever it runs (also emulated on pre-Ampere GPUs / CPU);
        fall back to fp32 only if a bf16 forward pass fails."""
        if self.amp is None:
            return
        nets = list((self.cnns or {}).values()) + ([self.den.net] if self.den is not None else []) + \
            ([self.oe] if self.oe is not None else [])
        try:
            with torch.no_grad(), torch.autocast(self.dev.type, dtype=self.amp):
                x = torch.zeros((1, H, W), device=self.dev)
                for net in nets:
                    out = net(x, self.m)
                    assert all(torch.isfinite(o.float()).all() for o in out)
        except Exception:
            _log(f'bf16 autocast unavailable on {self.dev}; falling back to fp32:\n{traceback.format_exc()}')
            self.amp = None
            if self.den is not None:
                self.den.amp = None

    # ------------------------------------------------------------------ extractors
    def _build_extractors(self):
        s = set(self.sources)
        self.stats = self.cnns = self.den = self.nfam = self.mfam = self.dirpower = self.oe = self.dif = None
        if 'stats' in s:
            from fwl_stats import Stats
            self.stats = Stats(self.mask, self.dev, edges=_np(_load('edges_stats.pt')))
        cn = [c for c in ('cnn0', 'cnn1') if c in s]
        if cn:
            from fwl_cnn import Net as CNN
            self.cnns = {}
            for c in cn:
                st = _load(f'{c}.pt')
                net = CNN(widths=tuple(int(w) for w in st['widths'])).to(self.dev).to(memory_format=torch.channels_last)
                net.load_state_dict(st['net']); net.eval()
                for p in net.parameters():
                    p.requires_grad_(False)
                self.cnns[c] = net
        self.den_want = {k for k in ('den_r', 'den_s', 'den_b') if k in s}
        if self.den_want:
            from fwl_den import DenFeats
            st = _load('unet.pt')
            self.den = DenFeats(dict(net=st['net'], widths=[int(w) for w in st['widths']]),
                                _np(_load('edges_denoised.pt')), st['probe'].numpy(), self.mask, self.dev, amp=self.amp)
        if 'dirpower' in s:
            import fwl_dirpower
            self.dirpower = fwl_dirpower.features
            self.mask_b = torch.as_tensor(self.mask, device=self.dev, dtype=torch.bool)
        if 'oe_head' in s:
            from fwl_oe import OENet
            st = _load('oe.pt')
            self.oe = OENet(int(st['nfam'])).to(self.dev).to(memory_format=torch.channels_last)
            self.oe.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st['net'].items()})
            self.oe.eval()
            for q in self.oe.parameters():
                q.requires_grad_(False)
        if 'difa' in s:
            from fwl_dif import DifFeats, levels_for
            need = sorted({lv for f in self.fams.values() for src, cols in f.sources if src == 'difa'
                           for lv in levels_for(cols)})
            self.dif = DifFeats(_load('dunet.pt'), self.m, self.dev, need)
        self.stx_want = sorted(k for k in s if k.startswith('stx_'))
        if any(k in ('stx_nois', 'stx_tail', 'stx_bisp') for k in self.stx_want):
            from fwl_famfeats import NoiseFam
            self.nfam = NoiseFam(self.mask, self.dev, _np(_load('edges_nois.pt')))
        if any(k in ('stx_scat', 'stx_prof', 'stx_pkcl', 'stx_ipow') for k in self.stx_want):
            from fwl_famfeats import Fam
            self.mfam = Fam(self.mask, self.dev, _np(_load('edges_fam.pt')))

    def _tick(self, name, t):
        if self.timing is not None:
            if self.dev.type == 'cuda':
                torch.cuda.synchronize()
            now = time.time()
            self.timing[name] = self.timing.get(name, 0.0) + now - t
            return now
        return t

    def _extractors(self):
        """Ordered (name, function of the padded batch -> dict of per-map feature tensors)."""
        ex = []
        if self.stats is not None:
            ex.append(('stats', lambda x: {'stats': self.stats(x)}))
        if self.cnns:
            def cnn(x):
                out = {}
                for c, net in self.cnns.items():
                    if self.amp is not None:
                        with torch.autocast(self.dev.type, dtype=self.amp):
                            pred, e = net(x, self.m)
                    else:
                        pred, e = net(x, self.m)
                    out[c] = torch.cat([e.float(), pred.float()], 1)
                return out
            ex.append(('cnn', cnn))
        if self.den is not None:
            ex.append(('denoiser', lambda x: self.den(x, self.den_want)))
        if self.dirpower is not None:
            ex.append(('dirpower', lambda x: {'dirpower': self.dirpower(x, self.mask_b)}))
        if self.oe is not None:
            from fwl_oe import oe_head
            ex.append(('oe', lambda x: {'oe_head': oe_head(self.oe, x, self.m, self.amp)}))
        if self.dif is not None:
            def dif(x):
                # the difa source was extracted with TF32 matmuls (band powers); match it here only
                old = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = True
                try:
                    return {'difa': self.dif(x)}
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = old
            ex.append(('diffusion', dif))
        if self.nfam is not None:
            def nfam(x):
                out = {}
                kp = self.nfam.pad(x.float())
                Kr = torch.fft.rfft2(kp)
                if 'stx_nois' in self.stx_want:
                    out['stx_nois'] = self.nfam.nois(kp, Kr)
                if 'stx_tail' in self.stx_want:
                    out['stx_tail'] = self.nfam.tail(kp, Kr)
                if 'stx_bisp' in self.stx_want:
                    out['stx_bisp'] = self.nfam.bisp(kp)
                return out
            ex.append(('stx_nois_bisp', nfam))
        if self.mfam is not None:
            want = tuple(k[4:] for k in self.stx_want if k[4:] in ('scat', 'prof', 'pkcl', 'ipow'))
            ex.append(('stx_other', lambda x: {'stx_' + k: v for k, v in self.mfam(x, want).items()}))
        return ex

    @torch.no_grad()
    def features(self, test_data, batch):
        n_total = len(test_data)
        acc = {}
        active = self._extractors()
        t0 = time.time()
        for lo in range(0, n_total, batch):
            n = min(batch, n_total - lo)
            full = torch.zeros((batch, H, W), device=self.dev)          # fixed shape, zero padding
            full[:n, self.sel] = torch.as_tensor(np.asarray(test_data[lo:lo + n], dtype=np.float32), device=self.dev)
            for name, fn in list(active):
                t = self._tick('_', time.time())
                try:
                    if name in os.environ.get('FWL_FAIL', '').split(','):   # testing hook: injected failure
                        raise ValueError('injected failure')
                    out = fn(full)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        raise
                    out = e
                except Exception as e:                               # degrade gracefully: drop this source
                    out = e
                if isinstance(out, Exception):
                    _log(f'feature source {name} FAILED and is dropped (its components are left out): {out!r}')
                    active.remove((name, fn))
                    for k in [k for k in acc if acc[k][0] == name]:
                        del acc[k]
                    continue
                for k, v in out.items():
                    acc.setdefault(k, [name, []])[1].append(v[:n].float().cpu().numpy())
                self._tick(name, t)
            if lo == 0 or (lo // batch) % 25 == 0:
                _log(f'features {lo + n}/{n_total} maps, {time.time() - t0:.0f}s')
        return {k: np.concatenate(v[1]).astype(np.float32) for k, v in acc.items()}

    # ------------------------------------------------------------------ scoring
    def predict(self, test_data):
        t0 = time.time()
        torch.backends.cuda.matmul.allow_tf32 = False      # likelihoods were evaluated without TF32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        det = bool(self.cfg.get('deterministic', False)) if 'FWL_DET' not in os.environ else os.environ['FWL_DET'] == '1'
        torch.backends.cudnn.deterministic = det
        if det and hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                pass
        n_total = len(test_data)
        batch = int(self.cfg.get('batch', 64))
        while True:
            try:
                feats = self.features(test_data, batch)
                break
            except RuntimeError as e:
                if batch <= 8 or 'out of memory' not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                batch //= 2
                _log(f'OOM: retrying feature extraction with batch {batch}')
        _log(f'features done in {time.time() - t0:.0f}s' + (f' stages {self.timing}' if self.timing else ''))
        chunk = int(os.environ.get('FWL_CHUNK', self.cfg.get('profile_chunk', 16384)))
        if self.dev.type != 'cuda':
            chunk = min(chunk, 256)        # CPU fallback: compute-bound, keep the fixed chunk small
        th, cop, xs = {}, {}, {}
        for f, fam in self.fams.items():
            t1 = self._tick('_', time.time())
            try:
                x, t, c = fam.profile(feats, chunk)
                xs[f], th[f], cop[f] = x, t, c
            except Exception:
                _log(f'family {f} failed:\n{traceback.format_exc()}')
            self._tick('profile_' + f, t1)
        preds = {}
        if 'cnn0' in feats:
            preds['pred0'] = feats['cnn0'][:, -5:]
        if 'cnn1' in feats:
            preds['pred1'] = feats['cnn1'][:, -5:]
        for c in self.comps.values():
            for v, r in c.get('ridge', {}).items():
                if v not in preds and r['src'] in feats:
                    preds[v] = ridge_view(r, feats[r['src']])
        cross = None
        if 'small' in th and 'large' in th:
            cross = {'small@large': self.fams['small'].copula_energy_at(xs['small'], th['large'], chunk),
                     'large@small': self.fams['large'].copula_energy_at(xs['large'], th['small'], chunk)}
        thn = {f: t.cpu().numpy().astype(np.float32) for f, t in th.items()}
        if os.environ.get('FWL_KEEP'):                       # debugging hook for parity checks only
            self.last_feats, self.last_th, self.last_cop, self.last_cross = feats, thn, cop, cross
            self.last_preds = preds
        _log(f'profiles done in {time.time() - t0:.0f}s')
        zs, ws = [], []
        self.last_components = {}
        for name, w in self.weights.items():
            try:
                c = self.comps[name]
                if c.get('kind') == 'feat_z':
                    # per-map robust z of one raw feature, constants frozen on calib3 (median, 1.4826 MAD)
                    z = (feats[c['src']][:, int(c['col'])].astype(np.float64) - float(c['med'])) / float(c['scale'])
                else:
                    z = component_z(name, c, thn, cop, preds, cross)
                z = np.asarray(z, dtype=np.float64)
                if z.shape != (n_total,):
                    raise ValueError(f'bad shape {z.shape}')
                self.last_components[name] = z
                zs.append(w * z); ws.append(w)
            except Exception:
                _log(f'component {name} failed and is left out:\n{traceback.format_exc()}')
        if not ws:
            _log('no component succeeded; returning zeros')
            return np.zeros(n_total, dtype=np.float64)
        scores = np.sum(zs, 0) / np.sum(ws)
        mem = f', peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB' if self.dev.type == 'cuda' else ''
        _log(f'done in {time.time() - t0:.0f}s ({len(ws)}/{len(self.weights)} components){mem}')
        return np.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=-1e6)
