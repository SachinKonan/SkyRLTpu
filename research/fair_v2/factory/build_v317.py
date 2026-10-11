"""Build package v3-17 = packages/v3-16b + two optional OE-ensemble members (weights 0 by default).

  python factory/build_v317.py

Ol3: OoD logit of three OE networks (run_all14 = the packaged oe.pt, run_all14_s1, run_phys); per net a robust z
     (median, 1.4826 MAD) frozen on all calib3 views, averaged over the nets, then robust-z'd again on calib3.
Oe3: per net the Mahalanobis distance of the 256-d pooled embedding to all fit3 views (mean + OAS covariance),
     robust z on calib3, averaged over the nets, robust-z'd again on calib3.
All constants come from the stored OE feature sources (fair-data/v3/oe, oe_s1, oe_phys), which the package recomputes
with the same networks under the same bf16 autocast. Writes packages/v3-17/ + v3-17.zip (no __pycache__) and the
reference test z of both members (from the stored test features) to packages/_verify/v3-17/ref_test_z.npz.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def compact(sd):
    """As package_full.compact: >= 2-D float weights in bf16 (lossless under the bf16 autocast), the rest as is."""
    return {k: (v.cpu().to(torch.bfloat16) if v.is_floating_point() and v.dim() >= 2 else v.cpu())
            for k, v in sd.items()}

DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data')
PKG = DATA / 'packages'
BASE, NAME = PKG / 'v3-16b', 'v3-17'
NETS = [('oe', 'run_all14', 'oe.pt'), ('oe_s1', 'run_all14_s1', 'oe_s1.pt'), ('oe_phys', 'run_phys', 'oe_phys.pt')]
EMB = slice(23, 279)


def robust(x):
    med = float(np.median(x))
    return med, float(1.4826 * np.median(np.abs(x - med)))


def oas(X):
    n, p = X.shape
    S = np.cov(X, rowvar=False, bias=True)
    mu = np.trace(S) / p
    alpha = np.mean(S ** 2)
    num, den = alpha + mu ** 2, (n + 1.0) * (alpha - mu ** 2 / p)
    sh = 1.0 if den == 0 else min(num / den, 1.0)
    C = (1.0 - sh) * S
    C.flat[::p + 1] += sh * mu
    return C, sh


def feats(src, part):
    with np.load(DATA / 'v3' / src / f'{part}.npz') as f:
        a = f['feats'].astype(np.float64)
    return a.reshape(-1, a.shape[-1])          # all views as rows


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cal = {s: feats(s, 'calib') for s, _, _ in NETS}
    test = {s: feats(s, 'test') for s, _, _ in NETS}
    # ---- Ol3
    meds, scales = zip(*[robust(cal[s][:, 0]) for s, _, _ in NETS])
    meds, scales = np.array(meds), np.array(scales)
    zm_cal = np.mean([(cal[s][:, 0] - meds[i]) / scales[i] for i, (s, _, _) in enumerate(NETS)], 0)
    med2, scale2 = robust(zm_cal)
    ol3 = dict(kind='oe3_logit', src='oe3', families=[], meds=torch.tensor(meds), scales=torch.tensor(scales),
               med2=med2, scale2=scale2)
    print('Ol3 per-net med/scale', meds.round(5).tolist(), scales.round(5).tolist(), 'second', round(med2, 5), round(scale2, 5))
    # ---- Oe3
    mus, Cis, emeds, escales, info = [], [], [], [], []
    dz_cal = []
    for s, _, _ in NETS:
        fit = feats(s, 'fit')[:, EMB]
        mu = fit.mean(0)
        C, sh = oas(fit)
        Ci = np.linalg.inv(C)
        d = lambda X: np.sqrt(np.maximum(np.einsum('nj,jk,nk->n', X - mu, Ci, X - mu), 0.0))  # noqa: E731
        dc = d(cal[s][:, EMB])
        m, sc = robust(dc)
        mus.append(mu); Cis.append(Ci); emeds.append(m); escales.append(sc)
        dz_cal.append((dc - m) / sc)
        info.append(dict(src=s, n_fit=len(fit), oas_shrinkage=round(sh, 5), med=round(m, 4), scale=round(sc, 4)))
    emed2, escale2 = robust(np.mean(dz_cal, 0))
    oe3 = dict(kind='oe3_emb', src='oe3', families=[], mus=torch.tensor(np.stack(mus)), Cis=torch.tensor(np.stack(Cis)),
               meds=torch.tensor(emeds), scales=torch.tensor(escales), med2=emed2, scale2=escale2)
    print('Oe3', info, 'second', round(emed2, 5), round(escale2, 5))
    # ---- reference test z from the stored test features (same formulas as model.py)
    zl = np.mean([(test[s][:, 0] - meds[i]) / scales[i] for i, (s, _, _) in enumerate(NETS)], 0)
    ref_ol3 = (zl - med2) / scale2
    ze = []
    for i, (s, _, _) in enumerate(NETS):
        X = test[s][:, EMB] - mus[i]
        dt = np.sqrt(np.maximum(np.einsum('nj,jk,nk->n', X, Cis[i], X), 0.0))
        ze.append((dt - emeds[i]) / escales[i])
    ref_oe3 = (np.mean(ze, 0) - emed2) / escale2
    vdir = PKG / '_verify' / NAME
    vdir.mkdir(parents=True, exist_ok=True)
    np.savez(vdir / 'ref_test_z.npz', Ol3=ref_ol3, Oe3=ref_oe3)
    # ---- package
    dst = PKG / NAME
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(BASE, dst, ignore=shutil.ignore_patterns('__pycache__'))
    for f in ('model.py', 'fwl_oe.py'):
        shutil.copy(HERE.parent / 'pack_full' / f, dst / f)
    for s, run, fn in NETS[1:]:
        st = torch.load(DATA / 'agents' / 'outlier' / run / 'ckpt.pt', map_location='cpu', weights_only=False)
        torch.save(dict(net=compact(st['net']), nfam=int(st['net']['fam.1.weight'].shape[0] - 1)), dst / fn)
    comps = torch.load(dst / 'components.pt', map_location='cpu', weights_only=True)
    comps['Ol3'], comps['Oe3'] = ol3, oe3
    torch.save(comps, dst / 'components.pt')
    cfg = json.loads((dst / 'config.json').read_text())
    cfg['weights'].update(Ol3=0.0, Oe3=0.0)
    (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
    zp = PKG / f'{NAME}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(dst.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                zf.write(f, f.relative_to(dst))
    rep = dict(package=str(zp), package_sha256=sha(zp), mb=round(zp.stat().st_size / 1e6, 1),
               weights=cfg['weights'], Ol3=dict(meds=meds.tolist(), scales=scales.tolist(), med2=med2, scale2=scale2),
               Oe3=dict(nets=info, med2=emed2, scale2=escale2))
    (vdir / 'build.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ('package', 'package_sha256', 'mb')}, indent=1))


if __name__ == '__main__':
    main()
