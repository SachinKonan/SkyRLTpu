"""Build packages v3-16a/b = v3-15d + 'Ol' (the OE classifier's own OoD logit as a direct member).

  python factory/build_v316.py

Ol: per-map robust z of oe_head column 0, (x - median) / (1.4826 MAD) with constants frozen on all calib3 maps
(both noise views) of the stored oe_head source, i.e. the same OE network the package runs. Each package is a copy
of packages/v3-15d with the updated model.py (feat_z components) and one extra entry in components.pt and
config.json; nothing else changes. Leaderboard files are written separately from the package's own GPU output.
"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data')
PKG = DATA / 'packages'
BASE = PKG / 'v3-15d'
VARIANTS = {'v3-16a': 1.5, 'v3-16b': 3.0}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    calib = np.load(DATA / 'v3' / 'oe_head' / 'calib.npz')['feats'][..., 0].reshape(-1).astype(np.float64)
    med = float(np.median(calib))
    scale = float(1.4826 * np.median(np.abs(calib - med)))
    print(f'Ol constants from {calib.size} calib3 views: median {med:.6f} scale {scale:.6f}')
    out = {}
    for name, w in VARIANTS.items():
        dst = PKG / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(BASE, dst)
        shutil.copy(HERE.parent / 'pack_full' / 'model.py', dst / 'model.py')
        meta = torch.load(dst / 'components.pt', map_location='cpu', weights_only=True)
        meta['Ol'] = dict(kind='feat_z', src='oe_head', col=0, med=med, scale=scale, families=[])
        torch.save(meta, dst / 'components.pt')
        cfg = json.loads((dst / 'config.json').read_text())
        cfg['weights']['Ol'] = w
        (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
        z = PKG / f'{name}.zip'
        z.unlink(missing_ok=True)
        with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(dst.iterdir()):
                if p.is_file() and '__pycache__' not in p.parts:
                    zf.write(p, p.name)
        out[name] = dict(Ol_weight=w, package=str(z), package_sha256=sha(z), mb=round(z.stat().st_size / 1e6, 1),
                         weights=cfg['weights'])
        print(name, out[name])
    (PKG / '_verify' / 'v3-16_build.json').write_text(json.dumps(dict(med=med, scale=scale, **out), indent=1))


if __name__ == '__main__':
    sys.exit(main())
