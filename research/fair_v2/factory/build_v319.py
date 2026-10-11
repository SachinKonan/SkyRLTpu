"""Package v3-19 = v3-16b (or 16c weights) with Ol taken from a low-FPR fine-tuned OE OoD head (Ol').

  python factory/build_v319.py TAG OL_WEIGHT NAME

The fine-tuned head (agents/outlier/ft/train_head.py, ft_pool/head_TAG.pt) is a second linear head on the same
frozen OE embedding: oe.pt gains ood2.1.weight/bias, fwl_oe.oe_head appends its logit as oe_head column 23, and the
Ol component reads column 23 with robust-z constants (median, 1.4826 MAD) frozen on all calib3 views of the stored
`oe` source. Oh keeps reading columns 0..22, so nothing else changes. The leaderboard file is written separately
from the package's own full GPU run.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/packages')
POOL = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier/ft_pool')
V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')

FWL_OE = '''"""Outlier-exposure CNN for inference (copied from agents/outlier/train_oe.py OENet; backbone = fwl_cnn.Net).

Source oe_head = [OoD logit (1), generator-family logits (nfam + 1), parameter predictions (5)] as
agents/outlier/extract_oe.py (first 23 columns of the `oe` source), bf16 autocast; when the checkpoint carries a
second OoD head (ood2: low-FPR fine-tuned on the frozen embedding, agents/outlier/ft/train_head.py) its logit is
appended as column 23.
"""
import torch
import torch.nn as nn

from fwl_cnn import Net


class OENet(nn.Module):
    def __init__(self, nfam, widths=(32, 64, 128, 192, 256, 256), ood2=False):
        super().__init__()
        self.base = Net(widths)
        self.ood = nn.Sequential(nn.SiLU(), nn.Linear(256, 1))
        self.fam = nn.Sequential(nn.SiLU(), nn.Linear(256, nfam + 1))
        self.ood2 = nn.Sequential(nn.SiLU(), nn.Linear(256, 1)) if ood2 else None

    def forward(self, x, m):
        e = self.base.embed(x, m)
        return self.base.head(e), self.ood(e)[:, 0], self.fam(e), e


def oe_head(net, x, m, amp):
    if amp is not None:
        with torch.autocast(x.device.type, dtype=amp):
            pred, logit, flog, e = net(x, m)
            l2 = net.ood2(e)[:, 0] if net.ood2 is not None else None
    else:
        pred, logit, flog, e = net(x, m)
        l2 = net.ood2(e)[:, 0] if net.ood2 is not None else None
    cols = [logit.float()[:, None], flog.float(), pred.float()]
    if l2 is not None:
        cols.append(l2.float()[:, None])
    return torch.cat(cols, 1)
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    tag, olw, name = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    h = torch.load(POOL / f'head_{tag}.pt', weights_only=False)
    w, b = h['w'].float().reshape(1, 256), h['b'].float().reshape(1)
    f = np.load(V3 / 'oe' / 'calib.npz')['feats']
    e = f.reshape(-1, f.shape[-1])[:, 23:279].astype(np.float32)
    raw = (e / (1 + np.exp(-e))) @ w.numpy()[0] + float(b[0])
    med = float(np.median(raw))
    scale = float(1.4826 * np.median(np.abs(raw - med)))
    dst = PKG / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(PKG / 'v3-16b', dst, ignore=shutil.ignore_patterns('__pycache__'))
    (dst / 'fwl_oe.py').write_text(FWL_OE)
    m = (dst / 'model.py').read_text()
    old = "            self.oe = OENet(int(st['nfam'])).to(self.dev).to(memory_format=torch.channels_last)\n"
    assert m.count(old) == 1
    m = m.replace(old, "            self.oe = OENet(int(st['nfam']), ood2='ood2.1.weight' in st['net']).to(self.dev).to(\n"
                       "                memory_format=torch.channels_last)\n")
    (dst / 'model.py').write_text(m)
    st = torch.load(dst / 'oe.pt', weights_only=False)
    dt = st['net']['ood.1.weight'].dtype
    st['net']['ood2.1.weight'] = w.to(dt)
    st['net']['ood2.1.bias'] = b.to(dt)
    torch.save(st, dst / 'oe.pt')
    comps = torch.load(dst / 'components.pt', weights_only=False)
    comps['Ol'] = dict(kind='feat_z', src='oe_head', col=23, med=med, scale=scale, families=[],
                       note=f'fine-tuned OoD head {tag} (step {h.get("step")}, val {h.get("val")})')
    torch.save(comps, dst / 'components.pt')
    cfg = json.loads((dst / 'config.json').read_text())
    cfg['weights']['Ol'] = olw
    (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
    print(json.dumps(dict(name=name, tag=tag, Ol_weight=olw, med=med, scale=scale, weights=cfg['weights']), indent=1))


if __name__ == '__main__':
    main()
