"""FAIR Universe Phase-2 OoD detector (code submission).

Per-map pipeline: summary statistics + CNN embeddings -> conditional-Gaussian
likelihood profiled over the 5 physical parameters -> standardized score ->
mean over detectors. Every map is scored independently (fixed-shape batches,
eval mode, no statistics across test maps).
"""
import json
import os
import sys

import numpy as np
import torch

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MODEL_DIR)
from fair_stats import Stats          # noqa: E402
from fair_cnn import Net as CNN       # noqa: E402
from fair_lik import Detector         # noqa: E402

BATCH = 64


class Model:
    def __init__(self):
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = json.load(open(os.path.join(MODEL_DIR, 'combo.json')))
        self.mask = np.load(os.path.join(MODEL_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
        self.m = torch.as_tensor(self.mask, device=self.dev, dtype=torch.float32)
        self.stats = Stats(self.mask, self.dev, edges=dict(np.load(os.path.join(MODEL_DIR, 'edges.npz'))))
        self.cnns = {}
        for src, fn in cfg['cnns'].items():
            st = torch.load(os.path.join(MODEL_DIR, fn), map_location=self.dev, weights_only=False)
            net = CNN(widths=tuple(st['widths'])).to(self.dev).to(memory_format=torch.channels_last)
            net.load_state_dict(st['net']); net.eval()
            self.cnns[src] = net
        self.dets = [Detector(torch.load(os.path.join(MODEL_DIR, fn), map_location='cpu', weights_only=False), self.dev)
                     for fn in cfg['detectors']]

    @torch.no_grad()
    def features(self, test_data):
        feats = {'stats': []}
        feats.update({src: [] for src in self.cnns})
        n_total = len(test_data)
        for lo in range(0, n_total, BATCH):
            n = min(BATCH, n_total - lo)
            full = torch.zeros((BATCH, 1424, 176), device=self.dev)
            full[:n, self.m > 0] = torch.as_tensor(np.asarray(test_data[lo:lo + n], dtype=np.float32), device=self.dev)
            feats['stats'].append(self.stats(full)[:n].cpu().numpy())
            for src, net in self.cnns.items():
                if self.dev.type == 'cuda':
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pred, e = net(full, self.m)
                else:
                    pred, e = net(full, self.m)
                feats[src].append(torch.cat([e.float(), pred.float()], 1)[:n].cpu().numpy())
        return {k: np.concatenate(v).astype(np.float32) for k, v in feats.items()}

    def predict(self, test_data):
        feats = self.features(test_data)
        z = np.stack([d.score(feats) for d in self.dets])
        scores = z.mean(0)
        return np.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=-1e6)
