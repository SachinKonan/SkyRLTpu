"""CNN definition for inference (copied from cnn.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.silu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.silu(h + (x if self.skip is None else self.skip(x)))


class Net(nn.Module):
    def __init__(self, widths=(32, 64, 128, 192, 256, 256), emb=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.SiLU())
        blocks, cin = [], widths[0]
        for i, w in enumerate(widths):
            blocks += [Block(cin, w, 1 if i == 0 else 2), Block(w, w, 1)]
            cin = w
        self.blocks = nn.Sequential(*blocks)
        self.emb = nn.Sequential(nn.Linear(2 * cin, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.head = nn.Sequential(nn.SiLU(), nn.Linear(emb, 5))

    def embed(self, x, m):
        h = self.blocks(self.stem(torch.stack([x / 0.03, m.expand_as(x)], 1)))
        mm = F.adaptive_avg_pool2d(m[None, None], h.shape[-2:])[0, 0]
        w = mm / mm.sum()
        mean = (h * w).sum((2, 3))
        sd = (((h - mean[:, :, None, None]) ** 2) * w).sum((2, 3)).clamp(min=1e-8).sqrt()
        return self.emb(torch.cat([mean, sd], 1))

    def forward(self, x, m):
        e = self.embed(x, m)
        return self.head(e), e
